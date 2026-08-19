use super::assets::*;
use super::cuda::*;
use super::ort::*;
use super::*;
use crate::config::OnnxExecutionProvider;
use std::io::Write;
use std::net::TcpListener;

#[test]
fn normalize_huggingface_repo_accepts_repo_ids_and_urls() {
    assert_eq!(
        normalize_huggingface_repo("Zenabius/CodeRankEmbed-onnx").unwrap(),
        "Zenabius/CodeRankEmbed-onnx"
    );
    assert_eq!(
        normalize_huggingface_repo("https://huggingface.co/Zenabius/CodeRankEmbed-onnx").unwrap(),
        "Zenabius/CodeRankEmbed-onnx"
    );
}

#[test]
fn coderankembed_preset_sets_required_query_prefix() {
    let config = LocalEmbeddingModelConfig::coderankembed();
    assert_eq!(
        config.query_text("find router code"),
        "Represent this query for searching relevant code: find router code"
    );
}

#[test]
fn coderankembed_repo_uses_coderank_defaults() {
    let config = LocalEmbeddingModelConfig::from_huggingface_repo("Zenabius/CodeRankEmbed-onnx");
    assert_eq!(config.pooling, LocalEmbeddingPooling::Cls);
    assert!(config.onnx_data_file.is_none());
}

#[test]
fn custom_sources_do_not_inherit_coderank_pooling_or_query_prefix() {
    // `default()` is CodeRankEmbed, but an arbitrary repo or a user-supplied
    // ONNX directory has no reason to inherit its CLS pooling and mandatory
    // query prefix — that would silently change their embeddings.
    for config in [
        LocalEmbeddingModelConfig::from_huggingface_repo("acme/custom-embeddings"),
        LocalEmbeddingModelConfig::from_directory(PathBuf::from("/models/custom")),
    ] {
        assert_eq!(config.pooling, LocalEmbeddingPooling::Mean);
        assert_eq!(config.query_prefix, None);
        assert_eq!(config.query_text("find router code"), "find router code");
    }
}

#[test]
fn parse_cuda_major_version_handles_paths_and_versions() {
    assert_eq!(
        parse_cuda_major_version(r#"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"#),
        Some(13)
    );
    assert_eq!(parse_cuda_major_version("/opt/cuda-13.2"), Some(13));
    assert_eq!(parse_cuda_major_version("12.9"), Some(12));
    assert_eq!(parse_cuda_major_version("12_6"), Some(12));
    assert_eq!(parse_cuda_major_version("CUDA Version 13.2.1"), Some(13));
}

#[test]
fn parse_cuda_major_from_cuda_version_metadata_supports_json_and_text() {
    assert_eq!(
        parse_cuda_major_from_cuda_version_metadata(
            r#"{"cuda":{"name":"CUDA SDK","version":"13.2.1"}}"#
        ),
        Some(13)
    );
    assert_eq!(
        parse_cuda_major_from_cuda_version_metadata("CUDA Version 12.8.0"),
        Some(12)
    );
}

#[test]
fn detect_cuda_major_from_cuda_path_env_vars_ignores_unrelated_values() {
    let vars = [
        ("SHLVL", "1"),
        ("TERM_PROGRAM_VERSION", "3.5.1"),
        ("PATH", "/usr/bin"),
    ];
    assert_eq!(detect_cuda_major_from_cuda_path_env_vars(vars), None);
}

#[test]
fn detect_cuda_major_from_cuda_path_env_vars_prefers_versioned_cuda_vars() {
    let vars = [
        ("SHLVL", "1"),
        ("CUDA_PATH_V13_2", "/opt/cuda"),
        ("TERM_PROGRAM_VERSION", "3.5.1"),
    ];
    assert_eq!(detect_cuda_major_from_cuda_path_env_vars(vars), Some(13));
}

#[test]
fn detect_cuda_major_from_cuda_path_value_reads_cuda_version_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        temp_dir.path().join("version.json"),
        r#"{"cuda":{"version":"13.2.1"}}"#,
    )
    .unwrap();

    assert_eq!(
        detect_cuda_major_from_cuda_path_value(temp_dir.path().to_string_lossy().as_ref()),
        Some(13)
    );
}

#[test]
fn parse_cuda_major_from_runtime_library_entry_handles_ldconfig_output() {
    assert_eq!(
        parse_cuda_major_from_runtime_library_entry(
            "libcudart.so.13 (libc6,x86-64) => /opt/cuda/lib64/libcudart.so.13"
        ),
        Some(13)
    );
    assert_eq!(
        parse_cuda_major_from_runtime_library_entry("/opt/cuda/lib64/libcublasLt.so.13"),
        Some(13)
    );
    assert_eq!(
        parse_cuda_major_from_runtime_library_entry("libcufft.so.12"),
        None
    );
}

#[test]
fn detect_cuda_major_from_library_entries_prefers_supported_runtime_libs() {
    let entries = [
        "libcufft.so.12 (libc6,x86-64) => /opt/cuda/lib64/libcufft.so.12",
        "libcublas.so.13 (libc6,x86-64) => /opt/cuda/lib64/libcublas.so.13",
        "libcublasLt.so.13 (libc6,x86-64) => /opt/cuda/lib64/libcublasLt.so.13",
        "libcudart.so.13 (libc6,x86-64) => /opt/cuda/lib64/libcudart.so.13",
    ];
    assert_eq!(detect_cuda_major_from_library_entries(entries), Some(13));
}

#[cfg(target_os = "linux")]
#[test]
fn detect_cuda_major_from_library_dir_groups_respects_group_order() {
    let temp_dir = tempfile::tempdir().unwrap();
    let cuda12_dir = temp_dir.path().join("cuda12");
    let cuda13_dir = temp_dir.path().join("cuda13");
    std::fs::create_dir_all(&cuda12_dir).unwrap();
    std::fs::create_dir_all(&cuda13_dir).unwrap();
    std::fs::write(cuda12_dir.join("libcudart.so.12"), b"").unwrap();
    std::fs::write(cuda13_dir.join("libcudart.so.13"), b"").unwrap();

    let groups = vec![vec![cuda12_dir], vec![cuda13_dir]];
    assert_eq!(detect_cuda_major_from_library_dir_groups(&groups), Some(12));
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn detect_cuda_major_from_ldconfig_entries_filters_non_native_arch_entries() {
    let entries = [
        "libcudart.so.13 (libc6,AArch64) => /usr/lib/aarch64-linux-gnu/libcudart.so.13",
        "libcudart.so.12 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudart.so.12",
    ];
    assert_eq!(detect_cuda_major_from_ldconfig_entries(entries), Some(12));
}

#[test]
fn cuda_ort_cache_dir_name_separates_cuda13_runtime() {
    assert_eq!(cuda_ort_cache_dir_name(None), "cuda");
    assert_eq!(cuda_ort_cache_dir_name(Some(12)), "cuda");
    assert_eq!(cuda_ort_cache_dir_name(Some(13)), "cuda13");
    assert_eq!(cuda_ort_cache_dir_name(Some(14)), "cuda13");
}

#[test]
fn cached_ort_library_path_reuses_cuda13_cache_when_detection_is_unknown() {
    let temp_dir = tempfile::tempdir().unwrap();
    let expected_path = temp_dir
        .path()
        .join("lib")
        .join("cuda13")
        .join(platform_ort_lib_name());
    std::fs::create_dir_all(expected_path.parent().unwrap()).unwrap();
    std::fs::write(&expected_path, b"").unwrap();

    let resolved =
        cached_ort_library_path_for_ep_in_home(temp_dir.path(), OnnxExecutionProvider::Cuda, None);
    assert_eq!(resolved, expected_path);
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn ort_platform_info_uses_cuda13_archive_and_plain_internal_gpu_dir() {
    let (_, archive_name, internal_path, _, _) =
        ort_platform_info_with_cuda_major(OnnxExecutionProvider::Cuda, Some(13)).unwrap();
    assert!(archive_name.contains("-gpu_cuda13-"));
    assert!(internal_path.contains("onnxruntime-linux-x64-gpu-"));
    assert!(!internal_path.contains("_cuda13"));
}

#[test]
fn wrap_ort_error_keeps_model_load_failures_specific() {
    let message = wrap_ort_error("failed to load embedding model C:\\Users\\me\\.vera\\model.onnx");
    assert!(message.contains("Failed to initialize ONNX session"));
    assert!(!message.contains("shared library not found"));
}

#[test]
fn wrap_ort_error_still_flags_missing_dlls() {
    let message = wrap_ort_error(
        "LoadLibrary failed for onnxruntime.dll: The specified module could not be found",
    );
    assert!(message.contains("ONNX Runtime shared library not found"));
}

#[test]
fn onnx_integrity_check_rejects_truncated_and_garbage_files() {
    let temp_dir = tempfile::tempdir().unwrap();
    let truncated = temp_dir.path().join("truncated.onnx");
    let garbage = temp_dir.path().join("garbage.onnx");
    std::fs::write(&truncated, [0x08]).unwrap();
    std::fs::write(&garbage, b"not an onnx model").unwrap();

    assert!(validate_file(&truncated, LocalModelAssetKind::Onnx).is_err());
    assert!(validate_file(&garbage, LocalModelAssetKind::Onnx).is_err());
}

#[test]
fn onnx_integrity_check_scans_fields_before_ir_version() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("model.onnx");
    // ModelProto.producer (field 2) may precede ir_version (field 1).
    std::fs::write(
        &path,
        [
            0x12, 0x08, b'p', b'r', b'o', b'd', b'u', b'c', b'e', b'r', 0x08, 0x01,
        ],
    )
    .unwrap();

    assert!(validate_file(&path, LocalModelAssetKind::Onnx).is_ok());
}

#[test]
fn onnx_integrity_check_accepts_multibyte_ir_version() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("model.onnx");
    std::fs::write(&path, [0x08, 0xac, 0x02]).unwrap();

    assert!(validate_file(&path, LocalModelAssetKind::Onnx).is_ok());
}

#[test]
fn onnx_integrity_check_rejects_zero_and_truncated_fields() {
    let temp_dir = tempfile::tempdir().unwrap();
    let zero = temp_dir.path().join("zero.onnx");
    let truncated = temp_dir.path().join("truncated-field.onnx");
    std::fs::write(&zero, [0x08, 0x00]).unwrap();
    std::fs::write(&truncated, [0x12, 0x03, b'p']).unwrap();

    assert!(validate_file(&zero, LocalModelAssetKind::Onnx).is_err());
    assert!(validate_file(&truncated, LocalModelAssetKind::Onnx).is_err());
}

#[test]
fn explicit_onnx_asset_kind_validates_custom_filename() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("model.bin");
    std::fs::write(&path, b"not an onnx model").unwrap();

    assert!(validate_file(&path, LocalModelAssetKind::Onnx).is_err());
}

#[test]
fn onnx_integrity_check_accepts_a_valid_model_proto_header() {
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("model.onnx");
    std::fs::write(&path, [0x08, 0x01]).unwrap();

    let status = inspect_asset("embedding-onnx", path, LocalModelAssetKind::Onnx);
    assert!(status.exists);
    assert_eq!(status.state, LocalModelAssetState::Valid);
    assert!(status.detail.is_none());
}

#[test]
fn inspect_asset_reports_missing_files_distinctly() {
    let temp_dir = tempfile::tempdir().unwrap();
    let status = inspect_asset(
        "embedding-onnx",
        temp_dir.path().join("missing.onnx"),
        LocalModelAssetKind::Onnx,
    );

    assert!(!status.exists);
    assert_eq!(status.state, LocalModelAssetState::Missing);
    assert_eq!(status.detail.as_deref(), Some("file not found"));
}

#[test]
fn required_dependency_status_ignores_optional_cuda_tensorrt_libraries() {
    let status = SharedLibraryDependencyStatus {
        inspected_files: Vec::new(),
        missing_details: vec![
            "libonnxruntime_providers_tensorrt.so: libnvinfer.so.10".to_string(),
            "libonnxruntime_providers_tensorrt.so: libnvonnxparser.so.10".to_string(),
            "libonnxruntime_providers_cuda.so: libcudnn.so.9".to_string(),
        ],
        missing_libraries: vec![
            "libcudnn.so.9".to_string(),
            "libnvinfer.so.10".to_string(),
            "libnvonnxparser.so.10".to_string(),
        ],
    };

    let filtered = required_dependency_status(OnnxExecutionProvider::Cuda, status);

    assert_eq!(
        filtered.missing_details,
        vec!["libonnxruntime_providers_cuda.so: libcudnn.so.9"]
    );
    assert_eq!(filtered.missing_libraries, vec!["libcudnn.so.9"]);
}

#[tokio::test]
async fn test_download_failure_cleanup() {
    let temp_dir = tempfile::tempdir().unwrap();
    let home = temp_dir.path().join(".vera");

    let listener = match TcpListener::bind("127.0.0.1:0") {
        Ok(listener) => listener,
        Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => return,
        Err(err) => panic!("failed to bind test listener: {err}"),
    };
    let port = listener.local_addr().unwrap().port();

    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            // Return a valid HTTP response header but truncate the body
            let response = "HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\nPartialData";
            let _ = stream.write_all(response.as_bytes());
            // abruptly close the connection
        }
    });

    let base_url = format!("http://127.0.0.1:{}", port);

    let res = ensure_model_file_impl(
        "test-repo",
        "test-file.bin",
        LocalModelAssetKind::Other,
        &base_url,
        Some(&home),
    )
    .await;

    assert!(res.is_err(), "Download should fail due to truncated stream");

    let target_dir = home.join("models").join("test-repo");
    let has_part_file = std::fs::read_dir(&target_dir)
        .unwrap()
        .filter_map(Result::ok)
        .any(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("test-file.part.")
        });
    assert!(
        !has_part_file,
        "Partial file should be cleaned up on failure"
    );
}

#[tokio::test]
async fn failed_onnx_download_preserves_invalid_cached_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    let home = temp_dir.path().join(".vera");
    let target = home.join("models").join("test-repo").join("model.onnx");
    std::fs::create_dir_all(target.parent().unwrap()).unwrap();
    let cached = b"cached-corrupt-model";
    std::fs::write(&target, cached).unwrap();

    let listener = match TcpListener::bind("127.0.0.1:0") {
        Ok(listener) => listener,
        Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => return,
        Err(err) => panic!("failed to bind test listener: {err}"),
    };
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let response = b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n";
            let _ = stream.write_all(response);
        }
    });

    let base_url = format!("http://127.0.0.1:{port}");
    let result = ensure_model_file_impl(
        "test-repo",
        "model.onnx",
        LocalModelAssetKind::Onnx,
        &base_url,
        Some(&home),
    )
    .await;

    assert!(result.is_err());
    assert_eq!(std::fs::read(&target).unwrap(), cached);
}

#[tokio::test]
async fn successful_onnx_download_replaces_invalid_cached_file() {
    let temp_dir = tempfile::tempdir().unwrap();
    let home = temp_dir.path().join(".vera");
    let target = home.join("models").join("test-repo").join("model.bin");
    std::fs::create_dir_all(target.parent().unwrap()).unwrap();
    std::fs::write(&target, b"cached-corrupt-model").unwrap();

    let listener = match TcpListener::bind("127.0.0.1:0") {
        Ok(listener) => listener,
        Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => return,
        Err(err) => panic!("failed to bind test listener: {err}"),
    };
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut request = [0u8; 4096];
            let _ = std::io::Read::read(&mut stream, &mut request);
            let response =
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n\x08\x01";
            let _ = stream.write_all(response);
            let _ = stream.flush();
            let _ = stream.shutdown(std::net::Shutdown::Write);
        }
    });

    let base_url = format!("http://127.0.0.1:{port}");
    let result = ensure_model_file_impl(
        "test-repo",
        "model.bin",
        LocalModelAssetKind::Onnx,
        &base_url,
        Some(&home),
    )
    .await
    .unwrap();

    assert_eq!(result, target);
    assert_eq!(std::fs::read(&target).unwrap(), [0x08, 0x01]);
}

#[cfg(target_os = "macos")]
#[test]
fn resolve_macos_rpath_loader_path_without_slash() {
    let inspected = std::path::Path::new("/tmp/vera/lib/libonnxruntime.dylib");
    assert_eq!(
        resolve_macos_rpath("@loader_path", inspected),
        std::path::Path::new("/tmp/vera/lib")
    );
}

#[cfg(target_os = "macos")]
#[test]
fn resolve_macos_rpath_loader_path_with_slash() {
    let inspected = std::path::Path::new("/tmp/vera/lib/libonnxruntime.dylib");
    assert_eq!(
        resolve_macos_rpath("@loader_path/subdir", inspected),
        std::path::Path::new("/tmp/vera/lib/subdir")
    );
}

#[cfg(target_os = "macos")]
#[test]
fn resolve_macos_rpath_executable_path_without_slash() {
    let inspected = std::path::Path::new("/tmp/vera/lib/libonnxruntime.dylib");
    let expected = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
        .unwrap_or_else(|| std::path::PathBuf::from("@executable_path"));
    assert_eq!(resolve_macos_rpath("@executable_path", inspected), expected);
}

#[cfg(target_os = "macos")]
#[test]
fn resolve_macos_rpath_executable_path_with_slash() {
    let inspected = std::path::Path::new("/tmp/vera/lib/libonnxruntime.dylib");
    let expected = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.join("subdir")))
        .unwrap_or_else(|| std::path::PathBuf::from("@executable_path/subdir"));
    assert_eq!(
        resolve_macos_rpath("@executable_path/subdir", inspected),
        expected
    );
}
