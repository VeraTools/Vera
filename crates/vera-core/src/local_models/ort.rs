//! ONNX Runtime library acquisition, provider dependency inspection,
//! and archive extraction.

use crate::config::OnnxExecutionProvider;
use anyhow::{Context, Result};
use reqwest::Client;
use std::io::Write;
use std::path::{Path, PathBuf};
use tokio::fs;

use super::cuda::*;
use super::*;

/// Ensure the ONNX Runtime shared library is loaded and initialized.
///
/// Accepts an optional pre-resolved library path (from `ensure_ort_library`).
/// Falls back to system library search if no path is provided.
///
/// Safe to call multiple times — only the first call takes effect.
pub fn ensure_ort_runtime(lib_path: Option<&std::path::Path>) -> Result<()> {
    let result = ORT_INIT_RESULT.get_or_init(|| {
        let lib_name = match lib_path {
            Some(p) => p.display().to_string(),
            None => ort_lib_filename(),
        };
        match ::ort::init_from(&lib_name) {
            Ok(builder) => {
                builder.commit();
                Ok(())
            }
            Err(e) => Err(format!(
                "ONNX Runtime shared library not found.\n\
                 Run `vera setup` to auto-download it, or use API mode instead.\n\
                 Original error: {e}"
            )),
        }
    });

    match result {
        Ok(()) => Ok(()),
        Err(msg) => anyhow::bail!("{msg}"),
    }
}

/// Return whether ONNX Runtime has already been initialized successfully.
pub(crate) fn ort_runtime_initialized() -> bool {
    ORT_INIT_RESULT
        .get()
        .is_some_and(std::result::Result::is_ok)
}

/// Returns the pip package name for EPs that require pip-based installation, or None
/// for EPs that have pre-built GitHub release archives.
pub(super) fn pip_package_for_ep(ep: OnnxExecutionProvider) -> Option<&'static str> {
    match ep {
        OnnxExecutionProvider::OpenVino => Some("onnxruntime-openvino"),
        OnnxExecutionProvider::Rocm => Some("onnxruntime-rocm"),
        _ => None,
    }
}

/// Try installing ORT via pip into a managed venv under `~/.vera/venv/`.
/// Returns the lib directory where .so files were copied on success.
#[cfg(target_os = "linux")]
pub(super) async fn try_pip_install_ort(
    ep: OnnxExecutionProvider,
    lib_dir: &std::path::Path,
) -> Result<()> {
    let pkg = pip_package_for_ep(ep).context("not a pip-based EP")?;
    let vera_home = vera_home_dir()?;
    let venv_dir = vera_home.join("venv");

    // Find python3
    let python = find_python3()
        .context("python3 not found. Install Python 3.11+ to enable automatic ORT installation.")?;

    eprintln!("Installing {pkg} via pip (this may take a minute)...");

    // Create venv if it doesn't exist
    if !venv_dir.join("bin").join("python3").exists() {
        eprintln!(
            "  Creating virtual environment at {}...",
            venv_dir.display()
        );
        let status = tokio::process::Command::new(&python)
            .args(["-m", "venv", &venv_dir.to_string_lossy()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .status()
            .await
            .context("failed to create venv")?;
        if !status.success() {
            anyhow::bail!(
                "failed to create virtual environment at {}",
                venv_dir.display()
            );
        }
    }

    let venv_pip = venv_dir.join("bin").join("pip");
    let venv_python = venv_dir.join("bin").join("python3");

    // Upgrade pip quietly, then install the package
    let _ = tokio::process::Command::new(&venv_python)
        .args(["-m", "pip", "install", "--upgrade", "pip"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await;

    eprintln!("  Running: pip install {pkg}");
    let output = tokio::process::Command::new(&venv_pip)
        .args(["install", pkg])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .await
        .context("failed to run pip install")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("pip install {pkg} failed:\n{stderr}");
    }

    // Find and copy .so files from the installed package
    let site_packages = find_site_packages(&venv_dir)?;
    let capi_dir = site_packages.join("onnxruntime").join("capi");
    if !capi_dir.exists() {
        anyhow::bail!(
            "pip install succeeded but onnxruntime/capi/ not found in {}",
            site_packages.display()
        );
    }

    copy_so_files_from_dir(&capi_dir, lib_dir).await?;
    Ok(())
}

/// Try downloading a wheel directly from PyPI and extracting .so files.
#[cfg(target_os = "linux")]
pub(super) async fn try_wheel_download_ort(
    ep: OnnxExecutionProvider,
    lib_dir: &std::path::Path,
) -> Result<()> {
    let pkg = pip_package_for_ep(ep).context("not a pip-based EP")?;
    if !cfg!(target_arch = "x86_64") {
        anyhow::bail!(
            "direct PyPI wheel fallback for {pkg} is only supported on Linux x86_64; install it manually"
        );
    }
    let pypi_name = pkg.replace('-', "_");

    eprintln!("Trying direct wheel download from PyPI...");

    crate::init_tls();
    let client = Client::new();

    // Query PyPI JSON API for the latest version's wheel URLs
    let api_url = format!("https://pypi.org/pypi/{pkg}/json");
    let resp = client
        .get(&api_url)
        .header("User-Agent", "vera")
        .send()
        .await?
        .error_for_status()
        .context("failed to query PyPI")?;
    let body: serde_json::Value = resp.json().await?;

    // Find a manylinux x86_64 wheel
    let urls = body["urls"]
        .as_array()
        .context("unexpected PyPI response format")?;
    let wheel_url = urls
        .iter()
        .filter_map(|entry| {
            let filename = entry["filename"].as_str()?;
            if filename.contains("linux") && filename.contains("x86_64") {
                entry["url"].as_str().map(|u| u.to_string())
            } else {
                None
            }
        })
        .next()
        .context("no compatible Linux x86_64 wheel found on PyPI")?;

    let version = body["info"]["version"].as_str().unwrap_or("unknown");
    eprintln!("  Downloading {pypi_name} v{version} wheel...");
    eprintln!("  {wheel_url}");

    let res = client
        .get(&wheel_url)
        .header("User-Agent", "vera")
        .send()
        .await?
        .error_for_status()?;
    let bytes = res.bytes().await?;

    // Wheels are zip files; extract .so files from onnxruntime/capi/
    let lib_dir_owned = lib_dir.to_path_buf();
    tokio::task::spawn_blocking(move || extract_wheel_libs(&bytes, &lib_dir_owned)).await??;

    Ok(())
}

/// Extract all shared libraries from `onnxruntime/capi/` inside a wheel (zip).
#[cfg(target_os = "linux")]
pub(super) fn extract_wheel_libs(data: &[u8], dest_dir: &std::path::Path) -> Result<()> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)?;
    let mut extracted = 0usize;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let path = entry.name().to_string();
        if !path.starts_with("onnxruntime/capi/") {
            continue;
        }
        let filename = path.rsplit('/').next().unwrap_or("");
        if !filename.contains(".so") {
            continue;
        }
        let local_name = strip_so_version(filename);
        let dest = dest_dir.join(&local_name);
        let mut out = std::fs::File::create(&dest)?;
        std::io::copy(&mut entry, &mut out)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755))?;
        }
        create_versioned_symlink(dest_dir, filename, &local_name);
        extracted += 1;
    }

    if extracted == 0 {
        anyhow::bail!("no shared libraries found in wheel");
    }
    eprintln!("  Extracted {extracted} libraries from wheel");
    Ok(())
}

/// Find a working python3 binary.
#[cfg(target_os = "linux")]
pub(super) fn find_python3() -> Option<String> {
    for name in ["python3", "python"] {
        if std::process::Command::new(name)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            return Some(name.to_string());
        }
    }
    None
}

/// Find the site-packages directory inside a venv.
#[cfg(target_os = "linux")]
pub(super) fn find_site_packages(venv_dir: &std::path::Path) -> Result<PathBuf> {
    let lib_dir = venv_dir.join("lib");
    if !lib_dir.exists() {
        anyhow::bail!("venv lib directory not found");
    }
    for entry in std::fs::read_dir(&lib_dir)? {
        let entry = entry?;
        let sp = entry.path().join("site-packages");
        if sp.exists() {
            return Ok(sp);
        }
    }
    anyhow::bail!("site-packages not found in venv")
}

/// Copy all .so files from a directory to the target lib directory.
#[cfg(target_os = "linux")]
pub(super) async fn copy_so_files_from_dir(
    src_dir: &std::path::Path,
    dest_dir: &std::path::Path,
) -> Result<()> {
    let mut extracted = 0usize;
    let mut entries = fs::read_dir(src_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
        if !filename.contains(".so") {
            continue;
        }
        let local_name = strip_so_version(filename);
        let dest = dest_dir.join(&local_name);
        fs::copy(&path, &dest).await?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755)).await?;
        }
        create_versioned_symlink(dest_dir, filename, &local_name);
        extracted += 1;
    }
    if extracted == 0 {
        anyhow::bail!("no .so files found in {}", src_dir.display());
    }
    eprintln!("  Copied {extracted} libraries from pip package");
    Ok(())
}

/// Ensure the ONNX Runtime shared library is available locally.
///
/// Returns the path to the library. Downloads it automatically if needed.
/// Respects `ORT_DYLIB_PATH` — if set, skips auto-download.
/// GPU execution providers download a different (larger) ORT build.
///
/// For OpenVINO and ROCm (no pre-built GitHub archives), tries in order:
/// 1. `pip install` into a managed venv at `~/.vera/venv/`
/// 2. Direct wheel download from PyPI
/// 3. Bail with manual instructions
pub async fn ensure_ort_library_for_ep(ep: OnnxExecutionProvider) -> Result<PathBuf> {
    if ort_dylib_path_from_env().is_some() {
        return ensure_ort_library_for_ep_with_cuda_major(ep, None).await;
    }

    let detected_cuda_major = match ep {
        OnnxExecutionProvider::Cuda => detected_cuda_major_for_ep(ep),
        _ => None,
    };

    ensure_ort_library_for_ep_with_cuda_major(ep, detected_cuda_major).await
}

pub(super) async fn ensure_ort_library_for_ep_with_cuda_major(
    ep: OnnxExecutionProvider,
    detected_cuda_major: Option<u32>,
) -> Result<PathBuf> {
    let target_path = ort_library_path_for_ep_with_cuda_major(ep, detected_cuda_major)?;
    if ort_dylib_path_from_env().is_some() {
        return Ok(target_path);
    }

    if target_path.exists() {
        return Ok(target_path);
    }

    let lib_dir = target_path
        .parent()
        .context("failed to determine ONNX Runtime directory")?
        .to_path_buf();

    fs::create_dir_all(&lib_dir).await?;

    // OpenVINO and ROCm: pip-based install with fallback chain
    #[cfg(target_os = "linux")]
    if pip_package_for_ep(ep).is_some() {
        return ensure_ort_via_pip_chain(ep, &lib_dir, &target_path).await;
    }

    // DirectML: distributed via NuGet, not GitHub release archives.
    #[cfg(target_os = "windows")]
    if matches!(ep, OnnxExecutionProvider::DirectMl) {
        return ensure_ort_via_nuget_directml(&lib_dir, &target_path).await;
    }

    // Standard path: download from GitHub releases
    let (ext, archive_name, lib_path_in_archive, local_lib_name, ort_version) =
        ort_platform_info_with_cuda_major(ep, detected_cuda_major)?;
    let is_gpu = ep != OnnxExecutionProvider::Cpu;

    let archive_filename = if ext == "tgz" {
        format!("{archive_name}.tgz")
    } else {
        format!("{archive_name}.zip")
    };
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{ort_version}/{archive_filename}"
    );

    eprintln!("Downloading ONNX Runtime v{ort_version} ({ep})...");
    eprintln!("  {url}");

    crate::init_tls();
    let client = Client::new();
    let res = client
        .get(&url)
        .header("User-Agent", "vera")
        .send()
        .await?
        .error_for_status()?;
    let bytes = res.bytes().await?;

    let lib_dir_clone = lib_dir.clone();
    let lib_path_in_archive_clone = lib_path_in_archive.clone();

    let extract_result = tokio::task::spawn_blocking(move || -> Result<()> {
        if ext == "tgz" {
            if is_gpu {
                extract_tgz_all_libs(&bytes, &lib_dir_clone)
            } else {
                extract_tgz_single(
                    &bytes,
                    &lib_path_in_archive_clone,
                    &lib_dir_clone.join(local_lib_name),
                )
            }
        } else if is_gpu {
            extract_zip_all_libs(&bytes, &archive_name, &lib_dir_clone)
        } else {
            extract_zip(
                &bytes,
                &lib_path_in_archive_clone,
                &lib_dir_clone.join(local_lib_name),
            )
        }
    })
    .await?;

    if let Err(e) = extract_result {
        return Err(e).context("Failed to extract ONNX Runtime from archive");
    }

    eprintln!(
        "ONNX Runtime v{ort_version} installed to {}",
        lib_dir.display()
    );
    Ok(target_path)
}

/// Re-fetch the ONNX Runtime library for the selected execution provider.
///
/// `vera setup` and `vera repair` call this for CUDA so switching between CUDA
/// toolkits refreshes the downloaded ORT build instead of reusing a stale one.
pub async fn refresh_ort_library_for_ep(ep: OnnxExecutionProvider) -> Result<PathBuf> {
    if let Some(path) = ort_dylib_path_from_env() {
        return Ok(path);
    }

    let detected_cuda_major = detected_cuda_major_for_ep(ep);
    let target_path = preferred_ort_library_path_for_ep_with_cuda_major(ep, detected_cuda_major)?;
    if ep == OnnxExecutionProvider::Cpu {
        if target_path.exists() {
            fs::remove_file(&target_path).await.with_context(|| {
                format!(
                    "failed to remove stale ONNX Runtime at {}",
                    target_path.display()
                )
            })?;
        }
    } else if let Some(dir) = target_path.parent() {
        if dir.exists() {
            fs::remove_dir_all(dir).await.with_context(|| {
                format!(
                    "failed to remove stale ONNX Runtime directory {}",
                    dir.display()
                )
            })?;
        }
    }

    ensure_ort_library_for_ep_with_cuda_major(ep, detected_cuda_major).await
}

/// Download ONNX Runtime DirectML from NuGet.
///
/// DirectML builds are not published as GitHub release archives; they are only
/// distributed via NuGet. The `.nupkg` is a zip containing DLLs at
/// `runtimes/win-x64/native/`.
#[cfg(target_os = "windows")]
pub(super) async fn ensure_ort_via_nuget_directml(
    lib_dir: &std::path::Path,
    target_path: &std::path::Path,
) -> Result<PathBuf> {
    let nuget_url = format!(
        "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/{ORT_VERSION}/microsoft.ml.onnxruntime.directml.{ORT_VERSION}.nupkg"
    );

    eprintln!("Downloading ONNX Runtime v{ORT_VERSION} (directml) from NuGet...");
    eprintln!("  {nuget_url}");

    crate::init_tls();
    let client = Client::new();
    let res = client
        .get(&nuget_url)
        .header("User-Agent", "vera")
        .send()
        .await?
        .error_for_status()
        .context("failed to download DirectML NuGet package")?;
    let bytes = res.bytes().await?;

    let lib_dir_owned = lib_dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        extract_nuget_native_dlls(&bytes, "runtimes/win-x64/native/", &lib_dir_owned)
    })
    .await??;

    eprintln!(
        "ONNX Runtime v{ORT_VERSION} (directml) installed to {}",
        lib_dir.display()
    );
    Ok(target_path.to_path_buf())
}

/// Extract native DLLs from a NuGet package (zip) at the given prefix.
#[cfg(target_os = "windows")]
pub(super) fn extract_nuget_native_dlls(
    data: &[u8],
    prefix: &str,
    dest_dir: &std::path::Path,
) -> Result<()> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)?;
    let mut extracted = 0usize;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let path = entry.name().to_string();
        if !path.starts_with(prefix) {
            continue;
        }
        let filename = path.rsplit('/').next().unwrap_or("");
        if !filename.ends_with(".dll") {
            continue;
        }
        let dest = dest_dir.join(filename);
        let mut out = std::fs::File::create(&dest)?;
        std::io::copy(&mut entry, &mut out)?;
        extracted += 1;
    }

    if extracted == 0 {
        anyhow::bail!("no DLLs found in NuGet package at {prefix}");
    }
    eprintln!("  Extracted {extracted} libraries from NuGet package");
    Ok(())
}

/// Pip-based fallback chain for OpenVINO and ROCm.
#[cfg(target_os = "linux")]
pub(super) async fn ensure_ort_via_pip_chain(
    ep: OnnxExecutionProvider,
    lib_dir: &std::path::Path,
    target_path: &std::path::Path,
) -> Result<PathBuf> {
    let pkg = pip_package_for_ep(ep).unwrap();

    // Option 1: pip install into managed venv
    match try_pip_install_ort(ep, lib_dir).await {
        Ok(()) => {
            eprintln!(
                "ONNX Runtime ({ep}) installed via pip to {}",
                lib_dir.display()
            );
            return Ok(target_path.to_path_buf());
        }
        Err(e) => {
            tracing::warn!("pip install failed, trying direct wheel download: {e:#}");
            eprintln!("  pip install failed, trying direct wheel download...");
        }
    }

    // Option 2: download wheel directly from PyPI
    match try_wheel_download_ort(ep, lib_dir).await {
        Ok(()) => {
            eprintln!(
                "ONNX Runtime ({ep}) installed via wheel to {}",
                lib_dir.display()
            );
            return Ok(target_path.to_path_buf());
        }
        Err(e) => {
            tracing::warn!("wheel download failed: {e:#}");
            eprintln!("  Wheel download also failed.");
        }
    }

    // Option 3: bail with manual instructions
    anyhow::bail!(
        "Could not automatically install ONNX Runtime with {ep} support.\n\
         Install manually:\n\
         \n\
         1. pip install {pkg}\n\
         2. Locate libonnxruntime.so inside the installed package:\n\
            python3 -c \"import onnxruntime; import os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))\"\n\
         3. Set ORT_DYLIB_PATH to the full path of libonnxruntime.so\n\
         4. Run `vera setup` again"
    )
}

pub fn ort_library_path_for_ep(ep: OnnxExecutionProvider) -> Result<PathBuf> {
    if let Some(path) = ort_dylib_path_from_env() {
        return Ok(path);
    }

    let detected_cuda_major = match ep {
        OnnxExecutionProvider::Cuda => detected_cuda_major_for_ep(ep),
        _ => None,
    };

    ort_library_path_for_ep_with_cuda_major(ep, detected_cuda_major)
}

pub(super) fn preferred_ort_library_path_for_ep_in_home(
    vera_home: &Path,
    ep: OnnxExecutionProvider,
    detected_cuda_major: Option<u32>,
) -> PathBuf {
    let lib_dir = match ep {
        OnnxExecutionProvider::Cpu => vera_home.join("lib"),
        OnnxExecutionProvider::Cuda => vera_home
            .join("lib")
            .join(cuda_ort_cache_dir_name(detected_cuda_major)),
        _ => vera_home.join("lib").join(ep.to_string()),
    };

    lib_dir.join(platform_ort_lib_name())
}

pub(super) fn cached_ort_library_path_for_ep_in_home(
    vera_home: &Path,
    ep: OnnxExecutionProvider,
    detected_cuda_major: Option<u32>,
) -> PathBuf {
    let preferred_path =
        preferred_ort_library_path_for_ep_in_home(vera_home, ep, detected_cuda_major);
    if matches!(ep, OnnxExecutionProvider::Cuda)
        && detected_cuda_major.is_none()
        && !preferred_path.exists()
    {
        let cuda13_path =
            preferred_ort_library_path_for_ep_in_home(vera_home, ep, Some(CUDA_13_ORT_MIN_MAJOR));
        if cuda13_path.exists() {
            return cuda13_path;
        }
    }
    preferred_path
}

pub(super) fn preferred_ort_library_path_for_ep_with_cuda_major(
    ep: OnnxExecutionProvider,
    detected_cuda_major: Option<u32>,
) -> Result<PathBuf> {
    if let Some(path) = ort_dylib_path_from_env() {
        return Ok(path);
    }

    let vera_home = vera_home_dir()?;
    Ok(preferred_ort_library_path_for_ep_in_home(
        &vera_home,
        ep,
        detected_cuda_major,
    ))
}

pub(super) fn ort_library_path_for_ep_with_cuda_major(
    ep: OnnxExecutionProvider,
    detected_cuda_major: Option<u32>,
) -> Result<PathBuf> {
    if let Some(path) = ort_dylib_path_from_env() {
        return Ok(path);
    }

    let vera_home = vera_home_dir()?;
    Ok(cached_ort_library_path_for_ep_in_home(
        &vera_home,
        ep,
        detected_cuda_major,
    ))
}

pub fn ensure_provider_dependencies(
    ep: OnnxExecutionProvider,
    runtime_path: &std::path::Path,
) -> Result<()> {
    // CPU mode only needs the core runtime library; skip provider dependency checks.
    if matches!(ep, OnnxExecutionProvider::Cpu) {
        return Ok(());
    }

    let Some(status) = inspect_provider_dependencies(ep, runtime_path)? else {
        return Ok(());
    };

    let mut missing_libraries = status.missing_libraries;
    missing_libraries.sort();
    missing_libraries.dedup();

    if missing_libraries.is_empty() {
        return Ok(());
    }

    let backend_name = match ep {
        OnnxExecutionProvider::Cpu => "CPU",
        OnnxExecutionProvider::Cuda => "CUDA",
        OnnxExecutionProvider::Rocm => "ROCm",
        OnnxExecutionProvider::DirectMl => "DirectML",
        OnnxExecutionProvider::CoreMl => "CoreML",
        OnnxExecutionProvider::OpenVino => "OpenVINO",
    };

    let mut message =
        format!("{backend_name} backend selected, but required libraries are missing:\n");
    for library in &missing_libraries {
        message.push_str(&format!("  {library}\n"));
    }
    if let Some(hint) = dependency_hint(ep) {
        message.push_str(&hint);
        message.push('\n');
    }
    message.push_str("Run `vera doctor --probe` for details.");
    anyhow::bail!("{}", message.trim_end());
}

pub fn inspect_shared_library_deps(
    runtime_path: &std::path::Path,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    inspect_shared_library_deps_impl(runtime_path, None)
}

pub fn inspect_provider_dependencies(
    ep: OnnxExecutionProvider,
    runtime_path: &std::path::Path,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    Ok(inspect_shared_library_deps_impl(runtime_path, Some(ep))?
        .map(|status| required_dependency_status(ep, status)))
}

pub(super) fn required_dependency_status(
    ep: OnnxExecutionProvider,
    mut status: SharedLibraryDependencyStatus,
) -> SharedLibraryDependencyStatus {
    if matches!(ep, OnnxExecutionProvider::Cuda) {
        // CUDA ORT archives also ship a TensorRT provider. TensorRT is optional
        // for Vera's CUDA path, so missing TensorRT libraries must not make the
        // CUDA backend look broken.
        status
            .missing_details
            .retain(|detail| !detail.starts_with("libonnxruntime_providers_tensorrt"));
        status.missing_libraries = status
            .missing_details
            .iter()
            .filter_map(|detail| detail.split(": ").nth(1))
            .map(str::to_string)
            .collect();
        status.missing_libraries.sort();
        status.missing_libraries.dedup();
    }
    status
}

#[cfg(target_os = "linux")]
pub(super) fn inspect_shared_library_deps_impl(
    runtime_path: &std::path::Path,
    _ep: Option<OnnxExecutionProvider>,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    if !runtime_path.exists() {
        return Ok(None);
    }

    if !command_exists("ldd", &["--version"]) {
        return Ok(None);
    }

    let inspected_files = collect_runtime_libraries(runtime_path, ".so");

    let mut missing_details = Vec::new();
    let mut missing_libraries = Vec::new();

    for inspected in &inspected_files {
        let output = std::process::Command::new("ldd")
            .arg(inspected)
            .output()
            .with_context(|| format!("failed to run `ldd` on {}", inspected.display()))?;
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let file_name = inspected
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");
        for line in text.lines().filter(|line| line.contains("not found")) {
            let library = line.split("=>").next().unwrap_or(line).trim().to_string();
            missing_details.push(format!("{file_name}: {library}"));
            missing_libraries.push(library);
        }
    }

    missing_details.sort();
    missing_details.dedup();
    missing_libraries.sort();
    missing_libraries.dedup();

    Ok(Some(SharedLibraryDependencyStatus {
        inspected_files,
        missing_details,
        missing_libraries,
    }))
}

#[cfg(target_os = "macos")]
pub(super) fn inspect_shared_library_deps_impl(
    runtime_path: &std::path::Path,
    _ep: Option<OnnxExecutionProvider>,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    if !runtime_path.exists() {
        return Ok(None);
    }

    if !command_exists("otool", &["-L", runtime_path.to_string_lossy().as_ref()]) {
        return Ok(None);
    }

    let inspected_files = collect_runtime_libraries(runtime_path, ".dylib");
    let mut missing_details = Vec::new();
    let mut missing_libraries = Vec::new();

    for inspected in &inspected_files {
        let dependencies = macos_dependencies(inspected)?;
        let rpaths = macos_rpaths(inspected)?;
        let file_name = inspected
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");
        for dependency in dependencies {
            if macos_dependency_exists(&dependency, inspected, &rpaths) {
                continue;
            }
            missing_details.push(format!("{file_name}: {dependency}"));
            missing_libraries.push(dependency);
        }
    }

    missing_details.sort();
    missing_details.dedup();
    missing_libraries.sort();
    missing_libraries.dedup();

    Ok(Some(SharedLibraryDependencyStatus {
        inspected_files,
        missing_details,
        missing_libraries,
    }))
}

#[cfg(target_os = "windows")]
pub(super) fn inspect_shared_library_deps_impl(
    runtime_path: &std::path::Path,
    ep: Option<OnnxExecutionProvider>,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    if !runtime_path.exists() {
        return Ok(None);
    }

    let inspected_files = collect_runtime_libraries(runtime_path, ".dll");

    // Use dumpbin if available (Visual Studio), otherwise fall back to a
    // known-list check for provider DLLs alongside onnxruntime.dll.
    let mut missing_details = Vec::new();
    let mut missing_libraries = Vec::new();

    if command_exists("dumpbin", &["/?"]) {
        for inspected in &inspected_files {
            let output = std::process::Command::new("dumpbin")
                .args(["/dependents", inspected.to_string_lossy().as_ref()])
                .output();
            let Ok(output) = output else { continue };
            let text = String::from_utf8_lossy(&output.stdout);
            let file_name = inspected
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown");
            for line in text.lines() {
                let dep = line.trim();
                if dep.ends_with(".dll") && !dep.contains(' ') {
                    // Check if the DLL can be found by the loader: same dir or system PATH.
                    let in_same_dir = inspected
                        .parent()
                        .map(|dir| dir.join(dep).exists())
                        .unwrap_or(false);
                    if !in_same_dir && which_dll(dep).is_none() {
                        missing_details.push(format!("{file_name}: {dep}"));
                        missing_libraries.push(dep.to_string());
                    }
                }
            }
        }
    } else {
        // No dumpbin: check that the selected provider DLLs exist alongside the runtime.
        if let Some(dir) = runtime_path.parent() {
            let expected_provider_libs = if matches!(ep, Some(OnnxExecutionProvider::DirectMl)) {
                [
                    "onnxruntime_providers_shared.dll",
                    "onnxruntime_providers_dml.dll",
                ]
            } else {
                [
                    "onnxruntime_providers_shared.dll",
                    "onnxruntime_providers_cuda.dll",
                ]
            };
            for lib in &expected_provider_libs {
                if !dir.join(lib).exists() {
                    missing_details.push(format!("onnxruntime.dll: {lib}"));
                    missing_libraries.push(lib.to_string());
                }
            }
        }
    }

    missing_details.sort();
    missing_details.dedup();
    missing_libraries.sort();
    missing_libraries.dedup();

    Ok(Some(SharedLibraryDependencyStatus {
        inspected_files,
        missing_details,
        missing_libraries,
    }))
}

#[cfg(target_os = "windows")]
pub(super) fn which_dll(name: &str) -> Option<std::path::PathBuf> {
    let path_var = std::env::var("PATH").unwrap_or_default();
    for dir in path_var.split(';') {
        let candidate = std::path::Path::new(dir).join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
pub(super) fn inspect_shared_library_deps_impl(
    _: &std::path::Path,
    _ep: Option<OnnxExecutionProvider>,
) -> Result<Option<SharedLibraryDependencyStatus>> {
    Ok(None)
}

pub(super) fn collect_runtime_libraries(
    runtime_path: &std::path::Path,
    suffix: &str,
) -> Vec<PathBuf> {
    let mut inspected_files = vec![runtime_path.to_path_buf()];
    if let Some(dir) = runtime_path.parent() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if (name.starts_with("libonnxruntime") || name.starts_with("onnxruntime"))
                    && name.contains(suffix)
                    && path != runtime_path
                {
                    inspected_files.push(path);
                }
            }
        }
    }
    inspected_files.sort();
    inspected_files.dedup();
    inspected_files
}

pub(super) fn command_exists(program: &str, args: &[&str]) -> bool {
    std::process::Command::new(program)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

#[cfg(target_os = "macos")]
pub(super) fn macos_dependencies(inspected: &std::path::Path) -> Result<Vec<String>> {
    let output = std::process::Command::new("otool")
        .args(["-L", inspected.to_string_lossy().as_ref()])
        .output()
        .with_context(|| format!("failed to run `otool -L` on {}", inspected.display()))?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(text
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().next())
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect())
}

#[cfg(target_os = "macos")]
pub(super) fn macos_rpaths(inspected: &std::path::Path) -> Result<Vec<String>> {
    let output = std::process::Command::new("otool")
        .args(["-l", inspected.to_string_lossy().as_ref()])
        .output()
        .with_context(|| format!("failed to run `otool -l` on {}", inspected.display()))?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let mut rpaths = Vec::new();
    let mut in_rpath = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed == "cmd LC_RPATH" {
            in_rpath = true;
            continue;
        }
        if in_rpath && trimmed.starts_with("path ") {
            if let Some(path) = trimmed
                .strip_prefix("path ")
                .and_then(|rest| rest.split(" (offset").next())
            {
                rpaths.push(path.trim().to_string());
            }
            in_rpath = false;
        }
    }
    Ok(rpaths)
}

#[cfg(target_os = "macos")]
pub(super) fn macos_dependency_exists(
    dependency: &str,
    inspected: &std::path::Path,
    rpaths: &[String],
) -> bool {
    if dependency.starts_with("/System/Library/") || dependency.starts_with("/usr/lib/") {
        return true;
    }

    if let Some(rest) = dependency.strip_prefix("@loader_path/") {
        return inspected
            .parent()
            .map(|parent| parent.join(rest).exists())
            .unwrap_or(false);
    }

    if let Some(rest) = dependency.strip_prefix("@executable_path/") {
        let exe_path = std::env::current_exe().ok();
        let exe_exists = exe_path
            .as_deref()
            .and_then(|exe| exe.parent())
            .map(|parent| parent.join(rest).exists())
            .unwrap_or(false);
        return exe_exists
            || inspected
                .parent()
                .map(|parent| parent.join(rest).exists())
                .unwrap_or(false);
    }

    if let Some(rest) = dependency.strip_prefix("@rpath/") {
        return rpaths
            .iter()
            .map(|rpath| resolve_macos_rpath(rpath, inspected))
            .any(|candidate| candidate.join(rest).exists());
    }

    if dependency.starts_with('@') {
        return false;
    }

    std::path::Path::new(dependency).exists()
}

#[cfg(target_os = "macos")]
pub(super) fn resolve_macos_rpath(rpath: &str, inspected: &std::path::Path) -> PathBuf {
    if rpath == "@loader_path" || rpath.starts_with("@loader_path/") {
        let rest = rpath.strip_prefix("@loader_path/").unwrap_or("");
        return inspected
            .parent()
            .unwrap_or_else(|| std::path::Path::new(""))
            .join(rest);
    }

    if rpath == "@executable_path" || rpath.starts_with("@executable_path/") {
        let rest = rpath.strip_prefix("@executable_path/").unwrap_or("");
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                return parent.join(rest);
            }
        }
    }

    PathBuf::from(rpath)
}

pub(super) fn dependency_hint(ep: OnnxExecutionProvider) -> Option<String> {
    match ep {
        OnnxExecutionProvider::Cpu => None,
        OnnxExecutionProvider::Cuda => Some(match detect_cuda_major_version() {
            Some(cuda_major) => format!(
                "Install the CUDA {cuda_major} toolkit and cuDNN 9, then ensure they're on the linker path."
            ),
            None => {
                "Install the CUDA toolkit and cuDNN 9, then ensure they're on the linker path."
                    .to_string()
            }
        }),
        OnnxExecutionProvider::Rocm => {
            Some("Install the ROCm userspace libraries, then ensure they're on the linker path.".to_string())
        }
        OnnxExecutionProvider::DirectMl => {
            Some("Install the DirectML runtime and required GPU drivers.".to_string())
        }
        OnnxExecutionProvider::CoreMl => {
            Some("Verify you are running on Apple Silicon with a supported macOS version.".to_string())
        }
        OnnxExecutionProvider::OpenVino => {
            Some("Install the Intel OpenVINO runtime or compute libraries, then ensure they're on the linker path.".to_string())
        }
    }
}

pub(super) fn platform_ort_lib_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// Extract a single shared library from a tgz archive (CPU builds).
pub(super) fn extract_tgz_single(
    data: &[u8],
    entry_path: &str,
    dest: &std::path::Path,
) -> Result<()> {
    use flate2::read::GzDecoder;

    let decoder = GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        if path.to_string_lossy() == entry_path {
            write_lib_file(&mut entry, dest)?;
            return Ok(());
        }
    }

    // Suffix fallback (archive structure may vary)
    let decoder2 = GzDecoder::new(data);
    let mut archive2 = tar::Archive::new(decoder2);
    let suffix = entry_path.rsplit('/').next().unwrap_or(entry_path);

    for entry in archive2.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let path_str = path.to_string_lossy();
        if path_str.ends_with(suffix) && path_str.contains("/lib/") {
            write_lib_file(&mut entry, dest)?;
            return Ok(());
        }
    }

    anyhow::bail!("Could not find {entry_path} in ORT archive")
}

/// Extract all shared libraries from a tgz archive (GPU builds need provider libs).
pub(super) fn extract_tgz_all_libs(data: &[u8], dest_dir: &std::path::Path) -> Result<()> {
    use flate2::read::GzDecoder;

    let decoder = GzDecoder::new(data);
    let mut archive = tar::Archive::new(decoder);
    let mut extracted = 0usize;

    for entry in archive.entries()? {
        let mut entry = entry?;
        // Skip symlinks — we want the real files
        if entry.header().entry_type() != tar::EntryType::Regular {
            continue;
        }
        let path = entry.path()?;
        let path_str = path.to_string_lossy();
        if !path_str.contains("/lib/") {
            continue;
        }
        let filename = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("")
            .to_string();
        // Extract .so, .dylib, .dll files (skip .pc and other non-library files)
        let is_lib =
            filename.contains(".so") || filename.ends_with(".dylib") || filename.ends_with(".dll");
        if !is_lib {
            continue;
        }
        // Normalize versioned names: libonnxruntime.so.1.23.2 → libonnxruntime.so
        let local_name = strip_so_version(&filename);
        let dest = dest_dir.join(&local_name);
        write_lib_file(&mut entry, &dest)?;
        create_versioned_symlink(dest_dir, &filename, &local_name);
        extracted += 1;
    }

    if extracted == 0 {
        anyhow::bail!("No shared libraries found in ORT archive");
    }
    Ok(())
}

/// Strip .so version suffix: "libonnxruntime.so.1.23.2" → "libonnxruntime.so"
pub(super) fn strip_so_version(name: &str) -> String {
    if let Some(pos) = name.find(".so.") {
        name[..pos + 3].to_string()
    } else {
        name.to_string()
    }
}

/// Create a versioned symlink if the original filename differs from the
/// stripped name. For example, if `original` is "libopenvino.so.2541" and
/// `stripped` is "libopenvino.so", creates a symlink
/// `dest_dir/libopenvino.so.2541 -> libopenvino.so`.
#[cfg(unix)]
pub(super) fn create_versioned_symlink(dest_dir: &std::path::Path, original: &str, stripped: &str) {
    if original == stripped {
        return;
    }
    let link = dest_dir.join(original);
    if link.exists() || link.symlink_metadata().is_ok() {
        return;
    }
    if let Err(e) = std::os::unix::fs::symlink(stripped, &link) {
        tracing::debug!(link = %link.display(), target = stripped, error = %e, "failed to create versioned symlink");
    }
}

#[cfg(not(unix))]
pub(super) fn create_versioned_symlink(
    _dest_dir: &std::path::Path,
    _original: &str,
    _stripped: &str,
) {
}

pub(super) fn write_lib_file(
    reader: &mut impl std::io::Read,
    dest: &std::path::Path,
) -> Result<()> {
    let attempt = MODEL_DOWNLOAD_ATTEMPT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temp_path = dest.with_extension(format!("part.{}.{}", std::process::id(), attempt));

    let result: Result<()> = (|| {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        let mut out = options.open(&temp_path)?;
        std::io::copy(reader, &mut out)?;
        out.flush()?;
        out.sync_all()?;
        drop(out);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&temp_path, std::fs::Permissions::from_mode(0o755))?;
        }

        #[cfg(windows)]
        if dest.exists() {
            std::fs::remove_file(dest)?;
        }
        std::fs::rename(&temp_path, dest)?;
        Ok(())
    })();

    if let Err(error) = result {
        let _ = std::fs::remove_file(&temp_path);
        return Err(error);
    }
    Ok(())
}

pub(super) fn extract_zip(data: &[u8], entry_path: &str, dest: &std::path::Path) -> Result<()> {
    // Windows .zip extraction using tar crate's zip support is not available,
    // so we use a minimal zip reader via the `zip` crate. Since we only compile
    // this path on Windows and want to avoid an extra dependency, we fall back
    // to extracting via the system `tar` command or manual parsing.
    //
    // For simplicity, use the zip crate. But since it's not added as a dep,
    // we'll use a raw approach: download the tgz variant if available, or
    // shell out to PowerShell on Windows.
    #[cfg(target_os = "windows")]
    {
        let temp_zip = dest.with_extension("zip");
        std::fs::write(&temp_zip, data)?;
        let output = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Add-Type -AssemblyName System.IO.Compression.FileSystem; \
                     $zip = [System.IO.Compression.ZipFile]::OpenRead('{}'); \
                     $entry = $zip.Entries | Where-Object {{ $_.FullName -eq '{}' }}; \
                     if ($entry) {{ \
                         $stream = $entry.Open(); \
                         $file = [System.IO.File]::Create('{}'); \
                         $stream.CopyTo($file); \
                         $file.Close(); $stream.Close(); \
                     }}; $zip.Dispose()",
                    temp_zip.display(),
                    entry_path.replace('/', "\\"),
                    dest.display()
                ),
            ])
            .output()?;
        let _ = std::fs::remove_file(&temp_zip);
        if !output.status.success() {
            anyhow::bail!(
                "Failed to extract ORT zip: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        if !dest.exists() {
            // Try with forward slashes
            let temp_zip2 = dest.with_extension("zip2");
            std::fs::write(&temp_zip2, data)?;
            let output2 = std::process::Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-Command",
                    &format!(
                        "Add-Type -AssemblyName System.IO.Compression.FileSystem; \
                         $zip = [System.IO.Compression.ZipFile]::OpenRead('{}'); \
                         $entry = $zip.Entries | Where-Object {{ $_.FullName -eq '{}' }}; \
                         if ($entry) {{ \
                             $stream = $entry.Open(); \
                             $file = [System.IO.File]::Create('{}'); \
                             $stream.CopyTo($file); \
                             $file.Close(); $stream.Close(); \
                         }}; $zip.Dispose()",
                        temp_zip2.display(),
                        entry_path,
                        dest.display()
                    ),
                ])
                .output()?;
            let _ = std::fs::remove_file(&temp_zip2);
            if !output2.status.success() || !dest.exists() {
                anyhow::bail!("Could not find {entry_path} in ORT zip archive");
            }
        }
        Ok(())
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = (data, entry_path, dest);
        anyhow::bail!("ZIP extraction not expected on this platform")
    }
}

/// Extract all DLL files from a zip archive's lib/ directory (GPU builds need provider libs).
pub(super) fn extract_zip_all_libs(
    data: &[u8],
    _archive_name: &str,
    dest_dir: &std::path::Path,
) -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        let temp_zip = dest_dir.join("_ort_download.zip");
        std::fs::write(&temp_zip, data)?;
        // Extract all .dll files from any lib/ subdirectory inside the archive.
        // We match loosely because the CUDA 13 archive filename contains `_cuda13`
        // but the internal directory does not.
        let script = format!(
            "Add-Type -AssemblyName System.IO.Compression.FileSystem; \
             $zip = [System.IO.Compression.ZipFile]::OpenRead('{zip}'); \
             $count = 0; \
             foreach ($entry in $zip.Entries) {{ \
                 if ($entry.FullName -match '/lib/[^/]+\\.dll$' -and $entry.Length -gt 0) {{ \
                     $name = $entry.Name; \
                     $dest = Join-Path '{dest}' $name; \
                     $stream = $entry.Open(); \
                     $file = [System.IO.File]::Create($dest); \
                     $stream.CopyTo($file); \
                     $file.Close(); $stream.Close(); \
                     $count++; \
                 }} \
             }}; \
             $zip.Dispose(); \
             if ($count -eq 0) {{ exit 1 }}",
            zip = temp_zip.display(),
            dest = dest_dir.display(),
        );
        let output = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", &script])
            .output()?;
        let _ = std::fs::remove_file(&temp_zip);
        if !output.status.success() {
            anyhow::bail!(
                "Failed to extract DLLs from ORT zip: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Ok(())
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = (data, _archive_name, dest_dir);
        anyhow::bail!("ZIP extraction not expected on this platform")
    }
}

/// Wrap an ort error with a user-friendly message suggesting alternatives.
pub fn wrap_ort_error(e: impl std::fmt::Display) -> String {
    let err_msg = e.to_string();
    let lower = err_msg.to_ascii_lowercase();
    if lower.contains("required libraries are missing") {
        return err_msg;
    }
    if lower.contains("libonnxruntime")
        || lower.contains("onnxruntime.dll")
        || lower.contains("libonnxruntime.so")
        || lower.contains("libonnxruntime.dylib")
        || lower.contains("loadlibrary")
        || lower.contains("shared library")
        || lower.contains("specified module could not be found")
        || lower.contains(".dll")
        || lower.contains(".dylib")
        || lower.contains(".so")
    {
        format!(
            "ONNX Runtime shared library not found.\n\
             Run `vera setup` to auto-download it, or use API mode instead.\n\
             Original error: {err_msg}"
        )
    } else {
        format!(
            "Failed to initialize ONNX session: {err_msg}\nRun `vera doctor --probe` for details."
        )
    }
}

#[cfg(test)]
mod finding_tests {
    use super::*;
    use std::io::{self, Read};

    struct FailingReader {
        yielded_data: bool,
    }

    impl Read for FailingReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if self.yielded_data {
                return Err(io::Error::other("synthetic extraction failure"));
            }
            self.yielded_data = true;
            buffer[..4].copy_from_slice(b"part");
            Ok(4)
        }
    }

    #[test]
    fn failed_library_write_leaves_destination_unchanged() {
        let temp_dir = tempfile::tempdir().unwrap();
        let destination = temp_dir.path().join("libonnxruntime.so");
        std::fs::write(&destination, b"existing").unwrap();

        let result = write_lib_file(
            &mut FailingReader {
                yielded_data: false,
            },
            &destination,
        );

        assert!(result.is_err());
        assert_eq!(std::fs::read(&destination).unwrap(), b"existing");
        assert_eq!(std::fs::read_dir(temp_dir.path()).unwrap().count(), 1);
    }
}
