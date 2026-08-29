//! `vera config` — Show or set configuration values.

use anyhow::{Context, bail};

use crate::state;

/// Run the `vera config` command.
pub fn run(args: &[String], json_output: bool) -> anyhow::Result<()> {
    let mut config = state::load_runtime_config()?;

    match args.first().map(|s| s.as_str()) {
        None | Some("show") => {
            // Show full configuration.
            if json_output {
                let json = serde_json::to_string_pretty(&config)
                    .map_err(|e| anyhow::anyhow!("failed to serialize config: {e}"))?;
                println!("{json}");
            } else {
                print_human_config(&config);
            }
        }
        Some("get") => {
            let key = match args.get(1) {
                Some(k) => k,
                None => bail!(
                    "missing key for `vera config get`.\n\
                     Hint: use `vera config get <key>`, \
                     e.g., `vera config get retrieval.default_limit`"
                ),
            };
            let value = get_config_value(&config, key);
            match value {
                Some(v) => {
                    if json_output {
                        println!("{v}");
                    } else {
                        println!("{key} = {v}");
                    }
                }
                None => bail!(
                    "unknown configuration key: {key}\n\
                     Hint: run `vera config show` to see all available keys."
                ),
            }
        }
        Some("set") => {
            let key = args.get(1);
            let value = args.get(2);
            match (key, value) {
                (Some(key), Some(value)) => {
                    set_config_value(&mut config, key, value)?;
                    state::save_runtime_config(&config)?;

                    if json_output {
                        let result = serde_json::json!({
                            "key": key,
                            "value": value,
                            "status": "saved"
                        });
                        println!("{}", serde_json::to_string_pretty(&result).unwrap());
                    } else {
                        println!("Saved: {key} = {value}");
                    }
                }
                _ => bail!(
                    "missing key or value for `vera config set`.\n\
                     Hint: use `vera config set <key> <value>`, \
                     e.g., `vera config set retrieval.default_limit 20`"
                ),
            }
        }
        Some(unknown) => bail!(
            "unknown config subcommand: {unknown}\n\
             Hint: valid subcommands are: show, get, set.\n\
             Run `vera config --help` for details."
        ),
    }

    Ok(())
}

/// Print human-readable configuration.
fn print_human_config(config: &vera_core::config::VeraConfig) {
    println!("Vera Configuration");
    println!();
    println!("  Indexing:");
    println!(
        "    max_chunk_lines           {}",
        config.indexing.max_chunk_lines
    );
    println!(
        "    max_file_size_bytes       {}",
        config.indexing.max_file_size_bytes
    );
    println!(
        "    default_excludes          {:?}",
        config.indexing.default_excludes
    );
    println!(
        "    extra_excludes            {:?}",
        config.indexing.extra_excludes
    );
    println!(
        "    max_chunk_bytes           {}",
        config.indexing.max_chunk_bytes
    );
    println!();
    println!("  Retrieval:");
    println!(
        "    default_limit             {}",
        config.retrieval.default_limit
    );
    println!(
        "    max_output_chars          {}",
        config.retrieval.max_output_chars
    );
    println!("    rrf_k                     {}", config.retrieval.rrf_k);
    println!(
        "    rerank_candidates         {}",
        config.retrieval.rerank_candidates
    );
    println!(
        "    reranking_enabled         {}",
        config.retrieval.reranking_enabled
    );
    println!(
        "    max_rerank_batch          {}",
        config.retrieval.max_rerank_batch
    );
    println!(
        "    reranker_protocol         {:?}",
        config.retrieval.reranker_protocol
    );
    println!(
        "    reranker_endpoint_path    {:?}",
        config.retrieval.reranker_endpoint_path
    );
    println!(
        "    reranker_task_instruction {:?}",
        config.retrieval.reranker_task_instruction
    );
    println!(
        "    reranker_task_field       {:?}",
        config.retrieval.reranker_task_field
    );
    println!(
        "    reranker_max_doc_chars    {}",
        config.retrieval.reranker_max_doc_chars
    );
    println!(
        "    reranker_timeout_secs     {}",
        config.retrieval.reranker_timeout_secs
    );
    println!(
        "    reranker_max_retries      {}",
        config.retrieval.reranker_max_retries
    );
    println!(
        "    reranker_rate_limit_wait_secs {:?}",
        config.retrieval.reranker_rate_limit_wait_secs
    );
    println!(
        "    reranker_return_documents {:?}",
        config.retrieval.reranker_return_documents
    );
    println!();
    println!("  Embedding:");
    println!(
        "    batch_size                {}",
        config.embedding.batch_size
    );
    println!(
        "    max_concurrent_requests   {}",
        config.embedding.max_concurrent_requests
    );
    println!(
        "    max_in_flight_inputs      {}",
        config.embedding.max_in_flight_inputs
    );
    println!(
        "    timeout_secs              {}",
        config.embedding.timeout_secs
    );
    println!(
        "    max_retries               {}",
        config.embedding.max_retries
    );
    println!(
        "    max_stored_dim            {}",
        config.embedding.max_stored_dim
    );
    println!(
        "    gpu_mem_limit_mb          {}",
        config.embedding.gpu_mem_limit_mb
    );
    println!(
        "    low_vram                  {}",
        config.embedding.low_vram
    );
    println!(
        "    query_prefix              {:?}",
        config.embedding.query_prefix
    );
    println!(
        "    document_prefix           {:?}",
        config.embedding.document_prefix
    );
    println!(
        "    model_aliases             {}",
        serde_json::to_string(&config.embedding.model_aliases).unwrap_or_else(|_| "[]".to_string())
    );
}

/// Get a configuration value by dot-notation key.
pub fn get_config_value(
    config: &vera_core::config::VeraConfig,
    key: &str,
) -> Option<serde_json::Value> {
    match key {
        "indexing.max_chunk_lines" => Some(serde_json::Value::Number(
            config.indexing.max_chunk_lines.into(),
        )),
        "indexing.max_file_size_bytes" => Some(serde_json::Value::Number(
            config.indexing.max_file_size_bytes.into(),
        )),
        "indexing.default_excludes" => serde_json::to_value(&config.indexing.default_excludes).ok(),
        "indexing.extra_excludes" => serde_json::to_value(&config.indexing.extra_excludes).ok(),
        "indexing.max_chunk_bytes" => Some(serde_json::Value::Number(
            config.indexing.max_chunk_bytes.into(),
        )),
        "retrieval.default_limit" => Some(serde_json::Value::Number(
            config.retrieval.default_limit.into(),
        )),
        "retrieval.rrf_k" => serde_json::to_value(config.retrieval.rrf_k).ok(),
        "retrieval.rerank_candidates" => Some(serde_json::Value::Number(
            config.retrieval.rerank_candidates.into(),
        )),
        "retrieval.reranking_enabled" => {
            Some(serde_json::Value::Bool(config.retrieval.reranking_enabled))
        }
        "retrieval.max_output_chars" => Some(serde_json::Value::Number(
            config.retrieval.max_output_chars.into(),
        )),
        "retrieval.max_rerank_batch" => Some(serde_json::Value::Number(
            config.retrieval.max_rerank_batch.into(),
        )),
        "retrieval.reranker_protocol" | "retrieval.rerank_protocol" => {
            serde_json::to_value(config.retrieval.reranker_protocol).ok()
        }
        "retrieval.reranker_endpoint_path"
        | "retrieval.rerank_endpoint_path"
        | "retrieval.endpoint_path" => {
            serde_json::to_value(&config.retrieval.reranker_endpoint_path).ok()
        }
        "retrieval.reranker_task_instruction"
        | "retrieval.rerank_task_instruction"
        | "retrieval.task_instruction" => {
            serde_json::to_value(&config.retrieval.reranker_task_instruction).ok()
        }
        "retrieval.reranker_task_field"
        | "retrieval.rerank_task_field"
        | "retrieval.task_field" => {
            serde_json::to_value(&config.retrieval.reranker_task_field).ok()
        }
        "retrieval.reranker_max_doc_chars"
        | "retrieval.rerank_max_doc_chars"
        | "retrieval.max_rerank_doc_chars" => Some(serde_json::Value::Number(
            config.retrieval.reranker_max_doc_chars.into(),
        )),
        "retrieval.reranker_timeout_secs"
        | "retrieval.rerank_timeout_secs"
        | "retrieval.timeout_secs" => Some(serde_json::Value::Number(
            config.retrieval.reranker_timeout_secs.into(),
        )),
        "retrieval.reranker_max_retries" | "retrieval.rerank_max_retries" => Some(
            serde_json::Value::Number(config.retrieval.reranker_max_retries.into()),
        ),
        "retrieval.reranker_rate_limit_wait_secs"
        | "retrieval.rerank_rate_limit_wait_secs"
        | "retrieval.rate_limit_wait_secs" => {
            serde_json::to_value(config.retrieval.reranker_rate_limit_wait_secs).ok()
        }
        "retrieval.reranker_return_documents"
        | "retrieval.rerank_return_documents"
        | "retrieval.return_documents" => {
            serde_json::to_value(config.retrieval.reranker_return_documents).ok()
        }
        "embedding.batch_size" => Some(serde_json::Value::Number(
            config.embedding.batch_size.into(),
        )),
        "embedding.max_concurrent_requests" => Some(serde_json::Value::Number(
            config.embedding.max_concurrent_requests.into(),
        )),
        "embedding.max_in_flight_inputs" => Some(serde_json::Value::Number(
            config.embedding.max_in_flight_inputs.into(),
        )),
        "embedding.timeout_secs" => Some(serde_json::Value::Number(
            config.embedding.timeout_secs.into(),
        )),
        "embedding.max_retries" => Some(serde_json::Value::Number(
            config.embedding.max_retries.into(),
        )),
        "embedding.max_stored_dim" => Some(serde_json::Value::Number(
            config.embedding.max_stored_dim.into(),
        )),
        "embedding.gpu_mem_limit_mb" => Some(serde_json::Value::Number(
            config.embedding.gpu_mem_limit_mb.into(),
        )),
        "embedding.low_vram" => Some(serde_json::Value::Bool(config.embedding.low_vram)),
        "embedding.query_prefix" => serde_json::to_value(&config.embedding.query_prefix).ok(),
        "embedding.document_prefix" => serde_json::to_value(&config.embedding.document_prefix).ok(),
        "embedding.model_aliases" => serde_json::to_value(&config.embedding.model_aliases).ok(),
        _ => None,
    }
}

fn set_config_value(
    config: &mut vera_core::config::VeraConfig,
    key: &str,
    value: &str,
) -> anyhow::Result<()> {
    match key {
        "indexing.max_chunk_lines" => {
            config.indexing.max_chunk_lines = parse_positive(key, value)?;
        }
        "indexing.max_file_size_bytes" => {
            config.indexing.max_file_size_bytes = parse_positive(key, value)?;
        }
        "indexing.default_excludes" => {
            config.indexing.default_excludes = serde_json::from_str(value).with_context(|| {
                format!("failed to parse {key} as JSON array of strings: {value}")
            })?;
        }
        "indexing.extra_excludes" => {
            config.indexing.extra_excludes = serde_json::from_str(value).with_context(|| {
                format!("failed to parse {key} as JSON array of strings: {value}")
            })?;
        }
        "indexing.max_chunk_bytes" => {
            config.indexing.max_chunk_bytes = parse_value(key, value)?;
        }
        "retrieval.default_limit" => {
            config.retrieval.default_limit = parse_positive(key, value)?;
        }
        "retrieval.rrf_k" => {
            config.retrieval.rrf_k = parse_positive_finite(key, value)?;
        }
        "retrieval.rerank_candidates" => {
            config.retrieval.rerank_candidates = parse_positive(key, value)?;
        }
        "retrieval.reranking_enabled" => {
            config.retrieval.reranking_enabled = parse_value(key, value)?;
        }
        "retrieval.max_output_chars" => {
            config.retrieval.max_output_chars = parse_value(key, value)?;
        }
        "retrieval.max_rerank_batch" => {
            config.retrieval.max_rerank_batch = parse_value(key, value)?;
        }
        "retrieval.reranker_protocol" | "retrieval.rerank_protocol" => {
            config.retrieval.reranker_protocol = parse_optional_protocol(key, value)?;
        }
        "retrieval.reranker_endpoint_path"
        | "retrieval.rerank_endpoint_path"
        | "retrieval.endpoint_path" => {
            config.retrieval.reranker_endpoint_path = parse_optional_string(key, value)?;
            if let Some(path) = &config.retrieval.reranker_endpoint_path
                && !path.starts_with('/')
            {
                bail!("{key} must start with '/' when set");
            }
        }
        "retrieval.reranker_task_instruction"
        | "retrieval.rerank_task_instruction"
        | "retrieval.task_instruction" => {
            config.retrieval.reranker_task_instruction = parse_optional_string(key, value)?;
        }
        "retrieval.reranker_task_field"
        | "retrieval.rerank_task_field"
        | "retrieval.task_field" => {
            config.retrieval.reranker_task_field = parse_optional_string(key, value)?;
            if let Some(field) = &config.retrieval.reranker_task_field {
                const RESERVED: &[&str] = &[
                    "model",
                    "query",
                    "documents",
                    "top_n",
                    "top_k",
                    "return_documents",
                ];
                if RESERVED.contains(&field.as_str()) {
                    bail!(
                        "{key} must not be a reserved reranker field (model/query/documents/top_n/top_k/return_documents)"
                    );
                }
            }
        }
        "retrieval.reranker_max_doc_chars"
        | "retrieval.rerank_max_doc_chars"
        | "retrieval.max_rerank_doc_chars" => {
            config.retrieval.reranker_max_doc_chars = parse_value(key, value)?;
        }
        "retrieval.reranker_timeout_secs"
        | "retrieval.rerank_timeout_secs"
        | "retrieval.timeout_secs" => {
            config.retrieval.reranker_timeout_secs = parse_positive(key, value)?;
        }
        "retrieval.reranker_max_retries" | "retrieval.rerank_max_retries" => {
            config.retrieval.reranker_max_retries = parse_value(key, value)?;
        }
        "retrieval.reranker_rate_limit_wait_secs"
        | "retrieval.rerank_rate_limit_wait_secs"
        | "retrieval.rate_limit_wait_secs" => {
            config.retrieval.reranker_rate_limit_wait_secs = parse_optional_u64(key, value)?;
        }
        "retrieval.reranker_return_documents"
        | "retrieval.rerank_return_documents"
        | "retrieval.return_documents" => {
            config.retrieval.reranker_return_documents = parse_optional_bool(key, value)?;
        }
        "embedding.batch_size" => {
            config.embedding.batch_size = parse_positive(key, value)?;
        }
        "embedding.max_concurrent_requests" => {
            config.embedding.max_concurrent_requests = parse_positive(key, value)?;
        }
        "embedding.max_in_flight_inputs" => {
            config.embedding.max_in_flight_inputs = parse_positive(key, value)?;
        }
        "embedding.timeout_secs" => {
            config.embedding.timeout_secs = parse_positive(key, value)?;
        }
        "embedding.max_retries" => {
            config.embedding.max_retries = parse_value(key, value)?;
        }
        "embedding.max_stored_dim" => {
            config.embedding.max_stored_dim = parse_value(key, value)?;
        }
        "embedding.gpu_mem_limit_mb" => {
            config.embedding.gpu_mem_limit_mb = parse_value(key, value)?;
        }
        "embedding.low_vram" => {
            config.embedding.low_vram = parse_value(key, value)?;
        }
        "embedding.query_prefix" => {
            config.embedding.query_prefix = parse_optional_string(key, value)?;
        }
        "embedding.document_prefix" => {
            config.embedding.document_prefix = parse_optional_string(key, value)?;
        }
        "embedding.model_aliases" => {
            config.embedding.model_aliases = serde_json::from_str(value).with_context(|| {
                format!("failed to parse {key} as JSON array of string arrays: {value}")
            })?;
        }
        _ => bail!(
            "unknown configuration key: {key}\n\
             Hint: run `vera config show` to see all available keys."
        ),
    }

    Ok(())
}

fn parse_value<T>(key: &str, value: &str) -> anyhow::Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|e| anyhow::anyhow!("failed to parse {key}: {e}"))
}

fn parse_positive<T>(key: &str, value: &str) -> anyhow::Result<T>
where
    T: std::str::FromStr + PartialOrd + Default,
    T::Err: std::fmt::Display,
{
    let parsed = parse_value(key, value)?;
    if parsed <= T::default() {
        bail!("{key} must be greater than 0")
    }
    Ok(parsed)
}

fn parse_positive_finite(key: &str, value: &str) -> anyhow::Result<f64> {
    let parsed: f64 = parse_value(key, value)?;
    if !parsed.is_finite() || parsed <= 0.0 {
        bail!("{key} must be finite and greater than 0")
    }
    Ok(parsed)
}

fn parse_optional_string(key: &str, value: &str) -> anyhow::Result<Option<String>> {
    if value == "null" {
        return Ok(None);
    }
    if let Ok(parsed) = serde_json::from_str::<String>(value) {
        return Ok(Some(parsed));
    }
    if value.is_empty() {
        return Ok(Some(String::new()));
    }
    if value.starts_with('"') || value.ends_with('"') {
        bail!("failed to parse {key} as a string: {value}")
    }
    Ok(Some(value.to_string()))
}

fn parse_optional_protocol(
    key: &str,
    value: &str,
) -> anyhow::Result<Option<vera_core::config::RerankerProtocol>> {
    if value == "null" {
        return Ok(None);
    }
    let trimmed = value.trim_matches('"');
    trimmed
        .parse::<vera_core::config::RerankerProtocol>()
        .map(Some)
        .map_err(|e| anyhow::anyhow!("failed to parse {key}: {e}"))
}

fn parse_optional_bool(key: &str, value: &str) -> anyhow::Result<Option<bool>> {
    if value == "null" {
        return Ok(None);
    }
    value
        .parse::<bool>()
        .map(Some)
        .map_err(|e| anyhow::anyhow!("failed to parse {key}: {e}"))
}

fn parse_optional_u64(key: &str, value: &str) -> anyhow::Result<Option<u64>> {
    if value == "null" {
        return Ok(None);
    }
    let parsed: u64 = parse_value(key, value)?;
    if parsed == 0 {
        Ok(None)
    } else {
        Ok(Some(parsed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_set_rejects_zero_default_limit() {
        let mut config = vera_core::config::VeraConfig::default();
        let error = set_config_value(&mut config, "retrieval.default_limit", "0")
            .expect_err("zero results should not be accepted");
        assert!(error.to_string().contains("greater than 0"));
    }

    #[test]
    fn config_set_rejects_non_finite_rrf_k() {
        let mut config = vera_core::config::VeraConfig::default();
        let error = set_config_value(&mut config, "retrieval.rrf_k", "NaN")
            .expect_err("non-finite fusion constants should not be accepted");
        assert!(error.to_string().contains("finite"));
    }

    #[test]
    fn config_set_allows_unlimited_output_and_no_rerank_batching() {
        let mut config = vera_core::config::VeraConfig::default();
        set_config_value(&mut config, "retrieval.max_output_chars", "0").unwrap();
        set_config_value(&mut config, "retrieval.max_rerank_batch", "0").unwrap();
        assert_eq!(config.retrieval.max_output_chars, 0);
        assert_eq!(config.retrieval.max_rerank_batch, 0);
    }

    #[test]
    fn config_set_and_get_include_issue_180_keys() {
        let mut config = vera_core::config::VeraConfig::default();
        set_config_value(&mut config, "indexing.extra_excludes", r#"["vendor"]"#).unwrap();
        set_config_value(&mut config, "embedding.gpu_mem_limit_mb", "2048").unwrap();
        set_config_value(&mut config, "embedding.low_vram", "true").unwrap();
        set_config_value(&mut config, "embedding.query_prefix", "Query:").unwrap();
        set_config_value(&mut config, "embedding.document_prefix", "Passage:").unwrap();

        assert_eq!(
            get_config_value(&config, "indexing.extra_excludes"),
            Some(serde_json::json!(["vendor"]))
        );
        assert_eq!(
            get_config_value(&config, "embedding.gpu_mem_limit_mb"),
            Some(serde_json::json!(2048))
        );
        assert_eq!(
            get_config_value(&config, "embedding.query_prefix"),
            Some(serde_json::json!("Query:"))
        );
        assert_eq!(
            get_config_value(&config, "embedding.document_prefix"),
            Some(serde_json::json!("Passage:"))
        );
    }

    #[test]
    fn config_set_and_get_reranker_protocol_round_trips() {
        let mut config = vera_core::config::VeraConfig::default();
        set_config_value(&mut config, "retrieval.reranker_protocol", "generic").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_protocol"),
            Some(serde_json::json!("generic"))
        );
        set_config_value(&mut config, "retrieval.reranker_protocol", "voyage").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_protocol"),
            Some(serde_json::json!("voyage"))
        );
        set_config_value(&mut config, "retrieval.reranker_protocol", "null").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_protocol"),
            Some(serde_json::Value::Null)
        );
        // alias
        set_config_value(&mut config, "retrieval.rerank_protocol", "generic").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.rerank_protocol"),
            Some(serde_json::json!("generic"))
        );
    }

    #[test]
    fn config_set_and_get_reranker_endpoint_path_round_trips() {
        let mut config = vera_core::config::VeraConfig::default();
        set_config_value(
            &mut config,
            "retrieval.reranker_endpoint_path",
            "/v1/reranking",
        )
        .unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_endpoint_path"),
            Some(serde_json::json!("/v1/reranking"))
        );
        set_config_value(&mut config, "retrieval.reranker_endpoint_path", "null").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_endpoint_path"),
            Some(serde_json::Value::Null)
        );
        // alias
        set_config_value(&mut config, "retrieval.rerank_endpoint_path", "/v1/rerank").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.rerank_endpoint_path"),
            Some(serde_json::json!("/v1/rerank"))
        );
    }

    #[test]
    fn config_set_and_get_reranker_task_instruction_round_trips() {
        let mut config = vera_core::config::VeraConfig::default();
        set_config_value(
            &mut config,
            "retrieval.reranker_task_instruction",
            "find relevant code",
        )
        .unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_task_instruction"),
            Some(serde_json::json!("find relevant code"))
        );
        set_config_value(&mut config, "retrieval.reranker_task_instruction", "null").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_task_instruction"),
            Some(serde_json::Value::Null)
        );
        // alias
        set_config_value(
            &mut config,
            "retrieval.task_instruction",
            "alias instruction",
        )
        .unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.task_instruction"),
            Some(serde_json::json!("alias instruction"))
        );
    }

    #[test]
    fn config_set_and_get_reranker_task_field_round_trips() {
        let mut config = vera_core::config::VeraConfig::default();
        set_config_value(&mut config, "retrieval.reranker_task_field", "instruction").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_task_field"),
            Some(serde_json::json!("instruction"))
        );
        set_config_value(&mut config, "retrieval.reranker_task_field", "null").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_task_field"),
            Some(serde_json::Value::Null)
        );
        // alias
        set_config_value(&mut config, "retrieval.rerank_task_field", "custom_field").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.rerank_task_field"),
            Some(serde_json::json!("custom_field"))
        );
    }

    #[test]
    fn config_set_and_get_reranker_max_doc_chars_round_trips() {
        let mut config = vera_core::config::VeraConfig::default();
        set_config_value(&mut config, "retrieval.reranker_max_doc_chars", "1234").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_max_doc_chars"),
            Some(serde_json::json!(1234))
        );
        set_config_value(&mut config, "retrieval.reranker_max_doc_chars", "0").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_max_doc_chars"),
            Some(serde_json::json!(0))
        );
        // alias
        set_config_value(&mut config, "retrieval.max_rerank_doc_chars", "5678").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.max_rerank_doc_chars"),
            Some(serde_json::json!(5678))
        );
    }

    #[test]
    fn config_set_and_get_reranker_timeout_secs_round_trips() {
        let mut config = vera_core::config::VeraConfig::default();
        set_config_value(&mut config, "retrieval.reranker_timeout_secs", "42").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_timeout_secs"),
            Some(serde_json::json!(42))
        );
        // alias
        set_config_value(&mut config, "retrieval.rerank_timeout_secs", "99").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.rerank_timeout_secs"),
            Some(serde_json::json!(99))
        );
    }

    #[test]
    fn config_set_and_get_reranker_max_retries_round_trips() {
        let mut config = vera_core::config::VeraConfig::default();
        set_config_value(&mut config, "retrieval.reranker_max_retries", "5").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_max_retries"),
            Some(serde_json::json!(5))
        );
        set_config_value(&mut config, "retrieval.reranker_max_retries", "0").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_max_retries"),
            Some(serde_json::json!(0))
        );
        // alias
        set_config_value(&mut config, "retrieval.rerank_max_retries", "3").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.rerank_max_retries"),
            Some(serde_json::json!(3))
        );
    }

    #[test]
    fn config_set_and_get_reranker_rate_limit_wait_secs_round_trips() {
        let mut config = vera_core::config::VeraConfig::default();
        set_config_value(&mut config, "retrieval.reranker_rate_limit_wait_secs", "15").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_rate_limit_wait_secs"),
            Some(serde_json::json!(15))
        );
        set_config_value(
            &mut config,
            "retrieval.reranker_rate_limit_wait_secs",
            "null",
        )
        .unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_rate_limit_wait_secs"),
            Some(serde_json::Value::Null)
        );
        set_config_value(&mut config, "retrieval.reranker_rate_limit_wait_secs", "0").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_rate_limit_wait_secs"),
            Some(serde_json::Value::Null)
        );
        // alias
        set_config_value(&mut config, "retrieval.rerank_rate_limit_wait_secs", "20").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.rerank_rate_limit_wait_secs"),
            Some(serde_json::json!(20))
        );
    }

    #[test]
    fn config_set_and_get_reranker_return_documents_round_trips() {
        let mut config = vera_core::config::VeraConfig::default();
        set_config_value(&mut config, "retrieval.reranker_return_documents", "true").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_return_documents"),
            Some(serde_json::json!(true))
        );
        set_config_value(&mut config, "retrieval.reranker_return_documents", "false").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_return_documents"),
            Some(serde_json::json!(false))
        );
        set_config_value(&mut config, "retrieval.reranker_return_documents", "null").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.reranker_return_documents"),
            Some(serde_json::Value::Null)
        );
        // alias
        set_config_value(&mut config, "retrieval.return_documents", "true").unwrap();
        assert_eq!(
            get_config_value(&config, "retrieval.return_documents"),
            Some(serde_json::json!(true))
        );
    }
}
