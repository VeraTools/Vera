//! `vera serve` — Start the Vera HTTP API server.

use anyhow::Result;
use vera_core::config::{InferenceBackend, VeraConfig};

/// Run the `vera serve` command.
pub fn run(
    host: &str,
    port: u16,
    api_key: Option<String>,
    backend: InferenceBackend,
    config: VeraConfig,
    idle_timeout_secs: i64,
) -> Result<()> {
    let cache_mode = vera_serve::CacheMode::from_idle_timeout_secs(idle_timeout_secs);
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(vera_serve::run_server(
        config, backend, api_key, host, port, cache_mode,
    ))
}
