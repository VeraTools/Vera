//! `vera repair` — repair the configured backend by re-fetching missing assets
//! or re-persisting API configuration from the current environment.

use vera_core::config::InferenceBackend;
use vera_core::local_models::LocalEmbeddingModelConfig;

use crate::commands::setup;
use crate::state;

/// The embedding model `vera repair` hands to `configure_backend`.
///
/// Deliberately the raw stored value, not `state::repaired_local_embedding_model`:
/// `configure_backend` persists whatever it is handed, so the in-memory pooling
/// repair would be laundered into `config.json` and an older Vera on the same
/// machine would then abort on every command, its `FromStr` knowing only `mean`
/// and `cls`. Unlike `vera setup`, nobody asked this command to change a
/// pooling mode.
///
/// The repair still reaches this command's runtime: `configure_backend` calls
/// `state::apply_saved_env_force`, which applies it on the way to the process
/// environment. Asset prefetching never reads `pooling`.
/// `pub(crate)` so the regression test can live beside the other stored-pooling
/// tests in `state`, which own the `VERA_HOME` lock this needs.
pub(crate) fn embedding_model_to_persist(
    effective_backend: InferenceBackend,
) -> anyhow::Result<Option<LocalEmbeddingModelConfig>> {
    if !effective_backend.is_onnx() {
        return Ok(None);
    }
    Ok(Some(
        state::load_saved_config()?
            .local_embedding_model
            .unwrap_or_default(),
    ))
}

pub fn run(backend: Option<InferenceBackend>, api: bool, json_output: bool) -> anyhow::Result<()> {
    let effective_backend = if api {
        InferenceBackend::Api
    } else if let Some(backend) = backend {
        backend
    } else if let Some(saved_backend) = state::saved_backend()? {
        saved_backend
    } else {
        vera_core::config::resolve_backend(None)
    };
    let local_embedding_model = embedding_model_to_persist(effective_backend)?;

    setup::configure_backend(
        effective_backend,
        local_embedding_model,
        None,
        json_output,
        "Vera repair complete.",
    )
}
