//! Embedding generation via external API providers.
//!
//! This module provides:
//! - [`EmbeddingProvider`] trait for abstracting embedding API calls
//! - [`OpenAiProvider`] for OpenAI-compatible embedding endpoints
//! - Batched embedding generation with configurable batch size
//! - Credential management (read from environment, never log)
//! - Error handling (auth failures, connection errors, rate limits)

mod provider;

pub(crate) use provider::embed_chunks_concurrent_with_progress_and_cancellation;
pub use provider::{
    CachedEmbeddingProvider, EmbeddingError, EmbeddingProvider, EmbeddingProviderConfig,
    OpenAiProvider, embed_chunks_concurrent, embed_chunks_concurrent_with_progress,
};

pub mod dynamic;
pub use dynamic::{DynamicProvider, create_dynamic_provider};

pub mod local_provider;

pub use local_provider::LocalEmbeddingProvider;

pub mod model2vec_provider;
pub use model2vec_provider::Model2VecProvider;

/// Test helpers for creating mock embedding providers.
#[cfg(test)]
pub(crate) mod test_helpers {
    pub use super::provider::test_helpers::MockProvider;

    use super::provider::EmbeddingProvider;
    use crate::storage::vector::VectorStore;
    use crate::types::Chunk;

    /// Embed chunks with the provider and insert the vectors into the store.
    pub(crate) async fn embed_and_insert_vectors(
        store: &VectorStore,
        provider: &impl EmbeddingProvider,
        chunks: &[Chunk],
    ) {
        let embeddings =
            super::embed_chunks_concurrent(provider, chunks, chunks.len().max(1), 4, 0)
                .await
                .unwrap();
        let batch: Vec<(&str, &[f32])> = embeddings
            .iter()
            .map(|(id, vec)| (id.as_str(), vec.as_slice()))
            .collect();
        store.insert_batch(&batch).unwrap();
    }
}

#[cfg(test)]
mod tests;
