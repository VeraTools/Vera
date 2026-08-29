use tokio_util::sync::CancellationToken as AsyncCancellationToken;

/// Typed cancellation marker.
///
/// Produced by [`CancellationToken::check`] and by embedding cancellation.
/// Using a typed error lets callers distinguish cancellation from other
/// failures without brittle substring checks on the error message.
#[derive(Debug, Clone, thiserror::Error)]
#[error("operation cancelled")]
pub struct Cancelled;

/// Cooperative cancellation shared across indexing pipeline stages.
///
/// Clones observe the same permanent cancellation signal. Indexing stages use
/// the async token to stop active provider work before publishing artifacts.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    inner: AsyncCancellationToken,
}

impl CancellationToken {
    /// Create a token in the active state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancel this token and every clone derived from it.
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    /// Return whether this token or one of its clones has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    /// Borrow the async signal used by provider requests.
    pub(crate) fn as_async_token(&self) -> &AsyncCancellationToken {
        &self.inner
    }

    pub(crate) fn check(&self) -> anyhow::Result<()> {
        if self.is_cancelled() {
            return Err(Cancelled.into());
        }
        Ok(())
    }
}

/// Whether `err` represents a cooperative cancellation.
///
/// Checks the error chain for the typed [`Cancelled`] marker or for
/// [`crate::embedding::EmbeddingError::Cancelled`], including
/// through `anyhow::Context` wrappers. It intentionally does NOT match
/// arbitrary messages that merely contain the word "cancel" – callers
/// previously used `err.to_string().contains("cancel")`, which
/// misclassified unrelated failures.
pub fn is_cancel_error(err: &anyhow::Error) -> bool {
    // Direct or via chain.
    for cause in err.chain() {
        if cause.is::<Cancelled>() {
            return true;
        }
        if let Some(embed) = cause.downcast_ref::<crate::embedding::EmbeddingError>()
            && matches!(embed, crate::embedding::EmbeddingError::Cancelled)
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_cancel_error_distinguishes_typed_from_substring() {
        // Typed cancellation via token check.
        let typed: anyhow::Error = Cancelled.into();
        assert!(
            is_cancel_error(&typed),
            "typed Cancelled must be recognised"
        );
        assert!(
            typed.to_string().to_lowercase().contains("cancel"),
            "sanity: typed message contains cancel"
        );

        // Typed cancellation via embedding error.
        let embed_typed: anyhow::Error = crate::embedding::EmbeddingError::Cancelled.into();
        assert!(
            is_cancel_error(&embed_typed),
            "EmbeddingError::Cancelled must be recognised"
        );

        // Wrapped via anyhow context must still be recognised.
        let wrapped = anyhow::Error::from(crate::embedding::EmbeddingError::Cancelled)
            .context("embedding generation failed");
        assert!(
            is_cancel_error(&wrapped),
            "wrapped Cancelled must be recognised through context"
        );
        let wrapped_token = anyhow::Error::from(Cancelled).context("indexing failed");
        assert!(
            is_cancel_error(&wrapped_token),
            "wrapped token Cancelled must be recognised"
        );

        // Non-cancellation errors that happen to contain the word "cancel"
        // must NOT be treated as cancellation.
        for msg in [
            "failed to cancel subscription: user requested cancel",
            "cannot cancel: invalid request containing cancel",
            "cancelled subscription already exists — not a pipeline cancel",
            "operation CANCEL is not supported",
        ] {
            let fake = anyhow::anyhow!(msg.to_string());
            assert!(
                fake.to_string().to_lowercase().contains("cancel"),
                "fixture must contain cancel"
            );
            assert!(
                !is_cancel_error(&fake),
                "non-cancellation error containing 'cancel' must NOT be treated as cancellation: {msg}"
            );
            // Also through a context wrapper.
            let fake_wrapped = fake.context("outer");
            assert!(
                !is_cancel_error(&fake_wrapped),
                "wrapped non-cancellation must stay non-cancellation: {msg}"
            );
        }
    }

    #[test]
    fn cancellation_token_check_produces_typed_error() {
        let token = CancellationToken::new();
        assert!(token.check().is_ok());

        token.cancel();
        let err = token.check().unwrap_err();
        assert!(
            is_cancel_error(&err),
            "CancellationToken::check must produce a typed is_cancel_error"
        );
        assert_eq!(err.to_string(), "operation cancelled");
    }
}
