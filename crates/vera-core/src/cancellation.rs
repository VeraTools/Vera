use tokio_util::sync::CancellationToken as AsyncCancellationToken;

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
        anyhow::ensure!(!self.is_cancelled(), "operation cancelled");
        Ok(())
    }
}
