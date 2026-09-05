//! A shared "stop what you are doing" flag.
//!
//! A batch over a folder of video can run for hours, and a model download over
//! a slow link for minutes. Both need a way for the UI to ask them to stop that
//! does not involve killing a thread mid-write. [`CancelToken`] is that: a
//! cheap, clonable flag polled at the points where stopping is safe — between
//! files, between frames, between download chunks.
//!
//! It is deliberately one-way. Once cancelled a token stays cancelled, so
//! there is no window where one worker sees a reset flag and keeps going after
//! its siblings have stopped; a new run gets a new token.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A clonable cancellation flag shared by every worker in one operation.
///
/// `Default` is a token that is never cancelled, so callers that do not care
/// about cancellation can ignore it entirely.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A fresh, un-cancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Idempotent, and safe from any thread.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Borrow the underlying flag, for APIs that take a plain `&AtomicBool`
    /// (`ob_models::FetchOptions::cancel`, for one).
    pub fn as_flag(&self) -> &AtomicBool {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_token_is_not_cancelled() {
        assert!(!CancelToken::new().is_cancelled());
    }

    #[test]
    fn cancellation_is_shared_by_clones_and_is_one_way() {
        let a = CancelToken::new();
        let b = a.clone();
        assert!(!b.is_cancelled());
        a.cancel();
        // A clone handed to a worker thread sees the parent's cancellation.
        assert!(b.is_cancelled());
        // Idempotent: a second request neither panics nor un-cancels.
        b.cancel();
        assert!(a.is_cancelled());
    }

    #[test]
    fn the_raw_flag_matches_the_token() {
        let t = CancelToken::new();
        assert!(!t.as_flag().load(Ordering::SeqCst));
        t.cancel();
        assert!(t.as_flag().load(Ordering::SeqCst));
    }
}
