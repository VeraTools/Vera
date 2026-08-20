//! One process-wide lock for tests that mutate the environment.
//!
//! `set_var`/`remove_var` are unsafe because another thread reading the
//! environment at the same moment is undefined behaviour, and `cargo test`
//! runs the whole crate's tests as threads of one process. A lock private to a
//! single test module therefore protects nothing: it does not exclude a test in
//! a different module. Every environment-mutating test in this crate takes the
//! lock defined here.

use std::ffi::OsString;
use std::sync::{Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Block until no other test is touching the environment.
///
/// Poisoning is ignored: a panicking test leaves the environment restored by
/// [`EnvVarGuard`], so the data the lock guards is still sound.
pub(crate) fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|err| err.into_inner())
}

/// Holds the environment lock and restores the variables it set on drop,
/// including while unwinding from a panic.
pub(crate) struct EnvVarGuard {
    _lock: MutexGuard<'static, ()>,
    saved: Vec<(&'static str, Option<OsString>)>,
}

impl EnvVarGuard {
    /// Take the lock, then set each variable, remembering its previous value.
    pub(crate) fn set(vars: &[(&'static str, &str)]) -> Self {
        let lock = env_lock();
        let saved = vars
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect();
        for (key, value) in vars {
            // Safety: the lock excludes every other environment-mutating test
            // in this crate, and no test spawns a thread that reads the
            // environment while holding it.
            unsafe {
                std::env::set_var(key, value);
            }
        }
        Self { _lock: lock, saved }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        for (key, value) in self.saved.drain(..) {
            // Safety: as in `set` — still under the lock, which is released
            // only after this runs.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "VERA_TEST_ENV_GUARD_PROBE";

    #[test]
    fn guard_restores_the_environment_while_unwinding() {
        let previous = {
            let _lock = env_lock();
            std::env::var_os(KEY)
        };
        assert!(
            previous.is_none(),
            "{KEY} must not be set outside this test"
        );

        let panicked = std::panic::catch_unwind(|| {
            let _guard = EnvVarGuard::set(&[(KEY, "leaked")]);
            assert_eq!(std::env::var(KEY).unwrap(), "leaked");
            panic!("simulated test failure");
        });
        assert!(panicked.is_err(), "the closure was supposed to panic");

        let _lock = env_lock();
        assert_eq!(
            std::env::var_os(KEY),
            None,
            "the guard must unset {KEY} even when the test panics"
        );
    }
}
