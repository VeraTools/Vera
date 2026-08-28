//! Filesystem lock for concurrent index builds.
//!
//! Prevents two `vera index` or eval harness processes from writing the same
//! repository's `.vera` directory at once. The lock is advisory and held for
//! the duration of a full index or update. Readers that would reuse an index
//! check the lock non-blockingly and treat a held lock as "not current" so
//! they never observe a half-written live index.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};

/// Path of the lock file for a repository.
///
/// Sibling to `.vera` (e.g. `<repo>/.vera.lock`) so it is not inside the
/// directory that gets atomically swapped during publication.
pub fn lock_path_for_index_dir(index_dir: &Path) -> PathBuf {
    // Canonicalize the parent when possible so that a lock acquired on
    // `index_repository("/abs/repo")` is visible to
    // `is_locked_for_index_dir(index_dir("relative/repo"))` and vice versa.
    let parent = index_dir
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let canonical_parent = if parent.exists() {
        parent.canonicalize().unwrap_or(parent)
    } else {
        parent
    };
    canonical_parent.join(format!(
        "{}.lock",
        index_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(".vera")
    ))
}

/// Convenience wrapper when the caller has the repo root.
pub fn lock_path_for_repo(repo_path: &Path) -> PathBuf {
    // Use the canonical index_dir helper to stay in sync with layout.
    // Canonicalize repo_path when possible so that lock paths match the
    // pipeline's canonical `repo_root`.
    let canonical_repo = if repo_path.exists() {
        repo_path
            .canonicalize()
            .unwrap_or_else(|_| repo_path.to_path_buf())
    } else {
        repo_path.to_path_buf()
    };
    let idx_dir = super::pipeline::index_dir(&canonical_repo);
    lock_path_for_index_dir(&idx_dir)
}

fn process_locks() -> &'static Mutex<HashSet<PathBuf>> {
    static LOCKS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    LOCKS.get_or_init(|| Mutex::new(HashSet::new()))
}

fn acquire_in_process_try(lock_path: &Path) -> bool {
    let mut set = process_locks().lock().unwrap();
    if set.contains(lock_path) {
        false
    } else {
        set.insert(lock_path.to_path_buf());
        true
    }
}

fn release_in_process(lock_path: &Path) {
    let mut set = process_locks().lock().unwrap();
    set.remove(lock_path);
}

fn is_in_process_locked(lock_path: &Path) -> bool {
    let set = process_locks().lock().unwrap();
    set.contains(lock_path)
}

/// Exclusive lock guard. Held until dropped.
pub struct IndexLock {
    file: File,
    path: PathBuf,
}

impl IndexLock {
    /// Try to acquire the lock without blocking. Returns `Ok(None)` if another
    /// process or thread holds it.
    pub fn try_acquire_for_index_dir(index_dir: &Path) -> Result<Option<Self>> {
        Self::try_acquire(&lock_path_for_index_dir(index_dir))
    }

    pub fn try_acquire_for_repo(repo_path: &Path) -> Result<Option<Self>> {
        Self::try_acquire(&lock_path_for_repo(repo_path))
    }

    fn try_acquire(lock_path: &Path) -> Result<Option<Self>> {
        // Fast-path in-process check.
        if is_in_process_locked(lock_path) {
            return Ok(None);
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lock_path)
            .with_context(|| format!("failed to open lock file {}", lock_path.display()))?;
        if !try_lock_exclusive(&file)? {
            return Ok(None);
        }
        // Now claim in-process slot. If another thread raced in between the
        // flock check and this claim, release flock and report contention.
        if !acquire_in_process_try(lock_path) {
            let _ = unlock(&file);
            return Ok(None);
        }
        Ok(Some(Self {
            file,
            path: lock_path.to_path_buf(),
        }))
    }

    /// Block until the lock is acquired.
    pub fn acquire_blocking_for_index_dir(index_dir: &Path) -> Result<Self> {
        Self::acquire_blocking(&lock_path_for_index_dir(index_dir))
    }

    pub fn acquire_blocking_for_repo(repo_path: &Path) -> Result<Self> {
        Self::acquire_blocking(&lock_path_for_repo(repo_path))
    }

    fn acquire_blocking(lock_path: &Path) -> Result<Self> {
        // Block on in-process contention first.
        loop {
            if acquire_in_process_try(lock_path) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lock_path)
            .with_context(|| format!("failed to open lock file {}", lock_path.display()))?;
        // `flock` is blocking; if in-process slot was claimed, this will
        // block on inter-process contention.
        let result = lock_exclusive_blocking(&file)
            .with_context(|| format!("failed to lock {}", lock_path.display()));
        if result.is_err() {
            // Ensure in-process slot is released on failure.
            release_in_process(lock_path);
        }
        result.map(|_| Self {
            file,
            path: lock_path.to_path_buf(),
        })
    }

    /// Non-blocking check: true if the index is currently locked for writing.
    pub fn is_locked_for_index_dir(index_dir: &Path) -> bool {
        let lock_path = lock_path_for_index_dir(index_dir);
        Self::is_locked(&lock_path)
    }

    pub fn is_locked_for_repo(repo_path: &Path) -> bool {
        Self::is_locked(&lock_path_for_repo(repo_path))
    }

    fn is_locked(lock_path: &Path) -> bool {
        if is_in_process_locked(lock_path) {
            return true;
        }
        let file = match OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lock_path)
        {
            Ok(f) => f,
            Err(_) => return false,
        };
        match try_lock_exclusive(&file) {
            Ok(true) => {
                // We acquired it, so it was not locked. Release immediately.
                let _ = unlock(&file);
                false
            }
            Ok(false) => true,
            Err(_) => false,
        }
    }
}

impl Drop for IndexLock {
    fn drop(&mut self) {
        let _ = unlock(&self.file);
        release_in_process(&self.path);
    }
}

#[cfg(unix)]
fn try_lock_exclusive(file: &File) -> Result<bool> {
    use std::os::unix::io::AsRawFd;
    let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if ret == 0 {
        Ok(true)
    } else {
        let err = std::io::Error::last_os_error();
        if err.kind() == std::io::ErrorKind::WouldBlock {
            Ok(false)
        } else {
            Err(err).context("flock failed")?
        }
    }
}

#[cfg(unix)]
fn lock_exclusive_blocking(file: &File) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error()).context("flock failed")?
    }
}

#[cfg(unix)]
fn unlock(file: &File) -> Result<()> {
    use std::os::unix::io::AsRawFd;
    let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error()).context("flock unlock failed")?
    }
}

#[cfg(not(unix))]
fn try_lock_exclusive(_file: &File) -> Result<bool> {
    // No advisory flock on this platform – treat as unlocked.
    Ok(true)
}

#[cfg(not(unix))]
fn lock_exclusive_blocking(_file: &File) -> Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn unlock(_file: &File) -> Result<()> {
    Ok(())
}
