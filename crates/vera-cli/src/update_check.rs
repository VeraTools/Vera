//! Non-blocking update hints printed to stderr after command execution.
//!
//! Two checks:
//! 1. **Skill staleness** — compares binary version against `.version` files
//!    written by `vera agent install` into each agent client's skill directory.
//! 2. **Binary staleness** — fetches the latest release tag from GitHub (cached
//!    for 24 hours in `~/.vera/update-check.json`) and compares against the
//!    running binary version.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const CHECK_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
const GITHUB_API_TIMEOUT: Duration = Duration::from_secs(5);
const REPO: &str = "VeraTools/Vera";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionCheckSource {
    Live,
    Cache,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstallMethodSource {
    Provenance,
    Heuristic,
    Ambiguous,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct InstallMethodResolution {
    pub install_method: Option<String>,
    pub detected_install_methods: Vec<String>,
    pub source: InstallMethodSource,
}

#[derive(Debug, Clone)]
pub struct BinaryVersionStatus {
    pub current_version: &'static str,
    pub latest_version: Option<String>,
    pub install_method: Option<String>,
    pub install_method_source: InstallMethodSource,
    pub detected_install_methods: Vec<String>,
    pub source: VersionCheckSource,
}

impl BinaryVersionStatus {
    pub fn update_available(&self) -> bool {
        self.latest_version
            .as_deref()
            .is_some_and(|latest| is_newer(latest, self.current_version))
    }

    pub fn update_command(&self) -> String {
        if self.install_method.is_some() && self.can_apply_update() {
            suggested_update_command(self.install_method.as_deref())
        } else {
            "vera upgrade".to_string()
        }
    }

    pub fn can_apply_update(&self) -> bool {
        matches!(
            self.install_method_source,
            InstallMethodSource::Provenance | InstallMethodSource::Heuristic
        ) && self.install_method.is_some()
    }
}

pub fn current_version() -> &'static str {
    CURRENT_VERSION
}

/// Run all update checks and print hints to stderr. Never fails — errors are
/// silently swallowed so the user's actual command output is never disrupted.
pub fn print_nudges() {
    if std::env::var("VERA_NO_UPDATE_CHECK").is_ok() {
        return;
    }
    check_skill_staleness();
    check_binary_staleness();
}

// ---------------------------------------------------------------------------
// Skill version check
// ---------------------------------------------------------------------------

fn check_skill_staleness() {
    let home = match dirs::home_dir() {
        Some(h) => h,
        None => return,
    };

    let cwd = std::env::current_dir().ok();
    let skill_dirs = match crate::commands::agent::all_skill_paths(cwd.as_deref(), &home) {
        Ok(dirs) => dirs,
        Err(_) => return,
    };
    let stale_installs = stale_skill_installs(&skill_dirs);
    if stale_installs.is_empty() {
        return;
    }

    // Auto-sync stale skills silently instead of nagging the user.
    match crate::commands::agent::sync_skills_only() {
        Ok(()) => {}
        Err(_) => {
            // Fall back to a hint if auto-sync fails.
            if let Some(hint) = format_skill_staleness_hint(&stale_installs, cwd.as_deref(), &home)
            {
                eprintln!("{hint}");
            }
        }
    }
}

fn read_skill_version(skill_dir: &Path) -> Option<String> {
    let version_file = skill_dir.join(".version");
    fs::read_to_string(version_file)
        .ok()
        .map(|s| s.trim().to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StaleSkillInstall {
    path: PathBuf,
    version: String,
}

fn stale_skill_installs(skill_dirs: &[PathBuf]) -> Vec<StaleSkillInstall> {
    skill_dirs
        .iter()
        .filter_map(|dir| {
            let version = read_skill_version(dir)?;
            (version != CURRENT_VERSION).then(|| StaleSkillInstall {
                path: dir.clone(),
                version,
            })
        })
        .collect()
}

fn format_skill_staleness_hint(
    stale_installs: &[StaleSkillInstall],
    cwd: Option<&Path>,
    home: &Path,
) -> Option<String> {
    let first = stale_installs.first()?;
    let description = describe_skill_install(&first.path, cwd, home);
    let remaining = stale_installs.len().saturating_sub(1);
    let suffix = if remaining > 0 {
        format!(" (+{remaining} more)")
    } else {
        String::new()
    };

    Some(format!(
        "hint: stale {}: `{}` is v{}, binary is v{}{}. Refresh with `{}`.",
        description.label(),
        description.path,
        first.version,
        CURRENT_VERSION,
        suffix,
        description.refresh_command(),
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkillInstallScope {
    Global,
    Project,
    Unknown,
}

struct SkillInstallDescription {
    scope: SkillInstallScope,
    path: String,
}

impl SkillInstallDescription {
    fn label(&self) -> &'static str {
        match self.scope {
            SkillInstallScope::Global => "global Vera skill",
            SkillInstallScope::Project => "project Vera skill",
            SkillInstallScope::Unknown => "Vera skill",
        }
    }

    fn refresh_command(&self) -> &'static str {
        "vera agent sync"
    }
}

fn describe_skill_install(path: &Path, cwd: Option<&Path>, home: &Path) -> SkillInstallDescription {
    if let Some(cwd) = cwd {
        if let Ok(relative) = path.strip_prefix(cwd) {
            return SkillInstallDescription {
                scope: SkillInstallScope::Project,
                path: format!("./{}", relative.display()),
            };
        }
    }

    if let Ok(relative) = path.strip_prefix(home) {
        return SkillInstallDescription {
            scope: SkillInstallScope::Global,
            path: format!("~/{}", relative.display()),
        };
    }

    SkillInstallDescription {
        scope: SkillInstallScope::Unknown,
        path: path.display().to_string(),
    }
}

// ---------------------------------------------------------------------------
// Binary version check (cached GitHub API)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
struct UpdateCache {
    latest_version: String,
    checked_at_secs: u64,
    #[serde(default)]
    install_method: Option<String>,
}

fn cache_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".vera").join("update-check.json"))
}

fn check_binary_staleness() {
    let status = binary_version_status(false);
    if let Some(latest) = status.latest_version.as_deref() {
        if status.update_available() {
            print_binary_nudge(latest, &status);
        }
    }
}

fn print_binary_nudge(latest: &str, status: &BinaryVersionStatus) {
    let update_cmd = status.update_command();
    if status.install_method_source == InstallMethodSource::Ambiguous {
        eprintln!(
            "hint: vera v{} is available (current: v{}). Multiple install methods were detected; run `vera upgrade` to choose the right update command.",
            latest, CURRENT_VERSION,
        );
    } else {
        eprintln!(
            "hint: vera v{} is available (current: v{}). Update: `{}`",
            latest, CURRENT_VERSION, update_cmd,
        );
    }
}

pub fn binary_version_status(force_refresh: bool) -> BinaryVersionStatus {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    binary_version_status_inner(cache_path(), force_refresh, now, &resolve_install_method)
}

/// `binary_version_status` with the cache location, the clock and the install
/// method fallback injected, so tests can drive it without touching `~/.vera`
/// or spawning package managers.
///
/// `resolve` is only called when the cache cannot answer. That matters: the
/// fallback ends in `detect_install_methods`, which spawns `npm`, `bun`, `pip`
/// and `uv`, and `print_nudges` runs synchronously before the process exits.
fn binary_version_status_inner(
    cache_file: Option<PathBuf>,
    force_refresh: bool,
    now: u64,
    resolve: &dyn Fn() -> InstallMethodResolution,
) -> BinaryVersionStatus {
    let Some(cache_file) = cache_file else {
        return status_from(None, resolve(), VersionCheckSource::Unavailable);
    };

    let cached = load_cache(&cache_file);

    if !force_refresh {
        if let Some(cached) = cached.as_ref() {
            if now.saturating_sub(cached.checked_at_secs) < CHECK_INTERVAL.as_secs() {
                let resolution = cached_install_method(cached).unwrap_or_else(resolve);
                return status_from(
                    Some(cached.latest_version.clone()),
                    resolution,
                    VersionCheckSource::Cache,
                );
            }
        }
    }

    let resolution = resolve();

    if let Some(latest) = fetch_latest_version() {
        let cache = UpdateCache {
            latest_version: latest.clone(),
            checked_at_secs: now,
            install_method: resolution.install_method.clone(),
        };
        let _ = save_cache(&cache_file, &cache);
        return status_from(Some(latest), resolution, VersionCheckSource::Live);
    }

    if let Some(cached) = cached {
        return BinaryVersionStatus {
            current_version: CURRENT_VERSION,
            latest_version: Some(cached.latest_version),
            install_method: cached.install_method.or(resolution.install_method),
            install_method_source: resolution.source,
            detected_install_methods: resolution.detected_install_methods,
            source: VersionCheckSource::Cache,
        };
    }

    status_from(None, resolution, VersionCheckSource::Unavailable)
}

fn status_from(
    latest_version: Option<String>,
    resolution: InstallMethodResolution,
    source: VersionCheckSource,
) -> BinaryVersionStatus {
    BinaryVersionStatus {
        current_version: CURRENT_VERSION,
        latest_version,
        install_method: resolution.install_method,
        install_method_source: resolution.source,
        detected_install_methods: resolution.detected_install_methods,
        source,
    }
}

/// Rebuild an install method resolution from a cache entry, so a fresh cache
/// answers without re-running detection.
///
/// The source is reported as `Heuristic`. `save_cache` only ever stores a
/// method that `resolve_install_method` returned as `Some`, which narrows the
/// original source to `Provenance` or `Heuristic`, but the cache does not
/// record which, so this reports the weaker of the two. `can_apply_update` and
/// `suggested_update_command` treat them identically; the only callers that
/// surface the distinction, `vera upgrade` and `vera doctor`, pass
/// `force_refresh = true` and never reach this branch.
fn cached_install_method(cached: &UpdateCache) -> Option<InstallMethodResolution> {
    let method = cached.install_method.clone()?;
    Some(InstallMethodResolution {
        detected_install_methods: vec![method.clone()],
        install_method: Some(method),
        source: InstallMethodSource::Heuristic,
    })
}

pub fn suggested_update_command(install_method: Option<&str>) -> String {
    match install_method {
        Some("npm") => "npm install -g @vera-ai/cli@latest && npx @vera-ai/cli install".to_string(),
        Some("bun") => {
            "bun install -g @vera-ai/cli@latest && bunx @vera-ai/cli install".to_string()
        }
        Some("pip") => "pip install --upgrade vera-ai && vera-ai install".to_string(),
        Some("uv") => "uvx vera-ai install".to_string(),
        _ => "vera upgrade".to_string(),
    }
}

/// Compare two semver-ish strings. Returns true if `latest` > `current`.
fn is_newer(latest: &str, current: &str) -> bool {
    let parse = |s: &str| -> (u32, u32, u32) {
        let s = s.strip_prefix('v').unwrap_or(s);
        let mut parts = s.split('.').map(|p| p.parse::<u32>().unwrap_or(0));
        (
            parts.next().unwrap_or(0),
            parts.next().unwrap_or(0),
            parts.next().unwrap_or(0),
        )
    };
    parse(latest) > parse(current)
}

fn fetch_latest_version() -> Option<String> {
    // Use a small blocking runtime since we're called from sync main().
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .ok()?;

    rt.block_on(async {
        let url = format!("https://api.github.com/repos/{}/releases/latest", REPO);
        let client = reqwest::Client::builder()
            .timeout(GITHUB_API_TIMEOUT)
            .build()
            .ok()?;
        let resp = client
            .get(&url)
            .header("User-Agent", format!("vera/{}", CURRENT_VERSION))
            .header("Accept", "application/vnd.github+json")
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let body: serde_json::Value = resp.json().await.ok()?;
        let tag = body.get("tag_name")?.as_str()?;
        Some(tag.strip_prefix('v').unwrap_or(tag).to_string())
    })
}

pub fn resolve_install_method() -> InstallMethodResolution {
    if let Ok(provenance) = crate::state::load_install_provenance() {
        if let Some(method) = provenance.install_method {
            return InstallMethodResolution {
                detected_install_methods: vec![method.clone()],
                install_method: Some(method),
                source: InstallMethodSource::Provenance,
            };
        }
    }

    if let Ok(config) = crate::state::load_saved_config() {
        if let Some(method) = config.install_method {
            return InstallMethodResolution {
                detected_install_methods: vec![method.clone()],
                install_method: Some(method),
                source: InstallMethodSource::Heuristic,
            };
        }
    }

    let detected_install_methods = detect_install_methods();
    match detected_install_methods.as_slice() {
        [] => InstallMethodResolution {
            install_method: None,
            detected_install_methods,
            source: InstallMethodSource::Unknown,
        },
        [method] => InstallMethodResolution {
            install_method: Some(method.clone()),
            detected_install_methods,
            source: InstallMethodSource::Heuristic,
        },
        _ => InstallMethodResolution {
            install_method: None,
            detected_install_methods,
            source: InstallMethodSource::Ambiguous,
        },
    }
}

pub fn detect_install_methods() -> Vec<String> {
    let mut methods = Vec::new();

    if command_succeeds("npm", &["list", "-g", "--depth=0", "@vera-ai/cli"]) {
        methods.push("npm".to_string());
    }

    if let Ok(output) = Command::new(shell_command("bun"))
        .args(["pm", "ls", "-g"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
    {
        if String::from_utf8_lossy(&output.stdout).contains("@vera-ai/cli") {
            methods.push("bun".to_string());
        }
    }

    if command_succeeds("pip", &["show", "vera-ai"]) {
        methods.push("pip".to_string());
    }

    if command_succeeds("uv", &["pip", "show", "vera-ai"]) {
        methods.push("uv".to_string());
    }

    methods
}

pub fn supported_update_methods() -> &'static [&'static str] {
    &["npm", "bun", "pip", "uv"]
}

pub fn apply_update(method: &str) -> Result<()> {
    match method {
        "npm" => {
            run_update_step("npm", &["install", "-g", "@vera-ai/cli@latest"])?;
            run_update_step("npx", &["@vera-ai/cli", "install"])?;
        }
        "bun" => {
            run_update_step("bun", &["install", "-g", "@vera-ai/cli@latest"])?;
            run_update_step("bunx", &["@vera-ai/cli", "install"])?;
        }
        "pip" => {
            run_update_step("pip", &["install", "--upgrade", "vera-ai"])?;
            run_update_step("vera-ai", &["install"])?;
        }
        "uv" => run_update_step("uvx", &["vera-ai", "install"])?,
        other => bail!("unsupported install method: {other}"),
    }

    Ok(())
}

fn command_succeeds(program: &str, args: &[&str]) -> bool {
    Command::new(shell_command(program))
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn run_update_step(program: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(shell_command(program))
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .with_context(|| format!("failed to start `{program}`"))?;
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!(
            "`{}` exited with status {}",
            format_command(program, args),
            status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        ))
    }
}

fn format_command(program: &str, args: &[&str]) -> String {
    std::iter::once(program)
        .chain(args.iter().copied())
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_command(program: &str) -> String {
    if cfg!(windows) {
        format!("{program}.cmd")
    } else {
        program.to_string()
    }
}

fn load_cache(path: &Path) -> Option<UpdateCache> {
    let data = fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

fn save_cache(path: &Path, cache: &UpdateCache) -> Option<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok()?;
    }
    let data = serde_json::to_string(cache).ok()?;
    fs::write(path, data).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn is_newer_works() {
        assert!(is_newer("0.4.0", "0.3.1"));
        assert!(is_newer("1.0.0", "0.99.99"));
        assert!(is_newer("0.3.2", "0.3.1"));
        assert!(!is_newer("0.3.1", "0.3.1"));
        assert!(!is_newer("0.3.0", "0.3.1"));
        assert!(is_newer("v0.4.0", "0.3.1"));
    }

    #[test]
    fn read_skill_version_missing_dir() {
        assert_eq!(read_skill_version(Path::new("/nonexistent/path")), None);
    }

    #[test]
    fn format_skill_staleness_hint_includes_path_and_count() {
        let hint = format_skill_staleness_hint(
            &[
                StaleSkillInstall {
                    path: PathBuf::from("/tmp/home/.codex/skills/vera"),
                    version: "0.9.18".to_string(),
                },
                StaleSkillInstall {
                    path: PathBuf::from("/tmp/project/.agents/skills/vera"),
                    version: "0.9.16".to_string(),
                },
            ],
            Some(Path::new("/tmp/project")),
            Path::new("/tmp/home"),
        )
        .unwrap();

        assert!(hint.contains("global Vera skill"));
        assert!(hint.contains("~/.codex/skills/vera"));
        assert!(hint.contains("0.9.18"));
        assert!(hint.contains("(+1 more)"));
        assert!(hint.contains("vera agent sync"));
    }

    #[test]
    fn suggested_update_command_known_methods() {
        assert!(suggested_update_command(Some("npm")).contains("npm install"));
        assert!(suggested_update_command(Some("bun")).contains("bunx"));
        assert!(suggested_update_command(Some("pip")).contains("pip install"));
        assert!(suggested_update_command(Some("uv")).contains("uvx"));
        assert_eq!(suggested_update_command(None), "vera upgrade");
        assert_eq!(suggested_update_command(Some("unknown")), "vera upgrade");
    }

    #[test]
    fn supported_update_methods_contains_all() {
        let methods = supported_update_methods();
        assert!(methods.contains(&"npm"));
        assert!(methods.contains(&"bun"));
        assert!(methods.contains(&"pip"));
        assert!(methods.contains(&"uv"));
    }

    #[test]
    fn binary_version_status_no_update_when_equal() {
        let status = BinaryVersionStatus {
            current_version: CURRENT_VERSION,
            latest_version: Some(CURRENT_VERSION.to_string()),
            install_method: Some("npm".to_string()),
            install_method_source: InstallMethodSource::Provenance,
            detected_install_methods: vec!["npm".to_string()],
            source: VersionCheckSource::Cache,
        };
        assert!(!status.update_available());
        assert!(status.can_apply_update());
    }

    #[test]
    fn binary_version_status_update_available() {
        let status = BinaryVersionStatus {
            current_version: "0.0.1",
            latest_version: Some("99.0.0".to_string()),
            install_method: Some("pip".to_string()),
            install_method_source: InstallMethodSource::Heuristic,
            detected_install_methods: vec!["pip".to_string()],
            source: VersionCheckSource::Live,
        };
        assert!(status.update_available());
        assert!(status.can_apply_update());
        assert!(status.update_command().contains("pip"));
    }

    #[test]
    fn binary_version_status_cannot_apply_when_ambiguous() {
        let status = BinaryVersionStatus {
            current_version: "0.0.1",
            latest_version: Some("99.0.0".to_string()),
            install_method: None,
            install_method_source: InstallMethodSource::Ambiguous,
            detected_install_methods: vec!["npm".to_string(), "pip".to_string()],
            source: VersionCheckSource::Live,
        };
        assert!(!status.can_apply_update());
        assert_eq!(status.update_command(), "vera upgrade");
    }

    #[test]
    fn binary_version_status_cannot_apply_when_unknown() {
        let status = BinaryVersionStatus {
            current_version: "0.0.1",
            latest_version: Some("99.0.0".to_string()),
            install_method: None,
            install_method_source: InstallMethodSource::Unknown,
            detected_install_methods: vec![],
            source: VersionCheckSource::Live,
        };
        assert!(!status.can_apply_update());
    }

    /// A resolver that records how often it ran and reports nothing, standing in
    /// for `resolve_install_method` without spawning any package manager.
    fn counting_resolver(calls: &Cell<usize>) -> impl Fn() -> InstallMethodResolution + '_ {
        || {
            calls.set(calls.get() + 1);
            InstallMethodResolution {
                install_method: None,
                detected_install_methods: Vec::new(),
                source: InstallMethodSource::Unknown,
            }
        }
    }

    fn write_cache(dir: &Path, install_method: Option<&str>, checked_at_secs: u64) -> PathBuf {
        let path = dir.join("update-check.json");
        save_cache(
            &path,
            &UpdateCache {
                latest_version: "99.0.0".to_string(),
                checked_at_secs,
                install_method: install_method.map(str::to_string),
            },
        )
        .expect("cache written");
        path
    }

    #[test]
    fn fresh_cache_with_install_method_skips_detection() {
        let dir = tempfile::tempdir().unwrap();
        let checked_at = 1_700_000_000;
        let cache_file = write_cache(dir.path(), Some("npm"), checked_at);

        let calls = Cell::new(0);
        let status = binary_version_status_inner(
            Some(cache_file),
            false,
            checked_at + 60,
            &counting_resolver(&calls),
        );

        assert_eq!(
            calls.get(),
            0,
            "install method detection ran even though a fresh cache carried the method"
        );
        assert_eq!(status.source, VersionCheckSource::Cache);
        assert_eq!(status.latest_version.as_deref(), Some("99.0.0"));
        assert_eq!(status.install_method.as_deref(), Some("npm"));
        assert_eq!(status.install_method_source, InstallMethodSource::Heuristic);
        assert_eq!(status.detected_install_methods, vec!["npm".to_string()]);
        assert!(status.can_apply_update());
    }

    #[test]
    fn fresh_cache_without_install_method_falls_back_to_detection() {
        let dir = tempfile::tempdir().unwrap();
        let checked_at = 1_700_000_000;
        let cache_file = write_cache(dir.path(), None, checked_at);

        let calls = Cell::new(0);
        let status = binary_version_status_inner(
            Some(cache_file),
            false,
            checked_at + 60,
            &counting_resolver(&calls),
        );

        assert_eq!(calls.get(), 1);
        assert_eq!(status.source, VersionCheckSource::Cache);
        assert_eq!(status.install_method, None);
        assert_eq!(status.install_method_source, InstallMethodSource::Unknown);
    }

    #[test]
    fn missing_cache_path_still_resolves_the_install_method() {
        let calls = Cell::new(0);
        let status =
            binary_version_status_inner(None, false, 1_700_000_000, &counting_resolver(&calls));

        assert_eq!(calls.get(), 1);
        assert_eq!(status.source, VersionCheckSource::Unavailable);
        assert_eq!(status.latest_version, None);
    }

    #[test]
    fn format_command_joins_args() {
        assert_eq!(format_command("npm", &["update", "-g"]), "npm update -g");
        assert_eq!(format_command("vera", &[]), "vera");
    }

    #[test]
    fn shell_command_returns_program() {
        // On non-Windows, shell_command returns the program as-is.
        if !cfg!(windows) {
            assert_eq!(shell_command("npm"), "npm");
        }
    }
}
