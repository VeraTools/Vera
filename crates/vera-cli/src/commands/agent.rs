//! `vera agent ...` — install and manage the Vera skill for coding agents.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use clap::ValueEnum;
use serde::Serialize;

use crate::skill_assets::{VERA_SKILL_FILES, VERA_SKILL_NAME};
use crate::state;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AgentCommand {
    Install,
    Status,
    Remove,
    Sync,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentClient {
    /// Install for all supported clients at once.
    All,
    /// Cross-agent `.agents/skills/` directory (Agent Skills open spec).
    Agents,
    /// Sourcegraph Amp CLI.
    Amp,
    /// Antigravity CLI.
    Antigravity,
    /// Augment Code CLI.
    Augment,
    /// Anthropic Claude Code.
    Claude,
    /// Cline CLI.
    Cline,
    /// Codebuff CLI.
    Codebuff,
    /// CodeBuddy CLI.
    Codebuddy,
    /// OpenAI Codex CLI.
    Codex,
    /// GitHub Copilot CLI.
    Copilot,
    /// Snowflake Cortex Code.
    Cortex,
    /// Crush CLI.
    Crush,
    /// Cursor editor.
    Cursor,
    /// Factory Droid CLI.
    Droid,
    /// Google Gemini CLI.
    Gemini,
    /// Block Goose CLI.
    Goose,
    /// iFlow CLI.
    Iflow,
    /// JetBrains Junie CLI.
    Junie,
    /// Kilo Code CLI.
    Kilo,
    /// Kiro CLI.
    Kiro,
    /// Moonshot Kimi CLI.
    Kimi,
    /// Mistral Vibe CLI.
    Vibe,
    /// Mux CLI.
    Mux,
    /// OpenCode CLI.
    Opencode,
    /// OpenHands CLI.
    Openhands,
    /// Pi CLI.
    Pi,
    /// Qwen Code CLI.
    Qwen,
    /// Roo Code CLI.
    Roo,
    /// Trae CLI.
    Trae,
    /// Windsurf editor.
    Windsurf,
    /// Zed editor.
    Zed,
}

impl AgentClient {
    /// All concrete (non-All) clients with display names, in display order.
    const META: &[(AgentClient, &str)] = &[
        (AgentClient::Agents, "Universal (.agents/skills/)"),
        (AgentClient::Amp, "Amp (Sourcegraph)"),
        (AgentClient::Antigravity, "Antigravity"),
        (AgentClient::Augment, "Augment Code"),
        (AgentClient::Claude, "Claude Code (Anthropic)"),
        (AgentClient::Cline, "Cline"),
        (AgentClient::Codebuff, "Codebuff"),
        (AgentClient::Codebuddy, "CodeBuddy"),
        (AgentClient::Codex, "Codex (OpenAI)"),
        (AgentClient::Copilot, "Copilot (GitHub)"),
        (AgentClient::Cortex, "Cortex Code (Snowflake)"),
        (AgentClient::Crush, "Crush"),
        (AgentClient::Cursor, "Cursor"),
        (AgentClient::Droid, "Droid (Factory)"),
        (AgentClient::Gemini, "Gemini CLI (Google)"),
        (AgentClient::Goose, "Goose (Block)"),
        (AgentClient::Iflow, "iFlow"),
        (AgentClient::Junie, "Junie (JetBrains)"),
        (AgentClient::Kilo, "Kilo Code"),
        (AgentClient::Kiro, "Kiro"),
        (AgentClient::Kimi, "Kimi (Moonshot)"),
        (AgentClient::Vibe, "Vibe (Mistral)"),
        (AgentClient::Mux, "Mux"),
        (AgentClient::Opencode, "OpenCode"),
        (AgentClient::Openhands, "OpenHands"),
        (AgentClient::Pi, "Pi"),
        (AgentClient::Qwen, "Qwen Code"),
        (AgentClient::Roo, "Roo Code"),
        (AgentClient::Trae, "Trae"),
        (AgentClient::Windsurf, "Windsurf"),
        (AgentClient::Zed, "Zed"),
    ];

    /// All concrete (non-All) client variants, in display order.
    fn all_concrete() -> impl Iterator<Item = AgentClient> + 'static {
        Self::META.iter().map(|(client, _)| *client)
    }

    fn display_name(&self) -> &'static str {
        if *self == AgentClient::All {
            return "All";
        }
        Self::META
            .iter()
            .find(|(client, _)| client == self)
            .map(|(_, name)| *name)
            .unwrap_or("All")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentScope {
    Global,
    Project,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectionPreset {
    Installed,
    All,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InstallWorkflowChoice {
    RefreshStale,
    Manage,
}

#[derive(Debug, Clone, Serialize)]
pub struct SkillLocationReport {
    client: AgentClient,
    scope: AgentScope,
    path: String,
    installed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    up_to_date: Option<bool>,
    /// Only set on the removal path: whether a skill was actually present and
    /// deleted, as opposed to the location merely having been checked.
    #[serde(skip_serializing_if = "Option::is_none")]
    removed: Option<bool>,
}

impl SkillLocationReport {
    pub(crate) fn was_removed(&self) -> bool {
        self.removed == Some(true)
    }

    pub(crate) fn path(&self) -> &str {
        &self.path
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScopeInstallStatus {
    scope: AgentScope,
    installed: bool,
    up_to_date: bool,
}

#[derive(Debug, Clone)]
struct ClientInstallStatus {
    client: AgentClient,
    scopes: Vec<ScopeInstallStatus>,
}

impl ClientInstallStatus {
    fn is_installed(&self) -> bool {
        self.scopes.iter().any(|scope| scope.installed)
    }

    fn is_stale(&self) -> bool {
        self.scopes
            .iter()
            .any(|scope| scope.installed && !scope.up_to_date)
    }

    fn install_scopes(&self) -> impl Iterator<Item = AgentScope> + '_ {
        self.scopes
            .iter()
            .filter(|scope| scope.installed)
            .map(|scope| scope.scope)
    }

    fn scopes_needing_install(&self) -> impl Iterator<Item = AgentScope> + '_ {
        self.scopes
            .iter()
            .filter(|scope| !scope.up_to_date)
            .map(|scope| scope.scope)
    }

    fn stale_scopes(&self) -> impl Iterator<Item = AgentScope> + '_ {
        self.scopes
            .iter()
            .filter(|scope| scope.installed && !scope.up_to_date)
            .map(|scope| scope.scope)
    }

    fn hint(&self) -> String {
        let installed_scopes: Vec<&'static str> = self
            .scopes
            .iter()
            .filter(|scope| scope.installed)
            .map(|scope| scope.scope.label())
            .collect();

        if installed_scopes.is_empty() {
            return String::new();
        }

        let stale = self
            .scopes
            .iter()
            .any(|scope| scope.installed && !scope.up_to_date);
        let installed_label = format!("installed ({})", installed_scopes.join(" + "));
        if stale {
            format!("{installed_label}, needs sync")
        } else {
            format!("{installed_label}, up to date")
        }
    }
}

impl AgentScope {
    fn label(&self) -> &'static str {
        match self {
            AgentScope::Global => "global",
            AgentScope::Project => "project",
            AgentScope::All => "both",
        }
    }
}

pub fn run(
    command: AgentCommand,
    client: Option<AgentClient>,
    scope: Option<AgentScope>,
    json_output: bool,
) -> anyhow::Result<()> {
    match command {
        AgentCommand::Install => install(client, scope, json_output),
        AgentCommand::Status => status(
            client.unwrap_or(AgentClient::All),
            scope.unwrap_or(AgentScope::All),
            json_output,
        ),
        AgentCommand::Remove => remove(client, scope, json_output),
        AgentCommand::Sync => sync(
            client.unwrap_or(AgentClient::All),
            scope.unwrap_or(AgentScope::All),
            json_output,
        ),
    }
}

fn install(
    client: Option<AgentClient>,
    scope: Option<AgentScope>,
    json_output: bool,
) -> anyhow::Result<()> {
    let (resolved_client, resolved_scope) = match (client, scope) {
        (Some(c), Some(s)) => (c, s),
        (Some(c), None) => (c, AgentScope::Global),
        (None, Some(s)) => (AgentClient::All, s),
        (None, None) if json_output => (AgentClient::All, AgentScope::Global),
        (None, None) => return install_interactive(),
    };

    let locations = resolve_locations(resolved_client, resolved_scope)?;
    do_install(&locations, json_output)?;
    if !json_output {
        let selected_clients = selected_clients_for(resolved_client);
        offer_agents_md_snippet(&selected_clients)?;
    }
    Ok(())
}

fn install_interactive() -> anyhow::Result<()> {
    cliclack::intro("vera agent install")?;

    let scope: AgentScope = cliclack::select("Install scope")
        .item(AgentScope::Global, "Global", "available in all projects")
        .item(AgentScope::Project, "Project", "current repo only")
        .item(AgentScope::All, "Both", "global and project")
        .interact()?;

    let cwd = std::env::current_dir().context("failed to resolve current directory")?;
    let home = state::user_home_dir()?;
    let statuses = collect_client_install_statuses(scope, &cwd, &home)?;
    let all_clients: Vec<AgentClient> = AgentClient::all_concrete().collect();
    let installed_clients: Vec<AgentClient> = statuses
        .iter()
        .filter(|status| status.is_installed())
        .map(|status| status.client)
        .collect();
    let stale_locations = stale_locations_from_statuses(&statuses, &cwd, &home)?;

    if !stale_locations.is_empty() {
        let stale_install_count = statuses.iter().filter(|status| status.is_stale()).count();
        let choice = cliclack::select(format!(
            "Detected {} stale Vera skill install(s) across {} agent(s)",
            stale_locations.len(),
            stale_install_count,
        ))
        .item(
            InstallWorkflowChoice::RefreshStale,
            "Refresh stale installs now",
            "update every stale Vera skill install in one step",
        )
        .item(
            InstallWorkflowChoice::Manage,
            "Manage installs manually",
            "open the full install/remove selector",
        )
        .initial_value(InstallWorkflowChoice::RefreshStale)
        .interact()?;

        if choice == InstallWorkflowChoice::RefreshStale {
            do_install(&stale_locations, false)?;
            cliclack::outro("Done!")?;
            return Ok(());
        }
    }

    let preset = cliclack::select("Starting selection")
        .item(
            SelectionPreset::Installed,
            "Keep installed",
            "preselect agents that already have Vera installed",
        )
        .item(
            SelectionPreset::All,
            "Enable all",
            "start with every supported agent selected",
        )
        .item(
            SelectionPreset::None,
            "Disable all",
            "start with nothing selected",
        )
        .initial_value(if installed_clients.is_empty() {
            SelectionPreset::None
        } else {
            SelectionPreset::Installed
        })
        .interact()?;

    let initial_selected = match preset {
        SelectionPreset::Installed => installed_clients.clone(),
        SelectionPreset::All => all_clients.to_vec(),
        SelectionPreset::None => Vec::new(),
    };

    let mut multi = cliclack::multiselect(
        "Select agents to install (space to toggle, enter applies installs and removals)",
    )
    .initial_values(initial_selected);
    for status in &statuses {
        multi = multi.item(status.client, status.client.display_name(), status.hint());
    }
    let selected: Vec<AgentClient> = multi.interact()?;

    let mut install_locations = Vec::new();
    let mut remove_locations = Vec::new();

    for status in &statuses {
        if selected.contains(&status.client) {
            for scope in status.scopes_needing_install() {
                install_locations.push(SkillLocation {
                    client: status.client,
                    scope,
                    path: skill_path_for(status.client, scope, &cwd, &home)?,
                });
            }
        } else {
            for scope in status.install_scopes() {
                let path = skill_path_for(status.client, scope, &cwd, &home)?;
                remove_locations.push(SkillLocation {
                    client: status.client,
                    scope,
                    path,
                });
            }
        }
    }

    if !remove_locations.is_empty() {
        do_remove(&remove_locations, false)?;
    }
    if !install_locations.is_empty() {
        do_install(&install_locations, false)?;
    }
    if selected.is_empty() {
        cliclack::outro("Done!")?;
        return Ok(());
    }
    if install_locations.is_empty() && remove_locations.is_empty() {
        cliclack::log::info("No skill changes needed. Installed selections are already current.")?;
    }
    offer_agents_md_snippet(&selected)?;
    cliclack::outro("Done!")?;
    Ok(())
}

fn do_install(locations: &[SkillLocation], json_output: bool) -> anyhow::Result<()> {
    if locations.is_empty() {
        return Ok(());
    }

    for location in locations {
        if location.path.exists() {
            fs::remove_dir_all(&location.path).with_context(|| {
                format!(
                    "failed to replace existing skill at {}",
                    location.path.display()
                )
            })?;
        }
        install_skill_to(&location.path)?;
    }

    let reports: Vec<SkillLocationReport> = locations
        .iter()
        .map(|location| SkillLocationReport {
            client: location.client,
            scope: location.scope,
            path: location.path.display().to_string(),
            installed: true,
            up_to_date: Some(true),
            removed: None,
        })
        .collect();

    if json_output {
        println!("{}", serde_json::to_string_pretty(&reports)?);
    } else {
        let green = console::Style::new().green();
        let dim = console::Style::new().dim();
        println!("Installed Vera skill:");
        println!();
        for report in &reports {
            let name = format!("{:?}", report.client).to_lowercase();
            let scope = format!("{:?}", report.scope).to_lowercase();
            println!(
                "  {} {:<7} {}",
                green.apply_to(format!("{:<14}", name)),
                scope,
                dim.apply_to(&report.path)
            );
        }
    }

    Ok(())
}

fn remove(
    client: Option<AgentClient>,
    scope: Option<AgentScope>,
    json_output: bool,
) -> anyhow::Result<()> {
    let (resolved_client, resolved_scope) = match (client, scope) {
        (Some(c), Some(s)) => (c, s),
        (Some(c), None) => (c, AgentScope::Global),
        (None, Some(s)) => (AgentClient::All, s),
        (None, None) if json_output => (AgentClient::All, AgentScope::All),
        (None, None) => return remove_interactive(),
    };

    let locations = resolve_locations(resolved_client, resolved_scope)?;
    do_remove(&locations, json_output)
}

fn remove_interactive() -> anyhow::Result<()> {
    let cwd = std::env::current_dir().context("failed to resolve current directory")?;
    let home = state::user_home_dir()?;
    let all_clients: Vec<AgentClient> = AgentClient::all_concrete().collect();

    let mut installed: Vec<(AgentClient, AgentScope, PathBuf)> = Vec::new();
    for &client in &all_clients {
        for scope in [AgentScope::Global, AgentScope::Project] {
            let path = skill_path_for(client, scope, &cwd, &home)?;
            if path.join("SKILL.md").exists() {
                installed.push((client, scope, path));
            }
        }
    }

    if installed.is_empty() {
        println!("No Vera skill installations found.");
        return Ok(());
    }

    cliclack::intro("vera agent remove")?;

    let mut multi = cliclack::multiselect("Select installations to remove");
    for (i, (c, s, p)) in installed.iter().enumerate() {
        let label = format!(
            "{} ({})",
            c.display_name(),
            format!("{:?}", s).to_lowercase()
        );
        let hint = p.display().to_string();
        multi = multi.item(i, label, hint);
    }
    let selected: Vec<usize> = multi.required(true).interact()?;

    let locations: Vec<SkillLocation> = selected
        .iter()
        .map(|&idx| {
            let (client, scope, path) = &installed[idx];
            SkillLocation {
                client: *client,
                scope: *scope,
                path: path.clone(),
            }
        })
        .collect();

    do_remove(&locations, false)?;
    cliclack::outro("Done!")?;
    Ok(())
}

fn do_remove(locations: &[SkillLocation], json_output: bool) -> anyhow::Result<()> {
    if locations.is_empty() {
        return Ok(());
    }

    let removal = remove_skill_locations(locations);

    // Report before failing: a location deleted earlier in the run stays visible
    // even when a later one cannot be deleted.
    if json_output {
        println!("{}", serde_json::to_string_pretty(&removal.reports)?);
    } else {
        write_removed_skill_locations(&removal, &mut std::io::stdout().lock())?;
    }

    let mut failures = removal.failures.into_iter();
    if let Some(first) = failures.next() {
        for error in failures {
            tracing::warn!("{error:#}");
        }
        return Err(first);
    }

    Ok(())
}

/// One report per location plus the deletions that failed. The two are returned
/// together because a failure partway through must not discard the locations
/// already deleted: callers report what happened, then decide whether to fail.
#[derive(Default)]
pub(crate) struct SkillRemoval {
    pub(crate) reports: Vec<SkillLocationReport>,
    pub(crate) failures: Vec<anyhow::Error>,
}

/// Delete the skill directory at each location, reporting per location whether
/// a skill was actually there and deleted. A location that cannot be inspected
/// or deleted is recorded as not removed, with the reason kept as a failure, and
/// does not stop the remaining locations.
fn remove_skill_locations(locations: &[SkillLocation]) -> SkillRemoval {
    let mut removal = SkillRemoval {
        reports: Vec::with_capacity(locations.len()),
        failures: Vec::new(),
    };

    for location in locations {
        let marker = location.path.join("SKILL.md");
        // `Path::exists` coerces a permission error into `false`, which reports an
        // installed skill as absent. `try_exists` keeps that call's symlink-following
        // semantics, so a broken `SKILL.md` symlink stays "not installed" exactly as
        // before, and only separates "cannot tell" from "not there".
        let installed = match marker.try_exists() {
            Ok(installed) => installed,
            Err(error) => {
                removal
                    .failures
                    .push(anyhow::Error::new(error).context(format!(
                        "failed to check for an installed skill at {}",
                        marker.display()
                    )));
                false
            }
        };
        let removed = installed
            && match fs::remove_dir_all(&location.path) {
                Ok(()) => true,
                Err(error) => {
                    removal
                        .failures
                        .push(anyhow::Error::new(error).context(format!(
                            "failed to remove installed skill at {}",
                            location.path.display()
                        )));
                    false
                }
            };
        removal.reports.push(SkillLocationReport {
            client: location.client,
            scope: location.scope,
            path: location.path.display().to_string(),
            installed: false,
            up_to_date: None,
            removed: Some(removed),
        });
    }

    removal
}

/// Remove every supported client/scope skill install under the given roots,
/// without printing anything. Used by `vera uninstall`, which folds the result
/// into its own single output document.
pub(crate) fn remove_all_skills(cwd: &Path, home: &Path) -> anyhow::Result<SkillRemoval> {
    let locations = resolve_locations_with_roots(AgentClient::All, AgentScope::All, cwd, home)?;
    Ok(remove_skill_locations(&locations))
}

pub(crate) fn write_removed_skill_locations(
    removal: &SkillRemoval,
    out: &mut dyn Write,
) -> std::io::Result<()> {
    let removed: Vec<&SkillLocationReport> = removal
        .reports
        .iter()
        .filter(|report| report.was_removed())
        .collect();
    // Only claim nothing was installed when nothing was found; a location that
    // was found and could not be deleted is a failure, not an absence.
    if removed.is_empty() && removal.failures.is_empty() {
        writeln!(out, "No Vera skill installations found.")?;
        return Ok(());
    }
    if removed.is_empty() {
        return Ok(());
    }

    let red = console::Style::new().red();
    let dim = console::Style::new().dim();
    writeln!(out, "Removed Vera skill from:")?;
    writeln!(out)?;
    for report in removed {
        let name = format!("{:?}", report.client).to_lowercase();
        let scope = format!("{:?}", report.scope).to_lowercase();
        writeln!(
            out,
            "  {} {:<7} {}",
            red.apply_to(format!("{:<14}", name)),
            scope,
            dim.apply_to(&report.path)
        )?;
    }

    Ok(())
}

fn status(client: AgentClient, scope: AgentScope, json_output: bool) -> anyhow::Result<()> {
    let cwd = std::env::current_dir().context("failed to resolve current directory")?;
    let home = state::user_home_dir()?;
    let statuses = collect_client_install_statuses(scope, &cwd, &home)?;

    let reports = statuses
        .into_iter()
        .filter(|status| client == AgentClient::All || status.client == client)
        .flat_map(|status| {
            let cwd = cwd.clone();
            let home = home.clone();
            status.scopes.into_iter().map(move |scope_status| {
                let path = skill_path_for(status.client, scope_status.scope, &cwd, &home)
                    .expect("status paths should always resolve");
                SkillLocationReport {
                    client: status.client,
                    scope: scope_status.scope,
                    path: path.display().to_string(),
                    installed: scope_status.installed,
                    up_to_date: scope_status.installed.then_some(scope_status.up_to_date),
                    removed: None,
                }
            })
        })
        .collect::<Vec<_>>();

    if json_output {
        println!("{}", serde_json::to_string_pretty(&reports)?);
    } else {
        let style = console::Style::new();
        let bold = style.clone().bold();
        let green = style.clone().green();
        let yellow = style.clone().yellow();
        let dim = style.clone().dim();

        let installed: Vec<_> = reports.iter().filter(|r| r.installed).collect();
        let missing: Vec<_> = reports.iter().filter(|r| !r.installed).collect();
        let stale: Vec<_> = installed
            .iter()
            .filter(|report| report.up_to_date == Some(false))
            .collect();

        if installed.is_empty() {
            println!("{}", dim.apply_to("No Vera skills installed."));
        } else {
            println!("{}", bold.apply_to("Installed:"));
            println!();
            for report in &installed {
                let name = format!("{:?}", report.client).to_lowercase();
                let scope = format!("{:?}", report.scope).to_lowercase();
                let marker = if report.up_to_date == Some(false) {
                    yellow.apply_to("stale")
                } else {
                    green.apply_to("current")
                };
                println!(
                    "  {} {:<7} {:<7} {}",
                    green.apply_to(format!("{:<14}", name)),
                    scope,
                    marker,
                    dim.apply_to(&report.path)
                );
            }
        }

        if !stale.is_empty() {
            println!();
            println!(
                "{} {}",
                bold.apply_to("Refresh:"),
                dim.apply_to("Run `vera agent sync` to update all stale installs.")
            );
        }

        if !missing.is_empty() {
            println!();
            let names: Vec<_> = missing
                .iter()
                .map(|r| {
                    format!(
                        "{} ({})",
                        format!("{:?}", r.client).to_lowercase(),
                        format!("{:?}", r.scope).to_lowercase()
                    )
                })
                .collect();
            println!(
                "{} {}",
                bold.apply_to("Not installed:"),
                dim.apply_to(names.join(", "))
            );
        }
    }

    Ok(())
}

fn install_skill_to(target_dir: &Path) -> anyhow::Result<()> {
    for file in VERA_SKILL_FILES {
        let path = target_dir.join(file.relative_path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(&path, file.contents)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    let version_path = target_dir.join(".version");
    fs::write(&version_path, env!("CARGO_PKG_VERSION"))
        .with_context(|| format!("failed to write {}", version_path.display()))?;
    Ok(())
}

#[derive(Debug, Clone)]
struct SkillLocation {
    client: AgentClient,
    scope: AgentScope,
    path: PathBuf,
}

fn resolve_locations(client: AgentClient, scope: AgentScope) -> anyhow::Result<Vec<SkillLocation>> {
    let cwd = std::env::current_dir().context("failed to resolve current directory")?;
    let home = state::user_home_dir()?;
    resolve_locations_with_roots(client, scope, &cwd, &home)
}

fn selected_clients_for(client: AgentClient) -> Vec<AgentClient> {
    match client {
        AgentClient::All => AgentClient::all_concrete().collect(),
        single => vec![single],
    }
}

pub(crate) fn all_skill_paths(cwd: Option<&Path>, home: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut paths = BTreeSet::new();
    let cwd_for_globals = cwd.unwrap_or(home);

    for client in AgentClient::all_concrete() {
        paths.insert(skill_path_for(
            client,
            AgentScope::Global,
            cwd_for_globals,
            home,
        )?);

        if let Some(cwd) = cwd {
            paths.insert(skill_path_for(client, AgentScope::Project, cwd, home)?);
        }
    }

    Ok(paths.into_iter().collect())
}

fn collect_client_install_statuses(
    scope: AgentScope,
    cwd: &Path,
    home: &Path,
) -> anyhow::Result<Vec<ClientInstallStatus>> {
    let current_version = env!("CARGO_PKG_VERSION");
    let scopes = match scope {
        AgentScope::All => vec![AgentScope::Global, AgentScope::Project],
        single => vec![single],
    };

    AgentClient::all_concrete()
        .map(|client| {
            let scopes = scopes
                .iter()
                .copied()
                .map(|scope| {
                    let path = skill_path_for(client, scope, cwd, home)?;
                    let installed = path.join("SKILL.md").exists();
                    let up_to_date = installed
                        && fs::read_to_string(path.join(".version"))
                            .unwrap_or_default()
                            .trim()
                            == current_version;
                    Ok(ScopeInstallStatus {
                        scope,
                        installed,
                        up_to_date,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok(ClientInstallStatus { client, scopes })
        })
        .collect()
}

fn stale_locations_from_statuses(
    statuses: &[ClientInstallStatus],
    cwd: &Path,
    home: &Path,
) -> anyhow::Result<Vec<SkillLocation>> {
    let mut locations = Vec::new();

    for status in statuses {
        for scope in status.stale_scopes() {
            locations.push(SkillLocation {
                client: status.client,
                scope,
                path: skill_path_for(status.client, scope, cwd, home)?,
            });
        }
    }

    Ok(locations)
}

fn resolve_locations_with_roots(
    client: AgentClient,
    scope: AgentScope,
    cwd: &Path,
    home: &Path,
) -> anyhow::Result<Vec<SkillLocation>> {
    let clients = match client {
        AgentClient::All => AgentClient::all_concrete().collect(),
        single => vec![single],
    };
    let scopes = match scope {
        AgentScope::All => vec![AgentScope::Global, AgentScope::Project],
        single => vec![single],
    };

    let mut locations = Vec::new();
    for client in clients {
        for scope in &scopes {
            locations.push(SkillLocation {
                client,
                scope: *scope,
                path: skill_path_for(client, *scope, cwd, home)?,
            });
        }
    }

    Ok(locations)
}

fn skill_path_for(
    client: AgentClient,
    scope: AgentScope,
    cwd: &Path,
    home: &Path,
) -> anyhow::Result<PathBuf> {
    if scope == AgentScope::All {
        bail!("scope=all is only valid before path resolution");
    }

    let base = match (client, scope) {
        (AgentClient::Agents, AgentScope::Global) => {
            home.join(".config").join("agents").join("skills")
        }
        (AgentClient::Agents, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Amp, AgentScope::Global) => {
            home.join(".config").join("agents").join("skills")
        }
        (AgentClient::Amp, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Antigravity, AgentScope::Global) => {
            home.join(".gemini").join("antigravity").join("skills")
        }
        (AgentClient::Antigravity, AgentScope::Project) => cwd.join(".agent").join("skills"),
        (AgentClient::Augment, AgentScope::Global) => home.join(".augment").join("skills"),
        (AgentClient::Augment, AgentScope::Project) => cwd.join(".augment").join("skills"),
        (AgentClient::Claude, AgentScope::Global) => home.join(".claude").join("skills"),
        (AgentClient::Claude, AgentScope::Project) => cwd.join(".claude").join("skills"),
        (AgentClient::Cline, AgentScope::Global) => home.join(".agents").join("skills"),
        (AgentClient::Cline, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Codebuff, AgentScope::Global) => home.join(".codebuff").join("skills"),
        (AgentClient::Codebuff, AgentScope::Project) => cwd.join(".codebuff").join("skills"),
        (AgentClient::Codebuddy, AgentScope::Global) => home.join(".codebuddy").join("skills"),
        (AgentClient::Codebuddy, AgentScope::Project) => cwd.join(".codebuddy").join("skills"),
        (AgentClient::Codex, AgentScope::Global) => home.join(".codex").join("skills"),
        (AgentClient::Codex, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Copilot, AgentScope::Global) => home.join(".copilot").join("skills"),
        (AgentClient::Copilot, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Cortex, AgentScope::Global) => {
            home.join(".snowflake").join("cortex").join("skills")
        }
        (AgentClient::Cortex, AgentScope::Project) => cwd.join(".cortex").join("skills"),
        (AgentClient::Crush, AgentScope::Global) => {
            home.join(".config").join("crush").join("skills")
        }
        (AgentClient::Crush, AgentScope::Project) => cwd.join(".crush").join("skills"),
        (AgentClient::Cursor, AgentScope::Global) => home.join(".cursor").join("skills"),
        (AgentClient::Cursor, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Droid, AgentScope::Global) => home.join(".factory").join("skills"),
        (AgentClient::Droid, AgentScope::Project) => cwd.join(".factory").join("skills"),
        (AgentClient::Gemini, AgentScope::Global) => home.join(".gemini").join("skills"),
        (AgentClient::Gemini, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Goose, AgentScope::Global) => {
            home.join(".config").join("goose").join("skills")
        }
        (AgentClient::Goose, AgentScope::Project) => cwd.join(".goose").join("skills"),
        (AgentClient::Iflow, AgentScope::Global) => home.join(".iflow").join("skills"),
        (AgentClient::Iflow, AgentScope::Project) => cwd.join(".iflow").join("skills"),
        (AgentClient::Junie, AgentScope::Global) => home.join(".junie").join("skills"),
        (AgentClient::Junie, AgentScope::Project) => cwd.join(".junie").join("skills"),
        (AgentClient::Kilo, AgentScope::Global) => home.join(".kilocode").join("skills"),
        (AgentClient::Kilo, AgentScope::Project) => cwd.join(".kilocode").join("skills"),
        (AgentClient::Kiro, AgentScope::Global) => home.join(".kiro").join("skills"),
        (AgentClient::Kiro, AgentScope::Project) => cwd.join(".kiro").join("skills"),
        (AgentClient::Kimi, AgentScope::Global) => {
            home.join(".config").join("agents").join("skills")
        }
        (AgentClient::Kimi, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Vibe, AgentScope::Global) => home.join(".vibe").join("skills"),
        (AgentClient::Vibe, AgentScope::Project) => cwd.join(".vibe").join("skills"),
        (AgentClient::Mux, AgentScope::Global) => home.join(".mux").join("skills"),
        (AgentClient::Mux, AgentScope::Project) => cwd.join(".mux").join("skills"),
        (AgentClient::Opencode, AgentScope::Global) => {
            home.join(".config").join("opencode").join("skills")
        }
        (AgentClient::Opencode, AgentScope::Project) => cwd.join(".agents").join("skills"),
        (AgentClient::Openhands, AgentScope::Global) => home.join(".openhands").join("skills"),
        (AgentClient::Openhands, AgentScope::Project) => cwd.join(".openhands").join("skills"),
        (AgentClient::Pi, AgentScope::Global) => home.join(".pi").join("agent").join("skills"),
        (AgentClient::Pi, AgentScope::Project) => cwd.join(".pi").join("skills"),
        (AgentClient::Qwen, AgentScope::Global) => home.join(".qwen").join("skills"),
        (AgentClient::Qwen, AgentScope::Project) => cwd.join(".qwen").join("skills"),
        (AgentClient::Roo, AgentScope::Global) => home.join(".roo").join("skills"),
        (AgentClient::Roo, AgentScope::Project) => cwd.join(".roo").join("skills"),
        (AgentClient::Trae, AgentScope::Global) => home.join(".trae").join("skills"),
        (AgentClient::Trae, AgentScope::Project) => cwd.join(".trae").join("skills"),
        (AgentClient::Windsurf, AgentScope::Global) => {
            home.join(".codeium").join("windsurf").join("skills")
        }
        (AgentClient::Windsurf, AgentScope::Project) => cwd.join(".windsurf").join("skills"),
        (AgentClient::Zed, AgentScope::Global) => home.join(".zed").join("skills"),
        (AgentClient::Zed, AgentScope::Project) => cwd.join(".zed").join("skills"),
        (AgentClient::All, _) => bail!("client=all is only valid before path resolution"),
        (_, AgentScope::All) => bail!("scope=all is only valid before path resolution"),
    };

    Ok(base.join(VERA_SKILL_NAME))
}

fn sync(client: AgentClient, scope: AgentScope, json_output: bool) -> anyhow::Result<()> {
    sync_with_options(client, scope, json_output, true, false)
}

/// Automatic staleness sync: refresh skill installs only, without touching
/// project markdown or writing over the user's command output.
pub(crate) fn sync_skills_only() -> anyhow::Result<()> {
    sync_with_options(AgentClient::All, AgentScope::All, false, false, true)
}

#[derive(Debug)]
struct SyncOutcome {
    updated: Vec<PathBuf>,
    refreshed_snippets: Vec<PathBuf>,
    /// `--scope all` was asked for but only the global half could run.
    project_scope_skipped: bool,
}

/// Whether project-scoped paths under `cwd` can be resolved at all.
///
/// `getcwd` still succeeds on Linux for a directory the process has no search
/// permission on, so `project_cwd` can be `Some` for a directory every
/// `Path::exists()` probe underneath reports as missing. Stat a path inside it
/// so that access error is not read as "nothing is installed".
fn project_root_is_searchable(cwd: &Path) -> bool {
    fs::metadata(cwd.join(".")).is_ok()
}

fn sync_to_roots(
    client: AgentClient,
    scope: AgentScope,
    project_cwd: Option<&Path>,
    home: &Path,
    refresh_project_snippets: bool,
) -> anyhow::Result<SyncOutcome> {
    let project_cwd = project_cwd.filter(|cwd| project_root_is_searchable(cwd));
    let scan_scope = match (scope, project_cwd) {
        (AgentScope::Project, None) => {
            bail!("--scope project needs a readable current directory")
        }
        (AgentScope::All, None) => AgentScope::Global,
        (scope, _) => scope,
    };
    let cwd = project_cwd.unwrap_or(home);
    let statuses: Vec<ClientInstallStatus> =
        collect_client_install_statuses(scan_scope, cwd, home)?
            .into_iter()
            .filter(|status| client == AgentClient::All || status.client == client)
            .collect();

    let mut updated = Vec::new();
    for location in stale_locations_from_statuses(&statuses, cwd, home)? {
        install_skill_to(&location.path)?;
        updated.push(location.path);
    }

    // Refresh managed markdown snippets whenever a project directory is
    // available, regardless of which skill installs were stale. They live in
    // the project directory, so a global-scoped sync must leave them alone.
    let refreshed_snippets = match project_cwd {
        Some(cwd) if refresh_project_snippets && scan_scope != AgentScope::Global => {
            refresh_existing_vera_snippets(&find_agent_configs(cwd))?
        }
        _ => Vec::new(),
    };

    Ok(SyncOutcome {
        updated,
        refreshed_snippets,
        project_scope_skipped: scope == AgentScope::All && scan_scope == AgentScope::Global,
    })
}

fn sync_with_options(
    client: AgentClient,
    scope: AgentScope,
    json_output: bool,
    refresh_project_snippets: bool,
    quiet: bool,
) -> anyhow::Result<()> {
    let home = state::user_home_dir()?;
    let project_cwd = std::env::current_dir().ok();
    let current_version = env!("CARGO_PKG_VERSION");
    let SyncOutcome {
        updated,
        refreshed_snippets,
        project_scope_skipped,
    } = sync_to_roots(
        client,
        scope,
        project_cwd.as_deref(),
        &home,
        refresh_project_snippets,
    )?;

    if quiet {
        return Ok(());
    }

    // Half of what `--scope all` asked for could not run. The global half is
    // still work the user wants, so proceed, but do not let the project half
    // vanish into a report that reads as a complete success.
    if project_scope_skipped {
        eprintln!(
            "Warning: --scope all needs a readable current directory for the project scope. \
             Synced global scope only."
        );
    }

    if json_output {
        let reports: Vec<_> = updated
            .iter()
            .map(|p| {
                serde_json::json!({
                    "path": p.display().to_string(),
                    "version": current_version,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&reports)?);
    } else if updated.is_empty() {
        println!("All installed skills are up to date (v{current_version}).");
    } else {
        let green = console::Style::new().green();
        let dim = console::Style::new().dim();
        println!(
            "Updated {} skill install(s) to v{current_version}:",
            updated.len()
        );
        println!();
        for path in &updated {
            println!("  {} {}", green.apply_to("✓"), dim.apply_to(path.display()));
        }
    }

    if !json_output && !refreshed_snippets.is_empty() {
        let green = console::Style::new().green();
        let dim = console::Style::new().dim();
        println!();
        println!("Refreshed Vera snippet in:");
        println!();
        for path in &refreshed_snippets {
            println!("  {} {}", green.apply_to("✓"), dim.apply_to(path.display()));
        }
    }

    Ok(())
}

/// The snippet Vera offers to inject into agent config files.
const VERA_SNIPPET_BEGIN_MARKER: &str = "<!-- vera:begin -->";
const VERA_SNIPPET_END_MARKER: &str = "<!-- vera:end -->";
const AGENTS_MD_SNIPPET_HEADING: &str = "## Code Search";
const AGENTS_MD_SNIPPET_INTRO: &str = "Use Vera before opening many files or running broad text search when you need to find where logic lives or how a feature works.";

const AGENTS_MD_SNIPPET: &str = r#"## Code Search

<!-- vera:begin -->

Use Vera before opening many files or running broad text search when you need to find where logic lives or how a feature works.

- `vera search "query"` for semantic code search. Describe behavior: "JWT validation", not "auth". If one phrasing misses, try 2-3 varied queries or add `--intent "goal"`.
- `vera search ... --changed`, `--since <rev>`, or `--base <rev>` when the task is limited to modified files or a PR diff
- `vera grep "pattern"` for exact text or regex in indexed files
- `vera structural definitions <symbol>`, `vera structural env <NAME>`, `vera structural routes`, or `vera structural impls <symbol>` for common structural tasks and explicit type relationships
- `vera explain-path path/to/file` to explain why a file is or is not indexed
- `vera references <symbol>` for callers and `vera references <symbol> --callees` for callees
- `vera overview` for a project summary (languages, entry points, hotspots). Add `--changed`, `--since <rev>`, or `--base <rev>` to scope it to modified files.
- `vera stats --json` for index health, including tree-sitter error, parse-failure, and Tier 0 fallback counts
- `vera search --deep "query"` for RAG-fusion query expansion + merged ranking
- Narrow `vera search` or `vera grep` with `--lang`, `--path`, `--type`, or `--scope docs`
- `vera watch .` to auto-update the index, or `vera update .` after edits (`vera index .` if `.vera/` is missing)
- For detailed usage, query patterns, and troubleshooting, read the Vera skill file installed by `vera agent install`
<!-- vera:end -->
"#;

#[derive(Debug, Clone, Copy)]
struct AgentConfigFile {
    name: &'static str,
    description: &'static str,
}

#[derive(Debug, Clone)]
struct DetectedAgentConfig {
    file: AgentConfigFile,
    path: PathBuf,
    mentions_vera: bool,
}

/// Known agent config filenames to check.
const AGENT_CONFIG_FILES: &[AgentConfigFile] = &[
    AgentConfigFile {
        name: "AGENTS.md",
        description: "shared agent instructions used by many tools",
    },
    AgentConfigFile {
        name: "CLAUDE.md",
        description: "Claude Code project instructions",
    },
    AgentConfigFile {
        name: "COPILOT.md",
        description: "GitHub Copilot coding agent instructions",
    },
    AgentConfigFile {
        name: ".cursorrules",
        description: "Cursor project rules",
    },
    AgentConfigFile {
        name: ".clinerules",
        description: "Cline project rules",
    },
    AgentConfigFile {
        name: ".windsurfrules",
        description: "Windsurf project rules",
    },
];

fn preferred_config_filename(selected_clients: &[AgentClient]) -> &'static str {
    if selected_clients.len() != 1 {
        return "AGENTS.md";
    }

    match selected_clients[0] {
        AgentClient::Claude => "CLAUDE.md",
        AgentClient::Copilot => "COPILOT.md",
        AgentClient::Cursor => ".cursorrules",
        AgentClient::Cline => ".clinerules",
        AgentClient::Windsurf => ".windsurfrules",
        _ => "AGENTS.md",
    }
}

fn find_agent_configs(cwd: &Path) -> Vec<DetectedAgentConfig> {
    AGENT_CONFIG_FILES
        .iter()
        .filter_map(|file| {
            let path = cwd.join(file.name);
            if !path.is_file() {
                return None;
            }

            let mentions_vera = fs::read_to_string(&path)
                .map(|content| {
                    let lower = content.to_lowercase();
                    lower.contains("vera search")
                        || lower.contains("vera grep")
                        || lower.contains("vera update")
                        || lower.contains("vera references")
                        || lower.contains("vera overview")
                        || lower.contains("vera watch")
                })
                .unwrap_or(false);

            Some(DetectedAgentConfig {
                file: *file,
                path,
                mentions_vera,
            })
        })
        .collect()
}

fn insert_vera_snippet(existing: &str, file_name: &str) -> String {
    if file_name.ends_with(".md") {
        return insert_vera_snippet_into_markdown(existing);
    }

    let mut content = String::new();
    content.push_str(AGENTS_MD_SNIPPET.trim_end());
    content.push_str("\n\n");
    content.push_str(existing.trim_start_matches('\n'));
    if !content.ends_with('\n') {
        content.push('\n');
    }
    content
}

fn refresh_existing_vera_snippets(
    existing: &[DetectedAgentConfig],
) -> anyhow::Result<Vec<PathBuf>> {
    let mut updated = Vec::new();

    for config in existing {
        let content = fs::read_to_string(&config.path)
            .with_context(|| format!("failed to read {}", config.path.display()))?;
        let Some(refreshed) = refresh_vera_snippet(&content, config.file.name) else {
            continue;
        };
        if refreshed == content {
            continue;
        }

        fs::write(&config.path, refreshed)
            .with_context(|| format!("failed to write {}", config.path.display()))?;
        updated.push(config.path.clone());
    }

    Ok(updated)
}

fn refresh_vera_snippet(existing: &str, file_name: &str) -> Option<String> {
    if file_name.ends_with(".md") {
        return refresh_vera_snippet_in_markdown(existing);
    }

    None
}

fn refresh_vera_snippet_in_markdown(existing: &str) -> Option<String> {
    if let Some(begin) = markdown_marker_start(existing, VERA_SNIPPET_BEGIN_MARKER) {
        let content_start = begin + VERA_SNIPPET_BEGIN_MARKER.len();
        let end_marker =
            markdown_marker_start(&existing[content_start..], VERA_SNIPPET_END_MARKER)?;
        let end_marker = content_start + end_marker;
        let mut content = String::with_capacity(existing.len());
        content.push_str(&existing[..content_start]);
        content.push_str(&adapt_line_endings(marked_snippet_body(), existing));
        content.push_str(&existing[end_marker..]);
        return Some(content);
    }

    let start = markdown_section_start(existing, AGENTS_MD_SNIPPET_HEADING)?;
    // The managed section ends at the next heading of equal or higher level
    // (`## ` or `# `). Deeper headings (`### ` and below) stay in the section.
    let body_start = start + AGENTS_MD_SNIPPET_HEADING.len();
    let end = existing[body_start..]
        .match_indices('\n')
        .map(|(idx, _)| body_start + idx + 1)
        .find(|&line_start| {
            let line = &existing[line_start..];
            line.starts_with("# ") || line.starts_with("## ")
        })
        .unwrap_or(existing.len());
    let section = &existing[start..end];

    let legacy_snippet = legacy_agents_md_snippet();
    let legacy_snippet = adapt_line_endings(&legacy_snippet, existing);
    if !section.contains(AGENTS_MD_SNIPPET_INTRO) || section.trim_end() != legacy_snippet.trim_end()
    {
        return None;
    }

    let section_without_trailing_whitespace = section.trim_end();
    let trailing_whitespace = &section[section_without_trailing_whitespace.len()..];
    let mut replacement = adapt_line_endings(AGENTS_MD_SNIPPET.trim_end(), existing).into_owned();
    if trailing_whitespace.is_empty() {
        replacement.push('\n');
    } else {
        replacement.push_str(trailing_whitespace);
    }

    let mut content = String::with_capacity(existing.len());
    content.push_str(&existing[..start]);
    content.push_str(&replacement);
    content.push_str(&existing[end..]);
    Some(content)
}

fn marked_snippet_body() -> &'static str {
    let (_, body) = AGENTS_MD_SNIPPET
        .split_once(VERA_SNIPPET_BEGIN_MARKER)
        .expect("AGENTS_MD_SNIPPET must contain the begin marker");
    let (body, _) = body
        .split_once(VERA_SNIPPET_END_MARKER)
        .expect("AGENTS_MD_SNIPPET must contain the end marker");
    body
}

fn legacy_agents_md_snippet() -> String {
    AGENTS_MD_SNIPPET
        .replace(&format!("{VERA_SNIPPET_BEGIN_MARKER}\n\n"), "")
        .replace(&format!("\n{VERA_SNIPPET_END_MARKER}"), "")
}

/// Match generated content's line endings to the target file's.
fn adapt_line_endings<'a>(generated: &'a str, file: &str) -> std::borrow::Cow<'a, str> {
    if file.contains("\r\n") {
        std::borrow::Cow::Owned(generated.replace('\n', "\r\n"))
    } else {
        std::borrow::Cow::Borrowed(generated)
    }
}

fn markdown_marker_start(existing: &str, marker: &str) -> Option<usize> {
    let mut offset = 0;
    for line in existing.split_inclusive('\n') {
        if line.trim_end() == marker {
            return Some(offset);
        }
        offset += line.len();
    }
    None
}

fn markdown_section_start(existing: &str, heading: &str) -> Option<usize> {
    let mut offset = 0;
    for line in existing.split_inclusive('\n') {
        if line.trim_end() == heading {
            return Some(offset);
        }
        offset += line.len();
    }
    None
}

fn insert_vera_snippet_into_markdown(existing: &str) -> String {
    let heading_insert_pos = existing
        .lines()
        .next()
        .filter(|line| line.trim_start().starts_with("# "))
        .map(|first_line| {
            let mut insert_pos = first_line.len();
            if existing.as_bytes().get(insert_pos) == Some(&b'\n') {
                insert_pos += 1;
            }
            while let Some(rest) = existing.get(insert_pos..) {
                if rest.is_empty() {
                    break;
                }
                let next_newline = rest.find('\n').map(|idx| idx + 1).unwrap_or(rest.len());
                let line = &rest[..next_newline];
                if line.trim().is_empty() {
                    insert_pos += next_newline;
                    continue;
                }
                break;
            }
            insert_pos
        })
        .unwrap_or(0);

    let (head, tail) = existing.split_at(heading_insert_pos);
    let mut content = String::new();
    content.push_str(head);
    if !content.is_empty() && !content.ends_with("\n\n") {
        if content.ends_with('\n') {
            content.push('\n');
        } else {
            content.push_str("\n\n");
        }
    }
    content.push_str(AGENTS_MD_SNIPPET.trim_end());
    if !tail.is_empty() {
        content.push_str("\n\n");
        content.push_str(tail.trim_start_matches('\n'));
    } else {
        content.push('\n');
    }
    if !content.ends_with('\n') {
        content.push('\n');
    }
    content
}

fn choose_existing_agent_config(
    existing: &[DetectedAgentConfig],
    preferred_name: &str,
) -> anyhow::Result<PathBuf> {
    if existing.len() == 1 {
        let name = existing[0].file.name;
        let yes: bool = cliclack::confirm(format!("Add Vera usage snippet to {name}?"))
            .initial_value(true)
            .interact()?;
        if !yes {
            return Ok(PathBuf::new());
        }
        return Ok(existing[0].path.clone());
    }

    let preferred_idx = existing
        .iter()
        .position(|config| config.file.name == preferred_name)
        .unwrap_or(0);
    let mut select = cliclack::select("Choose agent config file for the Vera usage snippet")
        .initial_value(preferred_idx);
    for (idx, config) in existing.iter().enumerate() {
        select = select.item(
            idx,
            config.file.name,
            format!("existing file: {}", config.file.description),
        );
    }
    let selected_idx: usize = select.interact()?;
    Ok(existing[selected_idx].path.clone())
}

fn choose_new_agent_config_path(cwd: &Path, preferred_name: &str) -> anyhow::Result<PathBuf> {
    let preferred_idx = AGENT_CONFIG_FILES
        .iter()
        .position(|file| file.name == preferred_name)
        .unwrap_or(0);
    let mut select = cliclack::select("Choose which agent config file Vera should create")
        .initial_value(preferred_idx);
    for (idx, file) in AGENT_CONFIG_FILES.iter().enumerate() {
        select = select.item(idx, file.name, file.description);
    }
    let selected_idx: usize = select.interact()?;
    Ok(cwd.join(AGENT_CONFIG_FILES[selected_idx].name))
}

/// After skill install, offer to add a Vera snippet to the project's agent config file.
fn offer_agents_md_snippet(selected_clients: &[AgentClient]) -> anyhow::Result<()> {
    let cwd = std::env::current_dir().context("failed to resolve current directory")?;
    let existing = find_agent_configs(&cwd);

    for path in refresh_existing_vera_snippets(&existing)? {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        cliclack::log::success(format!("Updated Vera snippet in {name}"))?;
    }

    if existing.iter().any(|config| config.mentions_vera) {
        return Ok(());
    }

    let preferred_name = preferred_config_filename(selected_clients);
    let target_path = if existing.is_empty() {
        let yes: bool = cliclack::confirm(
            "No agent config file found. Create one with Vera usage instructions?",
        )
        .initial_value(true)
        .interact()?;
        if !yes {
            return Ok(());
        }
        choose_new_agent_config_path(&cwd, preferred_name)?
    } else {
        choose_existing_agent_config(&existing, preferred_name)?
    };
    if target_path.as_os_str().is_empty() {
        return Ok(());
    }

    let action = if target_path.is_file() {
        "Updated"
    } else {
        "Created"
    };
    let content = if target_path.is_file() {
        let existing = fs::read_to_string(&target_path)?;
        let name = target_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        insert_vera_snippet(&existing, &name)
    } else {
        let mut content = AGENTS_MD_SNIPPET.trim_end().to_string();
        content.push('\n');
        content
    };
    fs::write(&target_path, &content)?;

    let name = target_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy();
    cliclack::log::success(format!("{action} Vera snippet in {name}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn resolve_locations_expands_scope_all() {
        let cwd = Path::new("/tmp/project");
        let home = Path::new("/tmp/home");
        let locations =
            resolve_locations_with_roots(AgentClient::Codex, AgentScope::All, cwd, home).unwrap();

        assert_eq!(locations.len(), 2);
        assert!(
            locations
                .iter()
                .any(|location| location.scope == AgentScope::Global)
        );
        assert!(
            locations
                .iter()
                .any(|location| location.scope == AgentScope::Project)
        );
    }

    #[test]
    fn copilot_project_skill_uses_agents_dir() {
        let cwd = Path::new("/tmp/project");
        let home = Path::new("/tmp/home");
        let path = skill_path_for(AgentClient::Copilot, AgentScope::Project, cwd, home).unwrap();
        assert_eq!(path, PathBuf::from("/tmp/project/.agents/skills/vera"));
    }

    #[test]
    fn all_concrete_clients_have_paths() {
        let cwd = Path::new("/tmp/project");
        let home = Path::new("/tmp/home");
        for client in AgentClient::all_concrete() {
            skill_path_for(client, AgentScope::Global, cwd, home)
                .unwrap_or_else(|_| panic!("no global path for {:?}", client));
            skill_path_for(client, AgentScope::Project, cwd, home)
                .unwrap_or_else(|_| panic!("no project path for {:?}", client));
        }
    }

    #[test]
    fn all_skill_paths_dedup_shared_directories() {
        let cwd = Path::new("/tmp/project");
        let home = Path::new("/tmp/home");
        let paths = all_skill_paths(Some(cwd), home).unwrap();

        assert!(paths.contains(&PathBuf::from("/tmp/home/.codex/skills/vera")));
        assert!(paths.contains(&PathBuf::from("/tmp/project/.agents/skills/vera")));
        assert_eq!(
            paths
                .iter()
                .filter(|path| path.as_os_str() == "/tmp/project/.agents/skills/vera")
                .count(),
            1
        );
    }

    #[test]
    fn preferred_config_filename_uses_agents_for_gemini() {
        assert_eq!(
            preferred_config_filename(&[AgentClient::Gemini]),
            "AGENTS.md"
        );
        assert_eq!(
            preferred_config_filename(&[AgentClient::Claude]),
            "CLAUDE.md"
        );
        assert_eq!(
            preferred_config_filename(&[AgentClient::Gemini, AgentClient::Claude]),
            "AGENTS.md"
        );
    }

    #[test]
    fn insert_vera_snippet_into_markdown_places_section_after_title() {
        let existing = "# Repository Guidelines\n\n## Build\n\nRun tests.\n";
        let updated = insert_vera_snippet_into_markdown(existing);
        let expected_prefix = "# Repository Guidelines\n\n## Code Search\n";
        assert!(updated.starts_with(expected_prefix), "{updated}");
        assert!(updated.contains("## Build"));
    }

    #[test]
    fn insert_vera_snippet_prepends_rules_files() {
        let existing = "- prefer concise answers\n- run tests first\n";
        let updated = insert_vera_snippet(existing, ".cursorrules");
        assert!(updated.starts_with("## Code Search\n"));
        assert!(updated.contains("- prefer concise answers"));
    }

    #[test]
    fn refresh_vera_snippet_in_markdown_refreshes_marked_region() {
        let existing = format!(
            "# Repository Guidelines\n\n## Code Search\n\nUser-owned context.\n{VERA_SNIPPET_BEGIN_MARKER}\n\nOld generated content.\n{VERA_SNIPPET_END_MARKER}\n\nUser-owned note.\n\n## Build\n\nRun tests.\n"
        );

        let updated = refresh_vera_snippet_in_markdown(&existing).unwrap();

        assert!(updated.contains(&format!(
            "{VERA_SNIPPET_BEGIN_MARKER}{}{VERA_SNIPPET_END_MARKER}",
            marked_snippet_body()
        )));
        assert!(updated.contains("User-owned context.\n"));
        assert!(updated.contains("User-owned note.\n"));
        assert!(!updated.contains("Old generated content."));
    }

    #[test]
    fn refresh_vera_snippet_in_markdown_migrates_identical_legacy_section() {
        let existing = format!(
            "# Repository Guidelines\n\n{}\n\n# Guidelines\n\nRun tests first.\n",
            legacy_agents_md_snippet().trim_end()
        );

        let updated = refresh_vera_snippet_in_markdown(&existing).unwrap();

        assert!(updated.contains(AGENTS_MD_SNIPPET.trim_end()));
        assert!(updated.contains(VERA_SNIPPET_BEGIN_MARKER));
        assert!(updated.contains(VERA_SNIPPET_END_MARKER));
        assert!(updated.contains("# Guidelines\n\nRun tests first.\n"));
    }

    #[test]
    fn refresh_vera_snippet_in_markdown_preserves_crlf_line_endings() {
        let existing = format!(
            "# Repo\r\n\r\n{VERA_SNIPPET_BEGIN_MARKER}\r\n\r\nOld generated content.\r\n{VERA_SNIPPET_END_MARKER}\r\n"
        );

        let updated = refresh_vera_snippet_in_markdown(&existing).unwrap();

        assert!(updated.contains(marked_snippet_body().lines().next().unwrap()));
        assert!(!updated.contains("Old generated content."));
        // No bare LF: every line feed is part of a CRLF pair.
        assert_eq!(
            updated.matches('\n').count(),
            updated.matches("\r\n").count()
        );
    }

    #[test]
    fn refresh_vera_snippet_in_markdown_migrates_crlf_legacy_section() {
        let legacy = legacy_agents_md_snippet().replace('\n', "\r\n");
        let existing = format!("# Repo\r\n\r\n{}\r\n", legacy.trim_end());

        let updated = refresh_vera_snippet_in_markdown(&existing).unwrap();

        assert!(updated.contains(VERA_SNIPPET_BEGIN_MARKER));
        assert_eq!(
            updated.matches('\n').count(),
            updated.matches("\r\n").count()
        );
    }

    #[test]
    fn refresh_vera_snippet_in_markdown_skips_edited_legacy_section() {
        let edited = legacy_agents_md_snippet().replace(
            "- `vera grep \"pattern\"` for exact text or regex in indexed files",
            "- Use my preferred search tool instead",
        );
        let existing = format!(
            "# Repository Guidelines\n\n{}\n\n## Build\n\nRun tests.\n",
            edited.trim_end()
        );

        assert!(refresh_vera_snippet_in_markdown(&existing).is_none());
    }

    #[test]
    fn refresh_vera_snippet_in_markdown_skips_similar_edited_heading() {
        let existing = format!(
            "# Repository Guidelines\n\n## Code Search (Vera)\n\n{AGENTS_MD_SNIPPET_INTRO}\n\n- Use my preferred search tool instead.\n\n## Build\n\nRun tests.\n"
        );

        assert!(refresh_vera_snippet_in_markdown(&existing).is_none());
    }

    #[test]
    fn collect_client_install_statuses_tracks_installed_and_stale_scopes() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let cwd = temp.path().join("project");
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&cwd).unwrap();

        let global = skill_path_for(AgentClient::Claude, AgentScope::Global, &cwd, &home).unwrap();
        fs::create_dir_all(&global).unwrap();
        fs::write(global.join("SKILL.md"), "test").unwrap();
        fs::write(global.join(".version"), "0.0.0").unwrap();

        let project =
            skill_path_for(AgentClient::Claude, AgentScope::Project, &cwd, &home).unwrap();
        fs::create_dir_all(&project).unwrap();
        fs::write(project.join("SKILL.md"), "test").unwrap();
        fs::write(project.join(".version"), env!("CARGO_PKG_VERSION")).unwrap();

        let statuses = collect_client_install_statuses(AgentScope::All, &cwd, &home).unwrap();
        let claude = statuses
            .iter()
            .find(|status| status.client == AgentClient::Claude)
            .unwrap();

        assert!(claude.is_installed());
        assert_eq!(
            claude.install_scopes().collect::<Vec<_>>(),
            vec![AgentScope::Global, AgentScope::Project]
        );
        assert_eq!(
            claude.scopes_needing_install().collect::<Vec<_>>(),
            vec![AgentScope::Global]
        );
    }

    #[test]
    fn stale_locations_only_include_installed_stale_scopes() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let cwd = temp.path().join("project");
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&cwd).unwrap();

        let stale = skill_path_for(AgentClient::Claude, AgentScope::Global, &cwd, &home).unwrap();
        fs::create_dir_all(&stale).unwrap();
        fs::write(stale.join("SKILL.md"), "test").unwrap();
        fs::write(stale.join(".version"), "0.0.0").unwrap();

        let fresh = skill_path_for(AgentClient::Claude, AgentScope::Project, &cwd, &home).unwrap();
        fs::create_dir_all(&fresh).unwrap();
        fs::write(fresh.join("SKILL.md"), "test").unwrap();
        fs::write(fresh.join(".version"), env!("CARGO_PKG_VERSION")).unwrap();

        let missing_skill_dir =
            skill_path_for(AgentClient::Gemini, AgentScope::Global, &cwd, &home).unwrap();
        fs::create_dir_all(missing_skill_dir.parent().unwrap()).unwrap();

        let statuses = collect_client_install_statuses(AgentScope::All, &cwd, &home).unwrap();
        let stale_locations = stale_locations_from_statuses(&statuses, &cwd, &home).unwrap();

        assert_eq!(stale_locations.len(), 1);
        assert_eq!(stale_locations[0].client, AgentClient::Claude);
        assert_eq!(stale_locations[0].scope, AgentScope::Global);
    }

    const SCOPED_SYNC_FIXTURE: &[(AgentClient, AgentScope)] = &[
        (AgentClient::Claude, AgentScope::Global),
        (AgentClient::Claude, AgentScope::Project),
        (AgentClient::Gemini, AgentScope::Global),
        (AgentClient::Gemini, AgentScope::Project),
    ];

    #[test]
    fn sync_to_roots_only_writes_the_requested_client_and_scope() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let cwd = temp.path().join("project");
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&cwd).unwrap();

        for &(client, scope) in SCOPED_SYNC_FIXTURE {
            let path = skill_path_for(client, scope, &cwd, &home).unwrap();
            fs::create_dir_all(&path).unwrap();
            fs::write(path.join("SKILL.md"), "stale").unwrap();
            fs::write(path.join(".version"), "0.0.0").unwrap();
        }

        let outcome = sync_to_roots(
            AgentClient::Claude,
            AgentScope::Project,
            Some(cwd.as_path()),
            &home,
            true,
        )
        .unwrap();

        let target = skill_path_for(AgentClient::Claude, AgentScope::Project, &cwd, &home).unwrap();
        assert_eq!(outcome.updated, vec![target.clone()]);

        for &(client, scope) in SCOPED_SYNC_FIXTURE {
            let path = skill_path_for(client, scope, &cwd, &home).unwrap();
            let version = fs::read_to_string(path.join(".version")).unwrap();
            let expected = if path == target {
                env!("CARGO_PKG_VERSION")
            } else {
                "0.0.0"
            };
            assert_eq!(
                version,
                expected,
                "{client:?}/{scope:?} at {} was rewritten by a sync scoped to claude/project",
                path.display()
            );
        }
    }

    #[test]
    fn sync_to_roots_leaves_project_markdown_alone_when_scoped_global() {
        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let cwd = temp.path().join("project");
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&cwd).unwrap();

        let stale = skill_path_for(AgentClient::Claude, AgentScope::Global, &cwd, &home).unwrap();
        fs::create_dir_all(&stale).unwrap();
        fs::write(stale.join("SKILL.md"), "stale").unwrap();
        fs::write(stale.join(".version"), "0.0.0").unwrap();

        let claude_md = cwd.join("CLAUDE.md");
        let managed = format!(
            "# Project\n\n## Code Search\n\n{VERA_SNIPPET_BEGIN_MARKER}\n\nOld generated content.\n{VERA_SNIPPET_END_MARKER}\n"
        );
        fs::write(&claude_md, &managed).unwrap();

        let outcome = sync_to_roots(
            AgentClient::All,
            AgentScope::Global,
            Some(cwd.as_path()),
            &home,
            true,
        )
        .unwrap();

        assert_eq!(outcome.updated, vec![stale]);
        assert!(outcome.refreshed_snippets.is_empty());
        assert_eq!(fs::read_to_string(&claude_md).unwrap(), managed);
    }

    /// Restores a directory's mode on unwind so a failing test cannot leave an
    /// unreadable directory behind for `TempDir` to fail cleaning up.
    #[cfg(unix)]
    struct RestoreMode(PathBuf);

    #[cfg(unix)]
    impl Drop for RestoreMode {
        fn drop(&mut self) {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o755));
        }
    }

    #[cfg(unix)]
    #[test]
    fn sync_to_roots_rejects_a_project_scope_it_cannot_search() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let cwd = temp.path().join("project");
        fs::create_dir_all(&home).unwrap();

        let stale = skill_path_for(AgentClient::Claude, AgentScope::Project, &cwd, &home).unwrap();
        fs::create_dir_all(&stale).unwrap();
        fs::write(stale.join("SKILL.md"), "stale").unwrap();
        fs::write(stale.join(".version"), "0.0.0").unwrap();

        fs::set_permissions(&cwd, fs::Permissions::from_mode(0o000)).unwrap();
        let _restore = RestoreMode(cwd.clone());
        if fs::metadata(cwd.join(".")).is_ok() {
            // Running as root, or on a filesystem that ignores mode bits.
            return;
        }

        let error = sync_to_roots(
            AgentClient::All,
            AgentScope::Project,
            Some(cwd.as_path()),
            &home,
            true,
        )
        .expect_err("an unsearchable project directory must not report success");
        assert!(
            error.to_string().contains("current directory"),
            "unexpected error: {error}"
        );
    }

    /// `--scope all` keeps the global half rather than failing outright, so the
    /// only thing standing between "half the request ran" and a report that
    /// reads as complete success is this flag.
    #[cfg(unix)]
    #[test]
    fn sync_to_roots_flags_a_project_scope_all_could_not_search() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempdir().unwrap();
        let home = temp.path().join("home");
        let cwd = temp.path().join("project");
        fs::create_dir_all(&home).unwrap();

        for &(client, scope) in SCOPED_SYNC_FIXTURE {
            let path = skill_path_for(client, scope, &cwd, &home).unwrap();
            fs::create_dir_all(&path).unwrap();
            fs::write(path.join("SKILL.md"), "stale").unwrap();
            fs::write(path.join(".version"), "0.0.0").unwrap();
        }

        let searchable = sync_to_roots(
            AgentClient::All,
            AgentScope::All,
            Some(cwd.as_path()),
            &home,
            true,
        )
        .unwrap();
        assert!(
            !searchable.project_scope_skipped,
            "a searchable project directory must not report a skipped scope"
        );

        for &(client, scope) in SCOPED_SYNC_FIXTURE {
            let path = skill_path_for(client, scope, &cwd, &home).unwrap();
            fs::write(path.join(".version"), "0.0.0").unwrap();
        }

        fs::set_permissions(&cwd, fs::Permissions::from_mode(0o000)).unwrap();
        let _restore = RestoreMode(cwd.clone());
        if fs::metadata(cwd.join(".")).is_ok() {
            // Running as root, or on a filesystem that ignores mode bits.
            return;
        }

        let outcome = sync_to_roots(
            AgentClient::All,
            AgentScope::All,
            Some(cwd.as_path()),
            &home,
            true,
        )
        .unwrap();

        assert!(
            outcome.project_scope_skipped,
            "--scope all silently degraded to global for an unsearchable project directory"
        );
        let global: Vec<PathBuf> = SCOPED_SYNC_FIXTURE
            .iter()
            .filter(|(_, scope)| *scope == AgentScope::Global)
            .map(|(client, scope)| skill_path_for(*client, *scope, &cwd, &home).unwrap())
            .collect();
        assert_eq!(
            outcome.updated, global,
            "the global half of --scope all must still run"
        );
    }
}
