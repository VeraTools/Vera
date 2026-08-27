//! Vera CLI — code indexing and retrieval for local and tool-driven workflows.
//!
//! # Commands
//!
//! - `vera index <path>` — Index a codebase for search
//! - `vera search <query>` — Search the indexed codebase
//! - `vera update <path>` — Incrementally update the index
//! - `vera stats` — Show index statistics
//! - `vera config` — Show or set configuration values
//! - `vera structural ...` — Agent-oriented structural search intents

mod commands;
mod helpers;
mod skill_assets;
mod state;
mod update_check;

use std::process;

use clap::Parser;

mod cli;

use cli::{Cli, Commands};

fn main() {
    // Initialize tracing subscriber (logs go to stderr).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("VERA_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    vera_core::init_tls();
    let cli = Cli::parse();
    if let Err(err) = state::apply_saved_env() {
        // Diagnose/repair commands must still run against a broken saved
        // config; everything else keeps failing fast with the parse error.
        let tolerates_broken_config = matches!(
            cli.command,
            Commands::Doctor { .. }
                | Commands::Setup { .. }
                | Commands::Backend { .. }
                | Commands::Repair { .. }
                | Commands::Config { .. }
                | Commands::Uninstall
        );
        if tolerates_broken_config {
            eprintln!("Warning: ignoring broken saved configuration: {err:#}");
        } else {
            eprintln!("Error: {err:#}");
            process::exit(1);
        }
    }

    let show_nudges = !matches!(
        cli.command,
        Commands::Mcp
            | Commands::Serve { .. }
            | Commands::Agent { .. }
            | Commands::Uninstall
            | Commands::Upgrade { .. }
            | Commands::Backend { .. }
    ) && !cli.json;

    let result = match cli.command {
        Commands::Mcp => {
            tracing::info!("starting MCP server");
            commands::mcp::run();
            Ok(())
        }
        Commands::Serve {
            port,
            host,
            mut api_key,
            idle_timeout,
            backend,
            api,
        } => {
            tracing::info!("starting HTTP serve");
            // Also check VERA_SERVE_KEY env var.
            if api_key.is_none() {
                api_key = std::env::var("VERA_SERVE_KEY").ok();
            }
            let resolved_backend = if api {
                vera_core::config::InferenceBackend::Api
            } else {
                vera_core::config::resolve_backend(backend.explicit_backend())
            };
            let config = match state::load_runtime_config() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Error loading config: {e:#}");
                    process::exit(1);
                }
            };
            commands::serve::run(&host, port, api_key, resolved_backend, config, idle_timeout)
        }
        Commands::Agent {
            command,
            client,
            scope,
        } => {
            tracing::info!("agent command");
            commands::agent::run(command, client, scope, cli.json)
        }
        Commands::Setup {
            backend,
            embedding,
            api,
            index,
            yes,
        } => {
            tracing::info!("setup command");
            commands::setup::run(
                backend.explicit_backend(),
                api,
                index,
                cli.json,
                yes,
                embedding,
                true,
            )
        }
        Commands::Backend {
            backend,
            embedding,
            api,
            yes,
        } => {
            tracing::info!("backend command");
            commands::setup::run(
                backend.explicit_backend(),
                api,
                None,
                cli.json,
                yes,
                embedding,
                false,
            )
        }
        Commands::Uninstall => {
            tracing::info!("uninstall command");
            commands::uninstall::run(cli.json)
        }
        Commands::Doctor { probe } => {
            tracing::info!("doctor command");
            commands::doctor::run(cli.json, probe)
        }
        Commands::Repair { backend, api } => {
            tracing::info!("repair command");
            commands::repair::run(backend.explicit_backend(), api, cli.json)
        }
        Commands::Upgrade { apply } => {
            tracing::info!("upgrade command");
            commands::upgrade::run(apply, cli.json)
        }
        Commands::Index {
            path,
            backend,
            exclude,
            no_ignore,
            no_default_excludes,
            no_progress,
            verbose,
            low_vram,
        } => {
            tracing::info!(path = %path, "indexing");
            commands::index::run(
                &path,
                cli.json,
                backend.resolve(),
                exclude,
                no_ignore,
                no_default_excludes,
                no_progress,
                verbose,
                low_vram,
            )
        }
        Commands::Search {
            queries,
            intent,
            filters,
            limit,
            deep,
            git_scope,
            compact,
            backend,
        } => {
            tracing::info!(queries = ?queries, deep, "searching");
            commands::search::run(
                &queries,
                intent.as_deref(),
                limit,
                &filters.to_filters(),
                cli.json,
                cli.raw,
                cli.timing,
                deep,
                git_scope.resolve(),
                compact,
                backend.resolve(),
            )
        }
        Commands::Structural {
            intent,
            query,
            filters,
            limit,
            git_scope,
            compact,
        } => {
            tracing::info!("structural query");
            commands::structural::run(
                intent,
                query.as_deref(),
                limit,
                &filters.to_filters(),
                cli.json,
                cli.raw,
                cli.timing,
                git_scope.resolve(),
                compact,
            )
        }
        Commands::Update {
            path,
            backend,
            exclude,
            no_ignore,
            no_default_excludes,
            no_progress,
            max_files,
        } => {
            tracing::info!(path = %path, "updating");
            commands::update::run(
                &path,
                cli.json,
                commands::update::CommandOptions {
                    backend: backend.resolve(),
                    exclude,
                    no_ignore,
                    no_default_excludes,
                    no_progress,
                    max_files: max_files.map(std::num::NonZeroUsize::get),
                },
            )
        }
        Commands::Overview { git_scope } => {
            tracing::info!("showing overview");
            commands::overview::run(cli.json, git_scope.resolve())
        }
        Commands::ExplainPath {
            path,
            exclude,
            no_ignore,
            no_default_excludes,
        } => {
            tracing::info!(path = %path, "explaining path");
            commands::explain_path::run(&path, cli.json, exclude, no_ignore, no_default_excludes)
        }
        Commands::References {
            symbol,
            callees,
            receiver,
            limit,
            git_scope,
            compact,
        } => {
            tracing::info!(symbol = %symbol, callees, "references query");
            commands::references::run(
                &symbol,
                callees,
                receiver.as_deref(),
                limit,
                git_scope.resolve(),
                cli.json,
                cli.raw,
                compact,
            )
        }
        Commands::Grep {
            pattern,
            filters,
            limit,
            ignore_case,
            context,
            git_scope,
            compact,
        } => {
            tracing::info!(pattern = %pattern, "grep");
            commands::grep::run(
                &pattern,
                limit,
                ignore_case,
                context,
                &filters.to_filters(),
                cli.json,
                cli.raw,
                cli.timing,
                git_scope.resolve(),
                compact,
            )
        }
        Commands::DeadCode => {
            tracing::info!("dead code analysis");
            commands::references::run_dead_code(cli.json)
        }
        Commands::Stats => {
            tracing::info!("showing stats");
            commands::stats::run(cli.json)
        }
        Commands::Config { args } => {
            tracing::info!("config command");
            commands::config::run(&args, cli.json)
        }
        Commands::Watch { path } => {
            tracing::info!(path = %path, "watching");
            commands::watch::run(&path, cli.json)
        }
    };

    // Print update hints after the command runs (skip for MCP/agent/uninstall).
    if show_nudges {
        update_check::print_nudges();
    }

    if let Err(err) = result {
        eprintln!("Error: {err:#}");
        process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// Parse argv and return the subcommand, panicking on parse failure.
    fn parse(argv: &[&str]) -> Commands {
        Cli::parse_from(argv).command
    }

    #[test]
    fn cli_parses_index_command() {
        assert!(
            matches!(parse(&["vera", "index", "/tmp/repo"]), Commands::Index { path, .. } if path == "/tmp/repo")
        );
    }

    #[test]
    fn cli_parses_agent_install_command() {
        match parse(&["vera", "agent", "install", "--client", "codex"]) {
            Commands::Agent {
                command,
                client,
                scope,
                ..
            } => {
                assert_eq!(command, commands::agent::AgentCommand::Install);
                assert_eq!(client, Some(commands::agent::AgentClient::Codex));
                assert_eq!(scope, None);
            }
            _ => panic!("expected Agent command"),
        }
    }

    #[test]
    fn cli_parses_setup_command() {
        match parse(&["vera", "setup", "--local", "--index", "."]) {
            Commands::Setup {
                backend,
                embedding,
                api,
                index,
                ..
            } => {
                assert!(backend.local);
                assert!(!embedding.code_rank_embed);
                assert!(!api);
                assert_eq!(index, Some(".".to_string()));
            }
            _ => panic!("expected Setup command"),
        }
    }

    #[test]
    fn cli_parses_onnx_jina_cpu_flag() {
        match parse(&["vera", "index", ".", "--onnx-jina-cpu"]) {
            Commands::Index { backend, .. } => {
                assert!(backend.onnx_jina_cpu);
                assert!(!backend.local);
            }
            _ => panic!("expected Index command"),
        }
    }

    #[test]
    fn cli_parses_potion_code_flag() {
        match parse(&["vera", "setup", "--potion-code", "--yes"]) {
            Commands::Setup { backend, .. } => {
                assert!(backend.potion_code);
                assert_eq!(
                    backend.resolve(),
                    vera_core::config::InferenceBackend::PotionCode
                );
            }
            _ => panic!("expected Setup command"),
        }
    }

    #[test]
    fn cli_parses_potion_cpu_alias() {
        match parse(&["vera", "repair", "--potion-cpu"]) {
            Commands::Repair { backend, .. } => assert!(backend.potion_code),
            _ => panic!("expected Repair command"),
        }
    }

    #[test]
    fn cli_parses_onnx_jina_cuda_flag() {
        match parse(&["vera", "search", "test", "--onnx-jina-cuda"]) {
            Commands::Search { backend, .. } => {
                assert!(backend.onnx_jina_cuda);
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn cli_local_flag_still_works() {
        // --local is a hidden backwards-compat alias for --onnx-jina-cpu
        let cli = Cli::parse_from(["vera", "index", ".", "--local"]);
        match cli.command {
            Commands::Index { backend, .. } => {
                assert!(backend.local);
                assert!(!backend.onnx_jina_cpu);
            }
            _ => panic!("expected Index command"),
        }
    }

    #[test]
    fn cli_parses_doctor_command() {
        assert!(matches!(
            parse(&["vera", "doctor"]),
            Commands::Doctor { probe: false }
        ));
    }

    #[test]
    fn cli_parses_doctor_probe_command() {
        assert!(matches!(
            parse(&["vera", "doctor", "--probe"]),
            Commands::Doctor { probe: true }
        ));
    }

    #[test]
    fn cli_parses_repair_command() {
        match parse(&["vera", "repair", "--onnx-jina-cuda"]) {
            Commands::Repair { backend, .. } => assert!(backend.onnx_jina_cuda),
            _ => panic!("expected Repair command"),
        }
    }

    #[test]
    fn cli_parses_setup_code_rank_embed_flag() {
        match parse(&["vera", "setup", "--code-rank-embed", "--onnx-jina-cuda"]) {
            Commands::Setup {
                backend, embedding, ..
            } => {
                assert!(backend.onnx_jina_cuda);
                assert!(embedding.code_rank_embed);
            }
            _ => panic!("expected Setup command"),
        }
    }

    #[test]
    fn cli_parses_upgrade_command() {
        assert!(matches!(
            parse(&["vera", "upgrade", "--apply"]),
            Commands::Upgrade { apply: true }
        ));
    }

    #[test]
    fn cli_parses_search_command() {
        let command = parse(&["vera", "search", "find auth"]);
        assert!(
            matches!(command, Commands::Search { queries, .. } if queries == vec!["find auth".to_string()])
        );
    }

    #[test]
    fn cli_parses_search_with_filters() {
        match parse(&[
            "vera",
            "search",
            "find auth",
            "--lang",
            "rust",
            "--limit",
            "5",
        ]) {
            Commands::Search {
                queries,
                filters,
                limit,
                ..
            } => {
                assert_eq!(queries, vec!["find auth".to_string()]);
                assert_eq!(filters.lang, Some("rust".to_string()));
                assert_eq!(limit, Some(5));
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn cli_parses_search_with_type_filter() {
        match parse(&["vera", "search", "find auth", "--type", "function"]) {
            Commands::Search {
                queries, filters, ..
            } => {
                assert_eq!(queries, vec!["find auth".to_string()]);
                assert_eq!(filters.r#type, Some("function".to_string()));
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn cli_parses_search_with_path_filter() {
        match parse(&[
            "vera",
            "search",
            "config",
            "--path",
            "src/**/*.rs",
            "--path",
            "tests/**/*.rs",
        ]) {
            Commands::Search {
                queries, filters, ..
            } => {
                assert_eq!(queries, vec!["config".to_string()]);
                assert_eq!(
                    filters.path,
                    vec!["src/**/*.rs".to_string(), "tests/**/*.rs".to_string()]
                );
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn cli_parses_search_with_all_filters() {
        match parse(&[
            "vera",
            "search",
            "handle request",
            "--lang",
            "typescript",
            "--path",
            "src/**/*.ts",
            "--type",
            "function",
            "--limit",
            "3",
        ]) {
            Commands::Search {
                queries,
                filters,
                limit,
                ..
            } => {
                assert_eq!(queries, vec!["handle request".to_string()]);
                assert_eq!(filters.lang, Some("typescript".to_string()));
                assert_eq!(filters.path, vec!["src/**/*.ts".to_string()]);
                assert_eq!(filters.r#type, Some("function".to_string()));
                assert_eq!(limit, Some(3));
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn cli_parses_search_with_multiple_queries_and_intent() {
        match parse(&[
            "vera",
            "search",
            "OAuth token refresh",
            "JWT expiry handling",
            "auth middleware",
            "--intent",
            "find where tokens are refreshed and validated",
        ]) {
            Commands::Search {
                queries, intent, ..
            } => {
                assert_eq!(
                    queries,
                    vec![
                        "OAuth token refresh".to_string(),
                        "JWT expiry handling".to_string(),
                        "auth middleware".to_string(),
                    ]
                );
                assert_eq!(
                    intent,
                    Some("find where tokens are refreshed and validated".to_string())
                );
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn cli_parses_search_timing_flag() {
        let cli = Cli::parse_from(["vera", "search", "find auth", "--timing"]);
        assert!(matches!(cli.command, Commands::Search { .. }));
        assert!(cli.timing);
    }

    #[test]
    fn cli_parses_search_raw_flag() {
        let cli = Cli::parse_from(["vera", "search", "find auth", "--raw"]);
        assert!(matches!(cli.command, Commands::Search { .. }));
        assert!(cli.raw);
    }

    #[test]
    fn cli_parses_global_search_timing_flag_before_subcommand() {
        let cli = Cli::parse_from(["vera", "--timing", "search", "find auth"]);
        assert!(matches!(cli.command, Commands::Search { .. }));
        assert!(cli.timing);
    }

    #[test]
    fn cli_parses_global_search_raw_flag_before_subcommand() {
        let cli = Cli::parse_from(["vera", "--raw", "search", "find auth"]);
        assert!(matches!(cli.command, Commands::Search { .. }));
        assert!(cli.raw);
    }

    #[test]
    fn cli_parses_search_scope_flags() {
        match parse(&[
            "vera",
            "search",
            "mod loader",
            "--scope",
            "runtime",
            "--include-generated",
        ]) {
            Commands::Search { filters, .. } => {
                assert_eq!(filters.scope, Some("runtime".to_string()));
                assert!(filters.include_generated);
            }
            _ => panic!("expected Search command"),
        }
    }

    #[test]
    fn cli_parses_update_command() {
        assert!(
            matches!(parse(&["vera", "update", "/tmp/repo"]), Commands::Update { path, .. } if path == "/tmp/repo")
        );
    }

    #[test]
    fn cli_parses_index_no_progress_flag() {
        let command = parse(&["vera", "index", "/tmp/repo", "--no-progress"]);
        assert!(matches!(
            command,
            Commands::Index {
                no_progress: true,
                ..
            }
        ));
    }

    #[test]
    fn cli_parses_update_no_progress_flag() {
        let command = parse(&["vera", "update", "/tmp/repo", "--no-progress"]);
        assert!(matches!(
            command,
            Commands::Update {
                no_progress: true,
                ..
            }
        ));
    }

    #[test]
    fn cli_parses_update_max_files_flag() {
        let command = parse(&["vera", "update", "/tmp/repo", "--max-files", "250"]);
        assert!(matches!(
            command,
            Commands::Update {
                max_files: Some(max_files),
                ..
            } if max_files.get() == 250
        ));
    }

    #[test]
    fn cli_rejects_zero_update_max_files() {
        assert!(Cli::try_parse_from(["vera", "update", "/tmp/repo", "--max-files", "0"]).is_err());
    }

    #[test]
    fn cli_parses_grep_scope_flags() {
        match parse(&[
            "vera",
            "grep",
            "keybind",
            "--scope",
            "docs",
            "--include-generated",
        ]) {
            Commands::Grep { filters, .. } => {
                assert_eq!(filters.scope, Some("docs".to_string()));
                assert!(filters.include_generated);
            }
            _ => panic!("expected Grep command"),
        }
    }

    #[test]
    fn cli_parses_grep_with_path_filter() {
        match parse(&[
            "vera",
            "grep",
            "queryClient|invalidateQueries",
            "--path",
            "frontend/src/**",
        ]) {
            Commands::Grep {
                pattern, filters, ..
            } => {
                assert_eq!(pattern, "queryClient|invalidateQueries");
                assert_eq!(filters.path, vec!["frontend/src/**".to_string()]);
            }
            _ => panic!("expected Grep command"),
        }
    }

    #[test]
    fn cli_parses_grep_with_all_filters() {
        match parse(&[
            "vera",
            "grep",
            "Authorization",
            "--lang",
            "rust",
            "--path",
            "src/**/*.rs",
            "--type",
            "function",
            "--scope",
            "source",
        ]) {
            Commands::Grep {
                pattern, filters, ..
            } => {
                assert_eq!(pattern, "Authorization");
                assert_eq!(filters.lang, Some("rust".to_string()));
                assert_eq!(filters.path, vec!["src/**/*.rs".to_string()]);
                assert_eq!(filters.r#type, Some("function".to_string()));
                assert_eq!(filters.scope, Some("source".to_string()));
            }
            _ => panic!("expected Grep command"),
        }
    }

    #[test]
    fn cli_parses_grep_timing_flag() {
        let cli = Cli::parse_from(["vera", "grep", "TODO", "--timing"]);
        assert!(matches!(cli.command, Commands::Grep { .. }));
        assert!(cli.timing);
    }

    #[test]
    fn cli_parses_grep_raw_flag() {
        let cli = Cli::parse_from(["vera", "grep", "TODO", "--raw"]);
        assert!(matches!(cli.command, Commands::Grep { .. }));
        assert!(cli.raw);
    }

    #[test]
    fn cli_parses_global_grep_timing_flag_before_subcommand() {
        let cli = Cli::parse_from(["vera", "--timing", "grep", "TODO"]);
        assert!(matches!(cli.command, Commands::Grep { .. }));
        assert!(cli.timing);
    }

    #[test]
    fn cli_parses_global_grep_raw_flag_before_subcommand() {
        let cli = Cli::parse_from(["vera", "--raw", "grep", "TODO"]);
        assert!(matches!(cli.command, Commands::Grep { .. }));
        assert!(cli.raw);
    }

    #[test]
    fn cli_parses_structural_definitions_command() {
        match parse(&["vera", "structural", "definitions", "parse_config"]) {
            Commands::Structural { intent, query, .. } => {
                assert!(matches!(
                    intent,
                    commands::structural::StructuralIntent::Definitions
                ));
                assert_eq!(query.as_deref(), Some("parse_config"));
            }
            _ => panic!("expected structural command"),
        }
    }

    #[test]
    fn cli_parses_structural_filters_and_git_scope() {
        match parse(&[
            "vera",
            "structural",
            "env",
            "DATABASE_URL",
            "--lang",
            "rust",
            "--path",
            "src/**/*.rs",
            "--type",
            "function",
            "--changed",
            "--compact",
        ]) {
            Commands::Structural {
                intent,
                query,
                filters,
                git_scope,
                compact,
                ..
            } => {
                assert!(matches!(
                    intent,
                    commands::structural::StructuralIntent::Env
                ));
                assert_eq!(query.as_deref(), Some("DATABASE_URL"));
                assert_eq!(filters.lang, Some("rust".to_string()));
                assert_eq!(filters.path, vec!["src/**/*.rs".to_string()]);
                assert_eq!(filters.r#type, Some("function".to_string()));
                assert!(git_scope.changed);
                assert!(compact);
            }
            _ => panic!("expected structural command"),
        }
    }

    #[test]
    fn cli_parses_references_limit_git_scope_and_compact() {
        match parse(&[
            "vera",
            "references",
            "parse_config",
            "--limit",
            "7",
            "--changed",
            "--compact",
        ]) {
            Commands::References {
                symbol,
                limit,
                git_scope,
                compact,
                ..
            } => {
                assert_eq!(symbol, "parse_config");
                assert_eq!(limit, Some(7));
                assert!(git_scope.changed);
                assert!(compact);
            }
            _ => panic!("expected references command"),
        }
    }

    #[test]
    fn cli_parses_watch_command() {
        assert!(
            matches!(parse(&["vera", "watch", "/tmp/repo"]), Commands::Watch { path } if path == "/tmp/repo")
        );
    }

    #[test]
    fn cli_parses_stats_command() {
        assert!(matches!(parse(&["vera", "stats"]), Commands::Stats));
    }

    /// `--idle-timeout -1` is the spelling the flag's own help text documents.
    /// Without `allow_negative_numbers`, clap reads the `-1` as an unknown flag
    /// and only the `=` form works.
    #[test]
    fn cli_parses_a_negative_idle_timeout_in_the_documented_form() {
        for argv in [
            vec!["vera", "serve", "--idle-timeout", "-1"],
            vec!["vera", "serve", "--idle-timeout=-1"],
        ] {
            let parsed = Cli::try_parse_from(argv.iter().copied())
                .unwrap_or_else(|e| panic!("{argv:?} must parse: {e}"));
            match parsed.command {
                Commands::Serve { idle_timeout, .. } => assert_eq!(idle_timeout, -1),
                _ => panic!("{argv:?} did not parse as serve"),
            }
        }

        assert!(matches!(
            parse(&["vera", "serve"]),
            Commands::Serve {
                idle_timeout: 300,
                ..
            }
        ));
        assert!(matches!(
            parse(&["vera", "serve", "--idle-timeout", "0"]),
            Commands::Serve {
                idle_timeout: 0,
                ..
            }
        ));

        // The parser is loosened for negative numbers only: a hyphenated
        // non-number is still an unknown flag rather than a swallowed value.
        assert!(Cli::try_parse_from(["vera", "serve", "--idle-timeout", "-abc"]).is_err());
        assert!(Cli::try_parse_from(["vera", "serve", "--idle-timeout", "--port"]).is_err());
    }

    #[test]
    fn cli_parses_json_flag() {
        let cli = Cli::parse_from(["vera", "--json", "stats"]);
        assert!(cli.json);
    }

    #[test]
    fn cli_help_mentions_global_output_flags() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("--raw"));
        assert!(help.contains("--timing"));
        assert!(help.contains("before or after the subcommand"));
    }

    #[test]
    fn cli_parses_config_command() {
        assert!(matches!(parse(&["vera", "config"]), Commands::Config { args } if args.is_empty()));
    }

    #[test]
    fn cli_parses_config_show() {
        match parse(&["vera", "config", "show"]) {
            Commands::Config { args } => {
                assert_eq!(args, vec!["show".to_string()]);
            }
            _ => panic!("expected Config command"),
        }
    }

    #[test]
    fn cli_parses_config_get() {
        match parse(&["vera", "config", "get", "retrieval.default_limit"]) {
            Commands::Config { args } => {
                assert_eq!(
                    args,
                    vec!["get".to_string(), "retrieval.default_limit".to_string()]
                );
            }
            _ => panic!("expected Config command"),
        }
    }

    #[test]
    fn cli_parses_config_set() {
        match parse(&["vera", "config", "set", "retrieval.default_limit", "20"]) {
            Commands::Config { args } => {
                assert_eq!(
                    args,
                    vec![
                        "set".to_string(),
                        "retrieval.default_limit".to_string(),
                        "20".to_string()
                    ]
                );
            }
            _ => panic!("expected Config command"),
        }
    }

    #[test]
    fn config_get_known_keys() {
        let config = vera_core::config::VeraConfig::default();
        assert!(commands::config::get_config_value(&config, "indexing.max_chunk_lines").is_some());
        assert!(commands::config::get_config_value(&config, "retrieval.default_limit").is_some());
        assert!(commands::config::get_config_value(&config, "retrieval.rrf_k").is_some());
        assert!(
            commands::config::get_config_value(&config, "retrieval.reranking_enabled").is_some()
        );
        assert!(commands::config::get_config_value(&config, "embedding.batch_size").is_some());
        assert!(commands::config::get_config_value(&config, "embedding.max_stored_dim").is_some());
    }

    #[test]
    fn config_get_unknown_key_returns_none() {
        let config = vera_core::config::VeraConfig::default();
        assert!(commands::config::get_config_value(&config, "nonexistent.key").is_none());
    }

    #[test]
    fn config_values_match_defaults() {
        let config = vera_core::config::VeraConfig::default();
        let val = commands::config::get_config_value(&config, "retrieval.default_limit").unwrap();
        assert_eq!(val, serde_json::json!(5));

        let val = commands::config::get_config_value(&config, "indexing.max_chunk_lines").unwrap();
        assert_eq!(val, serde_json::json!(200));

        let val =
            commands::config::get_config_value(&config, "retrieval.reranking_enabled").unwrap();
        assert_eq!(val, serde_json::json!(false));
    }
}
