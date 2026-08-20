//! MCP tool definitions and handler dispatch.
//!
//! Defines the tools that the Vera MCP server exposes:
//! - `search_code` — search indexed codebase (auto-indexes and watches on first use)
//! - `get_stats` — retrieve index statistics
//! - `get_overview` — architecture overview for agent onboarding
//! - `regex_search` — regex search over indexed files
//! - `structural_search` — agent-oriented structural search intents
//! - `find_references` — exact callers or callees from the persisted call graph
//! - `explain_path` — explain why a path is or is not indexed

use std::sync::{Arc, Mutex, OnceLock};

use serde_json::Value;
use vera_core::presentation::{CompactResult, truncate_to_budget};

use crate::protocol::{ToolCallResult, ToolDefinition};
use crate::watcher::WatchHandle;

/// Global watcher handle. Kept alive for the lifetime of the MCP server process.
static WATCHER: Mutex<Option<WatchHandle>> = Mutex::new(None);
static SEARCH_CONTEXT: Mutex<Option<CachedSearchContext>> = Mutex::new(None);
static SEARCH_RUNTIME: OnceLock<Result<tokio::runtime::Runtime, String>> = OnceLock::new();

struct CachedSearchContext {
    key: String,
    context: Arc<vera_core::retrieval::search_service::SearchContext>,
}

/// Default total output budget for MCP responses (chars).
const MCP_OUTPUT_BUDGET: usize = 20_000;

/// Serialize search results as compact JSON, applying a total character budget.
/// When `signatures_only` is true, function/class bodies are stripped before output.
fn compact_results_json(
    results: &[vera_core::types::SearchResult],
    budget: usize,
    signatures_only: bool,
) -> Result<String, serde_json::Error> {
    use vera_core::parsing::signatures::extract_signature_for_path;

    let signatures: Vec<String> = if signatures_only {
        results
            .iter()
            .map(|r| extract_signature_for_path(&r.content, r.language, &r.file_path))
            .collect()
    } else {
        Vec::new()
    };

    // Build a parallel vec of display content (signature or original).
    let display: Vec<&str> = results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            if signatures_only {
                signatures[i].as_str()
            } else {
                r.content.as_str()
            }
        })
        .collect();

    let mut remaining = budget;
    let mut compact: Vec<CompactResult> = Vec::with_capacity(results.len());
    for (i, r) in results.iter().enumerate() {
        if budget > 0 && remaining == 0 {
            break;
        }
        let content = if budget > 0 {
            let c = truncate_to_budget(display[i], remaining);
            remaining = remaining.saturating_sub(c.len());
            c
        } else {
            std::borrow::Cow::Borrowed(display[i])
        };
        compact.push(CompactResult {
            file_path: &r.file_path,
            line_start: r.line_start,
            line_end: r.line_end,
            content,
            symbol_name: r.symbol_name.as_deref(),
            symbol_type: r.symbol_type.as_ref(),
        });
    }
    serde_json::to_string(&compact)
}

/// Git-scope filter properties shared by tool schemas: `changed`, `since`,
/// `base`. `what` names the restricted operation (e.g. "search").
fn git_scope_properties(what: &str) -> serde_json::Map<String, Value> {
    serde_json::json!({
        "changed": {
            "type": "boolean",
            "description": format!("Restrict {what} to modified, staged, and untracked files.")
        },
        "since": {
            "type": "string",
            "description": format!("Restrict {what} to files changed since the given revision.")
        },
        "base": {
            "type": "string",
            "description": format!("Restrict {what} to files changed since merge-base(HEAD, revision).")
        }
    })
    .as_object()
    .expect("git scope properties")
    .clone()
}

/// Merge the shared git-scope properties into a tool schema's `properties`.
fn with_git_scope(mut schema: Value, what: &str) -> Value {
    schema
        .pointer_mut("/properties")
        .and_then(Value::as_object_mut)
        .expect("tool schema properties")
        .extend(git_scope_properties(what));
    schema
}

/// Shared `scope` filter property (source/docs/runtime corpus selection).
fn scope_prop() -> Value {
    serde_json::json!({
        "type": "string",
        "enum": ["source", "docs", "runtime", "all"],
        "description": "Coarse corpus scope. Defaults to source-first behavior."
    })
}

/// Shared `include_generated` filter property.
fn include_generated_prop() -> Value {
    serde_json::json!({
        "type": "boolean",
        "description": "Include generated or minified files such as dist bundles."
    })
}

/// Shared `path` property naming the project directory.
fn project_path_prop() -> Value {
    serde_json::json!({
        "type": "string",
        "description": "Path to the project directory (default: current dir)"
    })
}

/// Shared `limit` property; `what` is the counted noun ("results", "matches").
fn limit_prop(what: &str, default: usize) -> Value {
    serde_json::json!({
        "type": "integer",
        "description": format!("Maximum number of {what} (default: {default})")
    })
}

/// Return the list of tools the server advertises.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "search_code".to_string(),
            description: "Search the indexed codebase using hybrid BM25+vector \
                          retrieval with cross-encoder reranking. Returns ranked \
                          code snippets with file paths, line numbers, and content.\n\
                          \n\
                          WHEN TO USE: conceptual or behavioral queries (\"how is auth handled\", \
                          \"error retry logic\", \"database connection pooling\"). Understands \
                          synonyms and related concepts.\n\
                          WHEN NOT TO USE: exact string matching, regex patterns, or \
                          import statements. Use regex_search for those.\n\
                          \n\
                          TIPS: Use 2-3 varied queries to capture different aspects of what \
                          you are looking for. Set intent to describe your higher-level goal \
                          for better reranking."
                .to_string(),
            input_schema: with_git_scope(serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (keyword or natural language)"
                    },
                    "queries": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Multiple search queries to run in parallel and merge. Use 2-3 varied queries to capture different aspects (e.g., ['OAuth token refresh', 'JWT expiry handling', 'auth middleware']). Results are deduplicated and reranked."
                    },
                    "intent": {
                        "type": "string",
                        "description": "Higher-level goal for reranking (e.g., 'find where auth tokens are validated and refreshed'). Improves precision when the query is ambiguous."
                    },
                    "lang": {
                        "type": "string",
                        "description": "Filter by programming language (e.g., rust, python)"
                    },
                    "path": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "description": "Filter by file path glob, as a string or an array of strings. Repeated patterns use OR semantics (e.g., src/**/*.rs)"
                    },
                    "symbol_type": {
                        "type": "string",
                        "description": "Filter by symbol type (function, struct, class, etc.)"
                    },
                    "scope": scope_prop(),
                    "include_generated": include_generated_prop(),
                    "limit": limit_prop("results", 5),
                    "compact": {
                        "type": "boolean",
                        "description": "Return only function/class signatures (omit bodies). Use for broad exploration; fits more results in fewer tokens."
                    }
                },
                "anyOf": [{"required": ["query"]}, {"required": ["queries"]}]
            }), "search"),
        },
        ToolDefinition {
            name: "get_stats".to_string(),
            description: "Get index statistics: file count, chunk count, index size, \
                          and language breakdown."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": project_path_prop()
                }
            }),
        },
        ToolDefinition {
            name: "get_overview".to_string(),
            description: "Get architecture overview of the indexed project: languages, \
                          directories, entry points, symbol types, complexity hotspots, \
                          and detected project conventions (frameworks, patterns, config files). \
                          Useful for onboarding and understanding project structure."
                .to_string(),
            input_schema: with_git_scope(serde_json::json!({
                "type": "object",
                "properties": {
                    "path": project_path_prop(),
                }
            }), "overview"),
        },
        ToolDefinition {
            name: "regex_search".to_string(),
            description: "Search indexed files using a regex pattern. Returns matches \
                          with surrounding context lines.\n\
                          \n\
                          WHEN TO USE: exact string matching, regex patterns, import statements, \
                          TODOs, specific syntax, or known identifiers.\n\
                          WHEN NOT TO USE: conceptual or behavioral queries. Use search_code \
                          for those."
                .to_string(),
            input_schema: with_git_scope(serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "limit": limit_prop("matches", 20),
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive matching (default: false)"
                    },
                    "context": {
                        "type": "integer",
                        "description": "Context lines before and after each match (default: 2)"
                    },
                    "scope": scope_prop(),
                    "include_generated": include_generated_prop(),
                    "compact": {
                        "type": "boolean",
                        "description": "Return only function/class signatures (omit bodies). Use for broad exploration."
                    }
                },
                "required": ["pattern"]
            }), "regex search"),
        },
        ToolDefinition {
            name: "structural_search".to_string(),
            description: "Run agent-oriented structural search intents over indexed code.\n\
                          \n\
                          WHEN TO USE: symbol definitions, env var reads, \
                          HTTP route handlers, SQL execution sites, or explicit \
                          implementations/conformances/inheritance declarations.\n\
                          WHEN NOT TO USE: conceptual behavior queries or exact caller/callee \
                          lookups. Use search_code or find_references for those."
                .to_string(),
            input_schema: with_git_scope(serde_json::json!({
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["definitions", "env_reads", "route_handlers", "sql_queries", "implementations"],
                        "description": "Structural intent to run."
                    },
                    "query": {
                        "type": "string",
                        "description": "Required for definitions and implementation lookups. Optional for env_reads to narrow to one env var."
                    },
                    "lang": {
                        "type": "string",
                        "description": "Filter by programming language (e.g., rust, python)"
                    },
                    "path": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "description": "Filter by file path glob, as a string or an array of strings. Repeated patterns use OR semantics (e.g., src/**/*.rs)"
                    },
                    "symbol_type": {
                        "type": "string",
                        "description": "Filter by enclosing symbol type (function, class, method, etc.)"
                    },
                    "scope": scope_prop(),
                    "include_generated": include_generated_prop(),
                    "limit": limit_prop("results", 20),
                    "compact": {
                        "type": "boolean",
                        "description": "Return only function/class signatures (omit bodies)."
                    }
                },
                "required": ["kind"]
            }), "search"),
        },
        ToolDefinition {
            name: "find_references".to_string(),
            description: "Find exact callers or callees of a symbol using Vera's persisted call graph.\n\
                          \n\
                          WHEN TO USE: who calls a symbol, what a symbol calls, or when \
                          narrowing exact call relationships to a diff.\n\
                          WHEN NOT TO USE: conceptual behavior queries or heuristic structural \
                          scans. Use search_code or structural_search for those."
                .to_string(),
            input_schema: with_git_scope(serde_json::json!({
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbol name to look up."
                    },
                    "callees": {
                        "type": "boolean",
                        "description": "Return what the symbol calls instead of who calls it."
                    },
                    "limit": limit_prop("results", 20),
                    "compact": {
                        "type": "boolean",
                        "description": "For caller lookups, return only function/class signatures."
                    }
                },
                "required": ["symbol"]
            }), "references"),
        },
        ToolDefinition {
            name: "explain_path".to_string(),
            description: "Explain why a path is or is not indexed. Returns the decisive reason such as a default exclude, .veraignore, .gitignore, binary detection, size limit, or missing file."
                .to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative or absolute path to explain."
                    },
                    "exclude": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Extra exclusion globs to apply, matching CLI --exclude semantics."
                    },
                    "no_ignore": {
                        "type": "boolean",
                        "description": "Disable .gitignore and .veraignore parsing."
                    },
                    "no_default_excludes": {
                        "type": "boolean",
                        "description": "Disable Vera's built-in default exclusions."
                    }
                },
                "required": ["path"]
            }),
        },
    ]
}

/// Dispatch a tool call to the appropriate handler.
///
/// Returns a `ToolCallResult` — either success with JSON content or an
/// error with a descriptive message. This function never panics.
pub fn handle_tool_call(name: &str, arguments: &Value) -> ToolCallResult {
    match name {
        "search_code" => handle_search_code(arguments),
        "get_stats" => handle_get_stats(arguments),
        "get_overview" => handle_get_overview(arguments),
        "regex_search" => handle_regex_search(arguments),
        "structural_search" => handle_structural_search(arguments),
        "find_references" => handle_find_references(arguments),
        "explain_path" => handle_explain_path(arguments),
        _ => ToolCallResult::error(format!("Unknown tool: {name}")),
    }
}

fn git_scope_from_args(args: &Value) -> Result<Option<vera_core::git_scope::GitScope>, String> {
    let changed = args
        .get("changed")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let since = args
        .get("since")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let base = args
        .get("base")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    let selected = changed as u8 + since.is_some() as u8 + base.is_some() as u8;
    if selected > 1 {
        return Err("Only one of 'changed', 'since', or 'base' may be set".to_string());
    }

    Ok(if changed {
        Some(vera_core::git_scope::GitScope::Changed)
    } else if let Some(rev) = since {
        Some(vera_core::git_scope::GitScope::Since(rev))
    } else {
        base.map(vera_core::git_scope::GitScope::Base)
    })
}

fn scope_from_args(args: &Value) -> Result<Option<vera_core::types::SearchScope>, ToolCallResult> {
    match args.get("scope").and_then(|v| v.as_str()) {
        Some(value) => value
            .parse()
            .map(Some)
            .map_err(|()| ToolCallResult::error(format!("Invalid scope: {value}"))),
        None => Ok(None),
    }
}

fn current_working_dir() -> Result<std::path::PathBuf, ToolCallResult> {
    std::env::current_dir()
        .map_err(|e| ToolCallResult::error(format!("Failed to get working directory: {e}")))
}

fn exact_paths_from_args(
    args: &Value,
    cwd: &std::path::Path,
) -> Result<Option<Arc<std::collections::HashSet<String>>>, ToolCallResult> {
    match git_scope_from_args(args) {
        Ok(Some(scope)) => vera_core::git_scope::resolve_scope(cwd, &scope)
            .map(|paths| Some(Arc::new(paths)))
            .map_err(|err| ToolCallResult::error(format!("Failed to resolve git scope: {err}"))),
        Ok(None) => Ok(None),
        Err(err) => Err(ToolCallResult::error(err)),
    }
}

fn apply_git_scope_filters(
    args: &Value,
    cwd: &std::path::Path,
    filters: &mut vera_core::types::SearchFilters,
) -> Result<(), ToolCallResult> {
    filters.exact_paths = exact_paths_from_args(args, cwd)?;
    Ok(())
}

fn existing_index_dir(cwd: &std::path::Path) -> Result<std::path::PathBuf, ToolCallResult> {
    let index_dir = vera_core::indexing::index_dir(cwd);
    if !index_dir.exists() {
        Err(ToolCallResult::error(
            "No index found in current directory. Run search_code first to auto-index.",
        ))
    } else {
        Ok(index_dir)
    }
}

fn search_code_filters(
    args: &Value,
    scope: Option<vera_core::types::SearchScope>,
) -> vera_core::types::SearchFilters {
    let path_glob = match args.get("path") {
        Some(value) if value.is_string() => value
            .as_str()
            .map(|path| vec![path.to_string()])
            .unwrap_or_default(),
        Some(value) => value
            .as_array()
            .map(|paths| {
                paths
                    .iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default(),
        None => Vec::new(),
    };

    vera_core::types::SearchFilters {
        language: args.get("lang").and_then(|v| v.as_str()).map(String::from),
        path_glob,
        exact_paths: None,
        symbol_type: args
            .get("symbol_type")
            .and_then(|v| v.as_str())
            .map(String::from),
        scope,
        include_generated: Some(
            args.get("include_generated")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        ),
    }
}

fn regex_search_filters(
    args: &Value,
    scope: Option<vera_core::types::SearchScope>,
) -> vera_core::types::SearchFilters {
    // Keep regex_search intentionally small over MCP. search_code already
    // exposes richer corpus filters, so regex_search sticks to the
    // highest-value regex controls.
    vera_core::types::SearchFilters {
        scope,
        exact_paths: None,
        include_generated: Some(
            args.get("include_generated")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        ),
        ..Default::default()
    }
}

/// Handle the `search_code` tool.
fn handle_search_code(args: &Value) -> ToolCallResult {
    // Collect queries: support both single `query` and multi `queries`.
    let mut queries: Vec<String> = Vec::new();
    if let Some(q) = args.get("query").and_then(|v| v.as_str()) {
        queries.push(q.to_string());
    }
    if let Some(arr) = args.get("queries").and_then(|v| v.as_array()) {
        for item in arr {
            if let Some(q) = item.as_str() {
                queries.push(q.to_string());
            }
        }
    }
    if queries.is_empty() {
        return ToolCallResult::error(
            "Missing required parameter: provide 'query' (string) or 'queries' (array)",
        );
    }

    let intent = args.get("intent").and_then(|v| v.as_str());

    let scope = match scope_from_args(args) {
        Ok(scope) => scope,
        Err(err) => return err,
    };

    let mut filters = search_code_filters(args, scope);

    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    let backend = vera_core::config::resolve_backend(None);
    let mut config = crate::saved_config::load_saved_runtime_config();
    config.adjust_for_backend(backend);
    let result_limit = limit.unwrap_or(config.retrieval.default_limit);

    let cwd = match current_working_dir() {
        Ok(cwd) => cwd,
        Err(err) => return err,
    };
    if let Err(err) = apply_git_scope_filters(args, &cwd, &mut filters) {
        return err;
    }
    let index_dir = match ensure_index_and_watcher(&cwd) {
        Ok(index_dir) => index_dir,
        Err(err) => return err,
    };
    let rt = match search_runtime() {
        Ok(rt) => rt,
        Err(err) => return err,
    };
    let search_context = match cached_search_context(rt, &config, backend) {
        Ok(context) => context,
        Err(err) => return err,
    };

    // Run each query, collect all results.
    let mut all_results: Vec<vera_core::types::SearchResult> = Vec::new();
    let per_query_limit = if queries.len() > 1 {
        result_limit.max(10)
    } else {
        result_limit
    };

    for query in &queries {
        // Pass the raw query plus the optional intent. Core applies the intent
        // only to the semantic (embedding/rerank) side; BM25 gets the raw query
        // so Tantivy does not parse `intent:` as a field. See issue #20.
        match rt.block_on(search_context.search(
            &index_dir,
            query,
            intent,
            &config,
            &filters,
            per_query_limit,
        )) {
            Ok((results, _timings)) => all_results.extend(results),
            Err(e) => return ToolCallResult::error(format!("Search failed: {e}")),
        }
    }

    // Deduplicate by (file_path, line_start, line_end), keeping first occurrence.
    let mut seen = std::collections::HashSet::new();
    all_results.retain(|r| seen.insert(format!("{}:{}:{}", r.file_path, r.line_start, r.line_end)));
    all_results.truncate(result_limit);

    let signatures_only = args
        .get("compact")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    match compact_results_json(&all_results, MCP_OUTPUT_BUDGET, signatures_only) {
        Ok(json) => ToolCallResult::success(json),
        Err(e) => ToolCallResult::error(format!("Failed to serialize results: {e}")),
    }
}

fn ensure_index_and_watcher(cwd: &std::path::Path) -> Result<std::path::PathBuf, ToolCallResult> {
    let index_dir = vera_core::indexing::index_dir(cwd);

    if !index_dir.exists() {
        let (rt, provider, idx_config, model_name) = create_runtime_and_provider()?;
        rt.block_on(vera_core::indexing::index_repository(
            cwd,
            &provider,
            &idx_config,
            &model_name,
        ))
        .map_err(|e| ToolCallResult::error(format!("Auto-indexing failed: {e}")))?;
    }

    let mut guard = WATCHER.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        match crate::watcher::start_watching(cwd) {
            Ok(handle) => *guard = Some(handle),
            Err(e) => tracing::warn!("failed to start file watcher: {e}"),
        }
    }

    Ok(index_dir)
}

/// Create a tokio runtime, resolve backend config, and build an embedding provider.
fn create_runtime_and_provider() -> Result<
    (
        tokio::runtime::Runtime,
        vera_core::embedding::DynamicProvider,
        vera_core::config::VeraConfig,
        String,
    ),
    ToolCallResult,
> {
    let backend = vera_core::config::resolve_backend(None);
    let mut config = crate::saved_config::load_saved_runtime_config();
    config.adjust_for_backend(backend);

    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| ToolCallResult::error(format!("Failed to create runtime: {e}")))?;

    let (provider, model_name) = rt
        .block_on(vera_core::embedding::create_dynamic_provider(
            &config, backend,
        ))
        .map_err(|e| ToolCallResult::error(format!("Failed to create embedding provider: {e}")))?;

    Ok((rt, provider, config, model_name))
}

fn search_runtime() -> Result<&'static tokio::runtime::Runtime, ToolCallResult> {
    SEARCH_RUNTIME
        .get_or_init(|| tokio::runtime::Runtime::new().map_err(|err| err.to_string()))
        .as_ref()
        .map_err(|err| ToolCallResult::error(format!("Failed to create runtime: {err}")))
}

fn cached_search_context(
    rt: &tokio::runtime::Runtime,
    config: &vera_core::config::VeraConfig,
    backend: vera_core::config::InferenceBackend,
) -> Result<Arc<vera_core::retrieval::search_service::SearchContext>, ToolCallResult> {
    let key = search_context_key(config, backend);
    let mut guard = SEARCH_CONTEXT.lock().unwrap_or_else(|e| e.into_inner());

    if let Some(cached) = guard.as_ref().filter(|cached| cached.key == key) {
        return Ok(Arc::clone(&cached.context));
    }

    let context = Arc::new(
        rt.block_on(vera_core::retrieval::search_service::SearchContext::new(
            config, backend,
        )),
    );
    *guard = Some(CachedSearchContext {
        key,
        context: Arc::clone(&context),
    });
    Ok(context)
}

fn search_context_key(
    config: &vera_core::config::VeraConfig,
    backend: vera_core::config::InferenceBackend,
) -> String {
    let config_json = serde_json::to_string(config).unwrap_or_default();
    format!("{backend}|{config_json}")
}

/// Resolve an optional path argument to a validated directory path.
fn resolve_repo_path(args: &Value) -> Result<std::path::PathBuf, ToolCallResult> {
    let repo_path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => std::path::PathBuf::from(p),
        None => current_working_dir()?,
    };
    if !repo_path.exists() {
        return Err(ToolCallResult::error(format!(
            "Path does not exist: {}",
            repo_path.display()
        )));
    }
    Ok(repo_path)
}

/// Handle the `get_stats` tool.
fn handle_get_stats(args: &Value) -> ToolCallResult {
    let repo_path = match resolve_repo_path(args) {
        Ok(p) => p,
        Err(e) => return e,
    };
    match vera_core::stats::collect_stats(&repo_path) {
        Ok(stats) => match serde_json::to_string_pretty(&stats) {
            Ok(json) => ToolCallResult::success(json),
            Err(e) => ToolCallResult::error(format!("Failed to serialize stats: {e}")),
        },
        Err(e) => ToolCallResult::error(format!("Failed to collect stats: {e}")),
    }
}

/// Handle the `get_overview` tool.
fn handle_get_overview(args: &Value) -> ToolCallResult {
    let repo_path = match resolve_repo_path(args) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let exact_paths = match git_scope_from_args(args) {
        Ok(Some(scope)) => match vera_core::git_scope::resolve_scope(&repo_path, &scope) {
            Ok(paths) => Some(paths),
            Err(err) => {
                return ToolCallResult::error(format!("Failed to resolve git scope: {err}"));
            }
        },
        Ok(None) => None,
        Err(err) => return ToolCallResult::error(err),
    };
    match vera_core::stats::collect_overview_filtered(&repo_path, exact_paths.as_ref()) {
        Ok(overview) => match serde_json::to_string_pretty(&overview) {
            Ok(json) => ToolCallResult::success(json),
            Err(e) => ToolCallResult::error(format!("Failed to serialize overview: {e}")),
        },
        Err(e) => ToolCallResult::error(format!("Failed to collect overview: {e}")),
    }
}

/// Handle the `regex_search` tool.
fn handle_regex_search(args: &Value) -> ToolCallResult {
    let pattern = match args.get("pattern").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return ToolCallResult::error("Missing required parameter: pattern"),
    };
    let scope = match scope_from_args(args) {
        Ok(scope) => scope,
        Err(err) => return err,
    };

    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(20);
    let ignore_case = args
        .get("ignore_case")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let context = args
        .get("context")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(2);

    let cwd = match current_working_dir() {
        Ok(cwd) => cwd,
        Err(err) => return err,
    };
    let index_dir = match existing_index_dir(&cwd) {
        Ok(index_dir) => index_dir,
        Err(err) => return err,
    };

    let mut filters = regex_search_filters(args, scope);
    if let Err(err) = apply_git_scope_filters(args, &cwd, &mut filters) {
        return err;
    }

    match vera_core::retrieval::search_regex(
        &index_dir,
        pattern,
        limit,
        ignore_case,
        context,
        &filters,
    ) {
        Ok(results) => {
            let signatures_only = args
                .get("compact")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            match compact_results_json(&results, MCP_OUTPUT_BUDGET, signatures_only) {
                Ok(json) => ToolCallResult::success(json),
                Err(e) => ToolCallResult::error(format!("Failed to serialize results: {e}")),
            }
        }
        Err(e) => ToolCallResult::error(format!("Regex search failed: {e}")),
    }
}

fn handle_structural_search(args: &Value) -> ToolCallResult {
    let kind = match args.get("kind").and_then(|v| v.as_str()) {
        Some(value) => match value.parse::<vera_core::retrieval::StructuralSearchKind>() {
            Ok(kind) => kind,
            Err(()) => {
                return ToolCallResult::error(format!("Invalid structural kind: {value}"));
            }
        },
        None => return ToolCallResult::error("Missing required parameter: kind"),
    };

    let scope = match scope_from_args(args) {
        Ok(scope) => scope,
        Err(err) => return err,
    };
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(20);
    let query = args.get("query").and_then(|v| v.as_str());

    let cwd = match current_working_dir() {
        Ok(cwd) => cwd,
        Err(err) => return err,
    };

    let mut filters = search_code_filters(args, scope);
    if let Err(err) = apply_git_scope_filters(args, &cwd, &mut filters) {
        return err;
    }

    let index_dir = match ensure_index_and_watcher(&cwd) {
        Ok(index_dir) => index_dir,
        Err(err) => return err,
    };

    match vera_core::retrieval::search_structural(&index_dir, kind, query, limit, &filters) {
        Ok(results) => {
            let signatures_only = args
                .get("compact")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            match compact_results_json(&results, MCP_OUTPUT_BUDGET, signatures_only) {
                Ok(json) => ToolCallResult::success(json),
                Err(e) => ToolCallResult::error(format!("Failed to serialize results: {e}")),
            }
        }
        Err(e) => ToolCallResult::error(format!("Structural search failed: {e}")),
    }
}

fn handle_find_references(args: &Value) -> ToolCallResult {
    let symbol = match args.get("symbol").and_then(|v| v.as_str()) {
        Some(symbol) if !symbol.trim().is_empty() => symbol.trim(),
        Some(_) => return ToolCallResult::error("Parameter 'symbol' must not be empty"),
        None => return ToolCallResult::error("Missing required parameter: symbol"),
    };
    let callees = args
        .get("callees")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(20);

    let cwd = match current_working_dir() {
        Ok(cwd) => cwd,
        Err(err) => return err,
    };

    let exact_paths = match exact_paths_from_args(args, &cwd) {
        Ok(paths) => paths,
        Err(err) => return err,
    };
    let index_dir = match ensure_index_and_watcher(&cwd) {
        Ok(index_dir) => index_dir,
        Err(err) => return err,
    };

    if callees {
        match vera_core::stats::find_callees(&cwd, symbol) {
            Ok(mut results) => {
                if let Some(paths) = exact_paths.as_ref() {
                    results.retain(|result| paths.contains(&result.file_path));
                }
                results.truncate(limit);
                match serde_json::to_string(&results) {
                    Ok(json) => ToolCallResult::success(json),
                    Err(err) => {
                        ToolCallResult::error(format!("Failed to serialize references: {err}"))
                    }
                }
            }
            Err(err) => ToolCallResult::error(format!("Reference lookup failed: {err}")),
        }
    } else {
        let filters = vera_core::types::SearchFilters {
            scope: Some(vera_core::types::SearchScope::Source),
            exact_paths,
            include_generated: Some(false),
            ..Default::default()
        };
        match vera_core::retrieval::search_callers(&index_dir, symbol, limit, &filters) {
            Ok(results) => {
                let signatures_only = args
                    .get("compact")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                match compact_results_json(&results, MCP_OUTPUT_BUDGET, signatures_only) {
                    Ok(json) => ToolCallResult::success(json),
                    Err(err) => {
                        ToolCallResult::error(format!("Failed to serialize references: {err}"))
                    }
                }
            }
            Err(err) => ToolCallResult::error(format!("Reference lookup failed: {err}")),
        }
    }
}

fn handle_explain_path(args: &Value) -> ToolCallResult {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(path) => path,
        None => return ToolCallResult::error("Missing required parameter: path"),
    };

    let cwd = match current_working_dir() {
        Ok(cwd) => cwd,
        Err(err) => return err,
    };

    let mut config = crate::saved_config::load_saved_runtime_config();
    config.indexing.no_ignore = args
        .get("no_ignore")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    config.indexing.no_default_excludes = args
        .get("no_default_excludes")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    config.indexing.extra_excludes = args
        .get("exclude")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();

    match vera_core::discovery::explain_path(&cwd, std::path::Path::new(path), &config.indexing) {
        Ok(explanation) => match serde_json::to_string_pretty(&explanation) {
            Ok(json) => ToolCallResult::success(json),
            Err(e) => ToolCallResult::error(format!("Failed to serialize explanation: {e}")),
        },
        Err(e) => ToolCallResult::error(format!("Failed to explain path: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definitions_has_seven_tools() {
        let tools = tool_definitions();
        assert_eq!(tools.len(), 7);

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"search_code"));
        assert!(names.contains(&"get_stats"));
        assert!(names.contains(&"get_overview"));
        assert!(names.contains(&"regex_search"));
        assert!(names.contains(&"structural_search"));
        assert!(names.contains(&"find_references"));
        assert!(names.contains(&"explain_path"));
    }

    #[test]
    fn git_scope_schema_is_only_on_overview() {
        let tools = tool_definitions();
        let get_stats = tools.iter().find(|tool| tool.name == "get_stats").unwrap();
        let get_overview = tools
            .iter()
            .find(|tool| tool.name == "get_overview")
            .unwrap();

        let stats_props = get_stats.input_schema["properties"].as_object().unwrap();
        assert!(!stats_props.contains_key("changed"));
        assert!(!stats_props.contains_key("since"));
        assert!(!stats_props.contains_key("base"));

        let overview_props = get_overview.input_schema["properties"].as_object().unwrap();
        assert!(overview_props.contains_key("changed"));
        assert!(overview_props.contains_key("since"));
        assert!(overview_props.contains_key("base"));
    }

    #[test]
    fn git_scope_descriptions_name_the_right_operation() {
        let tools = tool_definitions();
        for (tool_name, noun) in [
            ("search_code", "search"),
            ("get_overview", "overview"),
            ("regex_search", "regex search"),
            ("structural_search", "search"),
            ("find_references", "references"),
        ] {
            let tool = tools.iter().find(|tool| tool.name == tool_name).unwrap();
            let description = tool.input_schema["properties"]["changed"]["description"]
                .as_str()
                .unwrap();
            assert!(
                description.starts_with(&format!("Restrict {noun} to")),
                "{tool_name} changed description should name '{noun}': {description}"
            );
        }
    }

    #[test]
    fn tool_definitions_have_valid_schemas() {
        let tools = tool_definitions();
        for tool in &tools {
            let schema = &tool.input_schema;
            assert_eq!(schema["type"], "object", "tool {} schema type", tool.name);
        }
    }

    #[test]
    fn handle_unknown_tool_returns_error() {
        let result = handle_tool_call("nonexistent", &serde_json::json!({}));
        assert!(result.is_error);
        assert!(result.content[0].text.contains("Unknown tool"));
    }

    #[test]
    fn search_code_missing_query_returns_error() {
        let result = handle_tool_call("search_code", &serde_json::json!({}));
        assert!(result.is_error);
        assert!(
            result.content[0]
                .text
                .contains("Missing required parameter")
        );
    }

    #[test]
    fn search_code_accepts_queries_array() {
        // No index and no embedding config, should get past parameter validation.
        let result = handle_tool_call(
            "search_code",
            &serde_json::json!({"queries": ["foo", "bar"]}),
        );
        // Should fail (either auto-index fails or embedding provider fails).
        assert!(result.is_error);
    }

    #[test]
    fn cached_search_context_reuses_context_for_same_key() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = vera_core::config::VeraConfig::default();
        let backend = vera_core::config::InferenceBackend::Api;

        let first = cached_search_context(&rt, &config, backend).unwrap();
        let second = cached_search_context(&rt, &config, backend).unwrap();

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn search_code_filters_include_lang_path_and_symbol_type() {
        let filters = search_code_filters(
            &serde_json::json!({
                "lang": "rust",
                "path": "src/**/*.rs",
                "symbol_type": "function",
                "include_generated": true,
            }),
            Some(vera_core::types::SearchScope::Source),
        );

        assert_eq!(filters.language.as_deref(), Some("rust"));
        assert_eq!(filters.path_glob, vec!["src/**/*.rs"]);
        assert_eq!(filters.symbol_type.as_deref(), Some("function"));
        assert_eq!(filters.scope, Some(vera_core::types::SearchScope::Source));
        assert_eq!(filters.include_generated, Some(true));
    }

    #[test]
    fn search_code_filters_accept_path_array() {
        let filters = search_code_filters(
            &serde_json::json!({
                "path": ["src/**/*.rs", "tests/**/*.py"],
            }),
            None,
        );

        assert_eq!(
            filters.path_glob,
            vec!["src/**/*.rs".to_string(), "tests/**/*.py".to_string()]
        );
    }

    #[test]
    fn regex_search_schema_stays_minimal() {
        let tools = tool_definitions();
        let regex_search = tools
            .iter()
            .find(|tool| tool.name == "regex_search")
            .unwrap();
        let properties = regex_search.input_schema["properties"].as_object().unwrap();

        assert!(properties.contains_key("pattern"));
        assert!(properties.contains_key("scope"));
        assert!(properties.contains_key("include_generated"));
        assert!(!properties.contains_key("lang"));
        assert!(!properties.contains_key("path"));
        assert!(!properties.contains_key("symbol_type"));
    }

    #[test]
    fn structural_search_schema_exposes_kind_and_git_scope() {
        let tools = tool_definitions();
        let structural = tools
            .iter()
            .find(|tool| tool.name == "structural_search")
            .unwrap();
        let properties = structural.input_schema["properties"].as_object().unwrap();

        assert!(properties.contains_key("kind"));
        assert!(properties.contains_key("query"));
        assert!(properties.contains_key("changed"));
        assert!(properties.contains_key("since"));
        assert!(properties.contains_key("base"));
        assert!(
            !properties
                .get("kind")
                .and_then(|kind| kind.get("enum"))
                .and_then(|value| value.as_array())
                .unwrap()
                .iter()
                .any(|value| value == "calls")
        );
    }

    #[test]
    fn references_schema_exposes_symbol_and_git_scope() {
        let tools = tool_definitions();
        let refs = tools
            .iter()
            .find(|tool| tool.name == "find_references")
            .unwrap();
        let properties = refs.input_schema["properties"].as_object().unwrap();

        assert!(properties.contains_key("symbol"));
        assert!(properties.contains_key("callees"));
        assert!(properties.contains_key("changed"));
        assert!(properties.contains_key("since"));
        assert!(properties.contains_key("base"));
    }

    #[test]
    fn removed_tools_return_unknown() {
        for tool in &[
            "index_project",
            "update_project",
            "watch_project",
            "find_dead_code",
        ] {
            let result = handle_tool_call(tool, &serde_json::json!({}));
            assert!(result.is_error);
            assert!(result.content[0].text.contains("Unknown tool"));
        }
    }

    #[test]
    fn get_stats_no_index_returns_error() {
        let result = handle_tool_call("get_stats", &serde_json::json!({"path": "/tmp"}));
        assert!(result.is_error);
        // Should mention no index found or similar.
        assert!(!result.content[0].text.is_empty());
    }
}
