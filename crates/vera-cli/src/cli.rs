//! CLI argument definitions (clap derive).

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "vera",
    about = "Hybrid code indexing and retrieval for CLI-first coding-agent workflows",
    long_about = "Vera is a code indexing and retrieval tool for source trees. It combines \
                  BM25 full-text search with vector similarity search using Reciprocal Rank \
                  Fusion (RRF) and optional cross-encoder reranking to return ranked code \
                  results for direct CLI use and installable agent skills. Vera always keeps \
                  the index local in `.vera/`; `vera setup` only chooses the model backend.\n\n\
                  Quick start:\n  \
                  vera agent install                   # Interactive: choose scope + agents\n  \
                  vera setup                          # Download built-in local models\n  \
                  vera index .                        # Index current directory\n  \
                  vera search \"auth\"                  # Search for authentication code\n  \
                  vera doctor                         # Check local setup and index health\n  \
                  vera repair                         # Re-fetch missing backend assets\n  \
                  vera upgrade                        # Show the binary update plan",
    after_long_help = "Output flags:\n  \
                  --json                             Structured JSON when supported\n  \
                  --raw                              Verbose search/grep output (before or after the subcommand)\n  \
                  --timing                           Search/grep timings to stderr (before or after the subcommand)",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Output command results as JSON when supported.
    ///
    /// Search commands emit compact machine-readable JSON; status and
    /// diagnostic commands emit structured JSON summaries.
    #[arg(long, global = true)]
    pub json: bool,

    /// Output all fields with pretty-printed verbose formatting.
    ///
    /// Search-style commands honor this flag whether it appears before or
    /// after the subcommand.
    #[arg(long, global = true)]
    pub raw: bool,

    /// Print timing information to stderr when supported.
    ///
    /// Search prints per-stage timings; grep prints total elapsed time.
    #[arg(long, global = true)]
    pub timing: bool,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the MCP (Model Context Protocol) server.
    #[command(long_about = "Start the MCP (Model Context Protocol) server.\n\n\
                      Runs a JSON-RPC 2.0 server over stdio so editors, assistants, and \
                      other tools can use Vera's indexing and search capabilities.\n\n\
                      The server reads JSON-RPC messages from stdin and writes responses \
                      to stdout. Logs go to stderr.\n\n\
                      Exposed tools:\n  \
                       search_code      — Hybrid search (auto-indexes and watches on first use)\n  \
                       get_stats        — Index statistics and health\n  \
                       get_overview     — Project summary for onboarding\n  \
                       regex_search     — Regex search over indexed files\n  \
                       explain_path     — Explain why a file is or is not indexed\n\n\
                      Examples:\n  \
                      vera mcp                       # Start MCP server on stdio")]
    Mcp,

    /// Start the Vera HTTP API server for remote embedding and reranking.
    #[command(long_about = "Start the Vera HTTP API server.\n\n\
                      Loads the embedding model at startup and keeps it (using the \
                      selected backend), then exposes it via HTTP so any unmodified vera \
                      client can use this host for compute.\n\n\
                      The reranker is built at startup too, to check it works, but that \
                      copy is discarded rather than kept: a server that only answers \
                      /v1/embeddings would otherwise hold it for nothing. Startup \
                      therefore pays for both models, and the first /v1/rerank request \
                      pays to load the reranker again.\n\n\
                      A loaded model is held in memory and reused across requests. It is \
                      unloaded after --idle-timeout seconds of inactivity and reloaded on \
                      the next request that needs it; see that flag for the values that \
                      keep it loaded indefinitely or rebuild it per request.\n\n\
                      Standard client setup (no client changes needed):\n  \
                      1. Start server:  vera serve --onnx-jina-cuda\n  \
                      2. Configure client:\n  \
                         export EMBEDDING_MODEL_BASE_URL=http://host:3000/v1\n  \
                         export EMBEDDING_MODEL_ID=<model-name-shown-at-startup>\n  \
                         export EMBEDDING_MODEL_API_KEY=<api-key-if-set>\n  \
                         vera setup --api\n  \
                      3. Use normally:  vera index . && vera search \"auth\"\n\n\
                      Authentication: --api-key (or VERA_SERVE_KEY env var).\n\n\
                      Endpoints:\n  \
                      POST /v1/embeddings    — OpenAI-compatible embeddings\n  \
                      POST /v1/rerank        — Cohere/Jina-compatible reranker\n  \
                      GET  /v1/health        — model info and liveness\n\n\
                      Examples:\n  \
                      vera serve                              # CPU (saved config), port 3000\n  \
                      vera serve --onnx-jina-cuda             # NVIDIA GPU\n  \
                      vera serve --onnx-jina-rocm             # AMD GPU\n  \
                      vera serve --host 0.0.0.0 --port 8080  # Expose to network\n  \
                      vera serve --api-key some-api-key             # Require bearer token")]
    Serve {
        /// TCP port to listen on.
        #[arg(long, default_value = "3000")]
        port: u16,

        /// Bind address.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Bearer token required from clients (or set VERA_SERVE_KEY env var).
        #[arg(long)]
        api_key: Option<String>,

        /// Seconds of inactivity before a model is unloaded from memory.
        /// 0 = rebuild the model on every request, and hold one live model per
        /// concurrent request; only useful to pick up model files replaced under
        /// a running server. Any negative value, -1 included, keeps models
        /// loaded indefinitely.
        #[arg(long, default_value_t = 300, allow_negative_numbers = true)]
        idle_timeout: i64,

        #[command(flatten)]
        backend: crate::helpers::LocalBackendFlags,

        /// Use API-backed embeddings/reranking (reads from environment).
        #[arg(long, group = "backend")]
        api: bool,
    },

    /// Install or manage the Vera skill for supported coding agents.
    #[command(
        long_about = "Install or manage the Vera skill for supported coding agents.\n\n\
                      This is the preferred agent integration path. Vera installs a \
                      CLI-centric skill bundle into known skill directories so agents \
                      can call `vera index`, `vera search`, `vera update`, and \
                      `vera stats` directly.\n\n\
                      `vera agent install` detects existing installs and lets you \
                      add or remove agents in one step. Deselecting an installed \
                      agent removes it. If stale installs are detected, the \
                      interactive flow can refresh them in one step before \
                      opening the full selector.\n\n\
                      `vera agent sync` refreshes all stale skill installs to match \
                      the current binary version and updates managed markdown agent \
                      config snippets in the current project, no prompts needed.\n\n\
                      Examples:\n  \
                      vera agent install                       # Interactive: choose scope and agents\n  \
                      vera agent install --client claude       # Install for Claude Code (global)\n  \
                      vera agent install --client all --scope project  # All agents, project only\n  \
                      vera agent sync                          # Update all stale skills\n  \
                      vera agent status                        # Show all install status\n  \
                      vera agent remove --client codex         # Remove the global Codex install"
    )]
    Agent {
        /// Agent command: install, status, remove, or sync.
        #[arg(value_enum)]
        command: crate::commands::agent::AgentCommand,
        /// Which agent client to target. Without this flag, interactive mode
        /// presents a checklist of all supported agents.
        #[arg(long, value_enum)]
        client: Option<crate::commands::agent::AgentClient>,
        /// Install scope: global, project, or all. Without this flag,
        /// interactive mode prompts for scope selection.
        #[arg(long, value_enum)]
        scope: Option<crate::commands::agent::AgentScope>,
    },

    /// Remove Vera: binary, models, config, agent skills, and PATH shim.
    #[command(
        long_about = "Remove Vera: binary cache, models, ONNX Runtime libs, config, \n\
                      credentials, agent skill files, and the PATH shim.\n\n\
                      Per-project indexes (.vera/ inside each project) are not touched.\n\n\
                      Examples:\n  \
                      vera uninstall\n  \
                      vera uninstall --json"
    )]
    Uninstall,

    /// Interactive first-time setup wizard.
    #[command(long_about = "Interactive first-time setup wizard.\n\n\
                      Walks through three steps:\n  \
                      1. Backend selection (Potion CPU, ONNX runtime + GPU, or API mode)\n  \
                      2. Agent skill installation (choose scope and agents)\n  \
                      3. Optional project indexing\n\n\
                      For backend-only changes, use `vera backend`. For skill-only \
                      changes, use `vera agent install`.\n\n\
                      Pass flags to skip the interactive wizard:\n  \
                      vera setup --potion-code         # CPU-only local mode\n  \
                      vera setup --onnx-jina-cuda      # NVIDIA GPU, skip wizard\n  \
                      vera setup --api                 # API mode from env vars\n  \
                      vera setup --yes                 # Auto-detect GPU, no prompts\n\n\
                      Examples:\n  \
                      vera setup                       # Full interactive wizard\n  \
                      vera setup --potion-code --index .       # CPU + index, no wizard\n  \
                      vera setup --onnx-jina-cuda --index .    # GPU + index, no wizard")]
    Setup {
        #[command(flatten)]
        backend: crate::helpers::LocalBackendFlags,
        #[command(flatten)]
        embedding: crate::helpers::LocalEmbeddingModelFlags,
        /// Configure Vera for API-backed mode using current env vars.
        #[arg(long, group = "backend")]
        api: bool,
        /// Optionally index a repository after saving config.
        #[arg(long)]
        index: Option<String>,
        /// Skip the confirmation prompt.
        #[arg(long)]
        yes: bool,
    },

    /// Select and manage the model backend.
    #[command(long_about = "Select and manage the model backend.\n\n\
                      This is the focused backend configuration command. It handles \
                      runtime selection, model downloads, and API credential persistence \
                      without touching agent skills or project indexes.\n\n\
                      With no flags, shows an interactive backend menu with auto-detected \
                      GPU as the default.\n\n\
                      Examples:\n  \
                      vera backend                     # Interactive backend selection\n  \
                      vera backend --potion-code        # CPU-only local mode\n  \
                      vera backend --onnx-jina-cuda    # NVIDIA GPU (skip menu)\n  \
                      vera backend --code-rank-embed   # Switch to CodeRankEmbed model\n  \
                      vera backend --api               # Persist API credentials from env\n  \
                      vera backend --yes               # Auto-detect GPU, no prompts")]
    Backend {
        #[command(flatten)]
        backend: crate::helpers::LocalBackendFlags,
        #[command(flatten)]
        embedding: crate::helpers::LocalEmbeddingModelFlags,
        /// Configure Vera for API-backed mode using current env vars.
        #[arg(long, group = "backend")]
        api: bool,
        /// Skip the confirmation prompt.
        #[arg(long)]
        yes: bool,
    },

    /// Inspect the current Vera setup for common configuration issues.
    #[command(
        long_about = "Inspect the current Vera setup for common configuration issues.\n\n\
                      Checks the persisted config, effective mode, local runtime or \
                      API environment variables, and whether the current repository \
                      has a `.vera/` index. `--probe` adds a deeper read-only local \
                      backend probe and never downloads or repairs missing assets.\n\n\
                      Exits 1 if any check fails, so `vera doctor && vera index .` \
                      stops on a broken setup. Warnings do not affect the exit \
                      code.\n\n\
                      Examples:\n  \
                      vera doctor\n  \
                      vera doctor --probe\n  \
                      vera doctor --json"
    )]
    Doctor {
        /// Run a deeper read-only probe of local backend init.
        #[arg(long, visible_alias = "deep")]
        probe: bool,
    },

    /// Repair the configured Vera backend.
    #[command(long_about = "Repair the configured Vera backend.\n\n\
                      For local backends, this re-fetches missing runtime and \
                      model assets for the selected backend. For API mode, it re-saves \
                      the current API environment variables into Vera's config.\n\n\
                      This is a write operation. Use `vera doctor --probe` for a read-only \
                      diagnostic check.\n\n\
                      Examples:\n  \
                      vera repair\n  \
                      vera repair --potion-code\n  \
                      vera repair --onnx-jina-cuda\n  \
                      vera repair --api")]
    Repair {
        #[command(flatten)]
        backend: crate::helpers::LocalBackendFlags,
        /// Repair API-backed mode using current env vars.
        #[arg(long, group = "backend")]
        api: bool,
    },

    /// Show the binary update plan, or apply it when the install method is known.
    #[command(
        long_about = "Show the binary update plan, or apply it when the install method is known.\n\n\
                      By default, `vera upgrade` is a dry run: it checks for a newer Vera \
                      release, resolves the saved or detected install method, and prints the \
                      exact command it would use.\n\n\
                      `--apply` runs the installer command only when Vera can determine a \
                      single install method. If multiple install methods are detected, Vera \
                      prints the manual options and refuses to guess.\n\n\
                      Examples:\n  \
                      vera upgrade\n  \
                      vera upgrade --apply\n  \
                      vera upgrade --json"
    )]
    Upgrade {
        /// Run the planned installer command instead of printing it only.
        #[arg(long)]
        apply: bool,
    },

    /// Index a codebase for search.
    #[command(long_about = "Index a codebase for search.\n\n\
                      Discovers source files (respecting .gitignore), parses them with \
                      tree-sitter for 60+ languages, creates searchable chunks at symbol \
                      boundaries, generates embeddings using the current Vera mode, and \
                      stores everything in a local `.vera/` index directory.\n\n\
                      Use `vera setup` for Vera's built-in local models, or `vera setup \
                      --api` for an OpenAI-compatible endpoint.\n\n\
                      Examples:\n  \
                      vera index .                  # Index current directory\n  \
                      vera index /path/to/repo      # Index a specific repo\n  \
                      vera index . --json           # Output summary as JSON")]
    Index {
        /// Path to the directory to index.
        path: String,
        #[command(flatten)]
        backend: crate::helpers::LocalBackendFlags,
        /// Exclude files matching this glob pattern (repeatable).
        #[arg(long = "exclude")]
        exclude: Vec<String>,
        /// Disable .gitignore and .veraignore parsing.
        #[arg(long)]
        no_ignore: bool,
        /// Disable smart default exclusions.
        #[arg(long)]
        no_default_excludes: bool,
        /// Disable the interactive indexing progress display.
        #[arg(long)]
        no_progress: bool,
        /// Show detailed information (e.g. paths of skipped files).
        #[arg(long, short = 'v')]
        verbose: bool,
        /// Reduce GPU memory usage (batch_size=1, conservative VRAM limit).
        #[arg(long)]
        low_vram: bool,
    },

    /// Run agent-oriented structural search intents.
    #[command(long_about = "Run agent-oriented structural search intents.\n\n\
                      Uses the existing index to answer common code-navigation \n\
                      questions without regex authoring or raw tree-sitter queries.\n\n\
                      Intents:\n  \
                      definitions   Find symbol definitions by name\n  \
                      env           Find environment variable reads\n  \
                      routes        Find common HTTP route registrations\n  \
                      sql           Find common SQL execution sites\n  \
                      impls         Find explicit implementations, conformances, and inheritance\n\n\
                      Examples:\n  \
                      vera structural definitions parse_config\n  \
                      vera structural env DATABASE_URL\n  \
                      vera structural routes --path \"src/**\"\n  \
                      vera structural sql --lang python\n  \
                      vera structural impls Loader")]
    Structural {
        /// Structural intent to run.
        #[arg(value_enum)]
        intent: crate::commands::structural::StructuralIntent,
        /// Query term. Required for definitions and impls, optional for env, rejected by routes and sql.
        query: Option<String>,
        #[command(flatten)]
        filters: crate::helpers::SearchFilterArgs,
        /// Maximum number of results (default: 20).
        #[arg(long, short = 'n')]
        limit: Option<usize>,
        #[command(flatten)]
        git_scope: crate::helpers::GitScopeFlags,
        /// Show only function/class signatures (omit bodies).
        #[arg(long)]
        compact: bool,
    },

    /// Search the indexed codebase.
    #[command(long_about = "Search the indexed codebase.\n\n\
                      Performs hybrid search combining BM25 keyword matching and vector \
                      similarity via Reciprocal Rank Fusion (RRF). Optional cross-encoder \
                      reranking for improved precision.\n\n\
                      Pass multiple quoted queries to merge different phrasings into a \
                      single result set. Use `--intent` when the query is short but your \
                      higher-level goal needs to guide reranking.\n\n\
                      Source files are favored by default. Use `--scope docs` for prose, \
                      `--scope runtime` for extracted runtime trees, and `--include-generated` \
                      when you intentionally want dist/minified artifacts.\n\n\
                      Falls back gracefully: if embedding API is unavailable, uses BM25-only \
                      search. If reranker is unavailable, returns unreranked hybrid results.\n\n\
                      Requires an existing index (run `vera index <path>` first).\n\n\
                      Examples:\n  \
                      vera search \"auth logic\"                                # Semantic search\n  \
                      vera search \"parse_config\"                               # Symbol lookup\n  \
                      vera search \"hotkeys\" --scope docs                       # Search docs only\n  \
                      vera search \"OAuth token refresh\" \"JWT expiry handling\" \"auth middleware\"\n  \
                      vera search \"config\" --intent \"find env-based DB loading\"\n  \
                      vera search \"error handling\" --lang rust                 # Filter by language\n  \
                      vera search \"routes\" --path \"src/**/*.ts\"                # Filter by path\n  \
                      vera search \"DB queries\" --type function                 # Filter by symbol type\n  \
                      vera search \"config\" --limit 5 --json --timing            # JSON output + timings")]
    Search {
        /// One or more search queries (keyword or natural language).
        ///
        /// Pass multiple quoted queries to merge different phrasings in one call.
        #[arg(required = true, num_args = 1..)]
        queries: Vec<String>,

        /// Higher-level goal used to disambiguate the query before reranking.
        #[arg(long)]
        intent: Option<String>,

        /// Maximum number of results to return (default: 5).
        #[arg(long, short = 'n')]
        limit: Option<usize>,
        /// Search symbol type. Note: function and method are treated as aliases.
        #[command(flatten)]
        filters: crate::helpers::SearchFilterArgs,

        /// Deep search: RAG-fusion query expansion + merged ranking when a completion
        /// endpoint is configured, otherwise iterative symbol-following search.
        #[arg(long)]
        deep: bool,

        #[command(flatten)]
        git_scope: crate::helpers::GitScopeFlags,

        /// Show only function/class signatures (omit bodies).
        ///
        /// Useful for broad exploration: fits more results in fewer tokens.
        /// Use default mode for targeted retrieval of full implementations.
        #[arg(long)]
        compact: bool,

        #[command(flatten)]
        backend: crate::helpers::LocalBackendFlags,
    },

    /// Incrementally update the index for changed files.
    #[command(long_about = "Incrementally update the index for changed files.\n\n\
                      Uses content hashing to detect files that have been added, modified, \
                      or deleted since the last index/update. Only changed files are \
                      re-processed, making updates much faster than a full re-index.\n\n\
                      Uses the saved Vera mode from `vera setup`, or the current shell \
                      environment if you are configuring providers manually.\n\n\
                      Examples:\n  \
                      vera update .                  # Update current directory\n  \
                      vera update /path/to/repo      # Update a specific repo\n  \
                      vera update . --max-files 250  # Bound work for this run\n  \
                      vera update . --json           # Output summary as JSON")]
    Update {
        /// Path to the directory to update.
        path: String,
        #[command(flatten)]
        backend: crate::helpers::LocalBackendFlags,
        /// Exclude files matching this glob pattern (repeatable).
        #[arg(long = "exclude")]
        exclude: Vec<String>,
        /// Disable .gitignore and .veraignore parsing.
        #[arg(long)]
        no_ignore: bool,
        /// Disable smart default exclusions.
        #[arg(long)]
        no_default_excludes: bool,
        /// Disable the interactive update progress display.
        #[arg(long)]
        no_progress: bool,
        /// Maximum added or modified files to process in this run.
        #[arg(long, value_name = "N")]
        max_files: Option<std::num::NonZeroUsize>,
    },

    /// Show architecture overview of the indexed project.
    #[command(long_about = "Show architecture overview of the indexed project.\n\n\
                      Returns a high-level summary of the codebase: languages with file \n\
                      and chunk counts, top-level directories, symbol type breakdown, \n\
                      likely entry points, and complexity hotspots.\n\n\
                      Useful for quick orientation when starting work on a new project.\n\n\
                      Examples:\n  \
                      vera overview             # Human-readable overview\n  \
                      vera overview --json      # Machine-readable JSON output")]
    Overview {
        #[command(flatten)]
        git_scope: crate::helpers::GitScopeFlags,
    },

    /// Explain why a path is or is not indexed.
    #[command(long_about = "Explain why a path is or is not indexed.\n\n\
                      Resolves the path relative to the current working directory, then explains \n\
                      the first decisive reason Vera would exclude it, such as a default exclude, \n\
                      --exclude flag, .veraignore, .ignore, .gitignore, binary detection, size \n\
                      limit, or RST include-fragment handling.\n\n\
                      Examples:\n  \
                      vera explain-path src/main.rs\n  \
                      vera explain-path dist/bundle.js\n  \
                      vera explain-path docs/includes/common.rst.inc --json")]
    ExplainPath {
        /// Path to explain.
        path: String,
        /// Exclude files matching this glob pattern (repeatable).
        #[arg(long = "exclude")]
        exclude: Vec<String>,
        /// Disable .gitignore and .veraignore parsing.
        #[arg(long)]
        no_ignore: bool,
        /// Disable smart default exclusions.
        #[arg(long)]
        no_default_excludes: bool,
    },

    /// Find callers or callees of a symbol.
    ///
    /// Queries the call graph built during indexing to find where a symbol
    /// is called from (callers) or what it calls (callees). Caller lookups
    /// return code snippets by default and support git-scoped filtering.
    ///
    /// Examples:
    ///   vera references parse_and_chunk
    ///   vera references parse_and_chunk --callees
    ///   vera references parse_and_chunk --changed
    ///   vera references parse_and_chunk --json
    References {
        /// Symbol name to look up.
        symbol: String,
        /// Show what this symbol calls instead of what calls it.
        #[arg(long)]
        callees: bool,
        /// Maximum number of results to return (default: 20).
        #[arg(long, short = 'n')]
        limit: Option<usize>,
        #[command(flatten)]
        git_scope: crate::helpers::GitScopeFlags,
        /// Show only caller signatures (omit bodies).
        #[arg(long)]
        compact: bool,
    },

    /// Regex pattern search over indexed files.
    #[command(long_about = "Regex pattern search over indexed files.\n\n\
                      Searches file contents using a regex pattern, returning matches \
                      with surrounding context lines. Only searches files that are in \
                      the Vera index, so .gitignore and .veraignore rules apply.\n\n\
                      Supports the same corpus filters as `vera search`: language, file \
                      path glob, symbol type, scope, and generated-file inclusion.\n\n\
                      Examples:\n  \
                      vera grep \"fn\\s+main\"                            # Find main functions\n  \
                      vera grep \"TODO|FIXME\" -i                         # Case-insensitive\n  \
                      vera grep \"queryClient|invalidateQueries\" --path \"frontend/src/**\"\n  \
                      vera grep \"Authorization\" --lang rust --type function\n  \
                      vera grep \"keybind\" --scope docs                  # Search docs first\n  \
                      vera grep \"use std::\" --context 0                 # No context lines")]
    Grep {
        /// Regex pattern to search for.
        pattern: String,
        #[command(flatten)]
        filters: crate::helpers::SearchFilterArgs,

        /// Maximum number of results (default: 20).
        #[arg(long, short = 'n')]
        limit: Option<usize>,

        /// Case-insensitive matching.
        #[arg(long, short = 'i')]
        ignore_case: bool,

        /// Number of context lines before and after each match (default: 2).
        #[arg(long, default_value = "2")]
        context: usize,

        #[command(flatten)]
        git_scope: crate::helpers::GitScopeFlags,

        /// Show only function/class signatures (omit bodies).
        #[arg(long)]
        compact: bool,
    },

    /// Find symbols with no callers (potential dead code).
    ///
    /// Scans the call graph for functions/methods that are never called.
    /// Excludes common entry points (main, new, default, etc.).
    ///
    /// Examples:
    ///   vera dead-code
    ///   vera dead-code --json
    DeadCode,

    /// Show index statistics.
    #[command(long_about = "Show index statistics.\n\n\
                      Displays file count, chunk count, index size on disk, and a breakdown \
                      of chunks by programming language for the current index.\n\n\
                      Looks for the index in the current working directory (`.vera/`).\n\n\
                      Examples:\n  \
                      vera stats             # Human-readable stats\n  \
                      vera stats --json      # Machine-readable JSON output")]
    Stats,

    /// Watch a project directory and auto-update the index on file changes.
    #[command(
        long_about = "Watch a project directory and auto-update the index on file changes.\n\n\
                      Starts a background file watcher that triggers incremental index updates \n\
                      when source files change. Changes are debounced (2s) to avoid redundant \n\
                      updates during rapid edits.\n\n\
                      Requires an existing index (run `vera index <path>` first). \n\
                      Blocks until interrupted with Ctrl-C.\n\n\
                      Examples:\n  \
                      vera watch .                   # Watch current directory\n  \
                      vera watch /path/to/repo       # Watch a specific repo\n  \
                      vera watch . --json            # JSON status output"
    )]
    Watch {
        /// Path to the directory to watch.
        path: String,
    },

    /// Show or set configuration values.
    #[command(long_about = "Show or set configuration values.\n\n\
                      Without arguments or with `show`, displays the full current \
                      configuration as a table (or JSON with --json).\n\n\
                      Use `get <key>` to read a specific value, or `set <key> <value>` \
                      to update it.\n\n\
                      Configuration keys use dot notation:\n  \
                      indexing.max_chunk_lines       Max lines per chunk (default: 200)\n  \
                      indexing.max_file_size_bytes   Max file size to index (default: 1000000)\n  \
                      retrieval.default_limit        Default result count (default: 5)\n  \
                      retrieval.rrf_k                RRF fusion constant (default: 60)\n  \
                      retrieval.rerank_candidates    Reranker candidate count (default: 50)\n  \
                      retrieval.reranking_enabled    Enable reranking (default: true)\n  \
                      retrieval.max_output_chars     Total output char budget (default: 12000)\n  \
                      embedding.batch_size           Embedding batch size (default: 128)\n  \
                      embedding.max_concurrent_requests  Concurrent API requests (default: 8)\n  \
                      embedding.timeout_secs         API timeout (default: 60)\n  \
                      embedding.max_retries          API retry count (default: 3)\n  \
                      embedding.max_stored_dim       Vector dimensionality (default: 1024)\n\n\
                      Examples:\n  \
                      vera config                                  # Show all settings\n  \
                      vera config show                             # Same as above\n  \
                      vera config get retrieval.default_limit      # Get one value\n  \
                      vera config set retrieval.default_limit 20   # Set a value\n  \
                      vera config --json                           # JSON output")]
    Config {
        /// Config action: show (default), get <key>, or set <key> <value>.
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
}
