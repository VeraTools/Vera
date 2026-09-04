# Query Guide

## Writing Good Queries

Vera works best with queries that describe what the code *does*, not just a keyword. Think of it like asking "find me the code that handles X" rather than searching for a variable name.

**Strong queries:**

```bash
vera search "authentication middleware"
vera search "JWT token validation"
vera search "request rate limiting" --lang rust
vera search "database connection pooling"
vera search "routes" --path "src/**/*.ts" --path "tests/**/*.ts"
vera search "handler" --type function --limit 5
```

**Weak queries:**

```bash
vera search "code"
vera search "utils"
vera search "file"
```

Single generic words return too many results. Be specific about the behavior or concept you're looking for.

## Narrowing Results

If your first search returns too much, add filters one at a time:

- `--lang rust`: restrict to a specific language
- `--path "src/**/*.ts"`: restrict to a file path pattern; repeat it to OR several patterns
- `--type function`: restrict to functions and methods (function and method are aliases in both directions)
- `--scope docs`: restrict to docs and markdown
- `--scope runtime`: restrict to extracted runtime trees and bundled app code
- `--include-generated`: include dist/minified/generated artifacts
- `--limit 5`: return fewer results

These stack, so you can combine them:

```bash
vera search "error handling" --lang rust --type function --limit 5
vera search "mod loader" --scope runtime --include-generated
```

## Git-Scoped Search

When the task is limited to your working tree or a PR diff, scope the search before broadening the query:

```bash
vera search "token validation" --changed
vera grep "TODO|FIXME" --changed
vera overview --base origin/main
```

Use:

- `--changed` for modified, staged, and untracked files
- `--since <rev>` for changes since a specific revision
- `--base <rev>` for changes since `merge-base(HEAD, <rev>)`

## Multi-Query Search

If one phrasing is too narrow, pass 2-3 varied queries in one call:

```bash
vera search "OAuth token refresh" "JWT expiry handling" "auth middleware"
```

Vera runs each query, then merges the results with reciprocal rank fusion. Downstream ranking signals can reorder the fused candidates (definition chunks get a boost by default).

## Intent

If the query is short but your goal is specific, add `--intent`:

```bash
vera search "config" --intent "find where database connection strings are loaded from environment variables"
```

This works best when the query alone is too ambiguous to steer ranking.

## Exact Symbol Names

If you already know the symbol name, search for it directly:

```bash
vera search "parse_config"
vera search "AuthMiddleware"
```

Vera's hybrid pipeline handles both natural language intent and exact symbol lookups.

## Default Search Bias

Vera favors source files by default. Docs, archives, runtime extracts, and generated files are still available, but they are no longer treated as equally good defaults for everyday coding tasks. Definition chunks get a ranking boost by default (`retrieval.ranking_definition_boost`).

Use `--scope docs` when you are reading guides or ADRs. Use `--scope runtime` when you're debugging extracted app bundles or decompiled runtime code. Add `--include-generated` when you intentionally want minified or generated files in the result set.

## When to Use `vera grep` vs `rg`

Use `vera grep` when you want exact text or regex matches limited to indexed files:

- `vera grep "EMBEDDING_MODEL_BASE_URL"`
- `vera grep "TODO\(" -i`
- `vera grep "queryClient|invalidateQueries" --path "frontend/src/**"`

Vera uses Rust regex syntax. Use `|` for alternation. `\|` matches a literal pipe. When every `--path` pattern matches zero-indexed files, `vera grep` prints a hint to stderr suggesting a wildcard directory alternative (for example `src/**/` instead of `src/`).

Use `rg` when you need:

- Counting occurrences
- Find-and-replace prep
- File name search
- Files outside the Vera index

## Structural Search

Use `vera structural` for the common structural cases where you want an intent, not raw query syntax:

```bash
vera structural definitions parse_config
vera structural env DATABASE_URL
vera structural routes --path "src/**"
vera structural sql
vera structural impls Loader
```

This is the better default for agents. `vera structural definitions` accepts bare symbol names for split symbols and deduplicates to the earliest declaration.

Use `vera structural impls <symbol>` for explicit inheritance or conformance declarations. It does not infer implicit interface satisfaction.

Use `vera references` for exact caller/callee questions:

```bash
vera references parse_config
vera references parse_config --callees
vera references parse_config --changed
vera references get --receiver app          # only calls written as app.get(...)
```

Callers are matched by symbol name, so two definitions sharing a name land in
one answer. When that happens the output names the receivers the calls went
through and their counts; rerun with `--receiver <name>` to keep only the calls
made through one of them. Calls written without a receiver (a plain
`parse_config()`) are not matched by any `--receiver` value. `vera references` resolves split-symbol call sites, and `vera dead-code` deduplicates split parts by (symbol, file).

## Missing Files Or Surprising Exclusions

If a file is missing from search results and you need the exact reason, ask Vera directly:

```bash
vera explain-path path/to/file
```

If `vera search` or `vera grep` warns that the index may be stale, refresh it:

```bash
vera update .
```

Use `vera watch .` when you want the index to stay current while you work.

Use `vera stats --json` when you want the repo-wide health view for parse failures, tree-sitter errors, and Tier 0 fallback.

## Output Format

See [features: output formats](features.md#multiple-output-formats) for all options (`--json`, `--raw`, `--timing`). `--raw` works with `vera search`, `vera grep`, and `vera references`; `--timing` works with `vera search` and `vera grep`. They can appear before or after the subcommand. `vera search --timing` prints per-stage timings; `vera grep --timing` prints total regex-search time. In JSON, `symbol_name` is the bare string and split symbols add a `part_index` field. `retrieval.max_output_chars` defaults to 0 (unlimited output) and when non-zero truncates lower-ranked results first.

## Keeping Results Fresh

If results feel stale after code changes, run `vera update .`. See [troubleshooting](troubleshooting.md#results-feel-stale-or-outdated) for details.
