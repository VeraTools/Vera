---
name: vera
description: Code search over the current repository. Your first search in a repository goes through Vera, not grep, rg, or file reads. Run `vera overview` in an unfamiliar repository, `vera search "<what the code does>"` for how or where something works, `vera structural definitions <symbol>` or `vera references <symbol>` when the question names a symbol, and `vera grep "pattern"` for exact text or regex. Fall back to grep only after a Vera call has missed.
---

# Vera

Ranked code search over an indexed repository. Results are markdown codeblocks: `path:line_start-line_end symbol_type:symbol_name` (split symbols render as `name (part N)` in that position, the bare name plus the part suffix), then the code.

## First search

The first search action in a task decides whether Vera gets used at all; later searches follow the first tool chosen. So:

1. Unfamiliar repository: `vera overview` (languages, entry points, hotspots) instead of `ls` and README skimming.
2. Then the question's first lookup goes through the table below. Do not "check with grep first".

## Pick the tool

| Question shape | Do this |
|----------------|---------|
| How or where does something work | `vera search "env file loading decision"` |
| The question names a function, class, or config key | `vera structural definitions make_config`, then `vera references make_config` for callers |
| Who calls or what is called by a symbol | `vera references make_config` / `vera references make_config --callees` |
| Exact text or regex | `vera grep "before_request"` (line numbers always shown; `--limit <N>` caps results) |
| Enumerate every route, env read, implementation | `vera structural routes` / `vera structural env` / `vera structural impls <symbol>` |
| Documentation on a topic | `vera search "deploying behind proxy" --scope docs` |
| Files changed in this branch | add `--changed`, `--since <rev>`, or `--base <rev>` to any of the above |
| Edit the same pattern in many files mechanically | `rg` |
| Read a file you already know the path and lines of | Read it directly |

`vera references` resolves split-symbol call sites, `vera structural definitions` finds split symbols by bare name and deduplicates to the earliest declaration, and `vera dead-code` deduplicates split parts by (symbol, file).

## Search well

- Search behavior, not nouns: `"JWT expiry handling"`, not `"auth"` or `"utils"`.
- Pass several angles in one call: `vera search "OAuth token refresh" "JWT expiry" "auth middleware"`.
- Start broad with `--compact` (signatures only, fewer tokens), then narrow with `--lang`, `--path`, `--type`, `--limit`.
- `--path` is relative to the repository root: `--path src/flask`, `--path "tests/**/*.py"`. Absolute paths inside the repository are accepted and rewritten; paths outside it match nothing.
- Add `--intent "<goal>"` when the query is vague but the goal is clear.
- `--deep` expands the query and merges rankings; use it only after normal search misses.

## Recover from a miss

- Top hit is a usage site, not the definition: `vera structural definitions <symbol>` with the name from the hit. Do not try more phrasings.
- Hit shows a call but not the caller chain: `vera references <symbol>`.
- Two searches returned the same region: stop searching and read that code.
- The code is in a dependency (site-packages, uv or pip caches, vendored trees outside the index): use `rg` there, then return to Vera for repository code.

## Treat hits as leads

- A hit is a lead, not evidence. Verify behavior against the hit's code; open the file only for lines the hit did not include.
- Cite `path:line` from code you actually read.
- A stale-index warning does not invalidate hits. After editing files, run `vera update .` before searching again.

## Recovery

| Symptom | Fix |
|---------|-----|
| `no index found` | `vera index .` from the repository root, then rerun the search |
| `no indexed file matches <path>` | Use a root-relative `--path`; stderr suggests a wildcard directory form when one applies |
| Stale results after edits | `vera update .` (or `vera watch .`) |
| A file is missing from results | `vera explain-path path/to/file` |
| Local model or ONNX error | `vera doctor --probe`, then `references/troubleshooting.md` |
| Missing local assets | `vera repair` |
| Install, API keys, backends | `references/install.md` |
| MCP server | `references/mcp.md` |

## References

- `references/install.md`: install, setup, API and local config, `.veraignore` rules
- `references/query-patterns.md`: more query examples and rg guidance
- `references/troubleshooting.md`: common errors and fixes
- `references/mcp.md`: optional MCP server usage
