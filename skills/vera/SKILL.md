---
name: vera
description: Code search over the current repository. Before reading files to answer "where is X", "how does Y work", "find Z", or "what calls W", run `vera search "<query>"` or `vera references <symbol>` first. Use `vera grep` for exact strings and regex, `vera structural` for definitions, routes, and env reads. Do not read multiple files hoping to find the right one; search first, then read the hit.
---

# Vera

Ranked code search over an indexed repository. Results are markdown codeblocks: `path:line_start-line_end symbol_type:symbol_name` (split symbols render as `name (part N)` in that position, the bare name plus the part suffix), then the code.

## Pick the tool

| You are about to... | Do this instead |
|---------------------|-----------------|
| Read files to find where something lives | `vera search "config object construction"` |
| Read files to understand how something works | `vera search "env file loading decision"` |
| Find documentation on a topic | `vera search "deploying behind proxy" --scope docs` |
| Find every occurrence of a pattern | `vera grep "TODO\|FIXME"` |
| `vera grep` with `--path` matches nothing | Check stderr: when every `--path` pattern matches zero indexed files, Vera suggests a wildcard directory alternative (for example `src/**/` instead of `src/`) |
| Find callers or callees of a symbol | `vera references make_config` |
| Find definitions, routes, env reads | `vera structural env` / `vera structural routes` |
| Edit the same pattern in many files | `rg` |
| Read a file you already know | Read it directly |

`vera references` resolves split-symbol call sites, `vera structural definitions` finds split symbols by bare name and deduplicates to the earliest declaration, and `vera dead-code` deduplicates split parts by (symbol, file).

## Do not use Vera when

- You already know the exact path and line: open the file.
- You are editing across many files mechanically: use `rg`.
- The answer is a literal string you can match: `vera grep` beats `vera search`.
- You have already run two searches that returned the same region: stop searching and read the code.

## Search well

- Search behavior, not nouns: `"JWT expiry handling"`, not `"auth"` or `"utils"`.
- Pass several angles in one call: `vera search "OAuth token refresh" "JWT expiry" "auth middleware"`.
- Start broad with `--compact` (signatures only, fewer tokens), then narrow with `--lang`, `--path`, `--type`, `--limit`.
- Add `--intent "<goal>"` when the query is vague but the goal is clear.
- Scope to a change with `--changed`, `--since <rev>`, or `--base <rev>` when reviewing a diff.
- `--deep` rewrites the query through an LLM; use it only after normal search misses.

## Treat hits as leads

- A search hit is a lead, not evidence. Before stating how something behaves, open the cited lines.
- Follow the call graph rather than re-searching: `vera references <symbol>` on a promising hit answers "who drives this" in one step.
- Cite `path:line` from code you actually read.
- After editing files, run `vera update .` before searching again.

## Recovery

| Symptom | Fix |
|---------|-----|
| `no index found` | `vera index .` |
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
