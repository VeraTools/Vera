# MCP

`vera mcp` runs a JSON-RPC MCP server over standard input and output. It exposes:

| Tool | Description |
|---|---|
| `search_code` | Hybrid BM25 and vector search for conceptual or behavioral queries, with filters, intent, multi-query input, and git scopes. |
| `get_stats` | Reports file count, chunk count, index size, language breakdown, and index health. |
| `get_overview` | Summarizes languages, directories, entry points, symbol types, complexity hotspots, and project conventions. |
| `regex_search` | Searches indexed files with a regular expression and surrounding context. |
| `structural_search` | Runs structural intents for definitions, environment-variable reads, routes, SQL sites, and explicit implementations. |
| `find_references` | Finds exact callers or callees through Vera's persisted call graph. |
| `explain_path` | Explains why a path is or is not indexed, including ignore, binary, size, and default-exclusion decisions. |

`search_code`, `structural_search`, and `find_references` auto-index and start a file watcher on first use when the project has no index. Search, references, and overview also accept changed-file git scopes.

## Client setup

### Claude Code

```bash
claude mcp add vera -- vera mcp
```

### Cursor

Add this server to the MCP configuration:

```json
{"mcpServers":{"vera":{"command":"vera","args":["mcp"]}}}
```

### Windsurf

Use the same JSON server entry:

```json
{"mcpServers":{"vera":{"command":"vera","args":["mcp"]}}}
```

### Codex

Add this TOML entry:

```toml
[mcp_servers.vera]
command = "vera"
args = ["mcp"]
```

### VS Code Copilot

Add the server to the MCP configuration used by VS Code:

```json
{"mcpServers":{"vera":{"command":"vera","args":["mcp"]}}}
```

For a local inference server, configure the backend separately and run `vera mcp`. See [llama.cpp setup](llama-cpp-setup.md) for the local endpoint example.

## Docker

The Docker images start `vera mcp` by default. Use the CPU, CUDA, ROCm, or OpenVINO variant described in the [Docker guide](docker.md), and mount the project at `/workspace`.

## Troubleshooting

If the server does not appear in the client, verify that `vera` is on the client's `PATH`. If the client has a restricted environment, use the absolute path to the Vera binary in the `command` field. Check the client logs and run `vera mcp` directly to confirm that the server starts.
