# Agent-Level Vera Benchmark

This benchmark measures whether a coding agent answers cross-file questions about Flask with less tool use and lower latency when a code-search tool is available. It records answer text, tool-call counts, token usage reported by the agent stream, and wall-clock time. Answer quality is graded separately by `judge.py`, which scores each answer blind (no arm label) against `flask/answer-key.md` with a 0-10 rubric.

There are three arms:

- `with-vera`: fresh Flask copy with a local Vera index and the project-scoped Droid skill installed. Its `PATH` resolves the release Vera binary first; a shim makes `semble` exit 127.
- `with-semble`: fresh Flask copy with an `AGENTS.md` at the repo root describing Semble's CLI usage and a pre-warmed Semble index cache inside the arm (`SEMBLE_CACHE_LOCATION` points there). A shim makes `vera` exit 127.
- `control`: otherwise identical copy with neither tool; shims make both `vera` and `semble` exit 127 with `<tool>: not available in this environment`.

Fairness: each tool arm gets that tool's own CLI integration. The Vera arm installs its Droid skill; the Semble arm gets a hand-written `AGENTS.md` covering the documented `semble search` and `semble find-related` usage, recorded in `setup.json`. Semble's installer text is not used because it leads with `mcp__semble__*` tool calls and no MCP server runs here; both tools are exercised through the shell only. Questions run sequentially, rotating the starting arm by question number so each arm leads roughly equally often.

Every cell runs with `FACTORY_HOME_OVERRIDE` pointed at a run-local config directory holding only credentials and model settings, and with `--disable-builtin-skills`. Without that, the operator's global `AGENTS.md`, personal skills, and custom droids would reach into all three arms and shift agent behavior for reasons unrelated to the tool under test.

## Questions

Put exactly 10 questions in `flask/questions.md` under `## Question 1` through `## Question 10` headings. The harness reads this file before setup or analysis and fails clearly if it is missing or malformed. Keep any private grading material outside all sandbox copies; the harness does not copy it into an arm and does not include it in prompts.

## Reproduce

From the Vera repository root, build the harness's release binary and create the three arms:

```bash
python3 benchmarks/agent-bench/run.py --setup-only
```

The setup phase copies `.bench/semble-repos/flask` into a timestamped directory under `~/.cache/agent-bench/`, excludes `.git`, `.vera`, and `.factory`, indexes only the Vera arm with `VERA_LOCAL=1` and installs the Droid skill there, pre-warms the Semble index in the Semble arm with one search, and writes that arm's `AGENTS.md`. It does not modify the source Flask checkout.

The command prints the run directory. Use that path for the question sweep:

```bash
python3 benchmarks/agent-bench/run.py --run ~/.cache/agent-bench/<timestamp>
```

For a smoke sweep, limit the arms to the first question:

```bash
python3 benchmarks/agent-bench/run.py --run ~/.cache/agent-bench/<timestamp> --questions 1
```

To parse or re-parse existing JSONL outputs without invoking an agent:

```bash
python3 benchmarks/agent-bench/run.py --analyze ~/.cache/agent-bench/<timestamp>
```

Analysis covers whatever arms exist in the run directory, so older two-arm runs still summarize cleanly. Pick the tested model and reasoning effort with `--model` and `--effort` (defaults: `claude-opus-5`, `medium`). Each model+effort pair writes its own transcripts (`qNN.<model>-<effort>.jsonl`) and `results.<model>-<effort>.json`, so several lanes can share one run directory. Grade a lane's answers with:

```bash
python3 benchmarks/agent-bench/judge.py ~/.cache/agent-bench/<timestamp> <model>-<effort>
```

Running the script with no mode performs setup, the full sweep, and analysis sequentially.

## Limitations

This is one model, one repository, and one set of 10 questions. The arms share provider, prompt, model, reasoning effort, and sequential execution, but provider timing and model variance still affect measurements. Tool-call count and token accounting depend on the stream schema and are not a measure of answer correctness. The experiment has no statistical power claims and should not be generalized beyond this workload.
