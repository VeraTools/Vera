#!/usr/bin/env python3
"""Run a small agent-level benchmark comparing tool arms for Vera."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, NoReturn


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = REPO_ROOT / ".bench" / "semble-repos" / "flask"
QUESTIONS_FILE = Path(__file__).resolve().parent / "flask" / "questions.md"
VERA_BINARY = REPO_ROOT / "target" / "release" / "vera"
RUNS_ROOT = Path(os.environ.get("AGENT_BENCH_RUNS", Path.home() / ".cache" / "agent-bench"))
ARMS = ("with-vera", "with-semble", "control")
# Shims make a binary exit 127 so an arm cannot reach another arm's tool.
ARM_SHIMS = {
    "with-vera": ("semble",),
    "with-semble": ("vera",),
    "control": ("vera", "semble"),
}
SEMBLE_PREWARM_QUERY = "authentication"
SEMBLE_PREWARM_TIMEOUT_S = 600
PROMPT_HEADER = """Answer the question about the codebase in the current directory.

This is a read-only task: do not modify files and do not install anything.
Cite evidence as path:line. Answer every subquestion. Finish with a per-subquestion confidence table.

"""
QUESTION_START = re.compile(
    r"^#{1,6}\s+Question\s+(\d+)\s*$",
    re.IGNORECASE,
)
TOKEN_KEYS = {
    "input": "tokens_in",
    "input_tokens": "tokens_in",
    "prompt_tokens": "tokens_in",
    "output": "tokens_out",
    "output_tokens": "tokens_out",
    "completion_tokens": "tokens_out",
    "cache_read": "cache_read",
    "cache_read_input_tokens": "cache_read",
    "cache_creation": "cache_creation",
    "cache_creation_input_tokens": "cache_creation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--setup-only",
        action="store_true",
        help="Create all arms and index the Vera arm, without agent runs",
    )
    mode.add_argument("--run", type=Path, metavar="RUN_DIR", help="Run agents in an existing run")
    mode.add_argument(
        "--analyze",
        type=Path,
        metavar="RUN_DIR",
        help="Parse existing JSONL outputs and write results.json",
    )
    parser.add_argument(
        "--questions",
        type=int,
        metavar="N",
        help="Use only the first N questions (1-10)",
    )
    parser.add_argument("--model", default="claude-opus-5", help="droid model ID")
    parser.add_argument(
        "--effort", default="medium", help="droid reasoning effort level"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run cells that already completed successfully (default: skip them)",
    )
    return parser.parse_args()


def fail(message: str) -> NoReturn:
    raise SystemExit(f"agent-bench: error: {message}")


def lane_slug(model: str, effort: str) -> str:
    """Filesystem-safe tag for a model/effort lane (model IDs may contain '/')."""
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{model}-{effort}")
    digest = hashlib.sha256(f"{model}\0{effort}".encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{digest}"


def load_questions(limit: int | None) -> list[dict[str, Any]]:
    if not QUESTIONS_FILE.is_file():
        fail(
            f"questions file is missing: {QUESTIONS_FILE}. "
            "Create it with exactly 10 numbered questions."
        )
    sections: list[tuple[int, str, list[str]]] = []
    current: tuple[int, str, list[str]] | None = None
    for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines():
        match = QUESTION_START.match(line)
        if match:
            if current is not None:
                sections.append(current)
            current = (int(match.group(1)), "", [])
        elif current is not None:
            current[2].append(line)
    if current is not None:
        sections.append(current)

    numbers = [number for number, _, _ in sections]
    if numbers != list(range(1, 11)):
        fail(
            f"{QUESTIONS_FILE} must contain exactly the numbered questions 1 through 10 "
            f"at top level; found {numbers or 'none'}"
        )
    questions = []
    for number, title, body in sections:
        text = "\n".join([title, *body]).strip()
        if not text:
            fail(f"question {number} in {QUESTIONS_FILE} is empty")
        questions.append({"number": number, "text": text})
    if limit is not None:
        if not 1 <= limit <= len(questions):
            fail(f"--questions must be between 1 and {len(questions)}")
        questions = questions[:limit]
    return questions


def ensure_binary() -> Path:
    if VERA_BINARY.is_file() and os.access(VERA_BINARY, os.X_OK):
        return VERA_BINARY
    print(f"Building {VERA_BINARY}...", file=sys.stderr)
    run_command(
        ["cargo", "build", "--release", "--bin", "vera"],
        cwd=REPO_ROOT,
        timeout=1800,
    )
    if not VERA_BINARY.is_file() or not os.access(VERA_BINARY, os.X_OK):
        fail(f"release binary was not produced: {VERA_BINARY}")
    return VERA_BINARY


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int,
    stdout: Any = subprocess.PIPE,
    stderr: Any = subprocess.PIPE,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        fail(f"required executable is unavailable: {exc.filename}")
    except subprocess.TimeoutExpired:
        fail(f"command timed out after {timeout}s: {' '.join(command)}")
    if result.returncode != 0:
        stderr_text = result.stderr if isinstance(result.stderr, str) else ""
        fail(f"command failed ({result.returncode}): {' '.join(command)}\n{stderr_text[-1000:]}")
    return result


def make_run_dir() -> Path:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = RUNS_ROOT / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = RUNS_ROOT / f"{stamp}-{suffix}"
        suffix += 1
    for arm in ARMS:
        (run_dir / arm / "repo").mkdir(parents=True)
        (run_dir / arm / "prompts").mkdir()
    return run_dir


def copy_repo(destination: Path) -> None:
    if not SOURCE_REPO.is_dir():
        fail(f"source repository is missing: {SOURCE_REPO}")
    if shutil.which("rsync") is None:
        fail("rsync is required to create benchmark copies")
    destination.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "rsync",
            "-a",
            "--delete",
            "--exclude=.git",
            "--exclude=.vera",
            "--exclude=.factory",
            "--exclude=answer-key.md",
            f"{SOURCE_REPO}/",
            f"{destination}/",
        ],
        cwd=REPO_ROOT,
        timeout=300,
    )


FACTORY_HOME_FILES = (
    "settings.json",
    "auth.v2.file",
    "auth.v2.key",
    "host.json",
    "last-startup-version",
)


def build_factory_home(run_dir: Path) -> Path:
    """Copy the minimum droid config into a run-local home.

    The real `~/.factory` carries global AGENTS.md instructions, personal
    skills, and custom droids. Any of those would reach into every arm and
    change agent behavior in ways that have nothing to do with the tool under
    test, so cells run against a home that holds only credentials and model
    settings.
    """
    home = run_dir / "factory-home"
    config = home / ".factory"
    config.mkdir(parents=True, exist_ok=True)
    source = Path.home() / ".factory"
    for name in FACTORY_HOME_FILES:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, config / name)
    return home


def environment_for(arm: str, run_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["FACTORY_HOME_OVERRIDE"] = str(build_factory_home(run_dir))
    path_parts: list[str] = []
    if arm == "with-vera":
        env["VERA_LOCAL"] = "1"
        path_parts.append(str(VERA_BINARY.parent))
    else:
        env.pop("VERA_LOCAL", None)
    if ARM_SHIMS[arm]:
        shim_dir = run_dir / arm / "bin"
        write_shims(shim_dir, ARM_SHIMS[arm])
        path_parts.insert(0, str(shim_dir))
    if arm == "with-semble":
        env["SEMBLE_CACHE_LOCATION"] = str(run_dir / arm / "cache")
    env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
    return env


def write_shims(shim_dir: Path, binaries: Iterable[str]) -> None:
    """Shim executables to exit 127 so an arm cannot use another arm's tool."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    for name in binaries:
        shim = shim_dir / name
        shim.write_text(
            f"#!/bin/sh\nprintf '%s\\n' '{name}: not available in this environment' >&2\nexit 127\n",
            encoding="ascii",
        )
        shim.chmod(0o755)


SEMBLE_AGENTS_MD = """\
## Semble Code Search

Use the `semble` CLI to find relevant code before reading files. Queries take
natural language or code identifiers.

```bash
semble search "how does the app factory work?" .
semble search "database host port" . --top-k 10
semble search "deployment guide" . --content docs
semble search "database host port" . --content config
semble search "rate limiting" . --content all
semble find-related flask/app.py 10 .
```

Results give file paths, line numbers, and snippets; open the file for full
context. The index builds on first search and is cached automatically.

rg/grep remain available for exact strings, symbol names, and regex search.
"""


def write_semble_agents_md(repo: Path) -> str:
    """Write the with-semble arm's AGENTS.md and return its provenance label.

    Uses a hand-written CLI-only snippet: this arm has no MCP server, so
    Semble's installer text (which leads with `mcp__semble__*` tool calls)
    would send the agent after tools that do not exist in this environment.
    """
    (repo / "AGENTS.md").write_text(SEMBLE_AGENTS_MD, encoding="utf-8")
    return "hand-written CLI snippet (SEMBLE_AGENTS_MD)"


def setup_run(questions: list[dict[str, Any]]) -> Path:
    binary = ensure_binary()
    run_dir = make_run_dir()
    with_repo = run_dir / "with-vera" / "repo"
    sem_repo = run_dir / "with-semble" / "repo"
    control_repo = run_dir / "control" / "repo"
    copy_repo(with_repo)
    copy_repo(sem_repo)
    copy_repo(control_repo)
    shutil.rmtree(sem_repo / ".vera", ignore_errors=True)
    shutil.rmtree(sem_repo / ".factory", ignore_errors=True)
    shutil.rmtree(control_repo / ".vera", ignore_errors=True)
    shutil.rmtree(control_repo / ".factory", ignore_errors=True)
    for arm in ARMS:
        if ARM_SHIMS[arm]:
            write_shims(run_dir / arm / "bin", ARM_SHIMS[arm])

    with_env = environment_for("with-vera", run_dir)
    index_log = run_dir / "with-vera" / "index.log"
    install_log = run_dir / "with-vera" / "agent-install.log"
    with index_log.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            ["vera", "index", "."],
            cwd=with_repo,
            env=with_env,
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
    if result.returncode != 0:
        fail(f"Vera indexing failed; see {index_log}")

    with install_log.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            ["vera", "agent", "install", "--client", "droid", "--scope", "project"],
            cwd=with_repo,
            env=with_env,
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    # `vera agent install` exits non-zero in this sandbox after a successful
    # install (a later connectivity check fails without a configured endpoint),
    # so judge success by the installed skill files, not the exit code.
    skill_dir = with_repo / ".factory" / "skills" / "vera"
    if not skill_dir.is_dir():
        fail(f"Vera agent installation failed; see {install_log}")
    if (control_repo / ".vera").exists() or (control_repo / ".factory").exists():
        fail("control arm contains Vera artifacts")
    if (sem_repo / ".vera").exists() or (sem_repo / ".factory").exists():
        fail("with-semble arm contains Vera artifacts")

    sem_env = environment_for("with-semble", run_dir)
    sem_cache = Path(sem_env["SEMBLE_CACHE_LOCATION"])
    sem_cache.mkdir(parents=True, exist_ok=True)
    # Pre-warm the Semble index so question runs measure search, not indexing.
    sem_index_log = run_dir / "with-semble" / "index.log"
    with sem_index_log.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            ["semble", "search", SEMBLE_PREWARM_QUERY, "."],
            cwd=sem_repo,
            env=sem_env,
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=SEMBLE_PREWARM_TIMEOUT_S,
            check=False,
        )
    if result.returncode != 0:
        fail(f"Semble index pre-warm failed; see {sem_index_log}")
    agents_md_source = write_semble_agents_md(sem_repo)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(SOURCE_REPO),
        "vera_binary": str(binary),
        "questions_file": str(QUESTIONS_FILE),
        "questions": questions,
        "arms": {
            "with-vera": {
                "repo": str(with_repo),
                "path_prefix": str(binary.parent),
                "shims": list(ARM_SHIMS["with-vera"]),
            },
            "with-semble": {
                "repo": str(sem_repo),
                "shims": list(ARM_SHIMS["with-semble"]),
                "cache": str(sem_cache),
                "agents_md_source": agents_md_source,
            },
            "control": {
                "repo": str(control_repo),
                "shims": list(ARM_SHIMS["control"]),
            },
        },
    }
    (run_dir / "setup.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Setup complete: {run_dir}")
    return run_dir


def write_prompts(run_dir: Path, questions: list[dict[str, Any]]) -> None:
    for arm in ARMS:
        prompt_dir = run_dir / arm / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for question in questions:
            path = prompt_dir / f"q{question['number']:02d}.md"
            path.write_text(PROMPT_HEADER + question["text"] + "\n", encoding="utf-8")


def run_question(
    run_dir: Path, arm: str, question: dict[str, Any], model: str, effort: str, force: bool
) -> None:
    arm_dir = run_dir / arm
    repo_dir = arm_dir / "repo"
    prompt = arm_dir / "prompts" / f"q{question['number']:02d}.md"
    slug = lane_slug(model, effort)
    suffix = f"q{question['number']:02d}.{slug}.jsonl"
    output_path = arm_dir / suffix
    stderr_path = arm_dir / f"q{question['number']:02d}.{slug}.stderr.log"
    meta_path = arm_dir / f"q{question['number']:02d}.{slug}.run.json"
    if not force and output_path.is_file() and meta_path.is_file():
        try:
            if json.loads(meta_path.read_text(encoding="utf-8")).get("returncode") == 0:
                print(f"  {arm} q{question['number']:02d}: skipped (already done)")
                return
        except json.JSONDecodeError:
            pass
    env = environment_for(arm, run_dir)
    for attempt in range(2):
        returncode, timed_out, wall_s = execute_cell(
            repo_dir, prompt, output_path, stderr_path, env, model, effort
        )
        if returncode == 0 or timed_out or not transient_provider_error(output_path):
            break
        print(f"  {arm} q{question['number']:02d}: provider error, retrying")
        time.sleep(60 * (attempt + 1))
    metadata = {"returncode": returncode, "wall_s": wall_s}
    if timed_out:
        metadata.update({"failed": True, "timed_out": True, "timeout_s": TIMEOUT_S})
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    status = "timeout" if timed_out else ("ok" if returncode == 0 else f"failed ({returncode})")
    print(f"  {arm} q{question['number']:02d}: {status}, {wall_s:.1f}s")


TIMEOUT_S = 7200


def transient_provider_error(transcript: Path) -> bool:
    """True when the run died on an upstream error worth retrying.

    A model gateway returning 429/503 says nothing about the arm under test,
    so retrying keeps a provider hiccup from leaving a hole in the data. A
    wrong model name or an exhausted quota is not retried: it would fail the
    same way every time.
    """
    try:
        text = transcript.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(
        marker in text
        for marker in ("service_overloaded", "temporarily overloaded", " 429 ", " 503 ")
    )


def execute_cell(
    repo_dir: Path,
    prompt: Path,
    output_path: Path,
    stderr_path: Path,
    env: dict[str, str],
    model: str,
    effort: str,
) -> tuple[int | None, bool, float]:
    start = time.monotonic()
    timeout_s = TIMEOUT_S
    returncode: int | None = None
    timed_out = False
    try:
        with output_path.open("w", encoding="utf-8") as output, stderr_path.open(
            "w", encoding="utf-8"
        ) as errors:
            result = subprocess.run(
                [
                    "droid",
                    "exec",
                    "--cwd",
                    str(repo_dir),
                    "--disable-builtin-skills",
                    "--auto",
                    "medium",
                    "-o",
                    "stream-json",
                    "-m",
                    model,
                    "-r",
                    effort,
                    "-f",
                    str(prompt),
                ],
                cwd=repo_dir,
                env=env,
                text=True,
                stdout=output,
                stderr=errors,
                timeout=timeout_s,
                check=False,
            )
            returncode = result.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    return returncode, timed_out, time.monotonic() - start


def run_agents(
    run_dir: Path, questions: list[dict[str, Any]], model: str, effort: str, force: bool
) -> None:
    for arm in ARMS:
        if not (run_dir / arm / "repo").is_dir():
            fail(f"run directory is missing {arm}/repo: {run_dir}")
    write_prompts(run_dir, questions)
    # Rotate the starting arm by question number so each arm leads roughly
    # equally often and no arm is systematically disadvantaged by position.
    arms_by_question = [
        ARMS[(question["number"] - 1) % len(ARMS) :]
        + ARMS[: (question["number"] - 1) % len(ARMS)]
        for question in questions
    ]
    print(f"Running {len(questions)} questions in {run_dir} with {model} ({effort})")
    for question, arm_order in zip(questions, arms_by_question):
        for arm in arm_order:
            run_question(run_dir, arm, question, model, effort, force)


def json_objects(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def first_nested(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
        for nested_value in value.values():
            found = first_nested(nested_value, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested_value in value:
            found = first_nested(nested_value, keys)
            if found is not None:
                return found
    return None


def event_type(event: dict[str, Any]) -> str:
    value = event.get("type")
    return value if isinstance(value, str) else ""


def tool_name(event: dict[str, Any]) -> str | None:
    event_kind = event_type(event).lower()
    if event_kind not in {"tool_call", "function_call", "tool_use", "function_use"}:
        return None
    direct = first_nested(event, {"tool_name", "toolName", "name"})
    return direct if isinstance(direct, str) and direct else event_kind


def add_usage(event: dict[str, Any], totals: Counter[str]) -> None:
    usage = first_nested(event, {"usage"})
    if not isinstance(usage, dict):
        return
    for key, total_key in TOKEN_KEYS.items():
        value = usage.get(key)
        if isinstance(value, (int, float)):
            totals[total_key] += value


def completion_text(event: dict[str, Any]) -> str:
    direct = event.get("finalText")
    if isinstance(direct, str):
        return direct
    direct = event.get("final_text")
    if isinstance(direct, str):
        return direct
    for key in ("text", "content", "output_text"):
        value = event.get(key)
        if isinstance(value, str):
            return value
    return ""


def parse_jsonl(path: Path, run_meta: Path | None = None) -> dict[str, Any]:
    calls: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    answer = ""
    duration_ms: float | None = None
    event_count = 0
    timed_out = False
    for event in json_objects(path):
        event_count += 1
        name = tool_name(event)
        if name is not None:
            calls[name] += 1
        if event_type(event) == "completion":
            add_usage(event, totals)
            duration = event.get("durationMs")
            if isinstance(duration, (int, float)):
                duration_ms = float(duration)
            text = completion_text(event)
            if text:
                answer = text
        final_text = event.get("finalText")
        if isinstance(final_text, str) and final_text:
            answer = final_text

    wall_s: float | None = None
    returncode: int | None = None
    if run_meta is not None and run_meta.is_file():
        metadata = json.loads(run_meta.read_text(encoding="utf-8"))
        wall_s = metadata.get("wall_s")
        returncode = metadata.get("returncode")
        timed_out = metadata.get("timed_out") is True
    return {
        "tool_calls": dict(sorted(calls.items())),
        "tokens_in": totals["tokens_in"],
        "tokens_out": totals["tokens_out"],
        "cache_read": totals["cache_read"],
        "cache_creation": totals["cache_creation"],
        "wall_s": wall_s,
        "duration_ms": duration_ms,
        "answer": answer,
        "event_count": event_count,
        **({"timed_out": True} if timed_out else {}),
        **({"returncode": returncode} if returncode is not None else {}),
    }


def analyze_run(
    run_dir: Path, questions: list[dict[str, Any]], model: str, effort: str
) -> dict[str, Any]:
    if not run_dir.is_dir():
        fail(f"run directory does not exist: {run_dir}")
    # Arms missing from an older run dir are simply absent from its summary.
    present_arms = [arm for arm in ARMS if (run_dir / arm / "repo").is_dir()]
    if not present_arms:
        fail(f"run directory contains no arm repositories: {run_dir}")
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "model": model,
        "effort": effort,
        "questions": {},
        "summary": {},
    }
    slug = lane_slug(model, effort)
    for question in questions:
        number = question["number"]
        result["questions"][f"q{number:02d}"] = {}
        for arm in present_arms:
            arm_dir = run_dir / arm
            parsed = parse_jsonl(
                arm_dir / f"q{number:02d}.{slug}.jsonl",
                arm_dir / f"q{number:02d}.{slug}.run.json",
            )
            # A nonzero droid exit means the cell is not a valid measurement;
            # mark it so aggregates and judging exclude it.
            if (
                parsed["event_count"] == 0
                or parsed.get("returncode") not in (None, 0)
                or parsed.get("timed_out")
            ):
                parsed["failed"] = True
            result["questions"][f"q{number:02d}"][arm] = parsed

    for arm in present_arms:
        rows = [result["questions"][f"q{q['number']:02d}"][arm] for q in questions]
        failed = sum(1 for row in rows if row.get("failed"))
        rows = [row for row in rows if not row.get("failed")]
        result["summary"][arm] = {
            "questions": len(rows),
            "failed": failed,
            "tool_calls": sum((Counter(row["tool_calls"]) for row in rows), Counter()),
            "tokens_in": sum(row["tokens_in"] for row in rows),
            "tokens_out": sum(row["tokens_out"] for row in rows),
            "cache_read": sum(row["cache_read"] for row in rows),
            "cache_creation": sum(row["cache_creation"] for row in rows),
            "wall_s_total": sum(row["wall_s"] or 0.0 for row in rows),
            "duration_ms_total": sum(row["duration_ms"] or 0.0 for row in rows),
        }
    result_path = run_dir / f"results.{slug}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print_comparison(result)
    print(f"Wrote {result_path}")
    return result


def print_comparison(result: dict[str, Any]) -> None:
    print("\nArm comparison")
    print("arm          questions  failed  tool calls  tokens in  tokens out  wall s  duration ms")
    for arm, summary in result["summary"].items():
        tool_count = sum(summary["tool_calls"].values())
        print(
            f"{arm:<12} {summary['questions']:>9}  {summary['failed']:>6}  {tool_count:>10}  "
            f"{summary['tokens_in']:>9}  {summary['tokens_out']:>10}  "
            f"{summary['wall_s_total']:>6.1f}  {summary['duration_ms_total']:>12.0f}"
        )


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions)
    if args.analyze is not None:
        analyze_run(args.analyze.resolve(), questions, args.model, args.effort)
        return
    if args.run is not None:
        run_agents(args.run.resolve(), questions, args.model, args.effort, args.force)
        analyze_run(args.run.resolve(), questions, args.model, args.effort)
        return
    run_dir = setup_run(questions)
    if args.setup_only:
        return
    run_agents(run_dir, questions, args.model, args.effort, args.force)
    analyze_run(run_dir, questions, args.model, args.effort)


if __name__ == "__main__":
    main()
