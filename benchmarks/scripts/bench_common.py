#!/usr/bin/env python3
"""Shared helpers for the benchmark scripts (run_baselines, run_vera_benchmarks,
run_final_benchmarks, run_local_binary_benchmarks).

Metric semantics (recall/mrr/nDCG matching rules) live here exactly once so the
scripts cannot drift apart. nDCG uses greedy best-relevance assignment with
each ground-truth entry consumed at most once.
"""

import json
import hashlib
import math
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TASKS_DIR = REPO_ROOT / "eval" / "tasks"


def load_tasks(
    task_ids: set[str] | None = None,
    categories: set[str] | None = None,
) -> list[dict]:
    """Load all benchmark tasks from eval/tasks/*.json."""
    tasks = []
    for task_file in sorted(TASKS_DIR.glob("*.json")):
        with open(task_file) as f:
            tasks.extend(json.load(f))
    task_ids = task_ids or set()
    categories = {category.replace("-", "_").lower() for category in (categories or set())}
    filtered = [
        task
        for task in tasks
        if (not task_ids or task["id"] in task_ids)
        and (not categories or task["category"] in categories)
    ]
    unknown_ids = sorted(task_ids - {task["id"] for task in tasks})
    if unknown_ids:
        raise ValueError(f"unknown task ID(s): {', '.join(unknown_ids)}")
    if not filtered and (task_ids or categories):
        raise ValueError("no benchmark tasks match the requested filters")
    return filtered


def task_set_identity(tasks: list[dict]) -> dict[str, str | int]:
    """Hash sorted task IDs without including ground truth or task content."""
    task_ids = sorted(task["id"] for task in tasks)
    serialized_ids = ("\n".join(task_ids) + "\n") if task_ids else ""
    digest = hashlib.sha256(serialized_ids.encode()).hexdigest()
    return {"count": len(task_ids), "task_ids_sha256": digest}


def environment_summary(env: dict[str, str]) -> dict[str, str]:
    """Return the benchmark environment with credential values redacted."""
    summary = {}
    for key, value in sorted({**os.environ, **env}.items()):
        if key.startswith(("EMBEDDING_", "RERANKER_", "VERA_")):
            if any(secret in key for secret in ("KEY", "TOKEN", "SECRET")):
                summary[key] = "<redacted>"
            else:
                summary[key] = value
    return summary


def load_secrets() -> dict[str, str]:
    """Load API credentials from secrets.env."""
    secrets_path = REPO_ROOT / "secrets.env"
    env = {}
    if secrets_path.exists():
        with open(secrets_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    env[key.strip()] = value.strip().strip("'\"")
    return env


def binary_version(binary: Path) -> str:
    proc = subprocess.run(
        [str(binary), "--version"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    return proc.stdout.strip() or proc.stderr.strip() or "unknown"


def git_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )
    return proc.stdout.strip()


def is_match(result: dict, gt: dict) -> bool:
    """Check if a retrieval result matches a ground truth entry (file + line overlap)."""
    return (
        result.get("file_path") == gt["file_path"]
        and result.get("line_start", 0) <= gt["line_end"]
        and result.get("line_end", 0) >= gt["line_start"]
    )


def recall_at_k(results: list[dict], ground_truth: list[dict], k: int) -> float:
    if not ground_truth:
        return 0.0
    top_k = results[:k]
    found = sum(1 for gt in ground_truth if any(is_match(r, gt) for r in top_k))
    return found / len(ground_truth)


def precision_at_k(results: list[dict], ground_truth: list[dict], k: int) -> float:
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for r in top_k if any(is_match(r, gt) for gt in ground_truth))
    return relevant / len(top_k)


def mrr(results: list[dict], ground_truth: list[dict]) -> float:
    for i, result in enumerate(results):
        if any(is_match(result, gt) for gt in ground_truth):
            return 1.0 / (i + 1)
    return 0.0


def matched_relevances(results: list[dict], ground_truth: list[dict], k: int) -> list[int]:
    """Assign each ranked result to at most one unmatched ground-truth entry."""
    used = [False] * len(ground_truth)
    relevances = []

    for result in results[:k]:
        best_idx = None
        best_rel = 0
        for idx, gt in enumerate(ground_truth):
            if used[idx] or not is_match(result, gt):
                continue
            rel = gt.get("relevance", 1)
            if rel > best_rel:
                best_idx = idx
                best_rel = rel

        if best_idx is not None:
            used[best_idx] = True
            relevances.append(best_rel)
        else:
            relevances.append(0)

    return relevances


def ndcg_at_k(results: list[dict], ground_truth: list[dict], k: int = 10) -> float:
    dcg = 0.0
    for i, relevance in enumerate(matched_relevances(results, ground_truth, k)):
        dcg += relevance / math.log2(i + 2.0)

    ideal_rels = sorted([gt.get("relevance", 1) for gt in ground_truth], reverse=True)[:k]
    ideal_dcg = sum(rel / math.log2(i + 2.0) for i, rel in enumerate(ideal_rels))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_task_metrics(results: list[dict], ground_truth: list[dict]) -> dict:
    """Compute all retrieval metrics for a single task."""
    return {
        "recall_at_1": recall_at_k(results, ground_truth, 1),
        "recall_at_5": recall_at_k(results, ground_truth, 5),
        "recall_at_10": recall_at_k(results, ground_truth, 10),
        "mrr": mrr(results, ground_truth),
        "ndcg": ndcg_at_k(results, ground_truth, 10),
        "precision_at_3": precision_at_k(results, ground_truth, 3),
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = p / 100.0 * (n - 1)
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    frac = rank - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac
