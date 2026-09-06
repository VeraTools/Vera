#!/usr/bin/env python3
"""Grade agent-bench answers against the answer key with agy opus, blind to arm."""
import json, re, subprocess, sys, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = HERE / "flask" / "answer-key.md"

def key_sections():
    text = KEY.read_text()
    parts = re.split(r"(?m)^##\s+Question\s+(\d+)\s*$", text)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        out[int(parts[i])] = parts[i + 1].strip()
    return out

def judge(question_text, key_text, answer, out_file):
    prompt = (
        "You are grading a coding agent's answer about the Flask codebase against a verified answer key. "
        "Be strict but fair: cite-based claims must match the key's verified behavior; use the rubric; "
        "do not penalize noted uncertainties. Judge ONLY this answer, you have no access to any other answer.\n\n"
        "=== QUESTION ===\n" + question_text + "\n\n"
        "=== VERIFIED ANSWER KEY AND RUBRIC ===\n" + key_text + "\n\n"
        "=== ANSWER TO GRADE ===\n" + answer + "\n\n"
        "Output exactly two lines:\nSCORE: <integer 0-10>\nJUSTIFICATION: <2-4 sentences citing what was right/wrong vs the key>"
    )
    proc = subprocess.run(
        ["agy", "--print", prompt, "--model", "claude-opus-4-6-thinking"],
        capture_output=True, text=True, timeout=1800,
    )
    out_file.write_text(proc.stdout + "\n=== STDERR ===\n" + proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"judge: agy exited {proc.returncode}; see {out_file}")
    m = re.search(r"(?m)^SCORE:\s*(\d+)\s*$", proc.stdout)
    score = int(m.group(1)) if m else -1
    if not 0 <= score <= 10:
        raise SystemExit(f"judge: no valid SCORE (0-10) in response; see {out_file}")
    return score

def main():
    run_dir = Path(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "claude-opus-5-medium"
    results_file = run_dir / f"results.{tag}.json"
    results = json.loads(results_file.read_text())
    keys = key_sections()
    questions = json.loads((run_dir / "setup.json").read_text())["questions"]
    qtext = {q["number"]: q["text"] for q in questions}
    judge_dir = run_dir / f"judge-{tag}"
    judge_dir.mkdir(exist_ok=True)
    # Grade in random order so position effects wash out; no arm labels shown to judge.
    # Arms come from the results file, so older 2-arm runs still grade fully.
    jobs = [(arm, q["number"]) for arm in results.get("summary", {}) for q in questions]
    random.Random(42).shuffle(jobs)
    for arm, qn in jobs:
        entry = results["questions"].get(f"q{qn:02d}", {}).get(arm)
        if not entry or not entry.get("answer"):
            print(f"skip {arm} q{qn}: no answer"); continue
        if entry.get("failed"):
            print(f"skip {arm} q{qn}: failed run"); continue
        out_file = judge_dir / f"{arm}-q{qn:02d}.txt"
        # Resume: a prior judge pass may have graded this cell (crash recovery,
        # or a re-run after fill-in cells refreshed the results file). Reuse the
        # recorded score instead of paying for the opus call again.
        if out_file.exists():
            m = re.search(r"(?m)^SCORE:\s*(\d+)\s*$", out_file.read_text())
            if m and 0 <= int(m.group(1)) <= 10:
                entry["score"] = int(m.group(1))
                print(f"resume {arm} q{qn:02d}: {entry['score']}", flush=True)
                continue
        score = judge(qtext[qn], keys[qn], entry["answer"], out_file)
        entry["score"] = score
        print(f"{arm} q{qn:02d}: {score}", flush=True)
        # Persist after every cell so a crash mid-run loses no graded scores.
        results_file.write_text(json.dumps(results, indent=2) + "\n")

if __name__ == "__main__":
    main()
