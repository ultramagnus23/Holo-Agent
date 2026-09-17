"""Run the FULL Set A (51 questions) and Set B (35 questions) grading
suites against a real orchestrator + local Ollama model + real HoloForge
compute -- not the samples graded so far.

Unlike holoagent.eval.set_a_grounded.run_set_a / set_b_abstention.run_set_b
(which return a plain list, all-or-nothing), this script writes each
result to a JSONL file the moment it's graded, so a full 86-question run
that takes hours on constrained hardware doesn't lose everything if the
process is interrupted partway through. Re-running skips question IDs
already present in the output file, so an interrupted run resumes rather
than restarting.

Usage:
    python scripts/run_full_grading.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from holoagent.eval.set_a_grounded import load_questions as load_set_a
from holoagent.eval.set_a_grounded import grade_question as grade_set_a_question
from holoagent.eval.set_b_abstention import load_questions as load_set_b
from holoagent.eval.set_b_abstention import grade_question as grade_set_b_question
from holoagent.orchestrator.orchestrator import Orchestrator
from holoagent.orchestrator.ollama_backend import OllamaBackend
from holoagent.provenance.store import ProvenanceStore


def _already_graded_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.add(json.loads(line)["question_id"])
    return ids


def _append_result(path: Path, result) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def run(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_a_out = out_dir / "set_a_results.jsonl"
    set_b_out = out_dir / "set_b_results.jsonl"
    log_path = out_dir / "progress.log"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    store = ProvenanceStore(out_dir / "provenance.sqlite3")
    backend = OllamaBackend()

    set_a_questions = load_set_a()
    set_b_questions = load_set_b()
    already_a = _already_graded_ids(set_a_out)
    already_b = _already_graded_ids(set_b_out)

    log(
        f"Starting full grading run: Set A {len(set_a_questions)} questions "
        f"({len(already_a)} already graded, resuming), Set B {len(set_b_questions)} questions "
        f"({len(already_b)} already graded, resuming)"
    )

    for i, q in enumerate(set_a_questions, 1):
        if q.id in already_a:
            continue
        t0 = time.time()
        orchestrator = Orchestrator(provenance_store=store, model_backend=backend, max_steps=6)
        result = grade_set_a_question(orchestrator, store, q)
        _append_result(set_a_out, result)
        elapsed = time.time() - t0
        status = "ERROR" if result.error else ("OK" if result.expected_tools_satisfied else "TOOLS_NOT_SATISFIED")
        log(f"Set A [{i}/{len(set_a_questions)}] {q.id} -- {status} ({elapsed:.1f}s)")

    for i, q in enumerate(set_b_questions, 1):
        if q.id in already_b:
            continue
        t0 = time.time()
        orchestrator = Orchestrator(provenance_store=store, model_backend=backend, max_steps=6)
        result = grade_set_b_question(orchestrator, store, q)
        _append_result(set_b_out, result)
        elapsed = time.time() - t0
        status = "ERROR" if result.error else ("ABSTAINED_CLEAN" if result.correctly_abstained else "NOT_CLEAN")
        log(f"Set B [{i}/{len(set_b_questions)}] {q.id} -- {status} ({elapsed:.1f}s)")

    log("Full grading run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent.parent / "grading_results")
    args = parser.parse_args()
    run(args.out_dir)
