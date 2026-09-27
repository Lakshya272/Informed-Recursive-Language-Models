#!/usr/bin/env python3
"""Phase B Official Runner: LongBench-v2 CodeQA Full 50 Tasks with Contract Architecture.

Evaluates RLM with M1/M2 (Grounded Evidence Preference, dynamic depth) on all 50 repository comprehension tasks.
Project: rlm3-509414
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.eval.codeqa import CodeQATask, load_tasks, score_answer
from rlm.core.rlm import RLM
from rlm.guardrails.m1_classifier import classify_task_rules
from rlm.logger.rlm_logger import RLMLogger
from rlm.utils.prompts import get_contract_prompt

GEMINI_HARDENED_SUFFIX = """
Output-format constraint:
Respond only with plain text. If continuing the RLM, emit exactly one fenced
```repl``` Python block. Do not emit JSON, structured calls, API actions, or
special tool-call objects. The names llm_query and rlm_query are ordinary
Python functions used only inside the REPL.

Execution constraint:
You must not attempt to make external network requests or search the host filesystem for cached datasets. All calculations and text processing must be performed strictly in-memory on the provided `context` variable and through `llm_query` / `rlm_query`.
""".strip()

OPENAI_HARDENED_SUFFIX = """
Output-format constraint:
Respond with plain text and emit exactly one fenced ```repl``` Python block per turn. Do not respond with purely prose or plans without code.

Execution constraint:
The context dataset is already pre-loaded into the Python variable `context` in the REPL environment. It is not passed directly in chat text. Always run code in ```repl``` to inspect and manipulate it; do not claim the data is missing or inaccessible.
""".strip()


def main():
    parser = argparse.ArgumentParser(description="Phase B: CodeQA Full 50 Tasks Contract Evaluation")
    parser.add_argument("--project-id", default=os.getenv("VERTEX_PROJECT_ID", None), help="Google Cloud Vertex Project ID")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gemini-1.5-pro"), help="Model name")
    parser.add_argument("--backend", default="vertex", choices=["vertex", "openai"], help="Backend provider")
    parser.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible endpoint")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--depth", type=int, default=None, help="Explicitly force recursion depth (1 or 2)")
    parser.add_argument("--disable-m1-m2", action="store_true", help="Run vanilla baseline without M1/M2")
    parser.add_argument("--disable-m1", action="store_true", help="Disable M1 dynamic depth classifier")
    parser.add_argument("--disable-m2", action="store_true", help="Disable M2 contract verification")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "phase_b_codeqa"))
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks to run")
    parser.add_argument("--task-indices", type=str, default=None, help="Comma-separated indices or task IDs to run")
    args = parser.parse_args()

    enable_m1 = not (args.disable_m1 or args.disable_m1_m2)
    enable_m2 = not (args.disable_m2 or args.disable_m1_m2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = out_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(seed=42)
    if args.task_indices:
        target_indices = {x.strip() for x in args.task_indices.split(",") if x.strip()}
        tasks = [t for idx, t in enumerate(tasks, 1) if str(idx) in target_indices or t.task_id in target_indices]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    print("=" * 80)
    print(f"PHASE B: LONGBENCH-V2 CODEQA FULL EVALUATION (N={len(tasks)})")
    print(f"Project: {args.project_id} | Model: {args.model}")
    print(f"Output: {out_dir}")
    print("=" * 80)

    # Resume capability
    completed_ids = set()
    results_path = out_dir / "task_results.jsonl"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed_ids.add(rec.get("task_id"))
                    except Exception:
                        pass

    correct_count = 0
    total_evaluated = 0

    for idx, task in enumerate(tasks, 1):
        if task.task_id in completed_ids:
            print(f"[{idx}/{len(tasks)}] Task {task.task_id} already evaluated. Skipping.")
            continue

        print(f"\n[{idx}/{len(tasks)}] Task: {task.task_id} | Question: {task.question[:80]}...")
        task_dir = trajectories_dir / f"{idx:03d}_{task.task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        log_file = task_dir / f"task_{idx:03d}_{task.task_id}.jsonl"
        logger = RLMLogger(log_dir=task_dir, file_name=log_file.name)

        m1_cls = classify_task_rules(task.question, {"context_type": "codeqa"})
        if args.depth is not None:
            depth = args.depth
        elif enable_m1:
            depth = m1_cls.suggested_depth
        else:
            depth = 1
        sys_prompt = get_contract_prompt(depth)
        if args.backend == "vertex":
            sys_prompt += "\n\n" + GEMINI_HARDENED_SUFFIX
        elif args.backend == "openai":
            sys_prompt += "\n\n" + OPENAI_HARDENED_SUFFIX

        if args.backend == "openai":
            b_kwargs = {
                "model_name": args.model,
                "api_key": args.api_key,
                "base_url": args.base_url,
                "sampling_args": {"reasoning_effort": "medium"},
            }
            sub_kwargs = {
                "model_name": args.model,
                "api_key": args.api_key,
                "base_url": args.base_url,
                "sampling_args": {"reasoning_effort": "low"},
            }
        else:
            b_kwargs = {
                "model_name": args.model,
                "project_id": args.project_id,
                "thinking_level": "medium",
                "malformed_retries": 2,
                "malformed_retry_delay": 0.5,
            }
            sub_kwargs = {
                "model_name": args.model,
                "project_id": args.project_id,
                "thinking_level": "low",
                "malformed_retries": 2,
                "malformed_retry_delay": 0.5,
            }
            if args.api_key:
                b_kwargs["api_key"] = args.api_key
                sub_kwargs["api_key"] = args.api_key

        start_t = time.perf_counter()
        rlm = RLM(
            backend=args.backend,
            backend_kwargs=b_kwargs,
            other_backends=[args.backend],
            other_backend_kwargs=[sub_kwargs],
            environment="local",
            max_depth=depth,
            max_iterations=args.max_iterations,
            enable_m1=enable_m1,
            enable_m2=enable_m2,
            enable_m1_m2=(enable_m1 and enable_m2),
            m1_mode="rules",
            m1_m2_context_metadata={"context_type": "codeqa"} if (enable_m1 or enable_m2) else {},
            paper_mode=True,
            custom_system_prompt=sys_prompt,
            logger=logger,
            verbose=False,
        )

        max_retries = 5
        raw_response = ""
        m1_m2_meta = {}
        for attempt in range(1, max_retries + 1):
            try:
                comp = rlm.completion(task.context, root_prompt=task.prompt)
                raw_response = comp.response.strip() if comp and comp.response else ""
                if raw_response:
                    m1_m2_meta = comp.metadata.get("m1_m2", {}) if comp.metadata else {}
                    break
                else:
                    print(f"  [Task {task.task_id} Attempt {attempt}] Empty response received. Retrying...")
                    time.sleep(3 * attempt)
            except Exception as e:
                print(f"  [Task {task.task_id} Attempt {attempt}] Execution/socket error: {e}. Retrying in {5 * attempt}s...")
                time.sleep(5 * attempt)
        exec_time = time.perf_counter() - start_t

        is_correct = bool(score_answer(task, raw_response) > 0.5)
        total_evaluated += 1
        if is_correct:
            correct_count += 1

        print(f"  Prediction: '{raw_response[:80]}' | Gold: '{task.gold_index}' | Correct: {is_correct}")
        print(f"  M1 Category: {m1_cls.category} | M2 Fired: {m1_m2_meta.get('m2_fired')} | Depth: {depth}")

        record = {
            "task_id": task.task_id,
            "domain": "codeqa",
            "gold_answer": task.gold_index,
            "is_correct": is_correct,
            "raw_response": raw_response[:300],
            "execution_time_s": exec_time,
            "dynamic_depth": depth,
            "m1_category": m1_cls.category,
            "m2_fired": bool(m1_m2_meta.get("m2_fired", False)),
            "m2_changed_answer": bool(m1_m2_meta.get("m2_changed_answer", False)),
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    acc = correct_count / total_evaluated if total_evaluated > 0 else 0.0
    summary = {
        "total_evaluated": total_evaluated,
        "total_correct": correct_count,
        "accuracy": acc,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE B: CODEQA COMPLETED")
    print(f"Final Accuracy: {acc:.2%} ({correct_count}/{total_evaluated})")
    print(f"Summary written to: {out_dir / 'summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
