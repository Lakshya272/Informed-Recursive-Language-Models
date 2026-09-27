#!/usr/bin/env python3
"""Phase B Official Runner: OOLONG-Pairs 64K with Contract Architecture (M1+M2).

Evaluates all 20 quadratic aggregation tasks with dynamic depth and deterministic boundary checks.
Project: rlm1-509413
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.eval.oolong_pairs import (
    OolongPairsTask,
    load_pairs_tasks,
    parse_predicted_pairs,
    score_pairs,
)
from rlm import RLM
from rlm.clients.vertex_gemini import DEFAULT_VERTEX_MODEL, DEFAULT_VERTEX_PROJECT_ID
from rlm.guardrails.paired_runner import PairedHarnessLogger, PairedTaskComparison, TaskRunRecord
from rlm.guardrails.m1_classifier import classify_task_rules
from rlm.logger import RLMLogger
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

BASELINE_PATH = (
    REPO_ROOT
    / "results"
    / "oolong"
    / "pairs"
    / "20260915T054641Z__vertex_openai__oolong_pairs__ctx65536__d2__n20__prlm2-508607"
    / "task_results.jsonl"
)
baseline_records: dict[int, dict[str, Any]] = {}
if BASELINE_PATH.exists():
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                t_num = int(rec.get("task_number", 0))
                baseline_records[t_num] = rec


def main():
    parser = argparse.ArgumentParser(description="Phase B: OOLONG-Pairs 64K Contract Evaluation")
    parser.add_argument("--project-id", default=os.getenv("VERTEX_PROJECT_ID", None), help="Google Cloud Vertex Project ID")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "gemini-1.5-pro"), help="Model name")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--depth", type=int, default=None, help="Explicitly force recursion depth (e.g. 1 or 2)")
    parser.add_argument("--disable-m1", action="store_true", help="Disable M1 complexity classifier and strategy nudge")
    parser.add_argument("--disable-m2", action="store_true", help="Disable M2 runtime intercept guardrail")
    parser.add_argument("--disable-m1-m2", action="store_true", help="Run vanilla RLM baseline without M1/M2 guardrails")
    parser.add_argument("--task-indices", type=str, default=None, help="Comma-separated task IDs to evaluate (e.g. '9' or '1,2,5')")
    parser.add_argument("--backend", default="vertex", choices=["vertex", "openai"], help="Backend provider ('vertex' or 'openai')")
    parser.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible endpoint")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "phase_b_oolong_pairs"))
    parser.add_argument("--max-iterations", type=int, default=50)
    args = parser.parse_args()

    enable_m1 = not (args.disable_m1 or args.disable_m1_m2)
    enable_m2 = not (args.disable_m2 or args.disable_m1_m2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_file = REPO_ROOT / "oolong_64k_context_0.json"
    tasks = load_pairs_tasks(data_file)
    paired_logger = PairedHarnessLogger(output_dir=out_dir)

    print("=" * 80)
    print(f"PHASE B: OOLONG-PAIRS 64K FULL EVALUATION (N={len(tasks)})")
    print(f"Project: {args.project_id} | Model: {args.model_name}")
    print(f"Output: {out_dir}")
    print(f"M1 Enabled: {enable_m1} | M2 Enabled: {enable_m2}")
    print("=" * 80)

    # Resume capability
    completed_task_ids = set()
    paired_file = out_dir / "paired_eval_results.jsonl"
    if paired_file.exists():
        for line in paired_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    if d.get("treatment", {}).get("score", 0.0) > 0.0:
                        completed_task_ids.add(d.get("task_id"))
                except Exception:
                    pass

    if args.task_indices:
        target_ids = {int(x.strip()) for x in args.task_indices.split(",") if x.strip()}
        tasks = [t for t in tasks if t.task_id in target_ids]
        completed_task_ids -= {f"pairs_task_{tid:02d}" for tid in target_ids}

    treatment_scores_list = []
    baseline_scores_list = []

    for idx, task in enumerate(tasks, 1):
        task_key = f"pairs_task_{task.task_id:02d}"
        base_rec = baseline_records.get(task.task_id, {})
        base_f1 = float(base_rec.get("f1", 0.0))
        base_prec = float(base_rec.get("precision", 0.0))
        base_rec_val = float(base_rec.get("recall", 0.0))
        base_pred_count = int(base_rec.get("pred_count", 0))
        gold_count = len(task.gold_pairs)

        if task_key in completed_task_ids:
            print(f"[{idx}/{len(tasks)}] Task {task.task_id} already evaluated. Skipping.")
            continue

        print(f"\n[{idx}/{len(tasks)}] Running Task {task.task_id} ({task.title})...")
        print(f"  Baseline: F1={base_f1:.4f} (P: {base_prec:.4f}, R: {base_rec_val:.4f}, Pred: {base_pred_count}, Gold: {gold_count})")

        m1_cls = classify_task_rules(task.query, {"context_type": "oolong_pairs"})
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

        task_log_dir = out_dir / f"task_{task.task_id:02d}"
        task_log_dir.mkdir(parents=True, exist_ok=True)
        logger = RLMLogger(log_dir=str(task_log_dir), file_name=f"task_{task.task_id:02d}.jsonl")

        if args.backend == "openai":
            b_kwargs = {
                "model_name": args.model_name,
                "api_key": args.api_key,
                "base_url": args.base_url,
                "sampling_args": {"reasoning_effort": "medium"},
            }
            sub_kwargs = {
                "model_name": args.model_name,
                "api_key": args.api_key,
                "base_url": args.base_url,
                "sampling_args": {"reasoning_effort": "low"},
            }
        else:
            b_kwargs = {
                "model_name": args.model_name,
                "project_id": args.project_id,
                "thinking_level": "medium",
                "malformed_retries": 2,
                "malformed_retry_delay": 0.5,
            }
            if args.api_key:
                b_kwargs["api_key"] = args.api_key

            sub_kwargs = {
                "model_name": args.model_name,
                "project_id": args.project_id,
                "thinking_level": "low",
                "malformed_retries": 2,
                "malformed_retry_delay": 0.5,
            }
            if args.api_key:
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
            m1_mode="rules",
            m1_m2_context_metadata={"context_type": "oolong_pairs"} if (enable_m1 or enable_m2) else {},
            paper_mode=True,
            custom_system_prompt=sys_prompt,
            logger=logger,
            verbose=False,
        )

        max_retries = 5
        final_ans = ""
        m1_m2_meta = {}
        for attempt in range(1, max_retries + 1):
            try:
                comp = rlm.completion(task.context, root_prompt=task.root_prompt)
                final_ans = comp.response.strip() if comp and comp.response else ""
                if final_ans:
                    m1_m2_meta = comp.metadata.get("m1_m2", {}) if comp.metadata else {}
                    break
                else:
                    print(f"  [Task {task.task_id} Attempt {attempt}] Empty response received. Retrying...")
                    time.sleep(3 * attempt)
            except Exception as e:
                print(f"  [Task {task.task_id} Attempt {attempt}] Execution/socket error: {e}. Retrying in {5 * attempt}s...")
                time.sleep(5 * attempt)
        exec_time = time.perf_counter() - start_t

        pred_pairs = parse_predicted_pairs(final_ans)
        scores = score_pairs(pred_pairs, task.gold_pairs)
        treat_f1 = scores["f1"]
        treat_prec = scores["precision"]
        treat_rec = scores["recall"]
        treat_count = len(pred_pairs)

        treatment_scores_list.append(treat_f1)
        baseline_scores_list.append(base_f1)

        print(f"  Treatment: F1={treat_f1:.4f} (P: {treat_prec:.4f}, R: {treat_rec:.4f}, Pred: {treat_count}, Gold: {gold_count})")
        print(f"  Delta F1:  {treat_f1 - base_f1:+.4f} | Dynamic Depth: {depth}")

        b_run = TaskRunRecord(
            condition="baseline",
            final_answer=f"pred_count={base_pred_count}",
            is_correct=(base_f1 > 0.95),
            score=base_f1,
            extra_metadata={"precision": base_prec, "recall": base_rec_val, "pred_count": base_pred_count, "gold_count": gold_count},
        )
        t_run = TaskRunRecord(
            condition="treatment",
            final_answer=f"pred_count={treat_count}",
            is_correct=(treat_f1 > 0.95),
            score=treat_f1,
            execution_time_s=exec_time,
            m1_category=m1_cls.category,
            m1_confidence=m1_cls.confidence,
            m1_reasoning=m1_cls.reasoning,
            m2_fired=bool(m1_m2_meta.get("m2_fired", False)),
            m2_check_type=m1_m2_meta.get("m2_check_type"),
            m2_changes_made=bool(m1_m2_meta.get("m2_changed_answer", False)),
            draft_answer_before_m2=m1_m2_meta.get("draft_answer_before_m2"),
            injected_m2_message=m1_m2_meta.get("injected_m2_message"),
            extra_metadata={"precision": treat_prec, "recall": treat_rec, "pred_count": treat_count, "gold_count": gold_count, "dynamic_depth": depth},
        )

        comparison = PairedTaskComparison(
            task_id=task_key,
            query_text=task.query[:240],
            context_metadata={"length": len(task.context), "domain": "oolong_pairs_64k"},
            m1_classification=m1_cls,
            baseline_run=b_run,
            treatment_run=t_run,
            m2_fired=bool(m1_m2_meta.get("m2_fired", False)),
            m2_changed_answer=bool(m1_m2_meta.get("m2_changed_answer", False)),
            accuracy_delta=(treat_f1 - base_f1),
        )
        paired_logger.record_pair(comparison)

        # Deduplicate records in paired_eval_results.jsonl so each task has only its latest record
        paired_file = out_dir / "paired_eval_results.jsonl"
        if paired_file.exists():
            records = {}
            for line in paired_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        d = json.loads(line)
                        records[d.get("task_id")] = line.strip()
                    except Exception:
                        pass
            with open(paired_file, "w", encoding="utf-8") as pf:
                for tid in sorted(records.keys()):
                    pf.write(records[tid] + "\n")

    paired_logger.save_summary()
    summary = paired_logger.get_summary()
    print("\n" + "=" * 80)
    print("PHASE B: OOLONG-PAIRS 64K COMPLETED")
    print(f"Summary written to: {out_dir / 'summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
