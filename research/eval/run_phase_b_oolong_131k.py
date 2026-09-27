#!/usr/bin/env python3
"""Phase B: OOLONG Single-Doc (131K / 262K / 1M) Contract Evaluation with M1/M2 Guardrails."""

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

from rlm import RLM
from rlm.logger import RLMLogger
from rlm.guardrails.m1_classifier import classify_task_rules
from rlm.utils.prompts import get_contract_prompt
from research.eval.oolong import load_tasks, score_answer, OolongTask

OPENAI_HARDENED_SUFFIX = """
Output-format constraint:
Respond with plain text only. In your responses, you may include executable Python code blocks wrapped in ```repl and ```.
Do not emit JSON or structured tool-call objects. The functions llm_query and rlm_query are available in the REPL.

Execution constraint:
The context dataset is already pre-loaded into the Python variable `context` in the REPL environment. Always run code in ```repl``` to inspect, search, and manipulate it.
""".strip()


def main():
    parser = argparse.ArgumentParser(description="Phase B: OOLONG Single-Doc Contract Evaluation")
    parser.add_argument("--model-name", default=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", None))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", None))
    parser.add_argument("--backend", default="openai", choices=["openai", "vertex"])
    parser.add_argument("--dataset-name", default="trec_coarse")
    parser.add_argument("--context-len", type=int, default=131072)
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=None, help="Explicitly force recursion depth (1 or 2)")
    parser.add_argument("--disable-m1", action="store_true", help="Disable M1 dynamic depth")
    parser.add_argument("--disable-m2", action="store_true", help="Disable M2 runtime intercept guardrail")
    parser.add_argument("--disable-m1-m2", action="store_true", help="Run vanilla RLM baseline")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "luna_eval" / "oolong_131k_depth1_vanilla"))
    parser.add_argument("--max-iterations", type=int, default=30)
    args = parser.parse_args()

    enable_m1 = not (args.disable_m1 or args.disable_m1_m2)
    enable_m2 = not (args.disable_m2 or args.disable_m1_m2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(
        dataset_name=args.dataset_name,
        context_len=args.context_len,
        num_examples=args.num_examples,
        seed=args.seed,
    )

    print("=" * 80)
    print(f"PHASE B: OOLONG SINGLE-DOC EVALUATION (N={len(tasks)}, Ctx={args.context_len})")
    print(f"Model: {args.model_name} | Backend: {args.backend}")
    print(f"Output: {out_dir}")
    print(f"M1 Enabled: {enable_m1} | M2 Enabled: {enable_m2}")
    print("=" * 80)

    # Resume capability
    completed_task_ids = set()
    results_file = out_dir / "task_results.jsonl"
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed_task_ids.add(rec.get("task_id"))
                    except Exception:
                        pass

    scores_list = []
    correct_count = 0
    total_evaluated = 0

    for idx, task in enumerate(tasks, 1):
        task_id = f"task_{idx:03d}_{task.source_id}"

        if task_id in completed_task_ids:
            print(f"[{idx}/{len(tasks)}] Task {task_id} already evaluated. Skipping.")
            continue

        print(f"\n[{idx}/{len(tasks)}] Running Task {task_id}...")
        print(f"  Question: {task.question[:120]}...")
        print(f"  Gold Answer: {task.answer} (Type: {task.answer_type})")

        ctx_meta = {"context_type": "oolong_single_doc", "answer_type": task.answer_type}
        m1_cls = classify_task_rules(task.question, ctx_meta)
        if args.depth is not None:
            depth = args.depth
        elif enable_m1:
            depth = m1_cls.suggested_depth
        else:
            depth = 1

        sys_prompt = get_contract_prompt(depth) + "\n\n" + OPENAI_HARDENED_SUFFIX

        task_log_dir = out_dir / f"task_{idx:03d}"
        task_log_dir.mkdir(parents=True, exist_ok=True)
        logger = RLMLogger(log_dir=str(task_log_dir), file_name=f"task_{idx:03d}.jsonl")

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
            m1_m2_context_metadata=ctx_meta if (enable_m1 or enable_m2) else {},
            paper_mode=True,
            custom_system_prompt=sys_prompt,
            logger=logger,
            verbose=False,
        )

        try:
            comp = rlm.completion(task.context, root_prompt=task.root_prompt)
            final_ans = comp.response.strip()
            exec_time = time.perf_counter() - start_t
            m1_m2_meta = comp.metadata.get("m1_m2", {}) if comp.metadata else {}
        except Exception as e:
            print(f"  Execution error on Task {task_id}: {e}")
            final_ans = ""
            exec_time = time.perf_counter() - start_t
            m1_m2_meta = {}

        score = score_answer(task, final_ans)
        is_correct = bool(score >= 1.0)
        scores_list.append(score)
        total_evaluated += 1
        if is_correct:
            correct_count += 1

        print(f"  Result: Score={score:.2f} | Correct={is_correct} | Time={exec_time:.1f}s | Depth={depth}")
        print(f"  Cumulative Accuracy: {correct_count}/{total_evaluated} ({correct_count/total_evaluated*100:.1f}%)")

        record = {
            "task_id": task_id,
            "index": idx,
            "source_id": task.source_id,
            "question": task.question,
            "answer_type": task.answer_type,
            "gold_answer": task.answer,
            "raw_response": final_ans,
            "score": score,
            "is_correct": is_correct,
            "execution_time_s": exec_time,
            "dynamic_depth": depth,
            "m1_category": m1_cls.category,
            "m2_fired": bool(m1_m2_meta.get("m2_fired", False)),
            "m2_changed_answer": bool(m1_m2_meta.get("m2_changed_answer", False)),
        }

        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    mean_acc = (sum(scores_list) / max(1, len(scores_list))) * 100.0
    print("\n" + "=" * 80)
    print(f"PHASE B: OOLONG SINGLE-DOC FINISHED (N={len(tasks)})")
    print(f"Mean Accuracy: {mean_acc:.2f}% ({correct_count}/{len(tasks)})")
    print("=" * 80)

    summary_file = out_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"total_tasks": len(tasks), "mean_accuracy": mean_acc, "correct": correct_count}, f, indent=2)


if __name__ == "__main__":
    main()
