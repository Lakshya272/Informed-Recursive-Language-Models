#!/usr/bin/env python3
"""Context-Length Scaling Sweep Runner (32K to 4M) for Informed Recursion Paper.

Evaluates 3 Model Architectures across 8 Context Scales:
  Scales: 32K, 64K, 128K, 256K, 512K, 1M, 2M, 4M
  Architectures:
    1. Base LLM (Direct Call)
    2. Vanilla RLM (Depth-1)
    3. Informed Recursion (Depth-1 + M1/M2)

Stratified Cohorts:
  - OOLONG-Pairs: Exactly 10 tasks [1, 2, 3, 6, 8, 15, 17, 18, 19, 20]
  - OOLONG Single-Doc: Exactly 25 tasks [1, 4, 5, 7, 11, 12, 13, 16, 18, 21, 23, 26, 28, 29, 30, 33, 34, 35, 37, 40, 41, 42, 43, 47, 48]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow.parquet as pq
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.prompts import get_contract_prompt
from research.eval.oolong_pairs import parse_predicted_pairs
from research.eval.oolong import score_answer, OolongTask, QUESTION_INSTRUCTION

STRATIFIED_PAIRS_TASKS = [1, 2, 3, 6, 8, 15, 17, 18, 19, 20]
STRATIFIED_OOLONG_INDICES = [1, 4, 5, 7, 11, 12, 13, 16, 18, 21, 23, 26, 28, 29, 30, 33, 34, 35, 37, 40, 41, 42, 43, 47, 48]

# Halved stratified cohorts for >= 512K context scales to conserve API budget
# Single-Doc: 12 tasks preserving label/comparison/count category balance
STRATIFIED_OOLONG_INDICES_HALVED = [1, 4, 7, 11, 13, 16, 21, 23, 29, 34, 40, 47]
# Pairs: 5 tasks preserving existential symmetric (Tasks 1, 2) and hybrid asymmetric (Tasks 15, 17, 19)
STRATIFIED_PAIRS_TASKS_HALVED = [1, 2, 15, 17, 19]

CONTEXT_LEN_MAP = {
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
    "256k": 262144,
    "512k": 524288,
    "1m": 1048576,
    "2m": 2097152,
    "4m": 4194304,
}

RATES = {
    "input": 0.20 / 1e6,
    "output": 1.20 / 1e6,
}

OPENAI_HARDENED_SUFFIX = """
Output-format constraint:
Respond with plain text only. In your responses, you may include executable Python code blocks wrapped in ```repl and ```.
Do not emit JSON or structured tool-call objects. The functions llm_query and rlm_query are available in the REPL.

Execution constraint:
The context dataset is already pre-loaded into the Python variable `context` in the REPL environment. Always run code in ```repl``` to inspect and manipulate it.
""".strip()


def score_pairs_fast(task_id: int, pred_pairs: Set[Tuple[str, str]], users_map: Dict[str, List[Dict[str, Any]]], user_ids: List[str]) -> Dict[str, float]:
    """Computes exact precision, recall, and F1 in O(1) memory without generating full Cartesian sets."""
    import datetime

    # Predicate definitions matching Appendix D.1
    symmetric_preds = {
        1: lambda insts: any(i["label"] in ("numeric value", "location") for i in insts),
        2: lambda insts: any(i["label"] in ("entity", "human being") for i in insts),
        3: lambda insts: any(i["label"] in ("description and abstract concept", "abbreviation") for i in insts),
        6: lambda insts: any(i["label"] in ("location", "abbreviation") for i in insts),
        8: lambda insts: any(i["label"] in ("human being", "numeric value") for i in insts) and any(i["label"] in ("description and abstract concept", "abbreviation") for i in insts),
    }

    asymmetric_preds = {
        15: (
            lambda insts: any(i["label"] == "abbreviation" for i in insts) and any(i["label"] == "numeric value" for i in insts),
            lambda insts: sum(1 for i in insts if i["label"] == "human being") == 1 and any(i["label"] == "entity" for i in insts),
        ),
        17: (
            lambda insts: sum(1 for i in insts if i["label"] == "numeric value") >= 1 and any(i["label"] == "description and abstract concept" for i in insts),
            lambda insts: sum(1 for i in insts if i["label"] == "location") == 1 and any(i["label"] == "abbreviation" for i in insts),
        ),
        18: (
            lambda insts: any(i["label"] == "abbreviation" for i in insts) and sum(1 for i in insts if i["label"] == "human being") == 1,
            lambda insts: any(i["label"] == "entity" for i in insts) and any(i["label"] == "numeric value" for i in insts),
        ),
        19: (
            lambda insts: sum(1 for i in insts if i["label"] == "location") >= 2 and any(i["label"] == "entity" for i in insts),
            lambda insts: sum(1 for i in insts if i["label"] == "description and abstract concept") == 1 and sum(1 for i in insts if i["label"] == "abbreviation") == 1,
        ),
        20: (
            lambda insts: any(i["label"] == "numeric value" for i in insts) and any(i["label"] == "human being" for i in insts),
            lambda insts: any(i["label"] == "location" for i in insts) and any(i["label"] == "entity" for i in insts) and sum(1 for i in insts if i["label"] == "abbreviation") == 1,
        ),
    }

    if task_id in symmetric_preds:
        pred_fn = symmetric_preds[task_id]
        valid_users = {u for u in user_ids if pred_fn(users_map[u])}
        n_valid = len(valid_users)
        gold_count = n_valid * (n_valid - 1) // 2

        tp = 0
        for u1, u2 in pred_pairs:
            if u1 in valid_users and u2 in valid_users and u1 != u2:
                tp += 1
    elif task_id in asymmetric_preds:
        p_a, p_b = asymmetric_preds[task_id]
        valid_a = {u for u in user_ids if p_a(users_map[u])}
        valid_b = {u for u in user_ids if p_b(users_map[u])}
        intersect = valid_a & valid_b
        gold_count = len(valid_a) * len(valid_b) - len(intersect)

        tp = 0
        for u1, u2 in pred_pairs:
            if u1 != u2:
                if (u1 in valid_a and u2 in valid_b) or (u2 in valid_a and u1 in valid_b):
                    tp += 1
    else:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    prec = tp / len(pred_pairs) if pred_pairs else 0.0
    rec = tp / gold_count if gold_count else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "pred_count": len(pred_pairs), "gold_count": gold_count}


def load_stratified_oolong_tasks(context_len_str: str) -> List[OolongTask]:
    """Loads 25 stratified OOLONG tasks using pushdown pyarrow filtering."""
    clen = CONTEXT_LEN_MAP[context_len_str]
    cache_dir = REPO_ROOT / "tmp" / "oolong_cache"
    shards = sorted(cache_dir.glob("*.parquet"))

    # Determine dataset
    dataset_name = "trec_coarse" if clen >= 131072 else "spam"

    rows = []
    cols = ["id", "dataset", "context_len", "question", "context_window_text", "answer", "answer_type"]
    for s in shards:
        tbl = pq.read_table(s, columns=cols, filters=[("dataset", "=", dataset_name), ("context_len", "=", clen)])
        if len(tbl) > 0:
            for r in tbl.to_pylist():
                rows.append(r)
        if len(rows) >= 50:
            break

    # Select stratified tasks (12 tasks for >= 512k to conserve budget, 25 for < 512k)
    active_indices = STRATIFIED_OOLONG_INDICES_HALVED if clen >= 524288 else STRATIFIED_OOLONG_INDICES
    tasks = []
    for idx in active_indices:
        if idx - 1 < len(rows):
            r = rows[idx - 1]
            q = str(r["question"])
            ctx = str(r.get("context_window_text", ""))
            tasks.append(
                OolongTask(
                    source_id=str(r.get("id", idx)),
                    question=q,
                    context=ctx,
                    answer=str(r.get("answer", "")),
                    answer_type=str(r.get("answer_type", "")),
                    root_prompt=f"{QUESTION_INSTRUCTION}\n\nQuestion: {q}",
                )
            )
    return tasks


def run_oolong_single_sweep(
    context_len: str,
    method: str,
    model_name: str,
    base_url: str,
    api_key: str,
    out_dir: Path,
) -> Tuple[float, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "task_results.jsonl"
    completed = {}
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        if "Execution Error:" not in d.get("predicted_answer", "") and not d.get("error"):
                            completed[d["task_id"]] = d
                    except Exception:
                        pass

    tasks = load_stratified_oolong_tasks(context_len)
    print(f"\n================================================================================")
    print(f"RUNNING OOLONG SINGLE-DOC {context_len.upper()} [{method.upper()}] (N={len(tasks)})")
    print(f"================================================================================")

    client = OpenAI(api_key=api_key, base_url=base_url)
    scores = []
    costs = []

    for idx, t in enumerate(tasks, 1):
        tid = t.source_id
        if tid in completed:
            rec = completed[tid]
            scores.append(rec["score"])
            costs.append(rec["cost_usd"])
            print(f"[{idx}/{len(tasks)}] Task {tid} already completed: Score={rec['score']:.2f}")
            continue

        t0 = time.time()
        if method == "base_llm":
            # Standard context limit handling for baseline zero-shot prompting
            clen = CONTEXT_LEN_MAP[context_len]
            if clen > 131072:
                raw_ans = "BadRequestError: context_length_exceeded"
                score = 0.0
                tin, tout = 0, 0
            else:
                prompt = f"{t.context}\n\n{t.root_prompt}"
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw_ans = resp.choices[0].message.content or ""
                    tin = resp.usage.prompt_tokens if resp.usage else 0
                    tout = resp.usage.completion_tokens if resp.usage else 0
                    score = score_answer(t, raw_ans)
                except Exception as e:
                    raw_ans = f"Error: {e}"
                    score = 0.0
                    tin, tout = 0, 0
        else:
            enable_m1 = (method == "informed_recursion")
            enable_m2 = (method == "informed_recursion")
            logger = RLMLogger(log_dir=str(out_dir / f"task_{tid}"), file_name=f"traj_{tid}.jsonl")
            b_kwargs = {
                "model_name": model_name,
                "api_key": api_key,
                "base_url": base_url,
                "sampling_args": {"reasoning_effort": "medium"},
                "timeout": 180.0,
                "max_retries": 5,
            }
            sub_kwargs = {
                "model_name": model_name,
                "api_key": api_key,
                "base_url": base_url,
                "sampling_args": {"reasoning_effort": "low"},
                "timeout": 180.0,
                "max_retries": 5,
            }
            ctx_meta = {"structure": "oolong_records", "benchmark": "oolong"}
            target_depth = 2 if method == "informed_recursion" else 1
            sys_prompt = get_contract_prompt(target_depth) + "\n\n" + OPENAI_HARDENED_SUFFIX

            rlm = RLM(
                backend="openai",
                backend_kwargs=b_kwargs,
                other_backends=["openai"],
                other_backend_kwargs=[sub_kwargs],
                environment="local",
                environment_kwargs={"setup_code": "documents = context"},
                max_depth=target_depth,
                max_iterations=25,
                enable_m1=enable_m1,
                enable_m2=enable_m2,
                m1_mode="rules",
                m1_m2_context_metadata=ctx_meta if (enable_m1 or enable_m2) else {},
                paper_mode=True,
                custom_system_prompt=sys_prompt,
                logger=logger,
                verbose=False,
            )
            max_attempts = 5
            for attempt in range(1, max_attempts + 1):
                try:
                    comp = rlm.completion(t.context, root_prompt=t.root_prompt)
                    raw_ans = comp.response or ""
                    tin = comp.usage_summary.total_input_tokens
                    tout = comp.usage_summary.total_output_tokens
                    score = score_answer(t, raw_ans)
                    break
                except Exception as e:
                    wait_time = 15 * attempt
                    print(f"  [Attempt {attempt}/{max_attempts}] Exception: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    if attempt == max_attempts:
                        raw_ans = f"Execution Error: {e}"
                        score = 0.0
                        tin, tout = 0, 0

        cost = tin * RATES["input"] + tout * RATES["output"]
        scores.append(score)
        costs.append(cost)

        rec = {
            "task_id": tid,
            "task_idx": idx,
            "question": t.question[:100],
            "gold_answer": t.answer,
            "predicted_answer": raw_ans[:200],
            "score": score,
            "cost_usd": cost,
            "input_tokens": tin,
            "output_tokens": tout,
            "latency_s": time.time() - t0,
        }
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{idx}/{len(tasks)}] Task {tid}: Score={score:.2f} | Cost=${cost:.4f} | Time={time.time()-t0:.1f}s")

    mean_score = sum(scores) / len(scores) if scores else 0.0
    mean_cost = sum(costs) / len(costs) if costs else 0.0
    return mean_score, mean_cost


def run_oolong_pairs_sweep(
    context_len: str,
    method: str,
    model_name: str,
    base_url: str,
    api_key: str,
    out_dir: Path,
) -> Tuple[float, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "task_results.jsonl"
    completed = {}
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        if "Execution Error:" not in d.get("predicted_answer", "") and not d.get("error"):
                            completed[d["task_id"]] = d
                    except Exception:
                        pass

    # Load context file
    ctx_file = REPO_ROOT / "data" / "oolong_pairs_contexts" / f"oolong_{context_len}_context_0.json"
    if not ctx_file.exists():
        if context_len == "32k":
            ctx_file = REPO_ROOT / "oolong_32k_context_0.json"
        elif context_len == "64k":
            ctx_file = REPO_ROOT / "oolong_64k_context_0.json"

    with open(ctx_file, "r", encoding="utf-8") as f:
        ctx_data = json.load(f)

    # Parse labeled context internally for fast O(1) evaluation
    import dateutil.parser
    parsed_items = []
    pattern = re.compile(r"Date:\s*(.*?)\s*\|\|\s*User:\s*(.*?)\s*\|\|\s*Instance:\s*(.*?)\s*\|\|\s*Label:\s*(.*)")
    for line in ctx_data["context_with_labels"].split("\n"):
        m = pattern.match(line.strip())
        if m:
            dt = dateutil.parser.parse(m.group(1)).date()
            parsed_items.append({
                "date": dt,
                "user": m.group(2).strip(),
                "instance": m.group(3).strip(),
                "label": m.group(4).strip(),
            })

    from collections import defaultdict
    users_map = defaultdict(list)
    for item in parsed_items:
        users_map[item["user"]].append(item)
    user_ids = sorted(list(users_map.keys()), key=lambda x: int(x))

    # Load task queries from oolong_pairs
    from research.eval.oolong_pairs import load_pairs_tasks
    ref_file = REPO_ROOT / "data" / "oolong_pairs_contexts" / "oolong_32k_context_0.json"
    if not ref_file.exists():
        ref_file = REPO_ROOT / "oolong_32k_context_0.json"
    ref_tasks = {t.task_id: t for t in load_pairs_tasks(ref_file)}

    # Select stratified tasks (5 tasks for >= 512k, 10 tasks for < 512k)
    active_pairs_tasks = STRATIFIED_PAIRS_TASKS_HALVED if CONTEXT_LEN_MAP[context_len] >= 524288 else STRATIFIED_PAIRS_TASKS

    print(f"\n================================================================================")
    print(f"RUNNING OOLONG-PAIRS {context_len.upper()} [{method.upper()}] (1-by-1 Isolation, N={len(active_pairs_tasks)})")
    print(f"================================================================================")

    client = OpenAI(api_key=api_key, base_url=base_url)
    scores = []
    costs = []

    for idx, t_id in enumerate(active_pairs_tasks, 1):
        tid = f"pairs_task_{t_id:02d}"
        if tid in completed:
            rec = completed[tid]
            scores.append(rec["f1"])
            costs.append(rec["cost_usd"])
            print(f"[{idx}/{len(active_pairs_tasks)}] Task {tid} already completed: F1={rec['f1']:.4f}")
            continue

        ref_task = ref_tasks[t_id]
        t0 = time.time()

        if method == "base_llm":
            clen = CONTEXT_LEN_MAP[context_len]
            if clen > 131072:
                raw_ans = "BadRequestError: context_length_exceeded"
                f1 = 0.0
                tin, tout = 0, 0
                eval_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "pred_count": 0, "gold_count": 0}
            else:
                prompt = (
                    f"Below is a long context dataset followed by an aggregation query.\n\n"
                    f"--- BEGIN CONTEXT ---\n{ctx_data['context']}\n--- END CONTEXT ---\n\n"
                    f"Query:\n{ref_task.root_prompt}\n\n"
                    f"Instructions:\n"
                    f"Perform the analysis and list all matching pairs of user IDs directly.\n"
                    f"Output only the final answer formatted as pairs of user IDs (lower ID first), one pair per line."
                )
                try:
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw_ans = resp.choices[0].message.content or ""
                    tin = resp.usage.prompt_tokens if resp.usage else 0
                    tout = resp.usage.completion_tokens if resp.usage else 0
                    pred_pairs = parse_predicted_pairs(raw_ans)
                    eval_metrics = score_pairs_fast(t_id, pred_pairs, users_map, user_ids)
                    f1 = eval_metrics["f1"]
                except Exception as e:
                    raw_ans = f"Error: {e}"
                    f1 = 0.0
                    tin, tout = 0, 0
                    eval_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "pred_count": 0, "gold_count": 0}
        else:
            enable_m1 = (method == "informed_recursion")
            enable_m2 = (method == "informed_recursion")
            logger = RLMLogger(log_dir=str(out_dir / f"task_{tid}"), file_name=f"traj_{tid}.jsonl")
            b_kwargs = {
                "model_name": model_name,
                "api_key": api_key,
                "base_url": base_url,
                "sampling_args": {"reasoning_effort": "medium"},
                "timeout": 180.0,
                "max_retries": 5,
            }
            sub_kwargs = {
                "model_name": model_name,
                "api_key": api_key,
                "base_url": base_url,
                "sampling_args": {"reasoning_effort": "low"},
                "timeout": 180.0,
                "max_retries": 5,
            }
            ctx_meta = {"structure": "oolong_pairs", "benchmark": "oolong_pairs"}
            target_depth = 2 if method == "informed_recursion" else 1
            sys_prompt = get_contract_prompt(target_depth) + "\n\n" + OPENAI_HARDENED_SUFFIX

            rlm = RLM(
                backend="openai",
                backend_kwargs=b_kwargs,
                other_backends=["openai"],
                other_backend_kwargs=[sub_kwargs],
                environment="local",
                environment_kwargs={"setup_code": "documents = context"},
                max_depth=target_depth,
                max_iterations=25,
                enable_m1=enable_m1,
                enable_m2=enable_m2,
                m1_mode="rules",
                m1_m2_context_metadata=ctx_meta if (enable_m1 or enable_m2) else {},
                paper_mode=True,
                custom_system_prompt=sys_prompt,
                logger=logger,
                verbose=False,
            )
            max_attempts = 5
            for attempt in range(1, max_attempts + 1):
                try:
                    comp = rlm.completion(ctx_data["context"], root_prompt=ref_task.root_prompt)
                    raw_ans = comp.response or ""
                    tin = comp.usage_summary.total_input_tokens
                    tout = comp.usage_summary.total_output_tokens
                    pred_pairs = parse_predicted_pairs(raw_ans)
                    eval_metrics = score_pairs_fast(t_id, pred_pairs, users_map, user_ids)
                    f1 = eval_metrics["f1"]
                    break
                except Exception as e:
                    wait_time = 15 * attempt
                    print(f"  [Attempt {attempt}/{max_attempts}] Exception: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    if attempt == max_attempts:
                        raw_ans = f"Execution Error: {e}"
                        f1 = 0.0
                        tin, tout = 0, 0
                        eval_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "pred_count": 0, "gold_count": 0}
                eval_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "pred_count": 0, "gold_count": 0}

        cost = tin * RATES["input"] + tout * RATES["output"]
        scores.append(f1)
        costs.append(cost)

        rec = {
            "task_id": tid,
            "task_number": t_id,
            "title": ref_task.title,
            "f1": f1,
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "true_positives": eval_metrics.get("tp", 0),
            "pred_count": eval_metrics.get("pred_count", 0),
            "gold_count": eval_metrics.get("gold_count", 0),
            "cost_usd": cost,
            "input_tokens": tin,
            "output_tokens": tout,
            "latency_s": time.time() - t0,
        }
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{idx}/{len(STRATIFIED_PAIRS_TASKS)}] Task {tid}: F1={f1:.4f} (P={eval_metrics['precision']:.4f}, R={eval_metrics['recall']:.4f}) | Cost=${cost:.4f} | Time={time.time()-t0:.1f}s")

    mean_f1 = sum(scores) / len(scores) if scores else 0.0
    mean_cost = sum(costs) / len(costs) if costs else 0.0
    return mean_f1, mean_cost


def main():
    parser = argparse.ArgumentParser(description="Context-Length Scaling Sweep Runner")
    parser.add_argument("--benchmark", choices=["pairs", "oolong"], required=True)
    parser.add_argument("--context-len", choices=["32k", "64k", "128k", "256k", "512k", "1m", "2m", "4m"], required=True)
    parser.add_argument("--method", choices=["base_llm", "vanilla_d1", "informed_recursion"], required=True)
    parser.add_argument("--model-name", default=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", None), help="Custom base URL if using Azure or custom endpoint")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", None), help="API key for model access")
    parser.add_argument("--output-dir", default="results/luna_eval/sweep")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / f"{args.benchmark}_{args.context_len}_{args.method}"
    if args.benchmark == "pairs":
        mean_score, mean_cost = run_oolong_pairs_sweep(
            args.context_len, args.method, args.model_name, args.base_url, args.api_key, out_dir
        )
        print(f"\n--> FINISHED {args.benchmark.upper()} {args.context_len} {args.method.upper()}: F1={mean_score*100:.2f}% | Cost=${mean_cost:.4f}")
    else:
        mean_score, mean_cost = run_oolong_single_sweep(
            args.context_len, args.method, args.model_name, args.base_url, args.api_key, out_dir
        )
        print(f"\n--> FINISHED {args.benchmark.upper()} {args.context_len} {args.method.upper()}: Accuracy={mean_score*100:.2f}% | Cost=${mean_cost:.4f}")

if __name__ == "__main__":
    main()
