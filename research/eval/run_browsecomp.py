#!/usr/bin/env python3
"""Official BrowseComp-Plus Generalization Runner for Informed Recursion.

Evaluates 150 randomly sampled queries with 1,000 documents per query.
Supports:
  - Vanilla RLM (Depth-1)
  - Informed Recursion (Depth-1 + M1/M2)
Uses the official LLM-as-a-Judge semantic evaluation prompt from Appendix B.2 of the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI
from datasets import load_dataset
from rlm import RLM
from rlm.logger import RLMLogger
from rlm.utils.prompts import get_contract_prompt

OPENAI_HARDENED_SUFFIX = """
Output-format constraint:
Respond with plain text only. In your responses, you may include executable Python code blocks wrapped in ```repl and ```.
Do not emit JSON or structured tool-call objects. The functions llm_query and rlm_query are available in the REPL.

Execution constraint:
The collection of 1,000 documents is already pre-loaded into the Python variable `context` (and `documents`) as a list of dictionaries with keys 'docid' and 'text'. Always run code in ```repl``` to inspect, search, and manipulate it.
When you find the final answer, output your answer clearly as 'FINAL ANSWER: <answer>' or 'ANSWER: <answer>'.
""".strip()

JUDGE_PROMPT_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}
[response]: {response}
[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:
extracted_final_answer: The final exact answer extracted from the [response].
correct_answer: Repeat the [correct_answer] given above.
reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.
correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
confidence: The extracted confidence score between 0% and 100%.
"""

def evaluate_with_judge(
    client: OpenAI,
    judge_model: str,
    question: str,
    response: str,
    correct_answer: str,
) -> tuple[bool, str, dict[str, int]]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )
    for attempt in range(5):
        try:
            res = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = res.choices[0].message.content or ""
            m = re.search(r"correct:\s*(yes|no)", content, re.IGNORECASE)
            is_correct = (m.group(1).lower() == "yes") if m else False
            usage = {
                "input_tokens": res.usage.prompt_tokens if res.usage else 0,
                "output_tokens": res.usage.completion_tokens if res.usage else 0,
            }
            return is_correct, content, usage
        except Exception as e:
            time.sleep(2 ** attempt)
            if attempt == 4:
                return False, f"Judge Error: {e}", {"input_tokens": 0, "output_tokens": 0}

def build_or_load_manifest(seed: int = 42, num_queries: int = 150, num_docs: int = 1000, pilot: bool = False) -> list[Any]:
    pilot_path = REPO_ROOT / "data" / "browsecomp_pilot_manifest.json"
    if pilot and pilot_path.exists():
        with open(pilot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data[:3]

    query_dir = REPO_ROOT / "data" / "browsecomp_queries"
    if query_dir.exists():
        files = sorted(query_dir.glob("query_*.json"))[:num_queries]
        if len(files) >= num_queries:
            return files[:3] if pilot else files

    manifest_path = REPO_ROOT / "data" / f"browsecomp_manifest_seed{seed}_n{num_queries}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data[:3] if pilot else data

    print(f"Loading full BrowseComp-Plus dataset to create reproducible manifest (seed={seed})...")
    ds = load_dataset("timchen0618/browsecomp-plus-benchmark", split="train")

    all_docs_map = {}
    for r in ds:
        for d in json.loads(r["gold_docs"]):
            all_docs_map[d["docid"]] = d["text"]
        for d in json.loads(r["evidence_docs"]):
            all_docs_map[d["docid"]] = d["text"]

    all_doc_ids = sorted(list(all_docs_map.keys()))
    rng = random.Random(seed)
    sampled_indices = rng.sample(range(len(ds)), num_queries)

    manifest = []
    for idx in sampled_indices:
        row = ds[idx]
        gold_docs = json.loads(row["gold_docs"])
        evidence_docs = json.loads(row["evidence_docs"])

        query_doc_map = {}
        for d in gold_docs:
            query_doc_map[d["docid"]] = d["text"]
        for d in evidence_docs:
            query_doc_map[d["docid"]] = d["text"]

        needed = num_docs - len(query_doc_map)
        distractor_pool = [did for did in all_doc_ids if did not in query_doc_map]
        sampled_distractor_ids = rng.sample(distractor_pool, min(needed, len(distractor_pool)))

        for did in sampled_distractor_ids:
            query_doc_map[did] = all_docs_map[did]

        doc_list = [{"docid": did, "text": text} for did, text in query_doc_map.items()]
        rng.shuffle(doc_list)

        manifest.append({
            "query_id": str(row["query_id"]),
            "query": str(row["query"]),
            "answer": str(row["answer"]),
            "num_docs": len(doc_list),
            "documents": doc_list,
        })

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved reproducible manifest with {len(manifest)} queries to {manifest_path}")
    return manifest[:3] if pilot else manifest

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="BrowseComp-Plus Generalization Evaluation")
    parser.add_argument("--model-name", default=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", None), help="OpenAI / Azure OpenAI API key")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", None), help="Custom base URL endpoint if applicable")
    parser.add_argument("--method", default="informed_recursion", choices=["vanilla_d1", "informed_recursion"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-queries", type=int, default=150)
    parser.add_argument("--pilot", action="store_true", help="Run 3-query pilot verification")
    parser.add_argument("--output-dir", default="results/luna_eval/browsecomp_generalization")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / f"{args.method}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_or_load_manifest(seed=args.seed, num_queries=args.num_queries, pilot=args.pilot)
    if args.pilot:
        print(f"PILOT MODE: Evaluating {len(manifest)} queries to verify pipeline...")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    pricing = {"input": 0.20 / 1e6, "output": 1.20 / 1e6}
    scores = []
    total_cost = 0.0

    # Resume capability
    results_file = out_dir / "task_results.jsonl"
    completed_qids = set()
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if "Execution Error:" not in rec.get("predicted_answer", "") and not rec.get("error"):
                            completed_qids.add(rec["query_id"])
                            scores.append(1.0 if rec.get("is_correct") else 0.0)
                            total_cost += rec.get("cost_usd", 0.0)
                    except Exception:
                        pass

    print("=" * 80)
    print(f"BROWSECOMP-PLUS GENERALIZATION EVALUATION")
    print(f"Method: {args.method.upper()} | Model: {args.model_name}")
    print(f"Queries: {len(manifest)} (Completed: {len(completed_qids)}) | Output: {out_dir}")
    print("=" * 80)

    for idx, item_ref in enumerate(manifest):
        if isinstance(item_ref, (str, Path)):
            with open(item_ref, "r", encoding="utf-8") as f:
                item = json.load(f)
        else:
            item = item_ref

        qid = item["query_id"]
        if qid in completed_qids:
            continue

        q_text = item["query"]
        gold_ans = item["answer"]
        docs = item["documents"]

        print(f"\n[{idx+1}/{len(manifest)}] Query {qid}: {q_text[:90]}...")
        t0 = time.time()

        root_prompt = f"Question: {q_text}\nFind the exact and definitive answer using the pre-loaded documents."

        # Setup RLM
        enable_m1 = (args.method == "informed_recursion")
        enable_m2 = (args.method == "informed_recursion")

        b_kwargs = {
            "model_name": args.model_name,
            "api_key": args.api_key,
            "base_url": args.base_url,
            "sampling_args": {"reasoning_effort": "medium"},
            "timeout": 180.0,
            "max_retries": 5,
        }
        sub_kwargs = {
            "model_name": args.model_name,
            "api_key": args.api_key,
            "base_url": args.base_url,
            "sampling_args": {"reasoning_effort": "low"},
            "timeout": 180.0,
            "max_retries": 5,
        }

        ctx_meta = {
            "structure": "browsecomp_docs",
            "benchmark": "browsecomp",
            "num_docs": len(docs),
        }

        sys_prompt = get_contract_prompt(2 if enable_m1 else 1) + "\n\n" + OPENAI_HARDENED_SUFFIX

        rlm = RLM(
            backend="openai",
            backend_kwargs=b_kwargs,
            other_backends=["openai"],
            other_backend_kwargs=[sub_kwargs],
            environment="local",
            environment_kwargs={"setup_code": "documents = context"},
            max_depth=2 if enable_m1 else 1,
            max_iterations=25,
            enable_m1=enable_m1,
            enable_m2=enable_m2,
            m1_mode="rules",
            m1_m2_context_metadata=ctx_meta if (enable_m1 or enable_m2) else {},
            paper_mode=True,
            custom_system_prompt=sys_prompt,
            logger=None,
            verbose=False,
        )

        max_attempts = 5
        comp = None
        for attempt in range(1, max_attempts + 1):
            try:
                comp = rlm.completion(docs, root_prompt=root_prompt)
                raw_response = comp.response.strip()
                tin = comp.usage_summary.total_input_tokens
                tout = comp.usage_summary.total_output_tokens
                break
            except Exception as e:
                wait_time = 15 * attempt
                print(f"  [Attempt {attempt}/{max_attempts}] Exception: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                if attempt == max_attempts:
                    raw_response = f"Execution Error: {e}"
                    tin, tout = 0, 0

        latency = time.time() - t0

        # Run semantic judge
        is_correct, judge_reasoning, judge_usage = evaluate_with_judge(
            client=client,
            judge_model=args.model_name,
            question=q_text,
            response=raw_response,
            correct_answer=gold_ans,
        )

        tin += judge_usage.get("input_tokens", 0)
        tout += judge_usage.get("output_tokens", 0)
        cost = tin * pricing["input"] + tout * pricing["output"]

        scores.append(1.0 if is_correct else 0.0)
        total_cost += cost

        print(f"  Result: Correct={is_correct} | Score={1.0 if is_correct else 0.0} | Cost=${cost:.4f} | Time={latency:.1f}s")
        safe_pred = raw_response[:80].encode('ascii', errors='replace').decode('ascii')
        safe_gold = str(gold_ans).encode('ascii', errors='replace').decode('ascii')
        print(f"  Predicted: {safe_pred}")
        print(f"  Gold:      {safe_gold}")
        print(f"  Cumulative Accuracy: {sum(scores)}/{len(scores)} ({sum(scores)/len(scores)*100:.1f}%)")

        rec = {
            "query_id": qid,
            "query": q_text,
            "gold_answer": gold_ans,
            "predicted_answer": raw_response,
            "is_correct": is_correct,
            "judge_reasoning": judge_reasoning,
            "input_tokens": tin,
            "output_tokens": tout,
            "cost_usd": cost,
            "latency_s": latency,
        }
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    mean_acc = sum(scores) / len(scores) if scores else 0.0
    mean_cost = total_cost / len(scores) if scores else 0.0
    summary = {
        "method": args.method,
        "seed": args.seed,
        "num_queries_evaluated": len(scores),
        "mean_accuracy": mean_acc,
        "mean_cost_usd": mean_cost,
        "total_cost_usd": total_cost,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE: {args.method.upper()}")
    print(f"Accuracy: {mean_acc*100:.2f}% | Total Cost: ${total_cost:.4f} | Mean Cost: ${mean_cost:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
