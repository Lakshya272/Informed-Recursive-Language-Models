#!/usr/bin/env python3
"""Run evaluation on LongCoT-mini across Depth-0, Depth-1, and Depth-2.

Paper-aligned evaluation strictly reproducing the RLM benchmark on LongCoT-mini
without decomposition hints, with depth-wise prompting, tool gating, and air-gapped sandboxing.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.eval.longcot_verifier.src._types import Question, VerifyOptions
from research.eval.longcot_verifier.src._verifier import verify
from research.eval.longcot_verifier.src._parsing import extract_solution

from rlm.core.rlm import RLM
from rlm.logger.rlm_logger import RLMLogger
from rlm.utils.prompts import (
    PAPER_RLM_DEPTH0_PROMPT,
    PAPER_RLM_DEPTH1_PROMPT,
    PAPER_RLM_DEPTH2_PROMPT,
)
from research.eval.trajectory import summarize_trajectory

DEFAULT_VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-1.5-pro")
DEFAULT_VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", None)

GEMINI_HARDENED_SUFFIX = """
Output-format constraint:
Respond only with plain text. If continuing the RLM, emit exactly one fenced
```repl``` Python block. Do not emit JSON, structured calls, API actions, or
special tool-call objects. The names llm_query, rlm_query, and llm_batch are ordinary
Python functions used only inside the REPL.

Execution constraint:
You must not attempt to make external network requests or search the host filesystem for cached datasets. All calculations and text processing must be performed strictly in-memory on the provided `context` variable and through `llm_query` / `rlm_query` / `llm_batch`.
""".strip()

DECOMPOSITION_HINTS_ENV_TIPS = """
<env_tips>

Orchestrate; don't solve. These problems drift on a single chain of
thought (lost partials, compounding sign errors) - "just think harder
in the REPL" scores ~0%
reasoner that can handle any individual sub-problem (competition math,
combinatorics, number theory, probability, geometry, algebra) given a
clear self-contained prompt. Trust it; don't write solver code for it.

Your job: (1) decompose into self-contained "nodes", (2) delegate all
reasoning to `llm_batch`, (3) memoize answers in a dict across turns,
(4) verify each answer before any child consumes it, (5) inline
verified parent values verbatim into child prompts, (6) assemble the
final answer by dict lookup only. You do NO math - if you're writing
Python that enumerates, solves, simulates, or picks among candidates
(vs. verifying one), STOP and delegate. Root compute = dict lookup +
string formatting + correctness checks.

## The only state that matters

Keep two variables alive across every REPL turn:

    answers = {}   # node_id -> VERIFIED answer (string)
    plan    = {}   # JSON structure returned by the planning sub-LM

If a value isn't in `answers`, it doesn't exist. Don't trust variables
from earlier turns, numbers in your own thinking, or pasted values -
context drifts. Memoize everything you'll reuse.

## Step 1 - Plan (turn 1, one `llm_batch` call)

Ask a sub-LM to extract structure as JSON - do not solve anything:

    planning_prompt = (
        "Read the following multi-step problem and return ONLY valid "
        "JSON of the form:\\n"
        '{"nodes":['
        '  {"id":"node_0","question":"<verbatim>","deps":[]},'
        '  {"id":"node_1","question":"<verbatim>","deps":["node_0"]},'
        '  ...'
        '],'
        ' "final":"<how to build the final answer from node answers, '
        '          including the exact output format>",'
        ' "cycles":["<ids of nodes referenced by their own transitive '
        '            deps; [] if none>"]}\\n'
        "Copy each node question VERBATIM - do NOT paraphrase or "
        "simplify wording. Do NOT solve anything.\\n"
        "---\\n"
    ) + FULL_PROBLEM_TEXT
    plan = json.loads(llm_batch([planning_prompt])[0])

For single self-contained puzzles, have the planner split into minimum
self-contained steps (e.g. "parse instance", "run algorithm X",
"format output"). Same workflow applies.

## Step 2 - Solve layer by layer (one `llm_batch` per DAG layer)

A node is "ready" when all its `deps` are in `answers`. Dispatch ALL
ready nodes in ONE `llm_batch` (parallel). Each sub-prompt must be
self-contained - the sub-LM never sees the global problem or the
`answers` dict, so copy the node question verbatim, inline every
parent's verified value verbatim, and ask for only the final value.

    def build_subprompt(node):
        ctx = "\\n".join(f"- {d} = {answers[d]}" for d in node["deps"])
        return (
            "Solve this subproblem in isolation.\\n\\n"
            "Verified parent values (use EXACTLY, do not recompute):\\n"
            f"{ctx or '(none)'}\\n\\n"
            f"Question:\\n{node['question']}\\n\\n"
            "Return ONLY the final value. No prose, no derivation."
        )

    pending = [n for n in plan["nodes"]
               if n["id"] not in plan.get("cycles", [])]
    while pending:
        ready = [n for n in pending
                 if all(d in answers for d in n["deps"])]
        if not ready:
            break  # cycle - see Step 4
        raw = llm_batch([build_subprompt(n) for n in ready])
        for n, a in zip(ready, raw):
            answers[n["id"]] = a.strip()
        pending = [n for n in pending if n["id"] not in answers]

Prefer many small per-layer `llm_batch` calls over one monolithic one.

## Step 3 - Verify every answer before it propagates

Use the cheapest definitive check: (a) independent second opinion -
re-dispatch the node via `llm_batch` with rephrased instructions,
accept only if both agree; (b) plausibility - range / sign / units /
integrality / shape expected downstream. On failure, re-dispatch JUST
that node with the failure reason appended, then re-verify. Never
propagate an unverified answer.

## Step 4 - Cycles

If `plan["cycles"]` is non-empty, pick a seed node `c`, set
`answers[c]` to a candidate, run Step 2 on the rest, check the
cycle-defining constraint. Use `llm_batch` (not hand computation) to
propose the next candidate from the previous miss. Cache trials to
avoid redoing downstream work:

    trials = {}   # candidate -> dict of downstream answers under it

Freeze answers once the constraint is satisfied.

## Step 5 - Assemble

Once every node in `plan["final"]` is verified in `answers`, build the
final string by dict lookup ONLY - no recomputation. You can use
`llm_batch` to aggregate if needed.

    with open("/task/answer.txt", "w") as f:
        f.write(final_answer)

## Red flags (you are off-track)

  - Python doing math (enumerate/solve/sum/factor/simulate/search/
    optimize/Monte Carlo/game trees/Z3/SAT/brute force) instead of
    `llm_batch` -> STOP, delete, delegate.
  - About to use an unverified node answer -> verify first.
  - > 2 turns in, < 3 `llm_batch` calls -> you're solving it yourself.
    Reset.
  - Code running > 30s or > 100 MB -> brute-forcing; delegate instead.
  - Remembering a value not in `answers` -> re-dispatch; working memory
    isn't reliable.
  - About to emit final but `answers` missing a node from
    `plan["final"]` -> dispatch the missing nodes.
  - Many turns on one node without a verified answer -> re-prompt
    `llm_batch` with clearer/longer sub-prompt and failure context.
    Do NOT switch to writing solver code.

## Output contract

Write your final answer to /task/answer.txt - that file is the only
thing scored. Assistant-message content is ignored.

</env_tips>
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RLM on LongCoT-mini.")
    parser.add_argument("--max-depth", type=int, default=1, choices=(0, 1, 2), help="Recursion depth (0=REPL only, 1=single sub-calls, 2=recursive).")
    parser.add_argument("--decomposition-hints", action="store_true", help="Append decomposition hints from Appendix C.3 of paper.")
    parser.add_argument("--domains", nargs="+", default=["all"], help="Domains to run: 'all', or subset of 'logic', 'math', 'cs', 'chemistry', 'chess'.")
    parser.add_argument("--num-per-domain", type=int, default=10, help="Number of questions per domain (-1 for all).")
    parser.add_argument("--model", default=DEFAULT_VERTEX_MODEL, help="Model name.")
    parser.add_argument("--project-id", default=DEFAULT_VERTEX_PROJECT_ID, help="GCP project ID for Vertex AI.")
    parser.add_argument("--location", default="global", help="Vertex AI location.")
    parser.add_argument("--root-thinking-level", default="medium", choices=("low", "medium", "high"), help="Thinking level for root model.")
    parser.add_argument("--sub-thinking-level", default="low", choices=("low", "medium", "high"), help="Thinking level for sub-model.")
    parser.add_argument("--max-iterations", type=int, default=30, help="Maximum REPL turns per task.")
    parser.add_argument("--prompt-variant", choices=("paper_gemini", "paper"), default="paper_gemini", help="Prompt variant.")
    parser.add_argument("--output-dir", default=None, help="Explicit output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-retries-per-task", type=int, default=3, help="Retries on transient network errors.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose execution traces.")
    return parser.parse_args()


def load_longcot_questions(domains: list[str], num_per_domain: int) -> list[Question]:
    all_domains = ["logic", "math", "cs", "chemistry", "chess"]
    target_domains = all_domains if "all" in domains else [d for d in domains if d in all_domains]
    
    questions: list[Question] = []
    data_root = Path(__file__).resolve().parent / "longcot_verifier" / "src" / "data"
    
    for dom in target_domains:
        json_path = data_root / dom / "easy.json"
        if not json_path.exists():
            print(f"Warning: data file not found at {json_path}")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dom_qs = data.get("questions", [])
        if num_per_domain > 0:
            dom_qs = dom_qs[:num_per_domain]
            
        for q in dom_qs:
            questions.append(
                Question(
                    question_id=str(q["question_id"]),
                    domain=dom,
                    difficulty="easy",
                    prompt=q["prompt"],
                    problem=q.get("problem"),
                    answer=q.get("answer"),
                )
            )
    return questions


def main() -> None:
    args = parse_args()
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    total_q_label = f"npd{args.num_per_domain}" if args.num_per_domain > 0 else "all"
    hints_label = "hints" if args.decomposition_hints else "nohints"
    run_id = f"{timestamp}__vertex_openai__longcot_mini__d{args.max_depth}__{total_q_label}__{hints_label}__seed{args.seed}"
    
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = REPO_ROOT / "results" / "longcot" / f"depth-{args.max_depth}_{'hints' if args.decomposition_hints else 'nohints'}"
        
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = run_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    
    questions = load_longcot_questions(args.domains, args.num_per_domain)
    print("=" * 90)
    print(f"STARTING LONGCOT-MINI RUN: Depth-{args.max_depth} (Hints: {args.decomposition_hints})")
    print(f"Run ID: {run_id}")
    print(f"Total Questions: {len(questions)} across domains: {args.domains}")
    print(f"Model: {args.model} | GCP Project: {args.project_id}")
    print("=" * 90)
    
    # Construct Depth-Wise System Prompt (Strictly Paper Matched, With/Without Decomposition Hints)
    if args.max_depth == 0:
        base_prompt = PAPER_RLM_DEPTH0_PROMPT
        suffix = "\n\n" + GEMINI_HARDENED_DEPTH0_SUFFIX if args.prompt_variant == "paper_gemini" else ""
    elif args.max_depth == 1:
        base_prompt = PAPER_RLM_DEPTH1_PROMPT
        suffix = "\n\n" + GEMINI_HARDENED_SUFFIX if args.prompt_variant == "paper_gemini" else ""
    else: # max_depth == 2
        base_prompt = PAPER_RLM_DEPTH2_PROMPT
        suffix = "\n\n" + GEMINI_HARDENED_SUFFIX if args.prompt_variant == "paper_gemini" else ""
        
    hints_block = ("\n\n" + DECOMPOSITION_HINTS_ENV_TIPS.replace("{", "{{").replace("}", "}}")) if args.decomposition_hints else ""
    system_prompt = base_prompt + hints_block + suffix
    
    config_record = {
        "run_id": run_id,
        "benchmark": "longcot_mini",
        "max_depth": args.max_depth,
        "decomposition_hints": args.decomposition_hints,
        "domains": args.domains,
        "num_per_domain": args.num_per_domain,
        "total_questions": len(questions),
        "model": args.model,
        "project_id": args.project_id,
        "max_iterations": args.max_iterations,
        "prompt_variant": args.prompt_variant,
        "timestamp": timestamp,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)
        
    results_path = run_dir / "task_results.jsonl"
    all_results = []
    
    domain_correct = {d: 0 for d in ["logic", "math", "cs", "chemistry", "chess"]}
    domain_total = {d: 0 for d in ["logic", "math", "cs", "chemistry", "chess"]}
    
    start_time_all = time.time()
    
    for idx, q in enumerate(questions, 1):
        task_dir = trajectories_dir / f"{idx:03d}_{q.question_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        log_file = task_dir / f"task_{idx:03d}_{q.question_id}.jsonl"
        logger = RLMLogger(log_dir=task_dir, file_name=log_file.name)
        
        print(f"\n[{idx}/{len(questions)}] Domain: {q.domain} | ID: {q.question_id}")
        
        record = None
        for attempt in range(1, args.max_retries_per_task + 1):
            rlm = None
            try:
                root_backend_kwargs = {
                    "model_name": args.model,
                    "project_id": args.project_id,
                    "location": args.location,
                    "thinking_level": args.root_thinking_level,
                    "malformed_retries": 2,
                    "malformed_retry_delay": 0.5,
                }
                sub_backend_kwargs = {
                    "model_name": args.model,
                    "project_id": args.project_id,
                    "location": args.location,
                    "thinking_level": args.sub_thinking_level,
                    "malformed_retries": 2,
                    "malformed_retry_delay": 0.5,
                }
                rlm = RLM(
                    backend="vertex",
                    backend_kwargs=root_backend_kwargs,
                    other_backends=["vertex"],
                    other_backend_kwargs=[sub_backend_kwargs],
                    max_depth=args.max_depth,
                    max_iterations=args.max_iterations,
                    custom_system_prompt=system_prompt,
                    orchestrator=False, # Pure RLM without OOLONG orchestrator
                    paper_mode=True,    # Enforce depth-wise tool availability strictly
                    logger=logger,
                    verbose=args.verbose,
                )
                
                # Context is the full problem statement.
                # Root prompt asks model to reason and submit final solution.
                root_instruction = (
                    f"You are given a problem from the '{q.domain}' domain in the `context` variable.\n"
                    "Analyze it carefully using the REPL environment, and submit your final answer inside a repl block by setting:\n"
                    "answer[\"content\"] = <final_solution>\n"
                    "answer[\"ready\"] = True\n"
                    f"Question ID: {q.question_id}"
                )
                
                t_start = time.time()
                completion = rlm.completion(q.prompt, root_prompt=root_instruction)
                exec_time = time.time() - t_start
                
                pred_ans = completion.response.strip()
                
                # Verify correctness using canonical verifier
                is_correct = False
                try:
                    is_correct = verify(q, pred_ans)
                except Exception:
                    is_correct = False

                if not is_correct:
                    sol_clean = extract_solution(pred_ans)
                    if sol_clean:
                        try:
                            is_correct = verify(q, sol_clean)
                        except Exception:
                            is_correct = False
                    if not is_correct and not pred_ans.startswith("solution ="):
                        try:
                            is_correct = verify(q, f"solution = {pred_ans}")
                        except Exception:
                            is_correct = False
                            
                score = 1.0 if is_correct else 0.0
                domain_total[q.domain] = domain_total.get(q.domain, 0) + 1
                if is_correct:
                    domain_correct[q.domain] = domain_correct.get(q.domain, 0) + 1
                    
                _, traj_summary = summarize_trajectory(completion.metadata)
                
                cost = getattr(completion.usage_summary, "total_cost", None) or 0.0
                print(f"   -> Result: {'CORRECT [PASS]' if is_correct else 'INCORRECT [FAIL]'} | Time: {exec_time:.1f}s | Cost: ${cost:.4f}")
                
                record = {
                    "question_id": q.question_id,
                    "domain": q.domain,
                    "difficulty": q.difficulty,
                    "status": "completed",
                    "score": score,
                    "is_correct": is_correct,
                    "prediction": pred_ans[:1000],
                    "gold": str(q.answer),
                    "execution_time_s": exec_time,
                    "total_cost_usd": cost,
                    "trajectory": traj_summary,
                    "trajectory_log": str(log_file.relative_to(REPO_ROOT)),
                }
                break
            except Exception as exc:
                err_str = str(exc)
                print(f"Attempt {attempt}/{args.max_retries_per_task} failed: {err_str[:120]}")
                time.sleep(min(60, 5 * (2 ** (attempt - 1))))
                if attempt == args.max_retries_per_task:
                    record = {
                        "question_id": q.question_id,
                        "domain": q.domain,
                        "difficulty": q.difficulty,
                        "status": "failed",
                        "error": err_str,
                        "score": 0.0,
                        "is_correct": False,
                        "prediction": "",
                        "gold": str(q.answer),
                        "trajectory_log": str(log_file.relative_to(REPO_ROOT)),
                    }
            finally:
                if rlm is not None:
                    rlm.close()
                    
        all_results.append(record)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    total_time = time.time() - start_time_all
    total_correct = sum(domain_correct.values())
    total_eval = sum(domain_total.values())
    overall_acc = total_correct / total_eval if total_eval > 0 else 0.0
    
    summary = {
        "run_id": run_id,
        "max_depth": args.max_depth,
        "total_questions": total_eval,
        "overall_accuracy": overall_acc,
        "domain_accuracies": {
            d: (domain_correct[d] / domain_total[d] if domain_total[d] > 0 else 0.0)
            for d in domain_total
        },
        "domain_counts": {d: {"correct": domain_correct[d], "total": domain_total[d]} for d in domain_total},
        "total_time_seconds": total_time,
    }
    
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 90)
    print(f"RUN COMPLETED: Depth-{args.max_depth}")
    print(f"Overall Accuracy: {overall_acc:.2%} ({total_correct}/{total_eval})")
    for d in domain_total:
        d_acc = domain_correct[d] / domain_total[d] if domain_total[d] > 0 else 0.0
        print(f"  {d:<12s}: {d_acc:.2%} ({domain_correct[d]}/{domain_total[d]})")
    print(f"Total Time: {total_time:.1f}s | Results: {results_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
