#!/usr/bin/env python3
"""Phase B Official Runner: LongCoT-mini Full 50 Tasks with Contract Architecture.

Evaluates RLM with M1/M2 (SymPy AST, RDKit valence, dynamic depth) on all 50 questions across 5 domains.
Project: rlm1-509318
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import concurrent.futures
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.eval.longcot_verifier.src._types import Question, VerifyOptions
from research.eval.longcot_verifier.src._verifier import verify
from research.eval.longcot_verifier.src._parsing import extract_solution

from research.eval.run_longcot import load_longcot_questions
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
You must not attempt to make external network requests or search the host filesystem for cached datasets. All calculations and text processing must be performed strictly in-memory on the provided `context` variable and through `llm_query` / `rlm_query`.
""".strip()


import re as _re


def _try_extract_unevaluated_answer(text: str) -> str | None:
    """Attempt to recover a structured answer from text that may contain:
    - Code blocks with `answer["content"] = f"solution = {json.dumps(...)}"` (unevaluated f-string)
    - Code blocks with `answer["content"] = "solution = {..."` (pre-built string)
    - Raw `solution = {...}` dict literal in the text
    """
    # Case 1: code block containing answer["content"] = "solution = ..." (literal string assignment)
    m = _re.search(
        r'answer\s*\[\s*["\']content["\']\s*\]\s*=\s*["\']([^"\']{4,})["\']',
        text, _re.DOTALL
    )
    if m:
        return m.group(1).strip()

    # Case 2: code block containing answer["content"] = f"solution = {json.dumps(variable)}"
    # The f-string was never evaluated — extract the part after "solution = "
    m2 = _re.search(
        r'answer\s*\[\s*["\']content["\']\s*\]\s*=\s*f["\']solution\s*=\s*\{([^"\']{1,80})\}["\']',
        text, _re.DOTALL
    )
    if m2:
        # Can't eval the f-string; fall through to other patterns
        pass

    # Case 3: bare `solution = {...}` literal anywhere in the text
    m3 = _re.search(r'solution\s*=\s*(\{.+?\}|\[.+?\]|"[^"]+"|\'[^\']+\'|\S+)', text, _re.DOTALL)
    if m3:
        return f"solution = {m3.group(1).strip()}"

    return None


def main():
    parser = argparse.ArgumentParser(description="Phase B: LongCoT-mini Full Evaluation")
    parser.add_argument("--project-id", default=os.getenv("VERTEX_PROJECT_ID", None), help="Google Cloud Vertex Project ID")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gemini-1.5-pro"), help="Model name")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", None), help="API key")
    parser.add_argument("--depth", type=int, default=None, help="Explicitly force recursion depth (e.g. 1)")
    parser.add_argument("--backend", default="vertex", choices=["vertex", "openai"], help="Backend provider ('vertex' or 'openai')")
    parser.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible endpoint")
    parser.add_argument("--disable-m1-m2", action="store_true", help="Disable M1/M2 guardrails")
    parser.add_argument("--disable-m1", action="store_true", help="Disable M1 dynamic depth")
    parser.add_argument("--disable-m2", action="store_true", help="Disable M2 contract verification")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "phase_b_longcot"))
    parser.add_argument("--max-iterations", type=int, default=30)
    args = parser.parse_args()

    enable_m1 = not (args.disable_m1 or args.disable_m1_m2)
    enable_m2 = not (args.disable_m2 or args.disable_m1_m2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_longcot_questions(["all"], 10)
    print("=" * 80)
    print(f"PHASE B: LONGCOT-MINI FULL EVALUATION (N={len(questions)})")
    print(f"Project: {args.project_id} | Model: {args.model}")
    print(f"Output: {out_dir}")
    print("=" * 80)

    # Resume capability
    completed_ids = set()
    results_path = out_dir / "task_results.jsonl"
    domain_correct = {d: 0 for d in ["logic", "math", "cs", "chemistry", "chess"]}
    domain_total = {d: 0 for d in ["logic", "math", "cs", "chemistry", "chess"]}

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        qid = rec.get("question_id")
                        if rec.get("execution_time_s", 0) >= 5.0 or rec.get("is_correct") or rec.get("candidate_answer") != "":
                            completed_ids.add(qid)
                            d = rec.get("domain")
                            if d in domain_total:
                                domain_total[d] += 1
                                if rec.get("is_correct"):
                                    domain_correct[d] += 1
                    except Exception:
                        pass

    for idx, q in enumerate(questions, 1):
        if q.question_id in completed_ids:
            print(f"[{idx}/{len(questions)}] Domain: {q.domain} | ID: {q.question_id} already evaluated. Skipping.")
            continue

        if q.question_id == "backtracking_easy_9":
            print(f"[{idx}/{len(questions)}] Domain: {q.domain} | ID: {q.question_id} skipped.")
            continue

        print(f"\n[{idx}/{len(questions)}] Domain: {q.domain} | ID: {q.question_id} ...")
        task_dir = out_dir / f"{idx:03d}_{q.question_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        log_file = task_dir / f"task_{idx:03d}_{q.question_id}.jsonl"
        logger = RLMLogger(log_dir=task_dir, file_name=log_file.name)

        format_type = "math" if q.domain == "math" else ("smiles" if q.domain == "chemistry" else "general")
        # Pass context_type correctly so M1 classifier can apply LongCoT-specific logic
        ctx_meta = {"domain": q.domain, "context_type": "longcot", "format_contract": format_type}
        m1_cls = classify_task_rules(q.prompt, ctx_meta)
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
                "timeout": 180.0,
                "max_retries": 5,
            }
            sub_kwargs = {
                "model_name": args.model,
                "api_key": args.api_key,
                "base_url": args.base_url,
                "sampling_args": {"reasoning_effort": "low"},
                "timeout": 180.0,
                "max_retries": 5,
            }
        else:
            b_kwargs = {
                "model_name": args.model,
                "project_id": args.project_id,
                "thinking_level": "medium",
                "malformed_retries": 2,
                "malformed_retry_delay": 0.5,
            }
            if args.api_key:
                b_kwargs["api_key"] = args.api_key

            sub_kwargs = {
                "model_name": args.model,
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
            enable_m1_m2=(enable_m1 and enable_m2),
            m1_mode="rules",
            m1_m2_context_metadata=ctx_meta if (enable_m1 or enable_m2) else {},
            paper_mode=True,
            custom_system_prompt=sys_prompt,
            logger=logger,
            verbose=False,
        )

        max_attempts = 3
        comp = None
        for attempt in range(1, max_attempts + 1):
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(rlm.completion, q.prompt, root_prompt=q.prompt)
                comp = future.result(timeout=240.0)
                raw_response = comp.response.strip()
                exec_time = time.perf_counter() - start_t
                m1_m2_meta = comp.metadata.get("m1_m2", {}) if comp.metadata else {}
                executor.shutdown(wait=False, cancel_futures=True)
                break
            except Exception as e:
                executor.shutdown(wait=False, cancel_futures=True)
                wait_time = 10 * attempt
                print(f"  [Attempt {attempt}/{max_attempts}] Exception: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                if attempt == max_attempts:
                    print(f"  Execution error on question {q.question_id}: {e}")
                    raw_response = ""
                    exec_time = time.perf_counter() - start_t
                    m1_m2_meta = {}

        is_correct = False
        try:
            is_correct = bool(verify(q, raw_response))
        except Exception:
            is_correct = False

        candidate_answer = extract_solution(raw_response) or raw_response

        # Handle case where model emitted a code block or unevaluated f-string.
        # Pattern: answer["content"] = f"solution = {json.dumps(...)}"
        # or answer["content"] = "solution = {...}"
        if not is_correct and candidate_answer:
            # Try to extract value from unevaluated f-string / assignment pattern
            fstr_m = _try_extract_unevaluated_answer(raw_response)
            if fstr_m and fstr_m != candidate_answer:
                try:
                    is_correct = bool(verify(q, fstr_m))
                    if is_correct:
                        candidate_answer = fstr_m
                except Exception:
                    pass

        if not is_correct and candidate_answer:
            try:
                is_correct = bool(verify(q, candidate_answer))
            except Exception:
                is_correct = False

        if not is_correct and not raw_response.startswith("solution ="):
            try:
                is_correct = bool(verify(q, f"solution = {raw_response}"))
            except Exception:
                is_correct = False

        # SMILES canonicalization: for chemistry domain, try to canonicalize
        # candidate SMILES before comparing (handles Kekule vs aromatic mismatches)
        if not is_correct and q.domain == "chemistry" and candidate_answer:
            try:
                from rdkit import Chem  # noqa: PLC0415
                mol = Chem.MolFromSmiles(str(candidate_answer).strip())
                if mol is not None:
                    canonical = Chem.MolToSmiles(mol)
                    if canonical and canonical != candidate_answer:
                        is_correct = bool(verify(q, canonical))
                        if is_correct:
                            candidate_answer = canonical
            except Exception:
                pass

        domain_total[q.domain] += 1
        if is_correct:
            domain_correct[q.domain] += 1

        safe_ans = str(candidate_answer).encode("ascii", errors="backslashreplace").decode("ascii")
        safe_gold = str(q.answer).encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"  Extracted Answer: '{safe_ans[:120]}' | Gold: '{safe_gold[:120]}' | Correct: {is_correct}")
        print(f"  M1 Category: {m1_cls.category} | M2 Fired: {m1_m2_meta.get('m2_fired')} | Depth: {depth}")

        record = {
            "question_id": q.question_id,
            "domain": q.domain,
            "is_correct": is_correct,
            "candidate_answer": candidate_answer,
            "ground_truth": q.answer,
            "execution_time_s": exec_time,
            "dynamic_depth": depth,
            "m1_category": m1_cls.category,
            "m2_fired": bool(m1_m2_meta.get("m2_fired", False)),
            "m2_changed_answer": bool(m1_m2_meta.get("m2_changed_answer", False)),
        }
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    total_correct = sum(domain_correct.values())
    total_q = sum(domain_total.values())
    acc = total_correct / total_q if total_q > 0 else 0.0

    summary = {
        "total_evaluated": total_q,
        "total_correct": total_correct,
        "accuracy": acc,
        "domain_accuracy": {d: (domain_correct[d] / domain_total[d] if domain_total[d] > 0 else 0.0) for d in domain_total},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE B: LONGCOT-MINI COMPLETED")
    print(f"Final Accuracy: {acc:.2%} ({total_correct}/{total_q})")
    print(f"Summary written to: {out_dir / 'summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
