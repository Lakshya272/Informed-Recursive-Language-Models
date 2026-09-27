"""Paper-aligned LongBench-v2 CodeQA loading and scoring."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "longbench_v2" / "data.json"

LETTER_TO_INDEX = {"A": "0", "B": "1", "C": "2", "D": "3"}
INDEX_TO_LETTER = {"0": "A", "1": "B", "2": "C", "3": "D"}


@dataclass(frozen=True)
class CodeQATask:
    task_id: str
    question: str
    choices: list[str]  # [choice_0, choice_1, choice_2, choice_3]
    gold_index: str     # "0", "1", "2", "3"
    gold_letter: str    # "A", "B", "C", "D"
    context: str        # Repository text
    prompt: str         # Full query prompt passed to RLM


def format_codeqa_query(question: str, choices: list[str]) -> str:
    """Format the CodeQA query exactly as described in Appendix E.4 of arXiv:2512.24601."""
    choices_str = "\n".join(f"{i}: {c}" for i, c in enumerate(choices))
    return (
        "You are a helpful assistant that can answer questions about code repositories. "
        f"You must answer the given question: {question} based on the stored context "
        "answer with exactly one number choice using only the choices provided:\n"
        f"{choices_str}\n"
        "(indexed from 0 to 3).\n"
        "Provide your final answer using FINAL(choice_number), e.g. FINAL(0)."
    )


def load_tasks(
    data_path: Path | str = DEFAULT_DATA_PATH,
    num_examples: int | None = None,
    seed: int = 42,
) -> list[CodeQATask]:
    """Load CodeQA tasks from LongBench-v2 data.json."""
    data_path = Path(data_path)
    cache_path = data_path.parent / "codeqa_tasks_cache.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif data_path.exists():
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        print(f"Local LongBench-v2 dataset not found at {data_path}. Downloading from Hugging Face ('THUDM/LongBench-v2')...")
        from datasets import load_dataset
        ds = load_dataset("THUDM/LongBench-v2", split="train")
        data = [dict(row) for row in ds]
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(data_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    # LongBench-v2 has domain/category/task fields
    tasks: list[CodeQATask] = []
    for item in data:
        domain = item.get("domain", "").lower()
        subtask = item.get("subtask", item.get("category", "")).lower()
        
        # Identify Code Repository Understanding (CodeQA)
        is_codeqa = (
            "code" in domain or 
            "code" in subtask or 
            "repository" in subtask or
            item.get("task", "").lower() == "codeqa"
        )
        if not is_codeqa:
            continue

        q_id = str(item.get("_id", item.get("id", len(tasks))))
        question = item.get("question", "")
        
        # Choices can be choice_A..D or choices list
        if "choice_A" in item:
            choices = [
                str(item.get("choice_A", "")),
                str(item.get("choice_B", "")),
                str(item.get("choice_C", "")),
                str(item.get("choice_D", "")),
            ]
        elif "choices" in item:
            choices = [str(c) for c in item["choices"]]
        else:
            continue

        raw_ans = str(item.get("answer", "")).strip().upper()
        if raw_ans in LETTER_TO_INDEX:
            gold_letter = raw_ans
            gold_index = LETTER_TO_INDEX[raw_ans]
        elif raw_ans in INDEX_TO_LETTER:
            gold_index = raw_ans
            gold_letter = INDEX_TO_LETTER[raw_ans]
        else:
            gold_letter = raw_ans
            gold_index = raw_ans

        context = item.get("context", "")
        prompt = format_codeqa_query(question, choices)

        tasks.append(
            CodeQATask(
                task_id=q_id,
                question=question,
                choices=choices,
                gold_index=gold_index,
                gold_letter=gold_letter,
                context=context,
                prompt=prompt,
            )
        )

    if seed is not None:
        random.seed(seed)
        random.shuffle(tasks)

    if num_examples is not None and num_examples > 0:
        tasks = tasks[:num_examples]

    return tasks


def score_answer(task: CodeQATask, prediction_text: str) -> float:
    """Score model prediction against task gold choice."""
    if not prediction_text:
        return 0.0

    pred = str(prediction_text).strip()

    # Direct match with index "0", "1", "2", "3" or letter "A", "B", "C", "D"
    if pred == task.gold_index or pred.upper() == task.gold_letter:
        return 1.0

    # Match FINAL(...) or FINAL_VAR(...)
    final_m = re.search(r"FINAL\s*\(\s*['\"]?([0-3A-Da-d])['\"]?\s*\)", pred)
    if final_m:
        val = final_m.group(1).upper()
        if val == task.gold_index or val == task.gold_letter:
            return 1.0

    # Match patterns like "Answer: 1", "Choice: 1", "Option 1", "Choice B", "1"
    match = re.search(r"(?:answer|choice|option|result|is|selected)\s*[:=]?\s*([0-3A-Da-d])\b", pred, re.IGNORECASE)
    if match:
        val = match.group(1).upper()
        if val == task.gold_index or val == task.gold_letter:
            return 1.0

    # Search for standalone single token at start or end of text
    tokens = re.findall(r"\b([0-3A-Da-d])\b", pred)
    if tokens:
        # Check the last mentioned choice
        last_tok = tokens[-1].upper()
        if last_tok == task.gold_index or last_tok == task.gold_letter:
            return 1.0

    return 0.0
