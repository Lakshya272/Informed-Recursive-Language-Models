"""Paper-aligned OOLONG-Pairs 32K task loader and evaluator.

Implements all 20 synthetic tasks from Appendix D.1 of the RLM paper (arXiv:2512.24601v3).
The tasks require quadratic pairwise reasoning over 220 users in a 32K context window.
Tasks 1-10 are symmetric constraints on both users.
Tasks 11-20 are asymmetric constraints where one user satisfies condition A and the other satisfies condition B.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import dateutil.parser

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "oolong_pairs_contexts" / "oolong_32k_context_0.json"
if not DEFAULT_DATA_PATH.exists():
    DEFAULT_DATA_PATH = Path("oolong_32k_context_0.json")


@dataclass(frozen=True)
class OolongPairsTask:
    task_id: int
    title: str
    query: str
    context: str  # Strictly unlabelled raw context string
    root_prompt: str
    gold_pairs: set[tuple[str, str]]

    @property
    def source_id(self) -> str:
        return f"pairs_task_{self.task_id:02d}"


def parse_predicted_pairs(text: str) -> set[tuple[str, str]]:
    """Extract all (user_id_1, user_id_2) pairs from model response.
    
    Normalizes each pair to (min_id, max_id) so ordering within a pair does not affect set matching.
    """
    pairs = set()
    matches = re.findall(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", text)
    for u1, u2 in matches:
        pair = (min(u1, u2, key=lambda x: int(x)), max(u1, u2, key=lambda x: int(x)))
        pairs.add(pair)
    return pairs


def score_pairs(
    pred_pairs: set[tuple[str, str]], gold_pairs: set[tuple[str, str]]
) -> dict[str, float]:
    """Compute exact Precision, Recall, and F1 score."""
    if not pred_pairs and not gold_pairs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_pairs or not gold_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_pairs.intersection(gold_pairs))
    precision = tp / len(pred_pairs)
    recall = tp / len(gold_pairs)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": float(tp),
        "pred_count": float(len(pred_pairs)),
        "gold_count": float(len(gold_pairs)),
    }


def _build_task_definitions() -> list[dict[str, Any]]:
    """Return all 20 task queries and predicate constructors."""
    return [
        {
            "id": 1,
            "title": "Task 01: symmetric (numeric value or location)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "where both users have at least one instance with a numeric value or location. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "symmetric",
            "pred": lambda insts: any(i["label"] in ("numeric value", "location") for i in insts),
        },
        {
            "id": 2,
            "title": "Task 02: symmetric (entity or human being)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "where both users have at least one instance with an entity or human being. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "symmetric",
            "pred": lambda insts: any(i["label"] in ("entity", "human being") for i in insts),
        },
        {
            "id": 3,
            "title": "Task 03: symmetric (description or abbreviation)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "where both users have at least one instance with a description and abstract concept or abbreviation. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "symmetric",
            "pred": lambda insts: any(i["label"] in ("description and abstract concept", "abbreviation") for i in insts),
        },
        {
            "id": 4,
            "title": "Task 04: symmetric (human or location, human after Jan 6 2023)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "where both users have at least one instance with a human being or location, and all instances "
                "that are a human being for both users must be after January 6, 2023. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "symmetric",
            "pred": lambda insts: any(i["label"] in ("human being", "location") for i in insts)
            and all(i["date"] > datetime.date(2023, 1, 6) for i in insts if i["label"] == "human being"),
        },
        {
            "id": 5,
            "title": "Task 05: symmetric (entity or numeric, entity before Mar 15 2023)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "where both users have at least one instance with an entity or numeric value, and all instances "
                "that are an entity for both users must be before March 15, 2023. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "symmetric",
            "pred": lambda insts: any(i["label"] in ("entity", "numeric value") for i in insts)
            and all(i["date"] < datetime.date(2023, 3, 15) for i in insts if i["label"] == "entity"),
        },
        {
            "id": 6,
            "title": "Task 06: symmetric (location or abbreviation)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "where both users have at least one instance with a location or abbreviation. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "symmetric",
            "pred": lambda insts: any(i["label"] in ("location", "abbreviation") for i in insts),
        },
        {
            "id": 7,
            "title": "Task 07: symmetric (description or numeric, numeric after Feb 1 2023)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "where both users have at least one instance with a description and abstract concept or numeric value, "
                "and all instances that are a numeric value for both users must be after February 1, 2023. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "symmetric",
            "pred": lambda insts: any(i["label"] in ("description and abstract concept", "numeric value") for i in insts)
            and all(i["date"] > datetime.date(2023, 2, 1) for i in insts if i["label"] == "numeric value"),
        },
        {
            "id": 8,
            "title": "Task 08: symmetric (human or description)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "where both users have at least one instance with a human being or description and abstract concept. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "symmetric",
            "pred": lambda insts: any(i["label"] in ("human being", "description and abstract concept") for i in insts),
        },
        {
            "id": 9,
            "title": "Task 09: symmetric (entity or location, location after Apr 10 2023)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "where both users have at least one instance with an entity or location, and all instances "
                "that are a location for both users must be after April 10, 2023. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "symmetric",
            "pred": lambda insts: any(i["label"] in ("entity", "location") for i in insts)
            and all(i["date"] > datetime.date(2023, 4, 10) for i in insts if i["label"] == "location"),
        },
        {
            "id": 10,
            "title": "Task 10: symmetric (numeric or abbreviation, abbreviation before May 20 2023)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "where both users have at least one instance with a numeric value or abbreviation, and all instances "
                "that are an abbreviation for both users must be before May 20, 2023. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "symmetric",
            "pred": lambda insts: any(i["label"] in ("numeric value", "abbreviation") for i in insts)
            and all(i["date"] < datetime.date(2023, 5, 20) for i in insts if i["label"] == "abbreviation"),
        },
        {
            "id": 11,
            "title": "Task 11: asymmetric (A: >=1 entity & >=1 abbr, B: ==1 entity)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "such that one user has at least one instance with entity and one with abbreviation, "
                "and the other user has exactly one instance with entity. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "asymmetric",
            "pred_a": lambda insts: any(i["label"] == "entity" for i in insts) and any(i["label"] == "abbreviation" for i in insts),
            "pred_b": lambda insts: sum(1 for i in insts if i["label"] == "entity") == 1,
        },
        {
            "id": 12,
            "title": "Task 12: asymmetric (A: >=2 numeric, B: >=1 location & >=1 human)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "such that one user has at least two instances with numeric value, and the other user has "
                "at least one instance with location and at least one instance with human being. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "asymmetric",
            "pred_a": lambda insts: sum(1 for i in insts if i["label"] == "numeric value") >= 2,
            "pred_b": lambda insts: any(i["label"] == "location" for i in insts) and any(i["label"] == "human being" for i in insts),
        },
        {
            "id": 13,
            "title": "Task 13: asymmetric (A: ==1 description, B: >=1 abbr & >=1 entity)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "such that one user has exactly one instance with description and abstract concept, "
                "and the other user has at least one instance with abbreviation and at least one instance with entity. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "asymmetric",
            "pred_a": lambda insts: sum(1 for i in insts if i["label"] == "description and abstract concept") == 1,
            "pred_b": lambda insts: any(i["label"] == "abbreviation" for i in insts) and any(i["label"] == "entity" for i in insts),
        },
        {
            "id": 14,
            "title": "Task 14: asymmetric (A: >=1 human & >=1 numeric, B: ==2 location)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "such that one user has at least one instance with human being and at least one instance with numeric value, "
                "and the other user has exactly two instances with location. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "asymmetric",
            "pred_a": lambda insts: any(i["label"] == "human being" for i in insts) and any(i["label"] == "numeric value" for i in insts),
            "pred_b": lambda insts: sum(1 for i in insts if i["label"] == "location") == 2,
        },
        {
            "id": 15,
            "title": "Task 15: asymmetric (A: >=1 entity & >=1 location & >=1 abbr, B: ==1 numeric)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "such that one user has at least one instance with entity, at least one instance with location, "
                "and at least one instance with abbreviation, and the other user has exactly one instance with numeric value. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "asymmetric",
            "pred_a": lambda insts: any(i["label"] == "entity" for i in insts)
            and any(i["label"] == "location" for i in insts)
            and any(i["label"] == "abbreviation" for i in insts),
            "pred_b": lambda insts: sum(1 for i in insts if i["label"] == "numeric value") == 1,
        },
        {
            "id": 16,
            "title": "Task 16: asymmetric (A: >=1 description & >=1 human, B: >=2 entity & ==1 abbr)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "such that one user has at least one instance with description and abstract concept and at least one instance "
                "with human being, and the other user has at least two instances with entity and exactly one instance with abbreviation. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "asymmetric",
            "pred_a": lambda insts: any(i["label"] == "description and abstract concept" for i in insts)
            and any(i["label"] == "human being" for i in insts),
            "pred_b": lambda insts: sum(1 for i in insts if i["label"] == "entity") >= 2
            and sum(1 for i in insts if i["label"] == "abbreviation") == 1,
        },
        {
            "id": 17,
            "title": "Task 17: asymmetric (A: ==1 numeric, B: >=1 location & >=1 description)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "such that one user has exactly one instance with numeric value, and the other user has "
                "at least one instance with location and at least one instance with description and abstract concept. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "asymmetric",
            "pred_a": lambda insts: sum(1 for i in insts if i["label"] == "numeric value") == 1,
            "pred_b": lambda insts: any(i["label"] == "location" for i in insts)
            and any(i["label"] == "description and abstract concept" for i in insts),
        },
        {
            "id": 18,
            "title": "Task 18: asymmetric (A: >=1 abbr & ==1 human, B: >=1 entity & >=1 numeric)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "such that one user has at least one instance with abbreviation and exactly one instance with human being, "
                "and the other user has at least one instance with entity and at least one instance with numeric value. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "asymmetric",
            "pred_a": lambda insts: any(i["label"] == "abbreviation" for i in insts)
            and sum(1 for i in insts if i["label"] == "human being") == 1,
            "pred_b": lambda insts: any(i["label"] == "entity" for i in insts)
            and any(i["label"] == "numeric value" for i in insts),
        },
        {
            "id": 19,
            "title": "Task 19: asymmetric (A: >=2 location & >=1 entity, B: ==1 description & ==1 abbr)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "such that one user has at least two instances with location and at least one instance with entity, "
                "and the other user has exactly one instance with description and abstract concept and exactly one instance with abbreviation. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "asymmetric",
            "pred_a": lambda insts: sum(1 for i in insts if i["label"] == "location") >= 2
            and any(i["label"] == "entity" for i in insts),
            "pred_b": lambda insts: sum(1 for i in insts if i["label"] == "description and abstract concept") == 1
            and sum(1 for i in insts if i["label"] == "abbreviation") == 1,
        },
        {
            "id": 20,
            "title": "Task 20: asymmetric (A: >=1 numeric & >=1 human, B: >=1 location & >=1 entity & ==1 abbr)",
            "query": (
                "In the above data, list all pairs of user IDs (no duplicate pairs, list lower ID first) "
                "such that one user has at least one instance with numeric value and at least one instance with human being, "
                "and the other user has at least one instance with location, at least one instance with entity, "
                "and exactly one instance with abbreviation. "
                "Each of the questions can be labelled as one of the labels (the data does not provide the labels, "
                "you need to figure out the label from the semantics of the question): description and abstract concept, "
                "entity, human being, numeric value, location, abbreviation. In your answer, list all pairs in the "
                "format (user_id_1, user_id_2), separated by newlines."
            ),
            "type": "asymmetric",
            "pred_a": lambda insts: any(i["label"] == "numeric value" for i in insts)
            and any(i["label"] == "human being" for i in insts),
            "pred_b": lambda insts: any(i["label"] == "location" for i in insts)
            and any(i["label"] == "entity" for i in insts)
            and sum(1 for i in insts if i["label"] == "abbreviation") == 1,
        },
    ]


def load_pairs_tasks(
    data_path: Path = DEFAULT_DATA_PATH,
    task_indices: list[int] | None = None,
) -> list[OolongPairsTask]:
    """Load the 32K context and build the 20 paper tasks with their gold pairs."""
    if not data_path.exists():
        raise FileNotFoundError(f"OOLONG-Pairs 32K dataset file not found at: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # context is the raw, unlabelled text of user instances
    raw_context = payload["context"]
    labeled_context = payload["context_with_labels"]

    # Parse labeled context internally to establish ground-truth sets
    parsed_items = []
    pattern = re.compile(
        r"Date:\s*(.*?)\s*\|\|\s*User:\s*(.*?)\s*\|\|\s*Instance:\s*(.*?)\s*\|\|\s*Label:\s*(.*)"
    )
    for line in labeled_context.split("\n"):
        m = pattern.match(line.strip())
        if m:
            dt = dateutil.parser.parse(m.group(1)).date()
            parsed_items.append({
                "date": dt,
                "user": m.group(2).strip(),
                "instance": m.group(3).strip(),
                "label": m.group(4).strip(),
            })

    users_map = defaultdict(list)
    for item in parsed_items:
        users_map[item["user"]].append(item)

    user_ids = sorted(list(users_map.keys()), key=lambda x: int(x))

    def _get_pairs_symmetric(predicate: Callable[[list[dict]], bool]) -> set[tuple[str, str]]:
        valid = [u for u in user_ids if predicate(users_map[u])]
        pairs = set()
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                u1, u2 = valid[i], valid[j]
                pairs.add((min(u1, u2, key=lambda x: int(x)), max(u1, u2, key=lambda x: int(x))))
        return pairs

    def _get_pairs_asymmetric(
        pred_a: Callable[[list[dict]], bool], pred_b: Callable[[list[dict]], bool]
    ) -> set[tuple[str, str]]:
        users_a = set(u for u in user_ids if pred_a(users_map[u]))
        users_b = set(u for u in user_ids if pred_b(users_map[u]))
        pairs = set()
        for u1 in user_ids:
            for u2 in user_ids:
                if int(u1) < int(u2):
                    if (u1 in users_a and u2 in users_b) or (u1 in users_b and u2 in users_a):
                        pairs.add((u1, u2))
        return pairs

    all_defs = _build_task_definitions()
    tasks: list[OolongPairsTask] = []

    for item in all_defs:
        tid = item["id"]
        if task_indices is not None and tid not in task_indices:
            continue

        if item["type"] == "symmetric":
            gold_pairs = _get_pairs_symmetric(item["pred"])
        else:
            gold_pairs = _get_pairs_asymmetric(item["pred_a"], item["pred_b"])

        # Root prompt gives the query clearly. Context is loaded strictly into REPL `context` variable.
        root_prompt = f"Question: {item['query']}"

        tasks.append(
            OolongPairsTask(
                task_id=tid,
                title=item["title"],
                query=item["query"],
                context=raw_context,
                root_prompt=root_prompt,
                gold_pairs=gold_pairs,
            )
        )

    return tasks
