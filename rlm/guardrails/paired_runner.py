"""Build 3: Logging & Paired Comparison Harness.

Per task:
- M1's label, confidence, reasoning
- whether the corresponding M2 check fired
- what it changed (if anything) between draft answer and final answer
- the final answer with/without the check active (paired comparison on the same task)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from rlm.guardrails.m1_classifier import M1Classification, classify_task
from rlm.guardrails.m2_intercepts import M2State


@dataclass
class TaskRunRecord:
    condition: str  # "baseline" (M1/M2 disabled) or "treatment" (M1/M2 enabled)
    final_answer: str
    score: float | None = None
    is_correct: bool | None = None
    execution_time_s: float = 0.0
    total_calls: int = 0
    m1_category: str = "none"
    m1_confidence: str = "low"
    m1_reasoning: str = ""
    m2_fired: bool = False
    m2_check_type: str | None = None
    m2_changes_made: bool = False
    draft_answer_before_m2: str | None = None
    injected_m2_message: str | None = None
    extra_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairedTaskComparison:
    task_id: str
    query_text: str
    context_metadata: dict[str, Any]
    m1_classification: M1Classification
    baseline_run: TaskRunRecord
    treatment_run: TaskRunRecord
    m2_fired: bool = False
    m2_changed_answer: bool = False
    accuracy_delta: float | None = None  # treatment_score - baseline_score

    def __post_init__(self):
        self.m2_fired = self.treatment_run.m2_fired
        self.m2_changed_answer = (
            self.baseline_run.final_answer.strip() != self.treatment_run.final_answer.strip()
        )
        if self.baseline_run.score is not None and self.treatment_run.score is not None:
            self.accuracy_delta = self.treatment_run.score - self.baseline_run.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "query_text": self.query_text,
            "context_metadata": self.context_metadata,
            "m1_classification": self.m1_classification.to_dict(),
            "m2_fired": self.m2_fired,
            "m2_changed_answer": self.m2_changed_answer,
            "accuracy_delta": self.accuracy_delta,
            "baseline": self.baseline_run.to_dict(),
            "treatment": self.treatment_run.to_dict(),
        }


class PairedHarnessLogger:
    """Manages recording and disk serialization of paired M1/M2 experiment evaluations."""

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[PairedTaskComparison] = []

    def record_pair(self, pair: PairedTaskComparison) -> None:
        self.records.append(pair)
        if self.output_dir:
            out_file = self.output_dir / "paired_eval_results.jsonl"
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(pair.to_dict()) + "\n")

    def get_summary(self) -> dict[str, Any]:
        total = len(self.records)
        if total == 0:
            return {"total_tasks": 0}

        m1_counts: dict[str, int] = {}
        m2_fired_count = 0
        m2_changed_count = 0
        baseline_correct = 0
        treatment_correct = 0

        for r in self.records:
            cat = r.m1_classification.category
            m1_counts[cat] = m1_counts.get(cat, 0) + 1
            if r.m2_fired:
                m2_fired_count += 1
            if r.m2_changed_answer:
                m2_changed_count += 1
            if r.baseline_run.is_correct:
                baseline_correct += 1
            if r.treatment_run.is_correct:
                treatment_correct += 1

        return {
            "total_tasks": total,
            "m1_distribution": m1_counts,
            "m2_fired_total": m2_fired_count,
            "m2_fired_rate": m2_fired_count / total,
            "m2_changed_answer_total": m2_changed_count,
            "m2_changed_answer_rate": m2_changed_count / total,
            "baseline_accuracy": baseline_correct / total if total > 0 else 0.0,
            "treatment_accuracy": treatment_correct / total if total > 0 else 0.0,
            "accuracy_lift": (treatment_correct - baseline_correct) / total if total > 0 else 0.0,
        }

    def save_summary(self, filename: str = "summary.json") -> Path | None:
        if not self.output_dir:
            return None
        summary_path = self.output_dir / filename
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2)
        return summary_path
