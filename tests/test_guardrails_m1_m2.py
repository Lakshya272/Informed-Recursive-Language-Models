"""Comprehensive unit and integration tests for M1/M2 Guardrail System.

Tests:
1. M1 Classifier (rule-based and LLM-based, parsing contract, confidence behavior)
2. M2 Intercept Checks (boundary_recheck, format_check, evidence_preference)
3. M2 Single One-Shot Invariant (one-bounce contract, no re-firing)
4. Paired Comparison Harness & Logging
5. End-to-End RLM Integration with MockLM (offline, no API calls)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rlm.core.rlm import RLM
from rlm.guardrails.m1_classifier import (
    M1Classification,
    classify_task,
    classify_task_llm,
    classify_task_rules,
)
from rlm.guardrails.m2_intercepts import (
    M2State,
    boundary_recheck,
    evidence_preference,
    format_check,
    evaluate_m2_intercept,
)
from rlm.guardrails.paired_runner import (
    PairedHarnessLogger,
    PairedTaskComparison,
    TaskRunRecord,
)
from rlm.guardrails.prompts import (
    M1_SYSTEM_PROMPT,
    M1_USER_PROMPT_TEMPLATE,
    M2_BOUNDARY_RECHECK_TEMPLATE,
    M2_EVIDENCE_PREFERENCE_TEMPLATE,
    M2_FORMAT_CHECK_TEMPLATE,
)
from tests.mock_lm import MockLM


# =========================================================================
# 1. M1 Classifier Tests
# =========================================================================

def test_m1_rules_nonmonotonic_predicate():
    queries = [
        "How many instances of user login are there in total?",
        "List all pairs where both satisfy exactly one numeric property",
        "What is the exact count of transactions before May 20?",
        "Find pairwise matching users with symmetric conditions",
    ]
    for q in queries:
        res = classify_task_rules(q)
        assert res.category == "nonmonotonic_predicate", f"Failed on: {q}"
        assert res.is_active_for_m2 is True
        assert res.confidence in {"medium", "high"}


def test_m1_rules_format_completeness():
    queries = [
        "Calculate the probability and express as a fraction in simplified form.",
        "Provide the final SMILES string for the synthesized molecule.",
        "Give the answer in the form X = 42 without extraneous text.",
        "Solve the differential equation and provide the exact LaTeX notation.",
    ]
    for q in queries:
        res = classify_task_rules(q)
        assert res.category == "format_completeness", f"Failed on: {q}"
        assert res.is_active_for_m2 is True
        assert res.confidence in {"medium", "high"}


def test_m1_rules_evidence_grounded():
    queries = [
        "Does this repo support external PETSc Krylov solvers in the build scripts?",
        "What does this codebase actually implement in the data loaders?",
        "Which classes in lvdm maintain temporal consistency versus motion dynamics?",
        "According to the provided documents, what are the advantages of this codebase?",
    ]
    for q in queries:
        res = classify_task_rules(q)
        assert res.category == "evidence_grounded", f"Failed on: {q}"
        assert res.is_active_for_m2 is True
        assert res.confidence in {"medium", "high"}


def test_m1_rules_none_category():
    queries = [
        "What is the capital of France?",
        "Explain the high-level concept of transformer attention mechanisms.",
        "Write a summary of this conversation.",
    ]
    for q in queries:
        res = classify_task_rules(q)
        assert res.category == "none"
        assert res.is_active_for_m2 is False


def test_m1_parsing_contract_low_confidence():
    """Parsing contract: treat confidence: 'low' the same as category: 'none' for M2."""
    c1 = M1Classification(
        category="nonmonotonic_predicate",
        confidence="low",
        reasoning="Ambiguous mention",
        raw_response="test",
    )
    assert c1.effective_category == "none"
    assert c1.is_active_for_m2 is False

    c2 = M1Classification(
        category="format_completeness",
        confidence="medium",
        reasoning="Clear format spec",
        raw_response="test",
    )
    assert c2.effective_category == "format_completeness"
    assert c2.is_active_for_m2 is True


def test_m1_llm_classification_mock():
    # Mock LLM returning valid JSON
    mock_resp = json.dumps({
        "category": "nonmonotonic_predicate",
        "confidence": "high",
        "reasoning": "Question asks for exact count of matching pairs."
    })
    mock_lm = MockLM(responses=[mock_resp])
    res = classify_task_llm(mock_lm, "How many pairs satisfy the count condition?")
    assert res.category == "nonmonotonic_predicate"
    assert res.confidence == "high"
    assert res.is_active_for_m2 is True


def test_m1_llm_classification_markdown_code_fence():
    # Mock LLM returning markdown code block
    mock_resp = "```json\n" + json.dumps({
        "category": "evidence_grounded",
        "confidence": "medium",
        "reasoning": "Checks repository implementation."
    }) + "\n```"
    mock_lm = MockLM(responses=[mock_resp])
    res = classify_task_llm(mock_lm, "Does this repository support CUDA?")
    assert res.category == "evidence_grounded"
    assert res.confidence == "medium"
    assert res.is_active_for_m2 is True


def test_m1_llm_classification_fallback_on_error():
    # Mock LLM returning completely unparseable garbage
    mock_lm = MockLM(responses=["NOT_JSON_AT_ALL"])
    res = classify_task_llm(mock_lm, "How many exact pairs exist in the dataset?")
    # Should fall back to rule-based classification cleanly
    assert res.category == "nonmonotonic_predicate"
    assert "LLM fallback" in res.reasoning


# =========================================================================
# 2. M2 Intercept Checks Tests
# =========================================================================

def test_m2_boundary_recheck_verbatim_prompt():
    disagreeing = [
        {"item": "User 14916", "pass_1": "Satisfies >=1 entity", "pass_2": "0 entity matches"},
        {"item": "User 22009", "pass_1": "Matches numeric=1", "pass_2": "Matches numeric=2"},
    ]
    res = boundary_recheck(disagreeing)
    assert res.should_bounce is True
    assert res.check_type == "boundary_recheck"
    assert "The following item(s) were classified inconsistently" in res.injected_message
    assert "User 14916: Pass 1 classified as 'Satisfies >=1 entity'; Pass 2 classified as '0 entity matches'" in res.injected_message
    assert "before setting answer['ready'] = True" in res.injected_message


def test_m2_boundary_recheck_empty_no_bounce():
    res = boundary_recheck([])
    assert res.should_bounce is False
    res2 = boundary_recheck(None)
    assert res2.should_bounce is False


def test_m2_format_check_fraction_mismatch():
    query = "Find the probability of rolling a prime number and express as a fraction."
    draft = "The probability is 0.5 because there are 3 primes out of 6."
    res = format_check(query, draft)
    assert res.should_bounce is True
    assert res.check_type == "format_check"
    assert "fraction" in res.injected_message
    assert draft in res.injected_message
    assert "Does this match the required form exactly" in res.injected_message


def test_m2_format_check_matching_no_bounce():
    query = "Find the probability and express as a fraction."
    draft = "1/2"
    res = format_check(query, draft)
    assert res.should_bounce is False


def test_m2_evidence_preference_verbatim_prompt():
    root_finding = "Grep in Extern/ confirmed only PETSc is supported in this repository configuration."
    subcall_claim = "Sub-LLM stated AMReX supports HYPRE, PETSc, and HPGMG external solvers."
    res = evidence_preference(root_finding, subcall_claim)
    assert res.should_bounce is True
    assert res.check_type == "evidence_preference"
    assert "you previously established the following" in res.injected_message
    assert root_finding in res.injected_message
    assert subcall_claim in res.injected_message
    assert "prefer your own directly-verified" in res.injected_message


# =========================================================================
# 3. M2 Single One-Shot Invariant Tests (No loops, strictly one-bounce)
# =========================================================================

def test_m2_single_one_shot_invariant():
    classification = M1Classification(
        category="format_completeness",
        confidence="high",
        reasoning="Format requested",
    )
    state = M2State(m1_classification=classification)

    assert state.can_intercept() is True
    assert state.bounce_triggered is False

    # Turn 1: Check fires
    res1 = evaluate_m2_intercept(
        state=state,
        query_text="Express as a fraction.",
        draft_answer="0.75",
        turn=1,
    )
    assert res1.should_bounce is True
    assert state.bounce_triggered is True
    assert state.bounce_turn == 1
    assert state.draft_answer_before == "0.75"

    # Turn 2: Attempting to finalize again
    assert state.can_intercept() is False
    res2 = evaluate_m2_intercept(
        state=state,
        query_text="Express as a fraction.",
        draft_answer="3/4",
        turn=2,
    )
    # MUST NOT BOUNCE AGAIN
    assert res2.should_bounce is False

    # Record finalization
    state.record_finalization("3/4")
    assert state.changes_made is True
    assert state.final_answer == "3/4"

    summary = state.get_summary()
    assert summary["m2_fired"] is True
    assert summary["m2_changes_made"] is True
    assert summary["draft_answer_before_m2"] == "0.75"
    assert summary["final_answer"] == "3/4"


# =========================================================================
# 4. Paired Comparison Harness Tests
# =========================================================================

def test_paired_comparison_harness(tmp_path: Path):
    logger = PairedHarnessLogger(output_dir=tmp_path)

    m1_cls = M1Classification("format_completeness", "high", "Requires fraction")
    baseline = TaskRunRecord(
        condition="baseline",
        final_answer="0.5",
        score=0.0,
        is_correct=False,
    )
    treatment = TaskRunRecord(
        condition="treatment",
        final_answer="1/2",
        score=1.0,
        is_correct=True,
        m2_fired=True,
        m2_changes_made=True,
        draft_answer_before_m2="0.5",
    )

    pair = PairedTaskComparison(
        task_id="task_001",
        query_text="Express probability as a fraction.",
        context_metadata={"num_chunks": 1, "context_total_length": 100},
        m1_classification=m1_cls,
        baseline_run=baseline,
        treatment_run=treatment,
    )

    logger.record_pair(pair)
    summary = logger.get_summary()

    assert summary["total_tasks"] == 1
    assert summary["m2_fired_total"] == 1
    assert summary["m2_changed_answer_total"] == 1
    assert summary["baseline_accuracy"] == 0.0
    assert summary["treatment_accuracy"] == 1.0
    assert summary["accuracy_lift"] == 1.0

    summary_file = logger.save_summary()
    assert summary_file.exists()
    saved = json.loads(summary_file.read_text())
    assert saved["accuracy_lift"] == 1.0


# =========================================================================
# 5. End-to-End RLM Integration with MockLM (Offline)
# =========================================================================

def test_rlm_completion_with_m1_m2_format_soft_bounce(monkeypatch: pytest.MonkeyPatch):
    """Tests end-to-end RLM execution with MockLM.

    Turn 1: Model outputs draft answer '0.75' and sets answer['ready'] = True.
    M1 detects 'format_completeness'. M2 format_check intercepts and soft bounces.
    Turn 2: Model receives M2 nudge and outputs corrected answer '3/4'.
    RLM accepts the answer without second bounce.
    """
    turn1_code = (
        "```repl\n"
        "answer['content'] = '0.75'\n"
        "answer['ready'] = True\n"
        "```"
    )
    turn2_code = (
        "```repl\n"
        "answer['content'] = '3/4'\n"
        "answer['ready'] = True\n"
        "```"
    )

    mock_responses = [
        f"I calculated the value: {turn1_code}",
        f"Correcting format as requested: {turn2_code}",
    ]

    mock_lm = MockLM(responses=mock_responses)
    monkeypatch.setattr("rlm.core.rlm.get_client", lambda *args, **kwargs: mock_lm)

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "mock-model"},
        environment="local",
        max_iterations=5,
        paper_mode=True,
        enable_m1_m2=True,
        m1_mode="rules",
    )

    result = rlm.completion("What is the probability of rolling a 1 or 2 on a 4-sided die? Express as a fraction.")

    assert result.response == "3/4"
    assert result.metadata is not None
    assert "m1_m2" in result.metadata

    m1_m2_meta = result.metadata["m1_m2"]
    assert m1_m2_meta["m1_classification"]["category"] == "format_completeness"
    assert m1_m2_meta["m2_fired"] is True
    assert m1_m2_meta["m2_check_type"] == "format_check"
    assert m1_m2_meta["draft_answer_before_m2"] == "0.75"
    assert m1_m2_meta["final_answer"] == "3/4"
    assert m1_m2_meta["m2_changes_made"] is True


def test_rlm_completion_with_m1_m2_evidence_grounded_bounce(monkeypatch: pytest.MonkeyPatch):
    """Tests end-to-end RLM execution with MockLM on evidence_grounded conflict."""
    turn1_code = (
        "```repl\n"
        "# Model chooses option 3 based on subcall\n"
        "answer['content'] = 'All of the above (Option 3)'\n"
        "answer['ready'] = True\n"
        "```"
    )
    turn2_code = (
        "```repl\n"
        "# Model heeds M2 evidence preference and switches to Option 1\n"
        "answer['content'] = 'PETSc Krylov only (Option 1)'\n"
        "answer['ready'] = True\n"
        "```"
    )

    mock_responses = [
        f"Initial thought: {turn1_code}",
        f"Revised following direct evidence: {turn2_code}",
    ]

    mock_lm = MockLM(responses=mock_responses)
    monkeypatch.setattr("rlm.core.rlm.get_client", lambda *args, **kwargs: mock_lm)

    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "mock-model"},
        environment="local",
        max_iterations=5,
        paper_mode=True,
        enable_m1_m2=True,
        m1_mode="rules",
        m2_grounded_finding="Direct inspection of Extern/ confirms only PETSc is supported in this checkout.",
        m2_subcall_claim="Sub-LLM asserted AMReX supports HYPRE, PETSc, and HPGMG.",
    )

    result = rlm.completion("Which external solvers does this repository support in AMReX?")

    assert result.response == "PETSc Krylov only (Option 1)"
    assert result.metadata is not None
    m1_m2_meta = result.metadata["m1_m2"]
    assert m1_m2_meta["m1_classification"]["category"] == "evidence_grounded"
    assert m1_m2_meta["m2_fired"] is True
    assert m1_m2_meta["m2_check_type"] == "evidence_preference"
    assert m1_m2_meta["draft_answer_before_m2"] == "All of the above (Option 3)"
    assert m1_m2_meta["final_answer"] == "PETSc Krylov only (Option 1)"
    assert m1_m2_meta["m2_changes_made"] is True
