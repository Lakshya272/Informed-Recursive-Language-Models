"""M1 (Classifier) and M2 (Targeted Intercept Checks) Guardrail System."""

from rlm.guardrails.m1_classifier import (
    M1Category,
    M1Classification,
    M1Confidence,
    classify_task,
    classify_task_llm,
    classify_task_rules,
)
from rlm.guardrails.m2_intercepts import (
    M2InterceptResult,
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
    M2_TARGETED_VERIFICATION_PROMPT,
)

__all__ = [
    "M1Category",
    "M1Classification",
    "M1Confidence",
    "classify_task",
    "classify_task_rules",
    "classify_task_llm",
    "M2InterceptResult",
    "M2State",
    "boundary_recheck",
    "format_check",
    "evidence_preference",
    "evaluate_m2_intercept",
    "PairedHarnessLogger",
    "PairedTaskComparison",
    "TaskRunRecord",
    "M1_SYSTEM_PROMPT",
    "M1_USER_PROMPT_TEMPLATE",
    "M2_BOUNDARY_RECHECK_TEMPLATE",
    "M2_FORMAT_CHECK_TEMPLATE",
    "M2_EVIDENCE_PREFERENCE_TEMPLATE",
    "M2_TARGETED_VERIFICATION_PROMPT",
]
