"""M2: Targeted Intercept Checks for RLM harnesses.

Three separate functions, each active ONLY if M1's label matches (and confidence != 'low'):
1. boundary_recheck() (fires on nonmonotonic_predicate)
2. format_check() (fires on format_completeness)
3. evidence_preference() (fires on evidence_grounded)

CRITICAL INVARIANT:
Each check is a SINGLE ONE-SHOT INJECTION. After it fires once, whatever the model
does next is accepted as final regardless of outcome (one-bounce contract).
Checks never re-fire on the same task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rlm.guardrails.m1_classifier import M1Classification
from rlm.guardrails.prompts import (
    M2_BOUNDARY_RECHECK_TEMPLATE,
    M2_EVIDENCE_PREFERENCE_TEMPLATE,
    M2_FORMAT_CHECK_TEMPLATE,
)


@dataclass
class M2InterceptResult:
    """Result of an M2 intercept check evaluation."""
    should_bounce: bool
    check_type: str | None = None
    injected_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class M2State:
    """Per-task state tracking for M1/M2 lifecycle and single one-shot injection."""
    m1_classification: M1Classification
    bounce_triggered: bool = False
    bounce_turn: int | None = None
    check_type: str | None = None
    injected_message: str | None = None
    draft_answer_before: str | None = None
    final_answer: str | None = None
    changes_made: bool = False
    audit_log: dict[str, Any] = field(default_factory=dict)

    def can_intercept(self) -> bool:
        """Enforces the one-shot contract: cannot intercept if already bounced once."""
        if self.bounce_triggered:
            return False
        return self.m1_classification.is_active_for_m2

    def record_bounce(
        self,
        check_type: str,
        injected_message: str,
        draft_answer: str,
        turn: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Mark the single one-shot bounce as triggered."""
        self.bounce_triggered = True
        self.bounce_turn = turn
        self.check_type = check_type
        self.injected_message = injected_message
        self.draft_answer_before = draft_answer
        self.audit_log["bounce_details"] = details or {}

    def record_finalization(self, final_answer: str) -> None:
        """Record the accepted final answer and check whether M2 changed it."""
        self.final_answer = final_answer
        if self.draft_answer_before is not None:
            self.changes_made = (self.draft_answer_before.strip() != final_answer.strip())

    def get_summary(self) -> dict[str, Any]:
        """Return standardized audit metadata for trajectory logging."""
        return {
            "m1_classification": self.m1_classification.to_dict(),
            "m2_fired": self.bounce_triggered,
            "m2_check_type": self.check_type,
            "m2_bounce_turn": self.bounce_turn,
            "m2_injected_message": self.injected_message,
            "draft_answer_before_m2": self.draft_answer_before,
            "final_answer": self.final_answer,
            "m2_changes_made": self.changes_made,
            "audit_log": self.audit_log,
        }


# -------------------------------------------------------------------------
# 1. Boundary Recheck (fires on nonmonotonic_predicate)
# -------------------------------------------------------------------------

def boundary_recheck(
    disagreeing_items: list[dict[str, Any]] | str | None,
) -> M2InterceptResult:
    """Evaluates whether to trigger boundary_recheck intercept.

    disagreeing_items can be:
      - A formatted string describing items and their candidate labels
      - A list of dicts, e.g. [{"item": "user_123", "pass_1": "match", "pass_2": "no_match"}]
    """
    if not disagreeing_items:
        return M2InterceptResult(should_bounce=False, check_type="boundary_recheck")

    if isinstance(disagreeing_items, list):
        formatted_lines = []
        for d in disagreeing_items:
            item_name = d.get("item", d.get("id", "item"))
            p1 = d.get("pass_1", d.get("label_1", "candidate A"))
            p2 = d.get("pass_2", d.get("label_2", "candidate B"))
            formatted_lines.append(f"- {item_name}: Pass 1 classified as '{p1}'; Pass 2 classified as '{p2}'")
        items_str = "\n".join(formatted_lines)
    else:
        items_str = str(disagreeing_items).strip()

    if not items_str:
        return M2InterceptResult(should_bounce=False, check_type="boundary_recheck")

    message = M2_BOUNDARY_RECHECK_TEMPLATE.format(
        list_of_disagreeing_items_with_both_candidate_labels=items_str
    )

    return M2InterceptResult(
        should_bounce=True,
        check_type="boundary_recheck",
        injected_message=message,
        details={"disagreeing_items": items_str},
    )


# -------------------------------------------------------------------------
# 2. Format / Completeness Check (fires on format_completeness)
# -------------------------------------------------------------------------

def detect_format_mismatch(
    query_text: str,
    draft_answer: str,
) -> tuple[bool, str]:
    """Detects whether draft_answer violates stated output format requirements in query_text.

    Returns:
      (is_mismatched, expected_format_description)
    """
    q_lower = query_text.lower()
    draft = draft_answer.strip()

    # SMILES string check via RDKit — only fire if parse fails, not just because
    # the format looks odd. This avoids bouncing on Kekule vs aromatic mismatches.
    if "smiles" in q_lower:
        candidate_smiles = draft.strip()
        candidate_smiles = re.sub(r"^[`'\"]+ |[`'\"]+$", "", candidate_smiles)
        # Extract the likely SMILES token (first non-space atom-like substring)
        m = re.search(r"([A-Za-z0-9@+\-\[\]\(\)\\/%=#$\.]{4,})", candidate_smiles)
        candidate = m.group(1) if m else candidate_smiles
        try:
            from rdkit import Chem  # noqa: PLC0415
            mol = Chem.MolFromSmiles(candidate)
            if mol is None:
                return True, f"a chemically valid SMILES string (RDKit failed to parse candidate '{candidate}')"
            # Valid mol → accept regardless of Kekule/aromatic style
        except Exception:
            # RDKit not available — only flag if the candidate looks totally wrong
            if " " in candidate.strip() or len(candidate.strip()) < 2:
                return True, "a valid chemical SMILES string"

    # LaTeX / math format mismatch — only fire when LaTeX is *explicitly required*
    # (not merely when "answer" appears in the query, which is nearly always)
    has_latex = any(k in draft for k in [r"\sqrt", r"\frac", r"^{", r"\\", r"\pi"])
    latex_explicitly_required = any(k in q_lower for k in ["latex", "tex format", "\\frac", "\\sqrt"])
    if has_latex and not latex_explicitly_required:
        # LaTeX in draft when plain text is expected — bounce
        if any(k in q_lower for k in ["solution =", "comma-separated", "plain text", "fraction"]):
            return True, "standard plain-text mathematical notation (e.g. sqrt(x) instead of \\sqrt{x}, a/b instead of \\frac{a}{b})"

    if any(k in q_lower for k in ["fraction", "rational number"]):
        if not re.search(r"\b-?\d+\s*/\s*\d+\b", draft) and r"\frac" not in draft:
            return True, "an exact fraction in the form a/b"

    # Specific form 'solution = ...' or 'X = value' — only bounce if the draft
    # contains NEITHER the variable prefix NOR any dict-like structure that might
    # carry the value. This avoids false positives on LongCoT answers that use
    # answer["content"] = {...} style instead of "solution = {...}" in the output.
    form_match = re.search(r"in\s+the\s+form(?:at)?(?::)?\s*([a-zA-Z_]+\s*=\s*[^,\.\n]+)", query_text, re.IGNORECASE)
    if form_match:
        expected_spec = form_match.group(1).strip()
        var_name = expected_spec.split("=")[0].strip()
        # Only bounce if truly missing — skip if draft already has dict-like or value content
        has_var_prefix = f"{var_name} =" in draft or f"{var_name}=" in draft
        has_dict_content = draft.startswith("{") or '"' in draft[:50] or "'" in draft[:50]
        if not has_var_prefix and not has_dict_content:
            return True, f"in the format: {expected_spec}"

    # Scientific notation check
    if "scientific notation" in q_lower:
        if not re.search(r"-?\d+(?:\.\d+)?[eE][+-]?\d+", draft) and r"\times 10^" not in draft:
            return True, "scientific notation (e.g. 1.23e-4 or 1.23 \\times 10^{-4})"

    # LaTeX equation check — only when LaTeX is explicitly required
    if latex_explicitly_required and "latex" in q_lower and ("equation" in q_lower or "expression" in q_lower):
        if "$" not in draft and "\\" not in draft:
            return True, "exact LaTeX notation (e.g. enclosed in $...$ with standard macros)"

    # JSON output check
    if "json" in q_lower and ("format" in q_lower or "valid json" in q_lower):
        if not (draft.startswith("{") and draft.endswith("}")) and not (draft.startswith("[") and draft.endswith("]")):
            return True, "valid JSON structure"

    # Verbose-answer check — only fire when "provide only" is explicitly requested,
    # not on generic "output format" or "exact notation" phrases (too broad for LongCoT).
    if "provide only" in q_lower and len(draft.split()) > 40:
        return True, "concise, exact format requested by the prompt without conversational explanation"

    return False, ""


def format_check(
    query_text: str,
    draft_answer: str,
    explicit_expected_format: str | None = None,
    force_check: bool = False,
) -> M2InterceptResult:
    """Evaluates whether to trigger format_check intercept."""
    if not draft_answer:
        return M2InterceptResult(should_bounce=False, check_type="format_check")

    if explicit_expected_format:
        is_mismatched = True
        expected_desc = explicit_expected_format
    else:
        is_mismatched, expected_desc = detect_format_mismatch(query_text, draft_answer)

    if not is_mismatched and not force_check:
        return M2InterceptResult(should_bounce=False, check_type="format_check")

    if not expected_desc:
        expected_desc = "exact representation and structure requested in the prompt"

    message = M2_FORMAT_CHECK_TEMPLATE.format(
        expected_format_description=expected_desc,
        current_draft_answer=draft_answer,
    )

    return M2InterceptResult(
        should_bounce=True,
        check_type="format_check",
        injected_message=message,
        details={
            "expected_format_description": expected_desc,
            "draft_answer": draft_answer,
        },
    )


# -------------------------------------------------------------------------
# 3. Evidence Preference (fires on evidence_grounded)
# -------------------------------------------------------------------------

def detect_evidence_conflict(
    environment: Any,
    subcall_records: list[dict[str, Any]] | None = None,
    draft_answer: str | None = None,
) -> tuple[bool, str, str]:
    """Detects whether direct REPL inspection findings conflict with a sub-call claim or parametric prior.
    Only fires when there is an explicit grounded conflict, avoiding false alarms on standard reads.
    """
    root_evidence = ""
    if hasattr(environment, "executed_outputs"):
        outputs = getattr(environment, "executed_outputs", [])
        for out in outputs:
            out_str = str(out).strip()
            if any(k in out_str.lower() for k in ["not found", "does not contain", "no such", "schema:", "def ", "class ", "table ", "column "]):
                root_evidence = out_str[:300]
                break

    subcall_claim = ""
    if subcall_records and root_evidence:
        for sc in subcall_records:
            resp = str(sc.get("response", sc.get("output", ""))).strip()
            if any(k in resp.lower() for k in ["yes", "supports", "implements", "contains"]) and not re.search(r"line\s+\d+|[a-zA-Z0-9_\.]+\.py:\d+", resp):
                if any(neg in root_evidence.lower() for neg in ["not", "no ", "none", "missing"]):
                    subcall_claim = resp[:300]
                    break

    if root_evidence and subcall_claim:
        return True, root_evidence, subcall_claim

    return False, "", ""


def evidence_preference(
    root_own_grounded_finding: str | None,
    subcall_conflicting_answer: str | None,
) -> M2InterceptResult:
    """Evaluates whether to trigger evidence_preference intercept."""
    if not root_own_grounded_finding or not subcall_conflicting_answer:
        return M2InterceptResult(should_bounce=False, check_type="evidence_preference")

    root_clean = str(root_own_grounded_finding).strip()
    subcall_clean = str(subcall_conflicting_answer).strip()

    if not root_clean or not subcall_clean:
        return M2InterceptResult(should_bounce=False, check_type="evidence_preference")

    message = M2_EVIDENCE_PREFERENCE_TEMPLATE.format(
        root_own_grounded_finding=root_clean,
        subcall_conflicting_answer=subcall_clean,
    )

    return M2InterceptResult(
        should_bounce=True,
        check_type="evidence_preference",
        injected_message=message,
        details={
            "root_finding": root_clean,
            "subcall_finding": subcall_clean,
        },
    )


# -------------------------------------------------------------------------
# Central M2 Intercept Dispatcher
# -------------------------------------------------------------------------

def evaluate_m2_intercept(
    state: M2State,
    query_text: str,
    draft_answer: str,
    turn: int,
    disagreeing_items: list[dict[str, Any]] | str | None = None,
    explicit_expected_format: str | None = None,
    root_grounded_finding: str | None = None,
    subcall_conflicting_answer: str | None = None,
    environment: Any | None = None,
    subcall_records: list[dict[str, Any]] | None = None,
) -> M2InterceptResult:
    """Main evaluation function called before accepting final answer.

    Enforces the single one-shot injection invariant.
    """
    if not state.can_intercept():
        return M2InterceptResult(should_bounce=False)

    cat = state.m1_classification.effective_category
    res = M2InterceptResult(should_bounce=False)

    # 1. Nonmonotonic predicate
    if cat == "nonmonotonic_predicate":
        res = boundary_recheck(disagreeing_items)

    # 2. Format completeness
    elif cat == "format_completeness":
        res = format_check(
            query_text=query_text,
            draft_answer=draft_answer,
            explicit_expected_format=explicit_expected_format,
        )

    # 3. Evidence grounded
    elif cat == "evidence_grounded":
        # If explicit findings passed, use them; otherwise try detecting from env/subcalls
        if not root_grounded_finding and environment is not None:
            has_conf, r_find, s_find = detect_evidence_conflict(environment, subcall_records)
            if has_conf:
                root_grounded_finding = r_find
                subcall_conflicting_answer = s_find

        res = evidence_preference(
            root_own_grounded_finding=root_grounded_finding,
            subcall_conflicting_answer=subcall_conflicting_answer,
        )

    # If check triggered, record the one-shot bounce in state
    if res.should_bounce and res.check_type and res.injected_message:
        state.record_bounce(
            check_type=res.check_type,
            injected_message=res.injected_message,
            draft_answer=draft_answer,
            turn=turn,
            details=res.details,
        )

    return res
