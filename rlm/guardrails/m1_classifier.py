"""M1: Task-structure classifier for RLM harnesses.

Runs once per task before the RLM loop starts, using only the query text plus
lightweight metadata (length, chunk count, context type). It does NOT read the payload.
Output: one of four labels:
  - nonmonotonic_predicate
  - format_completeness
  - evidence_grounded
  - none
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from rlm.guardrails.prompts import M1_SYSTEM_PROMPT, M1_USER_PROMPT_TEMPLATE

M1Category = Literal[
    "nonmonotonic_predicate",
    "format_completeness",
    "evidence_grounded",
    "none",
]

M1Confidence = Literal["low", "medium", "high"]

VALID_CATEGORIES: set[str] = {
    "nonmonotonic_predicate",
    "format_completeness",
    "evidence_grounded",
    "none",
}

VALID_CONFIDENCES: set[str] = {"low", "medium", "high"}


@dataclass
class M1Classification:
    category: str
    confidence: str
    reasoning: str
    raw_response: str = ""

    def __post_init__(self):
        if self.category not in VALID_CATEGORIES:
            self.category = "none"
        if self.confidence not in VALID_CONFIDENCES:
            self.confidence = "low"

    @property
    def effective_category(self) -> str:
        """Treat confidence: 'low' the same as category: 'none' per specification."""
        if self.confidence == "low":
            return "none"
        return self.category

    @property
    def is_active_for_m2(self) -> bool:
        """Whether this classification triggers any downstream M2 check."""
        return self.effective_category in {
            "nonmonotonic_predicate",
            "format_completeness",
            "evidence_grounded",
        }

    def get_strategy_nudge(self, actual_depth: int | None = None) -> str:
        """Advisory execution recommendation based on classified task topology.

        NOTE: The nudge is always consistent with actual depth — if
        depth < 2, rlm_query is never mentioned so the model isn't confused.
        """
        eff = self.effective_category
        depth = actual_depth if actual_depth is not None else self.suggested_depth

        if eff == "nonmonotonic_predicate":
            if depth >= 2:
                return (
                    "[STRATEGY ADVISORY: Quadratic Pair Aggregation — Depth-2 Mode]\n"
                    "This query requires exhaustive combinatorial pair enumeration over a large context. "
                    "Strategy: (1) Always use local REPL code in ```repl``` blocks to parse and filter user records — the REPL is active and pre-loaded with `context`. "
                    "(2) Use overlapping context chunks (e.g. stride of 200 chars overlap) to avoid "
                    "splitting records at chunk boundaries. "
                    "(3) Collect ALL candidate items per user before computing pairs — enumerate pairs "
                    "in a nested Python loop, not in an LLM call. "
                    "(4) Err on the side of INCLUSION for borderline matches (a missed pair costs more "
                    "than a spurious one in this dataset). "
                    "(5) rlm_query is available for complex sub-classification tasks where a single "
                    "llm_query would be unreliable."
                )
            else:
                return (
                    "[STRATEGY ADVISORY: Exact Count / Record Aggregation — Depth-1 Mode]\n"
                    "This query requires exact count or combinatorial matching over records. "
                    "Strategy: Always write code in ```repl``` blocks to inspect `context` and filter items programmatically — the REPL is active and pre-loaded. "
                    "Use local Python loops and llm_query to filter records. For borderline items, err on the side of inclusion rather than strict exclusion. "
                    "Enumerate pairs and verify final counts directly in Python code before submitting."
                )
        elif eff == "format_completeness":
            return (
                "[STRATEGY ADVISORY: Symbolic / Exact Format Contract]\n"
                "This query requires an exact output format (e.g. SMILES, JSON, fraction, LaTeX). "
                "Strategy: Always write code in ```repl``` blocks. Compute the answer value in Python first, then format it to the required "
                "representation. For SMILES, canonicalize via RDKit if available. "
                "For math, prefer plain-text notation (sqrt(x), a/b) unless LaTeX is explicitly required. "
                "Set answer[\"content\"] to the final formatted value — not to an unevaluated f-string or code block."
            )
        elif eff == "evidence_grounded":
            if depth >= 2:
                return (
                    "[STRATEGY ADVISORY: Open-Ended Multi-Document Investigation — Depth-2 Mode]\n"
                    "This query requires deep multi-hop evidence chaining across a large document collection. "
                    "Strategy: (1) Use REPL code in ```repl``` blocks to search documents using keyword indices, tf-idf, or grep — the REPL is active and pre-loaded with `documents`. "
                    "(2) Sub-agents (rlm_query) can be delegated to deeply analyze candidate documents or verify entity connections recursively. "
                    "(3) Verify and confirm evidence across multiple candidate documents before finalizing."
                )
            else:
                return (
                    "[STRATEGY ADVISORY: Grounded File Inspection Contract]\n"
                    "This query tests facts about an external codebase or document set. "
                    "Strategy: Always write code in ```repl``` blocks. Directly inspect the context via grep/search in the REPL. "
                    "Prioritize local grounded evidence over recalled knowledge. "
                    "Do not hallucinate file paths or function names — verify each claim in code."
                )
        return (
            "[STRATEGY ADVISORY: Exploratory Reasoning]\n"
            "Proceed with iterative decomposition. Always execute code in ```repl``` blocks to explore the preloaded context."
        )

    @property
    def strategy_nudge(self) -> str:
        return self.get_strategy_nudge()

    @property
    def suggested_depth(self) -> int:
        """Dynamic Epistemic Depth Allocation based on task topology."""
        eff = self.effective_category
        if eff == "nonmonotonic_predicate":
            # Combinatorial / quadratic pair aggregation requires recursive sub-REPLs (Depth 2)
            # per Section 3.2 to prevent Cartesian state explosion in a single linear REPL.
            reason_str = (self.reasoning or "").lower()
            raw_str = (self.raw_response or "").lower()
            if "pair" in reason_str or "combinatorial" in reason_str or "pairs" in raw_str:
                return 2
            return 1
        elif eff == "format_completeness":
            return 1
        elif eff == "evidence_grounded":
            # Open-ended multi-document collections (e.g. BrowseComp) require Depth-2 recursive exploration
            reason_str = (self.reasoning or "").lower()
            raw_str = (self.raw_response or "").lower()
            if any(k in reason_str or k in raw_str for k in ("browsecomp", "open-ended", "open_ended", "multi-document", "multi_document", "1000", "documents")):
                return 2
            # CodeQA / repository tasks run optimally at depth 1 per Section 3.4 Pareto analysis
            return 1
        elif eff == "none":
            # Single-doc comparison / extraction tasks routed to Depth-1
            return 1
        return 1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["effective_category"] = self.effective_category
        d["is_active_for_m2"] = self.is_active_for_m2
        d["strategy_nudge"] = self.strategy_nudge
        d["suggested_depth"] = self.suggested_depth
        return d


# -------------------------------------------------------------------------
# Rule-Based Classifier (Zero-cost, default for pilot)
# -------------------------------------------------------------------------

# Regular expressions for rule-based detection
_NONMONOTONIC_PATTERNS = [
    r"\bhow many\b",
    r"\bexact(?:ly)?\s+(?:one|two|three|\d+|N)\b",
    r"\bexact(?:ly)?\s+(?:count|match|number)\b",
    r"\btotal\s+number\s+of\b",
    r"\bcount\s+the\s+number\b",
    r"\blist\s+all\s+pairs\b",
    r"\bpairs?\s+where\s+both\b",
    r"\bpairwise\b",
    r"\bcombinatorial\b",
    r"\bsymmetric\b",
    r"\bhow\s+many\s+instances\b",
    r"\bnumber\s+of\s+users\b",
    r"\bhow\s+many\s+times\b",
    r"\bexact\s+count\b",
]

_FORMAT_PATTERNS = [
    r"\bexpress\s+as\s+a\s+fraction\b",
    r"\bgive\s+the\s+answer\s+in\s+the\s+form\b",
    r"\bin\s+the\s+form\s+[a-zA-Z]\s*=\b",
    r"\bprovide\s+the\s+final\s+smiles\b",
    r"\bsmiles\s+string\b",
    r"\blatex\b",
    r"\bproof-style\b",
    r"\bstep-by-step\s+derivation\b",
    r"\bexact\s+notation\b",
    r"\bsimplif(?:y|ied)\s+form\b",
    r"\boutput\s+format\b",
    r"\bformat:\s*",
    r"\bin\s+scientific\s+notation\b",
    r"\brounded\s+to\s+\d+\s+decimal\b",
    r"\bcomma-separated\s+list\b",
    r"\bjson\s+format\b",
    r"\bkey-value\b",
]

_EVIDENCE_PATTERNS = [
    r"\bdoes\s+this\s+repo(?:sitory)?\s+(?:support|implement|contain|use)\b",
    r"\bdoes\s+the\s+codebase\s+(?:support|implement|contain|use)\b",
    r"\bwhat\s+does\s+this\s+codebase\s+actually\s+implement\b",
    r"\baccording\s+to\s+the\s+provided\s+documents\b",
    r"\baccording\s+to\s+the\s+codebase\b",
    r"\bin\s+this\s+codebase\b",
    r"\bin\s+this\s+repo(?:sitory)?\b",
    r"\bwhich\s+external\s+solvers\b",
    r"\bwhich\s+class(?:es)?\b",
    r"\bwhich\s+function(?:s)?\b",
    r"\bcheck\s+out\s+in\s+the\s+repo\b",
    r"\bwhat\s+are\s+the\s+advantages\s+of\s+this\s+codebase\b",
    r"\bimplemented\s+in\b",
    r"\bdefined\s+in\b",
    r"\bsupported\s+in\s+this\s+version\b",
]


def classify_task_rules(
    query_text: str,
    context_metadata: dict[str, Any] | None = None,
) -> M1Classification:
    """Fast, zero-cost rule-based task structure classifier.

    Uses keyword, phrase, and regex heuristics on query_text plus context metadata.
    """
    if not query_text:
        return M1Classification(
            category="none",
            confidence="low",
            reasoning="Empty query text provided.",
            raw_response="rule_based_empty",
        )

    q_lower = query_text.lower()
    ctx_meta = context_metadata or {}
    ctx_type = str(ctx_meta.get("context_type", "")).lower()
    benchmark = str(ctx_meta.get("benchmark", "")).lower()
    structure = str(ctx_meta.get("structure", "")).lower()
    num_docs = ctx_meta.get("num_docs", 0)

    # Score evidence
    nonmono_matches = [p for p in _NONMONOTONIC_PATTERNS if re.search(p, q_lower)]
    format_matches = [p for p in _FORMAT_PATTERNS if re.search(p, q_lower)]
    evidence_matches = [p for p in _EVIDENCE_PATTERNS if re.search(p, q_lower)]

    # Domain/context hints
    if "browsecomp" in benchmark or "browsecomp" in structure or "browsecomp" in ctx_type or (isinstance(num_docs, int) and num_docs >= 100):
        evidence_matches.append("benchmark:browsecomp")
    if "codeqa" in ctx_type or "codebase" in ctx_type or "repo" in ctx_type:
        evidence_matches.append("context_type:codebase")
    if "pairs" in ctx_type or "oolong_pairs" in ctx_type:
        nonmono_matches.append("context_type:pairs")
    if "longcot" in ctx_type or "format" in ctx_type:
        format_matches.append("context_type:longcot")

    # Priority determination based on what determines whether FINAL answer is right/wrong:
    # 0. Open-ended multi-document investigation (BrowseComp) takes high priority for depth-2 exploration
    if "benchmark:browsecomp" in evidence_matches or "browsecomp" in benchmark or "browsecomp" in structure:
        confidence = "high"
        reason = "Open-ended multi-document investigation over large document collection (BrowseComp), requiring Depth-2 recursive exploration."
        return M1Classification(
            category="evidence_grounded",
            confidence=confidence,
            reasoning=reason,
            raw_response="rule_match:open_ended_browsecomp:depth_2",
        )

    # 1. Codebase / repository verification takes top priority when context_type is codebase
    if evidence_matches and ("context_type:codebase" in evidence_matches or "codeqa" in ctx_type or "repo" in q_lower or "codebase" in q_lower):
        confidence = "high"
        reason = f"Query queries specific features/artifacts in an external repository/document set ({', '.join(evidence_matches[:2])})."
        return M1Classification(
            category="evidence_grounded",
            confidence=confidence,
            reasoning=reason,
            raw_response=f"rule_match:evidence_grounded:{len(evidence_matches)}",
        )

    # 2. Format completeness (strict output format / longcot)
    if format_matches and ("context_type:longcot" in format_matches or "format:" in q_lower or "smiles" in q_lower or "solution =" in q_lower) and not ("pairs" in ctx_type or "oolong" in ctx_type):
        confidence = "high" if len(format_matches) >= 2 or "smiles" in q_lower or "fraction" in q_lower else "medium"
        reason = f"Query imposes strict structural or representational format requirements ({', '.join(format_matches[:2])})."
        return M1Classification(
            category="format_completeness",
            confidence=confidence,
            reasoning=reason,
            raw_response=f"rule_match:format_completeness:{len(format_matches)}",
        )

    # 3. Nonmonotonic predicate (exact counts / pairwise combinations)
    if nonmono_matches:
        confidence = "high" if len(nonmono_matches) >= 2 or "pairs" in q_lower or "exact" in q_lower else "medium"
        reason = f"Query specifies exact count or combinatorial matching conditions ({', '.join(nonmono_matches[:2])})."
        return M1Classification(
            category="nonmonotonic_predicate",
            confidence=confidence,
            reasoning=reason,
            raw_response=f"rule_match:nonmonotonic_predicate:{len(nonmono_matches)}",
        )

    # 2. Format completeness (specific derivation or exact representation demanded)
    if format_matches:
        confidence = "high" if len(format_matches) >= 2 or "smiles" in q_lower or "fraction" in q_lower else "medium"
        reason = f"Query imposes strict structural or representational format requirements ({', '.join(format_matches[:2])})."
        return M1Classification(
            category="format_completeness",
            confidence=confidence,
            reasoning=reason,
            raw_response=f"rule_match:format_completeness:{len(format_matches)}",
        )

    # 3. Evidence grounded (asking whether an external codebase/artifact implements/supports X)
    if evidence_matches:
        confidence = "high" if len(evidence_matches) >= 2 or "repo" in q_lower or "codebase" in q_lower else "medium"
        reason = f"Query queries specific features/artifacts in an external repository/document set ({', '.join(evidence_matches[:2])})."
        return M1Classification(
            category="evidence_grounded",
            confidence=confidence,
            reasoning=reason,
            raw_response=f"rule_match:evidence_grounded:{len(evidence_matches)}",
        )

    # 4. Fallback: none
    return M1Classification(
        category="none",
        confidence="medium",
        reasoning="Query does not exhibit exact-count, strict-format, or repository-evidence constraints.",
        raw_response="rule_match:none",
    )


# -------------------------------------------------------------------------
# LLM-Based Classifier (Prompted LLM call)
# -------------------------------------------------------------------------

def classify_task_llm(
    lm_client: Any,
    query_text: str,
    context_metadata: dict[str, Any] | None = None,
) -> M1Classification:
    """Prompted single-call task structure classifier using M1_SYSTEM_PROMPT.

    Parses JSON output strictly according to the parsing contract.
    Falls back to classify_task_rules if the LM fails or returns unparseable text.
    """
    ctx_meta = context_metadata or {}
    total_len = ctx_meta.get("context_total_length", ctx_meta.get("length", "unknown"))
    num_chunks = ctx_meta.get("num_chunks", ctx_meta.get("chunk_count", "unknown"))
    ctx_type = ctx_meta.get("context_type", "general")

    user_content = M1_USER_PROMPT_TEMPLATE.format(
        query_text=query_text,
        context_total_length=total_len,
        num_chunks=num_chunks,
        context_type=ctx_type,
    )

    prompt = [
        {"role": "system", "content": M1_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        if hasattr(lm_client, "completion"):
            raw_response = lm_client.completion(prompt)
        elif callable(lm_client):
            raw_response = lm_client(prompt)
        else:
            raise ValueError(f"Unsupported lm_client type: {type(lm_client)}")

        raw_str = str(raw_response).strip()

        # Extract JSON substring
        json_match = re.search(r"\{.*\}", raw_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            category = str(data.get("category", "none")).strip().lower()
            confidence = str(data.get("confidence", "low")).strip().lower()
            reasoning = str(data.get("reasoning", "")).strip()

            if category not in VALID_CATEGORIES:
                category = "none"
            if confidence not in VALID_CONFIDENCES:
                confidence = "low"

            return M1Classification(
                category=category,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=raw_str,
            )
        else:
            fb = classify_task_rules(query_text, context_metadata)
            fb.reasoning += " [LLM fallback due to: no JSON found]"
            return fb
    except Exception as err:
        # Fallback to rules on error
        fb = classify_task_rules(query_text, context_metadata)
        fb.reasoning += f" [LLM fallback due to: {err}]"
        return fb

    fb = classify_task_rules(query_text, context_metadata)
    fb.reasoning += " [LLM fallback due to: unexpected flow]"
    return fb


def classify_task(
    query_text: str,
    context_metadata: dict[str, Any] | None = None,
    mode: Literal["rules", "llm"] = "rules",
    lm_client: Any | None = None,
) -> M1Classification:
    """Primary entrypoint for M1 classification."""
    if mode == "llm" and lm_client is not None:
        return classify_task_llm(lm_client, query_text, context_metadata)
    return classify_task_rules(query_text, context_metadata)
