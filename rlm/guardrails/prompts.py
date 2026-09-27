"""Prompt templates for M1 (Task-Structure & Depth Classifier) and M2 (Targeted Verification).

These prompt templates define the operational prompts used by Informed Recursion,
free of benchmark-specific artifacts.
"""

# -------------------------------------------------------------------------
# M1 Classifier Prompt
# -------------------------------------------------------------------------

M1_SYSTEM_PROMPT = """You are an expert query complexity and structure analyzer for recursive reasoning architectures.
Analyze the user query and determine the minimal recursive execution depth required to answer it reliably.

Routing Rules:
- depth_0 (Flat Execution / No Sub-calls):
  * Simple, direct factual lookup or short-context single-step retrieval.
  * Queries that can be resolved in a single LLM generation without iterative inspection.

- depth_1 (Single-Level Recursion):
  * Multi-hop lookups over large contexts (e.g., standard document QA, localized code search).
  * Direct constraint matching or filtering where sub-problems are independent and do not require iterative sub-querying.

- depth_2 (Deep / Hierarchical Recursion):
  * Complex multi-predicate queries.
  * Open-ended, multi-document exploratory search .
  * Cross-file / multi-module dependency tracing or structural reasoning where each sub-call itself spawns further sub-examinations.

Query to Analyze:
{query}

Respond in JSON:
{
  "reasoning": "Brief explanation of query predicate topology and branching requirements",
  "assigned_depth": 0 | 1 | 2
}"""

M1_USER_PROMPT_TEMPLATE = """{query}"""


# -------------------------------------------------------------------------
# M2 Targeted Verification Prompts
# -------------------------------------------------------------------------

M2_TARGETED_VERIFICATION_PROMPT = """Your task is to inspect a candidate decision and determine whether it satisfies the required predicate or if it is a false positive / false negative.

Task Context & Query:
{task_query}

Extracted Candidate Answer / Sub-Decision:
{candidate_decision}

Supporting Context / Evidence Snippet:
{evidence_snippet}

Verification Instructions:
1. Identify the exact logical constraints demanded by the query.
2. Verify whether the candidate decision is strictly supported by the evidence or if critical constraints were overlooked, hallucinated, or partially met.
3. Check for borderline edge cases:
   - For pair/relation tasks: Did both entities independently satisfy their respective predicates?
   - For code/retrieval tasks: Does the selected code path or passage actually implement the requested behavior without conflicting side effects?

Respond in JSON:
{
  "is_verified": true | false,
  "confidence": 0.0 to 1.0,
  "criticism": "Precise explanation if flawed, ambiguous, or ungrounded; otherwise rationale for confirmation",
  "corrected_decision": "Corrected output if is_verified is false, or null if is_verified is true"
}"""

# Specialized intercept templates maintained for backward compatibility with harness intercepts
M2_BOUNDARY_RECHECK_TEMPLATE = """Before finalizing, note: this question depends on an exact count or exact match 
condition, where a single borderline classification can change the final answer. 
The following item(s) were classified inconsistently across two independent 
passes:

{list_of_disagreeing_items_with_both_candidate_labels}

Resolve each disagreement explicitly — state which classification you are 
using and why — before setting answer['ready'] = True. If you're not sure, 
treat the disagreement as genuine ambiguity and note it in your reasoning rather 
than silently picking one side.

IMPORTANT: Re-compute your updated classifications and pair generation programmatically in the REPL and assign the result to answer['content']. Do NOT attempt to print or output large lists of pairs manually in conversational text."""


M2_FORMAT_CHECK_TEMPLATE = """Before finalizing, verify: the question requires the answer in a specific form 
({expected_format_description}). Your current draft answer is:

{current_draft_answer}

Does this match the required form exactly (correct notation, correct level of 
simplification, correct structure)? If not, revise it to match before setting 
answer['ready'] = True. If it already matches, proceed."""


M2_EVIDENCE_PREFERENCE_TEMPLATE = """Before finalizing, note a conflict: you previously established the following 
via direct inspection of the provided material:

{root_own_grounded_finding}

A sub-call returned a different, conflicting claim:

{subcall_conflicting_answer}

Unless the sub-call's answer cites a specific location in the provided material 
that you had not already checked yourself, prefer your own directly-verified 
finding. State explicitly which one you are trusting and why before setting 
answer['ready'] = True."""
