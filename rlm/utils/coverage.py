"""Coverage tracking and line-index utilities for RLM environments and evaluations."""

from __future__ import annotations

import re
from typing import Any

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """Tokenize string into lower-cased words and punctuation."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def extract_atomic_units(raw_context: str) -> list[dict[str, Any]]:
    """Extract atomic units (lines) from raw context.

    Each unit contains:
      - 'index': 0-based integer index
      - 'raw_line': verbatim line string
      - 'clean_line': stripped line string
      - 'content': distinctive instance text (e.g. question following 'Instance:')
      - 'content_lower': lower-cased content
      - 'tokens': set of lower-cased tokens
    """
    if not isinstance(raw_context, str):
        raw_context = str(raw_context)

    lines = [l.strip() for l in raw_context.strip().split("\n") if l.strip()]
    atomic_units = []

    for idx, l in enumerate(lines):
        if "Instance:" in l:
            content = l.split("Instance:", 1)[1].strip()
        else:
            content = l

        atomic_units.append({
            "index": idx,
            "raw_line": l,
            "content": content,
            "content_lower": content.lower(),
            "tokens": set(tokenize(content)),
        })

    return atomic_units


def match_prompt_against_units(
    prompt: str,
    units: list[dict[str, Any]],
) -> set[int]:
    """Determine which atomic unit indices are present in a prompt.

    Uses exact substring matching first, with fallback to token overlap (>=90%).
    """
    matched = set()
    if not isinstance(prompt, str):
        prompt = str(prompt)

    prompt_lower = prompt.lower()
    prompt_tokens = set(tokenize(prompt))

    if len(prompt) < 15:
        return matched

    for u in units:
        idx = u["index"]
        c_lower = u["content_lower"]

        # 1. Substring matching of content part
        if len(c_lower) >= 10 and c_lower in prompt_lower:
            matched.add(idx)
            continue

        # 2. Token overlap fallback (>= 90% of unit's tokens present in prompt)
        u_tokens = u["tokens"]
        if not u_tokens:
            continue
        overlap = len(u_tokens & prompt_tokens)
        if overlap / len(u_tokens) >= 0.90:
            matched.add(idx)

    return matched


def compute_unvisited_ranges(total_lines: int, visited_indices: set[int]) -> list[list[int]]:
    """Convert unvisited line indices into contiguous [start_idx, end_idx] ranges."""
    if total_lines <= 0:
        return []

    unvisited = sorted(set(range(total_lines)) - visited_indices)
    if not unvisited:
        return []

    ranges = []
    start = unvisited[0]
    prev = start

    for idx in unvisited[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append([start, prev])
            start = idx
            prev = idx
    ranges.append([start, prev])
    return ranges
