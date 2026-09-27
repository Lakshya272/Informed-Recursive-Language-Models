"""Paper-style rlm_query(context, query) argument helpers."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator


def stitch_query_and_context(context: Any, query: Any) -> str:
    """Build a single LM prompt when recursion is unavailable.

    Paper ``rlm_query(context, query)`` loads ``context`` into the child REPL
    and asks ``query``. A terminal ``llm_query`` has no REPL, so both parts
    are concatenated.
    """
    context_text = "" if context is None else str(context)
    query_text = "" if query is None else str(query)
    if not query_text:
        return context_text
    if not context_text:
        return query_text
    return f"{query_text}\n\n{context_text}"


def iter_context_query_pairs(items: Iterable[Any]) -> Iterator[tuple[Any, str]]:
    """Yield ``(context, query)`` pairs from ``rlm_query_batched`` input.

    Each item may be a ``(context, query)`` pair (paper) or a bare context
    string (query defaults to empty).
    """
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            yield item[0], str(item[1])
        else:
            yield item, ""


def invoke_subcall_fn(fn: Callable[..., Any], context: Any, model: str | None = None, query: str | None = None) -> Any:
    """Call ``subcall_fn`` with paper ``(context, model, query)``.

    Older two-arg callbacks ``(prompt, model)`` still work.
    """
    try:
        return fn(context, model, query)
    except TypeError:
        return fn(context, model)
