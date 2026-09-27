"""Read RLM trajectory metadata and derive stable call-level statistics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _canonical_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _walk_calls(metadata: dict[str, Any], *, prefix: str = "root", parent_event_id: str | None = None):
    for iteration_index, iteration in enumerate(metadata.get("iterations", []), start=1):
        for block_index, block in enumerate(iteration.get("code_blocks", []), start=1):
            result = block.get("result") or {}
            for call_index, call in enumerate(result.get("rlm_calls", []), start=1):
                event_id = f"{prefix}.i{iteration_index}.b{block_index}.c{call_index}"
                prompt = _canonical_prompt(call.get("prompt", ""))
                event = {
                    "event_id": event_id,
                    "parent_event_id": parent_event_id,
                    "iteration": iteration_index,
                    "code_block": block_index,
                    "call_index": call_index,
                    "call_type": call.get("call_type", "unknown"),
                    "model": call.get("root_model"),
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "prompt_chars": len(prompt),
                    "response_chars": len(str(call.get("response", ""))),
                    "usage_summary": call.get("usage_summary", {}),
                    "execution_time_s": call.get("execution_time"),
                }
                yield event
                child_metadata = call.get("metadata")
                if isinstance(child_metadata, dict):
                    yield from _walk_calls(
                        child_metadata,
                        prefix=event_id,
                        parent_event_id=event_id,
                    )


def summarize_trajectory(metadata: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not metadata:
        return [], {
            "total_subcalls": 0,
            "repeated_subcalls": 0,
            "redundancy_rate": 0.0,
            "finalization_turn": None,
            "finalized_on_first_turn": False,
            "call_type_counts": {},
        }

    seen_hashes: set[str] = set()
    events: list[dict[str, Any]] = []
    for event in _walk_calls(metadata):
        event["is_repeated_prompt"] = event["prompt_sha256"] in seen_hashes
        seen_hashes.add(event["prompt_sha256"])
        events.append(event)

    finalization_turn = next(
        (
            index
            for index, iteration in enumerate(metadata.get("iterations", []), start=1)
            if iteration.get("final_answer") is not None
        ),
        None,
    )
    repeated = sum(event["is_repeated_prompt"] for event in events)
    total = len(events)
    return events, {
        "total_subcalls": total,
        "repeated_subcalls": repeated,
        "redundancy_rate": repeated / total if total else 0.0,
        "finalization_turn": finalization_turn,
        "finalized_on_first_turn": finalization_turn == 1,
        "call_type_counts": dict(Counter(event["call_type"] for event in events)),
    }


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part if isinstance(part, str) else str(part.get("text", part))
            if isinstance(part, dict)
            else str(part)
            for part in content
        )
    return str(content)


def _turn_input(prompt: Any) -> str:
    """Return only the new user message for a root/sub-RLM turn."""
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return _message_text(message.get("content"))
    return _canonical_prompt(prompt)


def _calls_from_block(block: dict[str, Any]) -> list[dict[str, Any]]:
    result = block.get("result") or {}
    calls = result.get("rlm_calls") or []
    return [call for call in calls if isinstance(call, dict)]


def _nested_call_entries(call: dict[str, Any]):
    child_metadata = call.get("metadata")
    if not isinstance(child_metadata, dict):
        return
    for iteration_index, iteration in enumerate(child_metadata.get("iterations", []), start=1):
        for block_index, block in enumerate(iteration.get("code_blocks", []), start=1):
            for call_index, child in enumerate(_calls_from_block(block), start=1):
                yield iteration_index, block_index, call_index, child


def _event(
    *,
    event_id: str,
    run_id: str,
    parent_run_id: str | None,
    depth: int,
    step: int,
    event_type: str,
    task_id: str | None,
    role: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_id": event_id,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "depth": depth,
        "step": step,
        "event_type": event_type,
        "role": role,
        **({"task_id": task_id} if task_id is not None else {}),
    }


def build_event_log(
    metadata: dict[str, Any] | None,
    *,
    task_id: str | None = None,
    result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a Fast-RLM-shaped event tree without changing RLM execution.

    The existing trajectory remains canonical.  This is a derived, flat NDJSON
    view with explicit run/parent/depth fields for a tree viewer or TUI.
    """
    metadata = metadata or {}
    run_metadata = metadata.get("run_metadata") or {}
    iterations = metadata.get("iterations") or []
    root_model = run_metadata.get("root_model") or run_metadata.get("backend_kwargs", {}).get("model_name")
    events: list[dict[str, Any]] = []
    root_run_id = "root"

    start = _event(
        event_id="root.start",
        run_id=root_run_id,
        parent_run_id=None,
        depth=0,
        step=0,
        event_type="run_start",
        task_id=task_id,
        role="root",
    )
    if root_model:
        start["model"] = root_model
    events.append(start)

    def emit_call(
        call: dict[str, Any],
        *,
        run_id: str,
        parent_run_id: str,
        depth: int,
        step: int,
        path: str,
    ) -> None:
        role = "sub_lm"
        call_start = _event(
            event_id=f"{run_id}.start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            depth=depth,
            step=step,
            event_type="agent_start",
            task_id=task_id,
            role=role,
        )
        call_start.update(
            {
                "path": path,
                "input": call.get("prompt", ""),
                "call_type": call.get("call_type"),
                "model": call.get("root_model"),
                "call_index": call.get("call_index"),
            }
        )
        events.append(call_start)

        child_metadata = call.get("metadata")
        if isinstance(child_metadata, dict) and child_metadata.get("iterations"):
            emit_agent(
                child_metadata,
                run_id=run_id,
                parent_run_id=parent_run_id,
                depth=depth,
                initial_input=call.get("prompt", ""),
                path=path,
            )
        else:
            call_end = _event(
                event_id=f"{run_id}.end",
                run_id=run_id,
                parent_run_id=parent_run_id,
                depth=depth,
                step=step,
                event_type="agent_end",
                task_id=task_id,
                role=role,
            )
            call_end.update(
                {
                    "path": path,
                    "output": call.get("response", ""),
                    "error": call.get("error"),
                    "call_type": call.get("call_type"),
                    "model": call.get("root_model"),
                    "usage": call.get("usage_summary", {}),
                    "execution_time_s": call.get("execution_time"),
                }
            )
            events.append(call_end)

    def emit_agent(
        agent_metadata: dict[str, Any],
        *,
        run_id: str,
        parent_run_id: str | None,
        depth: int,
        initial_input: Any = "",
        path: str = "ROOT MODEL",
    ) -> None:
        agent_iterations = agent_metadata.get("iterations") or []
        agent_role = "root" if depth == 0 else "sub_lm"
        for iteration_index, iteration in enumerate(agent_iterations, start=1):
            step_run_id = f"{run_id}.step.{iteration_index}"
            prompt = iteration.get("prompt", initial_input if iteration_index == 1 else "")
            step_start = _event(
                event_id=f"{step_run_id}.start",
                run_id=step_run_id,
                parent_run_id=run_id if parent_run_id is not None or depth > 0 else None,
                depth=depth,
                step=iteration_index,
                event_type="agent_start",
                task_id=task_id,
                role=agent_role,
            )
            step_start.update(
                {
                    "path": path,
                    "input": _turn_input(prompt),
                    "input_message_count": len(prompt) if isinstance(prompt, list) else None,
                    "full_input_chars": len(_canonical_prompt(prompt)),
                    "iteration": iteration.get("iteration", iteration_index),
                }
            )
            events.append(step_start)

            for block_index, block in enumerate(iteration.get("code_blocks", []), start=1):
                code_id = f"{step_run_id}.code.{block_index}"
                code = block.get("code") or block.get("source") or ""
                code_event = _event(
                    event_id=code_id,
                    run_id=step_run_id,
                    parent_run_id=step_run_id,
                    depth=depth,
                    step=iteration_index,
                    event_type="code_generated",
                    task_id=task_id,
                    role=agent_role,
                )
                code_event.update({"path": path, "code_block": block_index, "code": code})
                events.append(code_event)

                repl_result = block.get("result") or {}
                execution_id = f"{step_run_id}.execution.{block_index}"
                execution_event = _event(
                    event_id=execution_id,
                    run_id=step_run_id,
                    parent_run_id=code_id,
                    depth=depth,
                    step=iteration_index,
                    event_type="execution_result",
                    task_id=task_id,
                    role=agent_role,
                )
                execution_event.update(
                    {
                        "path": path,
                        "code_block": block_index,
                        "output": repl_result.get("stdout") or repl_result.get("output") or "",
                        "error": repl_result.get("stderr") or "",
                        "execution_time_s": repl_result.get("execution_time"),
                        "sub_lm_count": len(_calls_from_block(block)),
                    }
                )
                events.append(execution_event)

                for call_index, call in enumerate(_calls_from_block(block), start=1):
                    call_id = f"{execution_id}.sub.{call_index}"
                    emit_call(
                        call,
                        run_id=call_id,
                        parent_run_id=execution_id,
                        depth=depth + 1,
                        step=iteration_index,
                        path=f"{path} → SUB-LM #{call_index}",
                    )

            step_end = _event(
                event_id=f"{step_run_id}.end",
                run_id=step_run_id,
                parent_run_id=run_id if parent_run_id is not None or depth > 0 else None,
                depth=depth,
                step=iteration_index,
                event_type="agent_end",
                task_id=task_id,
                role=agent_role,
            )
            step_end.update(
                {
                    "path": path,
                    "output": iteration.get("response", ""),
                    "iteration_time_s": iteration.get("iteration_time"),
                    "final_answer": iteration.get("final_answer"),
                }
            )
            events.append(step_end)
            if iteration.get("final_answer") is not None:
                final_event = _event(
                    event_id=f"{step_run_id}.final",
                    run_id=step_run_id,
                    parent_run_id=step_run_id,
                    depth=depth,
                    step=iteration_index,
                    event_type="final_result",
                    task_id=task_id,
                    role=agent_role,
                )
                final_event.update({"path": path, "answer": iteration.get("final_answer")})
                events.append(final_event)

    emit_agent(metadata, run_id=root_run_id, parent_run_id=None, depth=0)

    end = _event(
        event_id="root.end",
        run_id=root_run_id,
        parent_run_id=None,
        depth=0,
        step=len(iterations),
        event_type="run_end",
        task_id=task_id,
        role="root",
    )
    if result:
        end.update(
            {
                "status": result.get("status"),
                "answer": result.get("answer"),
                "score": result.get("score"),
                "error": result.get("error"),
                "cost_usd": result.get("total_cost_usd"),
            }
        )
    events.append(end)
    return events


def write_event_log(
    path: Path,
    metadata: dict[str, Any] | None,
    *,
    task_id: str | None = None,
    result: dict[str, Any] | None = None,
) -> Path:
    """Persist the derived explicit event tree as newline-delimited JSON."""
    events = build_event_log(metadata, task_id=task_id, result=result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    return path
