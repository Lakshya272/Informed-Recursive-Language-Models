import copy
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any

from rlm.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from rlm.core.types import REPLResult, RLMChatCompletion
from rlm.utils.rlm_query_args import (
    invoke_subcall_fn,
    iter_context_query_pairs,
    stitch_query_and_context,
)
from rlm.environments.base_env import (
    RESERVED_TOOL_NAMES,
    NonIsolatedEnv,
    extract_tool_value,
    validate_custom_tools,
)
from rlm.utils.coverage import (
    compute_unvisited_ranges,
    extract_atomic_units,
    match_prompt_against_units,
)



class _AnswerDict(dict):
    """REPL-visible dict where ``answer["ready"] = True`` signals completion.

    Behaves exactly like ``dict`` for the model, but invokes ``on_ready`` the
    first time ``ready`` flips truthy. The callback receives the current
    ``content``, lets the env capture it (in-process attr, broker push, etc.),
    and the next ``execute_code`` will surface it as ``REPLResult.final_answer``.
    """

    def __init__(self, on_ready=None):
        super().__init__()
        super().__setitem__("content", "")
        super().__setitem__("ready", False)
        self._on_ready = on_ready

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "ready" and value and self._on_ready is not None:
            try:
                self._on_ready(self.get("content", ""))
            except Exception:
                pass


# =============================================================================
# Safe Builtins and Air-Gapped Sandbox Guards
# =============================================================================

import builtins

# Whitelisted standard library modules for RLM reasoning, text processing, and math.
# All networking (urllib, requests, socket, http, httpx, aiohttp), OS traversal
# (os, glob, pathlib, shutil, sys), external datasets (pyarrow, datasets, pandas, git),
# and process execution (subprocess, multiprocessing) are strictly blocked.
_ALLOWED_MODULES = {
    "math",
    "cmath",
    "re",
    "json",
    "collections",
    "itertools",
    "string",
    "random",
    "datetime",
    "time",
    "copy",
    "heapq",
    "bisect",
    "typing",
    "functools",
    "operator",
    "unicodedata",
    "csv",
    "io",
    "difflib",
    "decimal",
    "fractions",
    "statistics",
    "numbers",
    "textwrap",
    "sympy",
    "ast",
}

_REAL_IMPORT = builtins.__import__


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restricted import allowing only whitelisted safe standard-library modules."""
    root = name.split(".")[0]
    if root not in _ALLOWED_MODULES:
        raise PermissionError(
            f"Importing module '{name}' is disallowed in the evaluation sandbox. "
            f"Network requests, host filesystem inspection, and external packages are prohibited."
        )
    return _REAL_IMPORT(name, globals, locals, fromlist, level)


def _default_safe_open(file, *args, **kwargs):
    raise PermissionError("open() is only available within an initialized LocalREPL instance.")


# Safe builtins - blocks dangerous operations like eval/exec/input
_SAFE_BUILTINS = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": _safe_import,
    "open": _default_safe_open,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


class LocalREPL(NonIsolatedEnv):
    """
    Local REPL environment with persistent Python namespace.
    Executes code in a sandboxed namespace with access to context data.
    """

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        subcall_fn: Callable[[str, str | None], RLMChatCompletion] | None = None,
        custom_tools: dict[str, Any] | None = None,
        custom_sub_tools: dict[str, Any] | None = None,
        compaction: bool = False,
        max_concurrent_subcalls: int = 4,
        paper_mode: bool = False,
        max_depth: int = 1,
        **kwargs,
    ):
        super().__init__(
            persistent=persistent,
            depth=depth,
            max_concurrent_subcalls=max_concurrent_subcalls,
            **kwargs,
        )

        self.lm_handler_address = lm_handler_address
        self.subcall_fn = subcall_fn  # Callback for recursive RLM calls (depth > 1 support)
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
        self._lock = threading.Lock()
        self._context_count: int = 0
        self._history_count: int = 0
        self.compaction = compaction
        self.paper_mode = paper_mode
        self.max_depth = max_depth

        # Coverage tracking state for coverage gate
        self._context_lines: list[str] = []
        self._total_context_lines: int = 0
        self._visited_indices: set[int] = set()
        self._atomic_units: list[dict[str, Any]] = []

        # Grounded inspection tracking for evidence_preference M2 intercept
        self.executed_commands: list[str] = []
        self.executed_outputs: list[str] = []

        # Closed-list cache for anti-redundancy and cost reduction
        self._subcall_cache: dict[str, str] = {}
        self._subcall_cache_hits: int = 0


        # Custom tools: functions available in the REPL
        self.custom_tools = custom_tools or {}
        # Sub-tools: inherited from custom_tools if not specified
        self.custom_sub_tools = (
            custom_sub_tools if custom_sub_tools is not None else self.custom_tools
        )

        # Validate custom tools don't override reserved names
        validate_custom_tools(self.custom_tools)

        # Setup globals, locals, and modules in environment.
        self.setup()

        if compaction:
            self._compaction_history: list[Any] = []
            self.locals["history"] = self._compaction_history

        # Load context if provided
        if context_payload is not None:
            self.load_context(context_payload)

        # Run setup code if provided
        if setup_code:
            self.execute_code(setup_code)

    def _create_safe_open(self):
        """Create a restricted open() function scoped strictly to this REPL's temporary directory."""
        _real_open = builtins.open
        allowed_root = os.path.abspath(self.temp_dir).lower()

        def _safe_open(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                file_str = os.fspath(file)
                norm_p = file_str.replace("\\", "/").lower()
                if norm_p.startswith("/task/") or norm_p.startswith("task/"):
                    rel = norm_p.split("task/", 1)[1]
                    target_path = os.path.abspath(os.path.join(self.temp_dir, rel)).lower()
                elif not os.path.isabs(file_str):
                    target_path = os.path.abspath(os.path.join(self.temp_dir, file_str)).lower()
                else:
                    target_path = os.path.abspath(file_str).lower()
                try:
                    common = os.path.commonpath([allowed_root, target_path]).lower()
                    if common != allowed_root:
                        raise PermissionError(
                            f"Access to path '{file}' is denied. "
                            f"Evaluation REPL can only access files within its sandbox temporary directory."
                        )
                except ValueError:
                    raise PermissionError(
                        f"Access to path '{file}' is denied across drives."
                    )
            else:
                raise PermissionError("Direct file descriptors are not permitted in safe open.")

            return _real_open(file, *args, **kwargs)

        return _safe_open

    def setup(self):
        """Setup the environment."""
        # Create sandboxed globals
        safe_builtins = _SAFE_BUILTINS.copy()
        safe_builtins["open"] = self._create_safe_open()
        safe_builtins["__import__"] = _safe_import
        self.globals: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
            "open": safe_builtins["open"],
        }
        self.locals: dict[str, Any] = {}

        # Track LLM calls made during code execution
        self._pending_llm_calls: list[RLMChatCompletion] = []
        # Captured the first time the model sets ``answer["ready"] = True``.
        self._last_final_answer: str | None = None

        # Add helper functions
        if not self.paper_mode:
            self.globals["SHOW_VARS"] = self._show_vars
        if not self.paper_mode or self.max_depth > 0:
            self.globals["llm_query"] = self._llm_query
            if not self.paper_mode:
                self.globals["llm_query_batched"] = self._llm_query_batched
        # Appendix C (1a) exposes no recursive tool. In paper depth>1 mode,
        # RLM passes a callback and the 1c prompt explicitly exposes rlm_query.
        if (not self.paper_mode or self.subcall_fn is not None) and self.max_depth > 1:
            self.globals["rlm_query"] = self._rlm_query
            if not self.paper_mode:
                self.globals["rlm_query_batched"] = self._rlm_query_batched

        # Prime Intellect / LongCoT Appendix C.3 compatibility alias
        self.globals["llm_batch"] = self._llm_query_batched

        # The model marks completion via ``answer["ready"] = True``; the
        # custom dict captures the content as soon as that happens so we
        # don't have to probe the namespace after every cell.
        self.locals["answer"] = _AnswerDict(on_ready=self._capture_answer)

        # Coverage gate tracking functions
        self.globals["record_processed"] = self._record_processed
        self.globals["coverage_status"] = self._coverage_status

        # Add custom tools to globals
        # Tools can be either plain values or (value, description) tuples
        for name, entry in self.custom_tools.items():
            value = extract_tool_value(entry)
            if callable(value):
                self.globals[name] = value
            else:
                # For non-callable values (constants, data), add to locals
                self.locals[name] = value

    def _record_processed(self, start_idx: int, end_idx: int) -> None:
        """The model calls this after it has sent a chunk of lines to a sub-call.
        Appends those indices to an internal visited set for this run.
        """
        try:
            s = int(start_idx)
            e = int(end_idx)
            high = max(s + 1, e)
            for idx in range(max(0, s), min(self._total_context_lines, high)):
                self._visited_indices.add(idx)
        except Exception:
            pass

    def _coverage_status(self) -> dict[str, Any]:
        """Returns {"covered": k, "total": N, "fraction": k/N, "unvisited_ranges": [...]}.
        No LLM call involved, purely local bookkeeping, safe to call as often as the model likes.
        """
        total = self._total_context_lines
        covered = len(self._visited_indices)
        fraction = round(covered / total, 4) if total > 0 else 1.0
        unvisited = compute_unvisited_ranges(total, self._visited_indices)
        return {
            "covered": covered,
            "total": total,
            "fraction": fraction,
            "unvisited_ranges": unvisited,
        }

    def get_coverage_status(self) -> dict[str, Any]:
        """Return coverage status dictionary for the environment."""
        return self._coverage_status()

    def get_subcall_cache_stats(self) -> dict[str, Any]:
        """Return closed-list cache stats for anti-redundancy measurement."""
        return {
            "cache_size": len(getattr(self, "_subcall_cache", {})),
            "cache_hits": getattr(self, "_subcall_cache_hits", 0),
        }

    def reset_answer_ready(self) -> None:
        """Reset answer['ready'] to False (e.g. on coverage gate bounce)."""
        ans = self.locals.get("answer")
        if isinstance(ans, dict):
            dict.__setitem__(ans, "ready", False)
        self._last_final_answer = None

    def _capture_answer(self, content: Any) -> None:
        self._last_final_answer = str(content)

    def _show_vars(self) -> str:
        """Show all available variables in the REPL environment."""
        available = {
            k: type(v).__name__
            for k, v in self.locals.items()
            if not k.startswith("_") and k != "answer"
        }
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        """Query the LM with a single plain completion (no REPL, no recursion).

        This always makes a direct LM call via the handler, regardless of depth.

        Args:
            prompt: The prompt to send to the LM.
            model: Optional model name to use (if handler has multiple clients).
        """
        if not self.lm_handler_address:
            return "Error: No LM handler configured"

        cache_key = hashlib.sha256(f"{model}:{prompt}".encode("utf-8")).hexdigest()
        if hasattr(self, "_subcall_cache") and cache_key in self._subcall_cache:
            self._subcall_cache_hits += 1
            return self._subcall_cache[cache_key]

        try:
            request = LMRequest(prompt=prompt, model=model, depth=self.depth)
            response = send_lm_request(self.lm_handler_address, request)

            if not response.success:
                return f"Error: {response.error}"

            response.chat_completion.call_type = "llm_query"
            self._pending_llm_calls.append(response.chat_completion)

            # Automatic transparent coverage tracking (Step 6)
            if self._atomic_units:
                try:
                    matched = match_prompt_against_units(str(prompt), self._atomic_units)
                    self._visited_indices.update(matched)
                except Exception:
                    pass

            ans = response.chat_completion.response
            if hasattr(self, "_subcall_cache"):
                self._subcall_cache[cache_key] = ans
            return ans
        except Exception as e:
            return f"Error: LM query failed - {e}"

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Query the LM with multiple prompts concurrently (no REPL, no recursion).

        This always makes direct LM calls via the handler, regardless of depth.

        Args:
            prompts: List of prompts to send to the LM.
            model: Optional model name to use (if handler has multiple clients).

        Returns:
            List of responses in the same order as input prompts.
        """
        if not self.lm_handler_address:
            return ["Error: No LM handler configured"] * len(prompts)

        # Check subcall cache for batched prompts
        results = [None] * len(prompts)
        missed_indices = []
        missed_requests = []
        for idx, p in enumerate(prompts):
            ckey = hashlib.sha256(f"{model}:{p}".encode("utf-8")).hexdigest()
            if hasattr(self, "_subcall_cache") and ckey in self._subcall_cache:
                self._subcall_cache_hits += 1
                results[idx] = self._subcall_cache[ckey]
            else:
                missed_indices.append(idx)
                missed_requests.append(LMRequest(prompt=p, model=model, depth=self.depth))

        if not missed_requests:
            return results

        try:
            responses = send_lm_request_batched(
                self.lm_handler_address, missed_requests, max_workers=self.max_concurrent_subcalls
            )

            for i, resp in enumerate(responses):
                orig_idx = missed_indices[i]
                orig_prompt = prompts[orig_idx]
                ckey = hashlib.sha256(f"{model}:{orig_prompt}".encode("utf-8")).hexdigest()

                if not resp.success:
                    results[orig_idx] = f"Error: {resp.error}"
                else:
                    resp.chat_completion.call_type = "llm_query"
                    self._pending_llm_calls.append(resp.chat_completion)
                    if self._atomic_units:
                        try:
                            matched = match_prompt_against_units(str(resp.chat_completion.prompt), self._atomic_units)
                            self._visited_indices.update(matched)
                        except Exception:
                            pass
                    ans = resp.chat_completion.response
                    results[orig_idx] = ans
                    if hasattr(self, "_subcall_cache"):
                        self._subcall_cache[ckey] = ans

            return results
        except Exception as e:
            return [f"Error: LM query failed - {e}"] * len(prompts)

    def _rlm_query(self, context, query, model: str | None = None) -> str:
        """Spawn a recursive RLM sub-call (paper: ``rlm_query(context, query)``).

        When a subcall callback is available (max_depth > 1), this spawns a child
        RLM whose REPL ``context`` is ``context`` and whose question is ``query``.
        Falls back to a plain llm_query of query+context if recursion is off.
        """
        if self.subcall_fn is not None:
            try:
                completion = invoke_subcall_fn(self.subcall_fn, context, model, query)
                completion.call_type = "rlm_query"
                self._pending_llm_calls.append(completion)
                if self._atomic_units:
                    try:
                        combined_txt = f"{context}\n{query}"
                        matched = match_prompt_against_units(combined_txt, self._atomic_units)
                        self._visited_indices.update(matched)
                    except Exception:
                        pass
                return completion.response
            except Exception as e:
                return f"Error: RLM query failed - {e}"

        return self._llm_query(stitch_query_and_context(context, query), model)

    def _rlm_query_batched(self, items, model: str | None = None) -> list[str]:
        """Spawn recursive RLM sub-calls for ``(context, query)`` pairs in parallel.

        Each item may be a ``(context, query)`` pair or a bare context string.
        Falls back to llm_query_batched if no recursive capability is configured.
        """
        pairs = list(iter_context_query_pairs(items))
        if self.subcall_fn is not None:
            # For 0 or 1 prompts, no need for thread pool overhead
            if len(pairs) <= 1:
                results = []
                for context, query in pairs:
                    try:
                        completion = invoke_subcall_fn(self.subcall_fn, context, model, query)
                        completion.call_type = "rlm_query"
                        self._pending_llm_calls.append(completion)
                        if self._atomic_units:
                            try:
                                combined_txt = f"{context}\n{query}"
                                matched = match_prompt_against_units(combined_txt, self._atomic_units)
                                self._visited_indices.update(matched)
                            except Exception:
                                pass
                        results.append(completion.response)
                    except Exception as e:
                        results.append(f"Error: RLM query failed - {e}")
                return results

            # Parallel execution for multiple prompts
            max_workers = min(self.max_concurrent_subcalls, len(pairs))
            # Pre-allocate result slots to preserve ordering
            results: list[str] = [""] * len(pairs)
            completions: list[tuple[int, RLMChatCompletion]] = []
            lock = threading.Lock()

            def _run_subcall(index: int, context, query: str) -> None:
                try:
                    completion = invoke_subcall_fn(self.subcall_fn, context, model, query)
                    completion.call_type = "rlm_query"
                    with lock:
                        completions.append((index, completion))
                    results[index] = completion.response
                except Exception as e:
                    results[index] = f"Error: RLM query failed - {e}"

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_run_subcall, i, context, query)
                    for i, (context, query) in enumerate(pairs)
                ]
                # Wait for all futures to complete; exceptions are captured inside _run_subcall
                for future in as_completed(futures):
                    future.result()  # Re-raises unexpected executor errors

            # Append completions in original prompt order for deterministic metadata
            completions.sort(key=lambda x: x[0])
            for _, completion in completions:
                self._pending_llm_calls.append(completion)
                if self._atomic_units:
                    try:
                        prompt_txt = str(completion.prompt)
                        matched = match_prompt_against_units(prompt_txt, self._atomic_units)
                        self._visited_indices.update(matched)
                    except Exception:
                        pass

            return results

        # Fall back to plain batched LM call if no recursive capability
        return self._llm_query_batched(
            [stitch_query_and_context(context, query) for context, query in pairs],
            model,
        )

    def load_context(self, context_payload: dict | list | str):
        """Load context into the environment as context_0 (and 'context' alias)."""
        self.add_context(context_payload, 0)

    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int:
        """
        Add a context with versioned variable name.

        Args:
            context_payload: The context data to add
            context_index: Optional explicit index. If None, auto-increments.

        Returns:
            The context index used.
        """
        if context_index is None:
            context_index = self._context_count

        var_name = f"context_{context_index}"

        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
            with open(context_path, "w", encoding="utf-8") as f:
                f.write(context_payload)
            self.execute_code(f"with open(r'{context_path}', 'r', encoding='utf-8') as f:\n    {var_name} = f.read()")
        else:
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.json")
            with open(context_path, "w", encoding="utf-8") as f:
                json.dump(context_payload, f)
            self.execute_code(
                f"import json\nwith open(r'{context_path}', 'r', encoding='utf-8') as f:\n    {var_name} = json.load(f)"
            )

        # Alias context_0 as 'context' for backward compatibility
        if context_index == 0:
            self.execute_code(f"context = {var_name}")
            raw_str = (
                context_payload
                if isinstance(context_payload, str)
                else json.dumps(context_payload)
            )
            self._context_lines = [
                l.strip() for l in raw_str.strip().split("\n") if l.strip()
            ]
            self._total_context_lines = len(self._context_lines)
            self._visited_indices = set()
            self._atomic_units = extract_atomic_units(raw_str)


        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def update_handler_address(self, address: tuple[str, int]) -> None:
        """Update the LM handler address for a new completion call."""
        self.lm_handler_address = address

    def get_context_count(self) -> int:
        """Return the number of contexts loaded."""
        return self._context_count

    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int:
        """
        Store a conversation's message history as a versioned variable.

        Args:
            message_history: The list of message dicts from a completion call
            history_index: Optional explicit index. If None, auto-increments.

        Returns:
            The history index used.
        """
        if history_index is None:
            history_index = self._history_count

        var_name = f"history_{history_index}"

        # Store deep copy to avoid reference issues with nested dicts
        self.locals[var_name] = copy.deepcopy(message_history)

        # Alias history_0 as 'history' for convenience
        if history_index == 0:
            self.locals["history"] = self.locals[var_name]

        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        """Return the number of conversation histories stored."""
        return self._history_count

    def append_compaction_entry(self, entry: list[dict[str, Any]] | dict[str, Any]) -> None:
        """
        Append a trajectory segment or a summary to the compaction history.

        Entry is either a list of message dicts (trajectory segment) or
        a dict with "type": "summary" and "content": str.
        """
        if not self.compaction:
            return
        self._compaction_history.append(copy.deepcopy(entry))

    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_cwd(self):
        """Temporarily change to temp directory for execution."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def _restore_scaffold(self) -> None:
        """Restore scaffold names after execution so overwrites (e.g. context = 'x') don't persist."""
        for blocked_name in ("open", "__import__", "__builtins__"):
            self.locals.pop(blocked_name, None)

        safe_open = self._create_safe_open()
        if isinstance(self.globals.get("__builtins__"), dict):
            self.globals["__builtins__"]["open"] = safe_open
            self.globals["__builtins__"]["__import__"] = _safe_import
        self.globals["open"] = safe_open

        for name in RESERVED_TOOL_NAMES:
            if name == "llm_query":
                if not self.paper_mode or self.max_depth > 0:
                    self.globals["llm_query"] = self._llm_query
            elif name == "llm_query_batched":
                if not self.paper_mode:
                    self.globals["llm_query_batched"] = self._llm_query_batched
            elif name == "rlm_query":
                if (not self.paper_mode or self.subcall_fn is not None) and self.max_depth > 1:
                    self.globals["rlm_query"] = self._rlm_query
            elif name == "rlm_query_batched":
                if not self.paper_mode and self.max_depth > 1:
                    self.globals["rlm_query_batched"] = self._rlm_query_batched
            elif name == "SHOW_VARS":
                if not self.paper_mode:
                    self.globals["SHOW_VARS"] = self._show_vars
            elif name == "answer":
                current = self.locals.get("answer")
                # If the model rebound ``answer`` to a plain dict, the
                # _AnswerDict callback never fired; capture content here if
                # ``ready=True``, then re-wrap so the next cell signals.
                if not isinstance(current, _AnswerDict):
                    replacement = _AnswerDict(on_ready=self._capture_answer)
                    if isinstance(current, dict):
                        for k, v in current.items():
                            dict.__setitem__(replacement, k, v)
                        if current.get("ready") and self._last_final_answer is None:
                            self._last_final_answer = str(current.get("content", ""))
                    self.locals["answer"] = replacement
            elif name == "context" and "context_0" in self.locals:
                self.locals["context"] = self.locals["context_0"]
            elif name == "history" and "history_0" in self.locals and not self.compaction:
                self.locals["history"] = self.locals["history_0"]
            elif name == "history" and self.compaction:
                self.locals["history"] = self._compaction_history
            elif name == "record_processed":
                self.globals["record_processed"] = self._record_processed
            elif name == "coverage_status":
                self.globals["coverage_status"] = self._coverage_status


    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the persistent namespace and return result."""
        start_time = time.perf_counter()

        # Clear pending LLM calls from previous execution
        self._pending_llm_calls = []

        with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
            try:
                combined = {**self.globals, **self.locals}
                safe_open = self._create_safe_open()
                if isinstance(combined.get("__builtins__"), dict):
                    combined["__builtins__"]["open"] = safe_open
                    combined["__builtins__"]["__import__"] = _safe_import
                def _run_exec():
                    exec(code, combined, combined)

                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_run_exec)
                    try:
                        future.result(timeout=90.0)
                    except TimeoutError:
                        stderr_buf.write("\nTimeoutError: Code execution timed out after 90.0 seconds.")

                # Update locals with new variables
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value

                # Restore scaffold so model overwrites (context = ..., llm_query = ...) don't persist
                self._restore_scaffold()

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"

        self.executed_commands.append(code)
        self.executed_outputs.append(stdout)

        final_answer = self._last_final_answer
        self._last_final_answer = None
        if not final_answer:
            ans_file = os.path.join(self.temp_dir, "answer.txt")
            if os.path.exists(ans_file):
                try:
                    with builtins.open(ans_file, "r", encoding="utf-8") as f:
                        txt = f.read().strip()
                    if txt:
                        final_answer = txt
                except Exception:
                    pass

        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
            rlm_calls=self._pending_llm_calls.copy(),
            final_answer=final_answer,
        )

    def get_grounded_evidence(self) -> str:
        """Extract grounded evidence from REPL execution (e.g. stdout from search/inspections)."""
        inspections = []
        for cmd, out in zip(self.executed_commands, self.executed_outputs):
            if any(k in cmd for k in ["grep", "find", "search", "context", "open(", "read()", "print"]):
                out_clean = out.strip()
                if out_clean and not out_clean.startswith("Error:"):
                    inspections.append(f"Command: {cmd.strip()[:100]}\nOutput:\n{out_clean[:250]}")
        if inspections:
            return "\n---\n".join(inspections[-3:])
        return ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        """Clean up temp directory and reset state."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        if hasattr(self, "globals"):
            self.globals.clear()
        if hasattr(self, "locals"):
            self.locals.clear()

    def __del__(self):
        self.cleanup()
