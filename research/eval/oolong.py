"""Paper-aligned OOLONG trec_coarse loading and scoring."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import dateutil.parser

PAPER_DATASET_NAME = "trec_coarse"
PAPER_CONTEXT_LEN = 131_072
COMPARISON_PHRASES = ("more common than", "less common than", "same frequency as")

# Public Hugging Face layout for oolongbench/oolong-synth validation.
# Full validation is ~4.4 GB across 9 shards; paper trec_coarse@131072 lives only
# in shard 7 (~72 MB). Prefer direct parquet loads over streaming the whole split.
HF_DATASET = "oolongbench/oolong-synth"
HF_DATA_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/data"
VALIDATION_SHARDS = [f"validation-{i:05d}-of-00009.parquet" for i in range(9)]
# Empirically measured: 50 matching rows are all in shard 7.
PAPER_PREFERRED_SHARDS = ("validation-00007-of-00009.parquet",)

QUESTION_INSTRUCTION = (
    "The following lines contain 3182 general-knowledge questions, one per line. "
    "Each line has a User ID, which is not necessarily unique, i.e. each User ID "
    "can be associated with multiple questions. Each question has an answer that "
    "can be described as one of 6 categories: 'numeric value', 'entity', "
    "'location', 'description and abstract concept', 'abbreviation', 'human being' "
    "(the data does not provide the labels, you need to figure out the label from "
    "the semantics of the question). You will be asked to answer questions about "
    "the aggregate label statistics across all examples in this dataset. Do not "
    "try to guess, estimate, or approximate the result. Calculate the exact answer "
    "given these datapoints."
)

FILTER_COLUMNS = ("id", "dataset", "context_len", "answer_type")
TASK_COLUMNS = (
    "id",
    "dataset",
    "context_len",
    "question",
    "context_window_text",
    "answer",
    "answer_type",
)


@dataclass(frozen=True)
class OolongTask:
    source_id: str
    question: str
    context: str
    answer: str
    answer_type: str
    root_prompt: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _find_comparison_phrase(output: str) -> str | None:
    output_lower = output.lower()
    hits = [(output_lower.rfind(phrase), phrase) for phrase in COMPARISON_PHRASES if phrase in output_lower]
    return max(hits)[1] if hits else None


def _attempt_answer_parse(answer: str) -> tuple[str, str]:
    comparison = _find_comparison_phrase(answer)
    if comparison is not None:
        return comparison, "high"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, "low"
        return answer.split()[-1], "low"
    candidate = answer.split(":")[-1].strip().replace("*", "").replace("[", "").replace("]", "")
    if len(candidate) < 20:
        return candidate, "vhigh"
    return candidate, "med"


def score_answer(task: OolongTask, output: str) -> float:
    """Matches the repository's OOLONG training-environment scorer."""
    try:
        if "datetime" in task.answer:
            gold: Any = datetime.strptime(task.answer, "[datetime.date(%Y, %m, %d)]")
        else:
            gold = ast.literal_eval(task.answer)[0]
    except Exception:
        gold = task.answer

    trimmed, _ = _attempt_answer_parse(output)
    gold_text = str(gold)
    if str(trimmed) == gold_text or str(trimmed).lower() == gold_text.lower():
        return 1.0

    if task.answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            return 0.75 ** abs(int(gold) - int(trimmed))
        except Exception:
            return 0.0

    if task.answer_type == "ANSWER_TYPE.DATE":
        try:
            return 1.0 if dateutil.parser.parse(trimmed) == gold else 0.0
        except Exception:
            return 0.0

    if gold_text and gold_text.lower() not in [phrase.lower() for phrase in COMPARISON_PHRASES]:
        if gold_text.lower() in output.lower():
            return 1.0
    return 0.0


def _default_cache_dir() -> Path:
    env = os.environ.get("OOLONG_CACHE_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "tmp" / "oolong_cache"


def _ensure_shard(shard_name: str, cache_dir: Path, *, attempts: int = 5) -> Path:
    """Download a validation shard with retries; keep partials for resume."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / shard_name
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path

    url = f"{HF_DATA_BASE}/{shard_name}?download=true"
    tmp_path = local_path.with_suffix(local_path.suffix + ".partial")
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            existing = tmp_path.stat().st_size if tmp_path.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            print(
                f"Downloading OOLONG shard {shard_name} "
                f"(attempt {attempt}/{attempts}, resume_from={existing}) -> {local_path}",
                flush=True,
            )
            # Prefer requests (already an eval/runtime dep) for resume + timeouts.
            try:
                import requests
            except ImportError:
                requests = None  # type: ignore[assignment]

            if requests is not None:
                with requests.get(url, headers=headers, stream=True, timeout=60) as response:
                    # 200 = full body, 206 = partial content. Anything else is failure.
                    if response.status_code not in (200, 206):
                        raise RuntimeError(
                            f"HTTP {response.status_code} downloading {shard_name}"
                        )
                    mode = "ab" if response.status_code == 206 and existing else "wb"
                    if mode == "wb" and existing:
                        existing = 0
                    expected = response.headers.get("Content-Length")
                    written = 0
                    with tmp_path.open(mode) as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            written += len(chunk)
                    if expected is not None and written < int(expected):
                        raise RuntimeError(
                            f"short read for {shard_name}: got {written} of {expected} bytes"
                        )
            else:
                # Fallback without resume if requests is unavailable.
                urlretrieve(url, tmp_path)

            if not tmp_path.exists() or tmp_path.stat().st_size <= 0:
                raise RuntimeError(f"empty download for {shard_name}")
            tmp_path.replace(local_path)
            return local_path
        except Exception as exc:  # network blips are common on large HF files
            last_error = exc
            print(f"Shard download failed ({type(exc).__name__}: {exc})", flush=True)

    raise RuntimeError(
        f"Failed to download OOLONG shard {shard_name} after {attempts} attempts"
    ) from last_error


def _rows_from_table(table: Any) -> list[dict[str, Any]]:
    return [dict(zip(table.column_names, row, strict=True)) for row in zip(*table.to_pydict().values(), strict=True)]


def _load_matching_from_shards(
    *,
    dataset_name: str,
    context_len: int,
    cache_dir: Path,
    shard_names: list[str],
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for targeted OOLONG loading. "
            "Install evaluation dependencies with `uv sync --extra eval`."
        ) from exc

    matches: list[dict[str, Any]] = []
    for shard_name in shard_names:
        path = _ensure_shard(shard_name, cache_dir)
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        filter_cols = [col for col in FILTER_COLUMNS if col in available]
        if "dataset" not in filter_cols or "context_len" not in filter_cols:
            continue
        filter_table = parquet.read(columns=filter_cols)
        filter_rows = _rows_from_table(filter_table)
        keep_indices = [
            index
            for index, row in enumerate(filter_rows)
            if row.get("dataset") == dataset_name and int(row.get("context_len", -1)) == context_len
        ]
        if not keep_indices:
            continue
        task_cols = [col for col in TASK_COLUMNS if col in available]
        full_table = parquet.read(columns=task_cols)
        full_rows = _rows_from_table(full_table)
        for index in keep_indices:
            matches.append(full_rows[index])
    return matches


def _candidate_shards(dataset_name: str, context_len: int) -> list[str]:
    if dataset_name == PAPER_DATASET_NAME:
        if context_len == 524_288:
            return ["validation-00007-of-00009.parquet", "validation-00006-of-00009.parquet"]
        if context_len in (PAPER_CONTEXT_LEN, 262_144, 1_048_576):
            return list(PAPER_PREFERRED_SHARDS)
    return list(VALIDATION_SHARDS)


def _examples_to_tasks(examples: list[dict[str, Any]]) -> list[OolongTask]:
    tasks: list[OolongTask] = []
    for index, example in enumerate(examples):
        question = str(example["question"])
        context = str(example.get("context_window_text", example.get("context", "")))
        source_id = str(example.get("id", index))
        tasks.append(
            OolongTask(
                source_id=source_id,
                question=question,
                context=context,
                answer=str(example.get("answer", "")),
                answer_type=str(example.get("answer_type", "")),
                root_prompt=f"{QUESTION_INSTRUCTION}\n\nQuestion: {question}",
            )
        )
    return tasks


def load_tasks(
    *,
    dataset_name: str = PAPER_DATASET_NAME,
    context_len: int = PAPER_CONTEXT_LEN,
    num_examples: int = 50,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> list[OolongTask]:
    """Load public validation data and deterministically select benchmark tasks.

    Prefer targeted parquet shard downloads (cached under ``tmp/oolong_cache`` by
    default). This avoids the multi-GB streaming/index cost of the full validation
    split while still matching the paper's trec_coarse @ 131072 setting.

    The context remains out of the root model's message history. Only the question and
    instruction become ``root_prompt``; this preserves the RLM context-offloading setup.
    """
    resolved_cache = cache_dir or _default_cache_dir()
    selected = _load_matching_from_shards(
        dataset_name=dataset_name,
        context_len=context_len,
        cache_dir=resolved_cache,
        shard_names=_candidate_shards(dataset_name, context_len),
    )

    # If the preferred-shard shortcut missed (dataset layout changed), scan all shards once.
    if not selected and _candidate_shards(dataset_name, context_len) != list(VALIDATION_SHARDS):
        selected = _load_matching_from_shards(
            dataset_name=dataset_name,
            context_len=context_len,
            cache_dir=resolved_cache,
            shard_names=list(VALIDATION_SHARDS),
        )

    if not selected:
        # Last-resort fallback for environments where direct parquet access fails.
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Install the evaluation dependencies with `uv sync --extra eval` "
                "(sets up datasets/pyarrow for OOLONG loading)."
            ) from exc
        stream = load_dataset(HF_DATASET, split="validation", streaming=True)
        for example in stream:
            if example.get("dataset") != dataset_name or example.get("context_len") != context_len:
                continue
            selected.append(dict(example))

    if not selected:
        raise ValueError(
            f"No matching OOLONG examples found for dataset={dataset_name!r}, "
            f"context_len={context_len}."
        )

    rng = random.Random(seed)
    ordered = list(selected)
    rng.shuffle(ordered)
    if num_examples > len(ordered):
        raise ValueError(
            f"Only {len(ordered)} matching OOLONG examples found for "
            f"dataset={dataset_name!r}, context_len={context_len}; requested {num_examples}."
        )
    return _examples_to_tasks(ordered[:num_examples])


def manifest_entry(task: OolongTask) -> dict[str, str]:
    return {
        "source_id": task.source_id,
        "question_sha256": sha256_text(task.question),
        "context_sha256": sha256_text(task.context),
        "root_prompt_sha256": sha256_text(task.root_prompt),
        "answer_sha256": sha256_text(task.answer),
    }
