from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def summarize_training_splits(
    results: list[dict[str, Any]],
    top_ks: list[int],
    summarize_fn: Callable[[list[dict[str, Any]], list[int]], dict[str, Any]],
    cfg: Any,
) -> dict[str, Any] | None:
    manifest_path = configured_manifest_path(cfg)
    if manifest_path is None:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trained_source_rows = {
        int(index) for index in manifest.get("source_row_indices", [])
    }
    trained_question_keys = manifest_question_keys(manifest)

    trained_row_results = [
        item for item in results if int(item["example_index"]) in trained_source_rows
    ]
    untrained_row_results = [
        item for item in results if int(item["example_index"]) not in trained_source_rows
    ]
    trained_question_results = [
        item for item in results if result_question_key(item) in trained_question_keys
    ]
    untrained_question_results = [
        item for item in results if result_question_key(item) not in trained_question_keys
    ]

    return {
        "train_data_manifest_path": str(manifest_path),
        "num_training_samples": manifest.get("num_training_samples"),
        "num_trained_source_rows": len(trained_source_rows),
        "num_trained_question_groups": len(trained_question_keys),
        "by_source_row": {
            "trained_source_rows": summarize_fn(trained_row_results, top_ks),
            "untrained_source_rows": summarize_fn(untrained_row_results, top_ks),
        },
        "by_exact_question": {
            "trained_questions": summarize_fn(trained_question_results, top_ks),
            "untrained_questions": summarize_fn(untrained_question_results, top_ks),
        },
    }


def manifest_question_keys(manifest: dict[str, Any]) -> set[tuple[int, int]]:
    explicit_keys = {
        (int(item["source_row_index"]), int(item["qa_index"]))
        for item in manifest.get("source_question_keys", [])
        if item.get("source_row_index") is not None and item.get("qa_index") is not None
    }
    if explicit_keys:
        return explicit_keys

    return {
        (int(group["source_row_index"]), int(group["qa_index"]))
        for group in manifest.get("question_groups", [])
        if group.get("source_row_index") is not None and group.get("qa_index") is not None
    }


def configured_manifest_path(cfg: Any) -> Path | None:
    evaluate_cfg = cfg.get("evaluate", {}) if hasattr(cfg, "get") else {}
    value = evaluate_cfg.get("train_data_manifest_path", None)
    if value in {None, "", "null", "None"}:
        return None
    path = Path(str(value)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Training data manifest not found: {path}")
    return path


def result_question_key(item: dict[str, Any]) -> tuple[int, int]:
    return int(item["example_index"]), int(item.get("qa_index", 0))
