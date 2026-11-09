"""
Training-data loader (REWRITTEN for the rebuttal package).

The original `src/data/load_train_data.py` of implicit_version_03 was never committed. This file
re-implements the two loaders the training scripts select with `train.data.name`:

  * oppose_gen_grouped  (the PAPER setting; v11con_2400 rows)
        row = {query, gold_session_fact: [8 str], general_fact: [8 str], memory_id, ...}
        -> one sample per gold fact:  {query, positive=gold_i, negative=general[pairing(i)],
                                        group_id="oppose_gen_grouped:<memory_id>", ...}
        negative_pairing: cycle_negatives (default; general[i % len]) | all_pairs | aligned
        Emits NORMALIZED rows (keys query/positive/negative) -> emits_normalized_rows()==True.
  * oppose_gen          (legacy triplet rows; pass-through, *_field names the emitted keys)

What MUST match the original for the rebuttal to be valid:
  - the validation split: split_validation() (in the entry point, unchanged) shuffles the SORTED
    set of group_id strings with Random(42) and takes the first 10%. Group ids are
    "<prefix>:<memory_id>"; the prefix is common to all rows, so the sorted order -- and therefore
    the val split -- depends only on memory_id. Reproduced exactly.
  - the sample count per row (one per gold fact under cycle_negatives).
Verify with shared/tools/check_loader_against_manifest.py against any ORIGINAL adapter directory
(it holds train_data_manifest.json with num_training_samples and source_row_indices).
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any

NORMALIZED_FIELDS = {"query": "query", "positive": "positive", "negative": "negative"}
NORMALIZING_LOADERS = {"oppose_gen_grouped"}


def emits_normalized_rows(cfg: Any) -> bool:
    return str(cfg.train.data.get("name", "")) in NORMALIZING_LOADERS


def load_train_data(cfg: Any) -> list[dict[str, Any]]:
    name = str(cfg.train.data.get("name", ""))
    if name == "oppose_gen_grouped":
        return load_train_oppose_gen_grouped(cfg)
    if name == "oppose_gen":
        return load_train_oppose_gen_triplet(cfg)
    raise ValueError(f"Unsupported train.data.name={name!r} (oppose_gen_grouped | oppose_gen)")


# --------------------------------------------------------------------------- helpers

def resolve_fact_jsonl_path(cfg: Any) -> Path:
    path = Path(str(cfg.train.data.path)).expanduser()
    file = cfg.train.data.get("file", None)
    if path.is_dir():
        if file:
            path = path / str(file)
        else:
            cands = sorted(path.glob("*.jsonl"))
            if not cands:
                raise FileNotFoundError(f"No .jsonl under {path}")
            path = cands[0]
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    return path


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def source_row_sample_seed(cfg: Any) -> int:
    return int(cfg.train.data.get("source_row_sample_seed", 42) or 42)


def select_source_rows(rows: list[dict[str, Any]], cfg: Any) -> list[tuple[int, dict[str, Any]]]:
    data = cfg.train.data
    indexed = list(enumerate(rows))
    start = int(data.get("start_source_row", 0) or 0)
    indexed = indexed[start:]
    ratio = data.get("source_row_ratio", None)
    if ratio not in (None, "", "null"):
        ratio = float(ratio)
        if not 0 < ratio <= 1:
            raise ValueError("train.data.source_row_ratio must be in (0, 1]")
        n = max(1, int(round(len(indexed) * ratio)))
        if str(data.get("source_row_sampling", "random")) == "first":
            indexed = indexed[:n]
        else:
            rng = random.Random(source_row_sample_seed(cfg))
            indexed = sorted(rng.sample(indexed, n), key=lambda x: x[0])
    max_rows = data.get("max_source_rows", None)
    if max_rows not in (None, "", "null"):
        indexed = indexed[: int(max_rows)]
    return indexed


def require_row_text(row: dict[str, Any], field: str, index: int) -> str:
    value = row.get(field)
    if value is None or not str(value).strip():
        raise ValueError(f"Row {index}: missing/empty field {field!r}")
    return str(value)


def normalize_fact_list(value: Any, field: str, index: int, min_len: int = 1) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        parsed = None
        if text.startswith("["):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(text)
                    break
                except Exception:  # noqa: BLE001
                    continue
        value = parsed if isinstance(parsed, list) else [text]
    if value is None:
        value = []
    facts = [str(v).strip() for v in value if v is not None and str(v).strip()]
    if len(facts) < min_len:
        raise ValueError(f"Row {index}: field {field!r} has {len(facts)} facts (< {min_len})")
    return facts


def resolve_source_row_index(row: dict[str, Any], local_index: int) -> int:
    for key in ("source_row_index", "row_index"):
        if row.get(key) is not None:
            return int(row[key])
    return int(local_index)


def _pair(golds: list[str], negs: list[str], mode: str, strict: bool, index: int) -> list[tuple[int, str, str]]:
    if mode == "cycle_negatives":
        return [(i, g, negs[i % len(negs)]) for i, g in enumerate(golds)]
    if mode == "all_pairs":
        return [(i, g, n) for i, g in enumerate(golds) for n in negs]
    if mode == "aligned":
        if strict and len(golds) != len(negs):
            raise ValueError(f"Row {index}: aligned pairing needs equal counts ({len(golds)} vs {len(negs)})")
        return [(i, g, n) for i, (g, n) in enumerate(zip(golds, negs))]
    raise ValueError(f"Unknown negative_pairing={mode!r}")


# --------------------------------------------------------------------------- loaders

def load_train_oppose_gen_grouped(cfg: Any) -> list[dict[str, Any]]:
    data = cfg.train.data
    rows = select_source_rows(read_jsonl_rows(resolve_fact_jsonl_path(cfg)), cfg)
    query_field = str(data.get("query_field", "query"))
    pos_field = str(data.get("positive_field", "gold_session_fact"))
    neg_field = str(data.get("negative_field", "general_fact"))
    if pos_field in ("positive_memory", "user_memory_fact"):
        pos_field = "gold_session_fact"          # grouped loader expects LIST fields
    if neg_field in ("surface_negative", "random_negative"):
        neg_field = "general_fact"
    group_source_field = str(data.get("group_source_field", "memory_id"))
    mode = str(data.get("negative_pairing", "cycle_negatives"))
    strict = bool(data.get("strict_pair_count", False))

    out: list[dict[str, Any]] = []
    for local_i, row in rows:
        query = require_row_text(row, query_field, local_i)
        golds = normalize_fact_list(row.get(pos_field), pos_field, local_i, 1)
        negs = normalize_fact_list(row.get(neg_field), neg_field, local_i, 1)
        sri = resolve_source_row_index(row, local_i)
        group_source = row.get(group_source_field, sri)
        for fact_index, pos, neg in _pair(golds, negs, mode, strict, local_i):
            out.append({
                "query": query,
                "positive": pos,
                "negative": neg,
                "group_id": f"oppose_gen_grouped:{group_source}",
                "answer": str(row.get("answer", "") or ""),
                "query_topic": row.get("query_topic", row.get("surface_topic")),
                "source_row_index": sri,
                "qa_index": 0,
                "source_qa_index": 0,
                "local_qa_index": 0,
                "fact_pair_index": fact_index,
            })
    max_samples = data.get("max_expanded_samples", None)
    if max_samples not in (None, "", "null"):
        out = out[: int(max_samples)]
    return out


def load_train_oppose_gen_triplet(cfg: Any) -> list[dict[str, Any]]:
    data = cfg.train.data
    rows = select_source_rows(read_jsonl_rows(resolve_fact_jsonl_path(cfg)), cfg)
    group_source_field = str(data.get("group_source_field", "memory_id"))
    out = []
    for local_i, row in rows:
        r = dict(row)
        sri = resolve_source_row_index(row, local_i)
        r["group_id"] = f"oppose_gen:{row.get(group_source_field, sri)}"
        r["source_row_index"] = sri
        r.setdefault("qa_index", 0)
        out.append(r)
    return out
