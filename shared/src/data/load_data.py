"""
Evaluation-data loader (REWRITTEN for the rebuttal package).

The original `src/data/load_data.py` of implicit_version_03 was never committed (the repo's
`src/data/` directory is missing). This file re-implements it from how every caller uses it:

    examples = load_data(cfg)          # cfg.data.{name,type,split,path}, cfg.run.{max_rows,sample_ratio}
    example["id"]                      # str
    example["conversations"]           # list[session] ; session = list[{"user","assistant"}]
    example["qa"]["questions"]         # list[str]
    example["qa"]["answers"]           # list[str]
    example["qa"]["gold_session_ids"]  # list[list[int]]   (parallel to questions)
    example["useful_indices"]          # list[list[int]]   (SAME as gold_session_ids; read by
                                       #  src.evaluate.retrieval.get_useful_indices)

Gold-session semantics are copied from the sibling baselines repo (actmem/src/data/data.py and
tacitree/framework/build_qa_gold_session_facts.py), which produced the fact bags we evaluate on:
  * ImplexConv  : each QA item's `retrieved_conv_ids` (0-based session positions, sessions ordered
                  by natural key of the `conversation` dict).
  * ActMemEval  : positions of `answer_session_ids` inside `haystack_session_ids`.
Fact bags are keyed by `row_index` == position of the example in the list returned here, so the
ROW ORDER MUST NOT CHANGE (no shuffling / filtering other than max_rows / sample_ratio prefixes).

SANITY CHECK (do this once on a new server): evaluating the paper adapter on ImplexConv-Opposed must
reproduce session_recall@10 = 0.3322 (Table 1). E1/run.sh runs this check automatically.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

HF_IDS = {
    "opposed": "Kaylee0501/ImplexConv-opposed",
    "supported": "Kaylee0501/ImplexConv-supportive",
    "supportive": "Kaylee0501/ImplexConv-supportive",
}
_RETRIEVED_ID_KEYS = (
    "retrieved_conv_ids", "retrieved_conv_id",
    "retrieved_session_ids", "retrieved_session_id",
    "gold_session_ids", "gold_session_id",
)
_MISSING = {"", "none", "null", "nan"}


def load_data(cfg: Any) -> list[dict[str, Any]]:
    data_cfg = cfg.dataset if "dataset" in cfg else cfg.data
    name = str(data_cfg.get("name"))
    if name == "implexconv":
        rows = _load_implexconv_rows(data_cfg)
        examples = [_normalize_implexconv(row, i) for i, row in enumerate(rows)]
    elif name == "actmemeval":
        path = data_cfg.get("path")
        if not path:
            raise ValueError("data.name=actmemeval requires data.path=<ActMemEval*.json>")
        rows = _coerce_rows(json.loads(Path(str(path)).read_text(encoding="utf-8")))
        examples = [_normalize_actmemeval(row, i) for i, row in enumerate(rows)]
    else:
        raise ValueError(f"Unsupported data.name={name!r} (expected implexconv | actmemeval)")

    run_cfg = cfg.get("run", {}) or {}
    max_rows = run_cfg.get("max_rows", None)
    if max_rows is not None:
        examples = examples[: int(max_rows)]
    ratio = run_cfg.get("sample_ratio", None)
    if ratio not in (None, "", "null") and float(ratio) < 1.0:
        examples = examples[: max(1, int(math.ceil(len(examples) * float(ratio))))]
    return examples


# --------------------------------------------------------------------------- ImplexConv

def _load_implexconv_rows(data_cfg: Any) -> list[dict[str, Any]]:
    from datasets import load_dataset  # lazy: only needed for ImplexConv

    dtype = str(data_cfg.get("type", "opposed") or "opposed")
    if dtype not in HF_IDS:
        raise ValueError(f"data.type must be one of {sorted(HF_IDS)} for implexconv, got {dtype!r}")
    local = data_cfg.get("path", None)
    if local:  # optional local copy (json list) when the HF hub is unreachable
        return _coerce_rows(json.loads(Path(str(local)).read_text(encoding="utf-8")))
    data = load_dataset(HF_IDS[dtype])
    split = data_cfg.get("split", None)
    key = split if (split and split in data) else next(iter(data.keys()))
    return list(data[key])


def _natural_key(value: Any) -> list[Any]:
    return [int(p) if p.isdigit() else p for p in str(value).replace("-", "_").split("_")]


def _ordered(value: Any) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return [value]
    if isinstance(value, dict):
        return [value[k] for k in sorted(value.keys(), key=_natural_key) if value[k] is not None]
    if isinstance(value, list):
        if value and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in value):
            return [x[1] for x in value]
        return value
    return [value]


def _parse_ids(value: Any) -> list[int]:
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _MISSING:
            return []
        if "," in text and not text.startswith(("[", "(", "{")):
            return [i for part in text.split(",") for i in _parse_ids(part)]
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if parsed != text:
                    return _parse_ids(parsed)
            except Exception:  # noqa: BLE001
                pass
        m = re.fullmatch(r"[A-Za-z_ -]*(?P<i>-?\d+)", text)
        return [int(m.group("i"))] if m else []
    if isinstance(value, dict):
        return [i for v in value.values() for i in _parse_ids(v)]
    if isinstance(value, (list, tuple, set)):
        out: list[int] = []
        for v in value:
            for i in _parse_ids(v):
                if i not in out:
                    out.append(i)
        return out
    return []


def _qa_items(raw_qa: Any) -> list[dict[str, Any]]:
    items = _ordered(raw_qa)
    if items and all(isinstance(x, dict) for x in items):
        return items
    # columnar dict {"question": [...], "answer": [...], "retrieved_conv_ids": [...]}
    if isinstance(raw_qa, dict) and any(isinstance(v, list) for v in raw_qa.values()):
        n = max(len(v) for v in raw_qa.values() if isinstance(v, list))
        return [{k: (v[i] if isinstance(v, list) and i < len(v) else v) for k, v in raw_qa.items()}
                for i in range(n)]
    return [x for x in items if isinstance(x, dict)]


def _normalize_implexconv(row: dict[str, Any], index: int) -> dict[str, Any]:
    conversations = []
    for session in _ordered(row["conversation"]):
        dialog = str(session).split("\n\n")
        conversations.append([
            {"user": dialog[2 * t], "assistant": dialog[2 * t + 1]} for t in range(len(dialog) // 2)
        ])
    questions, answers, golds = [], [], []
    for qa in _qa_items(row["qa"]):
        questions.append(str(qa.get("question") or qa.get("query") or ""))
        answers.append(str(qa.get("answer") or qa.get("ground_truth") or ""))
        ids: list[int] = []
        for key in _RETRIEVED_ID_KEYS:
            if key in qa:
                ids = _parse_ids(qa[key])
                break
        golds.append(ids)
    return {
        "id": str(row.get("id") or row.get("uid") or index),
        "conversations": conversations,
        "qa": {"questions": questions, "answers": answers, "gold_session_ids": golds},
        "useful_indices": golds,
    }


# --------------------------------------------------------------------------- ActMemEval

def _maybe_literal(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return value
    return value


def _normalize_actmemeval(row: dict[str, Any], index: int) -> dict[str, Any]:
    sessions = _maybe_literal(row["haystack_sessions"])
    haystack_ids = list(_maybe_literal(row.get("haystack_session_ids", [])) or [])
    answer_ids = list(_maybe_literal(row.get("answer_session_ids", [])) or [])
    position = {sid: i for i, sid in enumerate(haystack_ids)}
    gold = [position[a] for a in answer_ids if a in position]
    conversations = []
    for sub in sessions:
        conversations.append([
            {"user": sub[2 * t]["content"], "assistant": sub[2 * t + 1]["content"]}
            for t in range(len(sub) // 2)
        ])
    return {
        "id": str(row.get("id") or row.get("question_id") or index),
        "question_id": row.get("question_id"),
        "conversations": conversations,
        "qa": {"questions": [str(row["question"])], "answers": [str(row.get("answer", ""))],
               "gold_session_ids": [gold]},
        "useful_indices": [gold],
    }


def _coerce_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "examples", "test", "train", "validation", "rows"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("Unsupported dataset JSON shape (expected a list, or a dict holding a list).")
