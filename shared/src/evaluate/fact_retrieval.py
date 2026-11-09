from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.data.load_data import load_data
from src.evaluate.split_summary import summarize_training_splits
from src.evaluate.retrieval import (
    LocalEmbeddingModel,
    cache_dataset_slug,
    cache_model_slug,
    get_useful_indices,
)
from src.utils import atomic_write_json


SOURCE_ROW_INDEX_KEYS = ("source_row_index", "row_index", "original_row_index")
SOURCE_QA_INDEX_KEYS = (
    "source_qa_index",
    "qa_index",
    "source_question_index",
    "original_qa_index",
    "original_question_index",
    "question_index",
)


@dataclass
class FactItem:
    index: int
    session_index: int
    fact: str
    evidence: str | None = None


@dataclass
class FactRows:
    by_row: dict[int, dict[str, Any]]
    by_question: dict[tuple[int, int], dict[str, Any]]
    granularity: str

    def get(self, example_index: int, qa_index: int | None = None) -> dict[str, Any] | None:
        if qa_index is not None:
            question_row = self.by_question.get((int(example_index), int(qa_index)))
            if question_row is not None:
                return question_row
        return self.by_row.get(int(example_index))

    def __len__(self) -> int:
        return len(self.by_row) + len(self.by_question)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.max_rows is not None:
        _set_max_rows(cfg, args.max_rows)
    if args.print_fact_cache_path:
        path = resolve_fact_cache_path(cfg)
        if path is None:
            print("DISABLED")  # cache off -> no path to check
            raise SystemExit(2)
        exists = path.exists()
        print(str(path))
        raise SystemExit(0 if exists else 1)
    payload = evaluate_fact_retrieval(cfg)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


def evaluate_fact_retrieval(cfg: DictConfig) -> dict[str, Any]:
    examples = load_data(cfg)
    fact_rows = load_fact_rows(Path(str(cfg.evaluate.fact_bag_path)))
    embedder = LocalEmbeddingModel(cfg.evaluate.embedding)
    top_ks = top_ks_from_cfg(cfg)
    cache = FactEmbeddingCache(cfg, embedder, examples, fact_rows)

    # STAGE 1: embed EVERY query in one batched pass, with the LoRA adapter (the facts below are
    # embedded by the frozen base). Encoding one question per forward pass -- "Embedding queries:
    # 0/1", once per question -- wasted most of each batch. Cost here is tiny: a few thousand
    # queries at fp32 is ~tens of MB, versus the fact embeddings the cache already holds.
    query_slot: dict[tuple[int, int], int] = {}
    all_questions: list[str] = []
    for example_index, example in enumerate(examples):
        for qa_index, question in enumerate(example["qa"]["questions"]):
            query_slot[(example_index, qa_index)] = len(all_questions)
            all_questions.append(question)
    query_embeddings = embedder.encode(all_questions, is_query=True, desc="Embedding queries")

    # OPTIONAL: pre-encode ALL facts in one big batched pass (see evaluate.prebatch_facts). The
    # per-example loop below otherwise encodes ~one user's facts at a time -- tiny batches that
    # starve a big GPU. Here we gather every UNIQUE fact row (by cache key), flatten their facts
    # into one list, and encode it in embedder.batch_size-sized batches -> the GPU stays saturated.
    # Per-fact embeddings are identical to the per-row path (encoding is per-sequence), so results
    # are unchanged. Disabled automatically when the on-disk cache is enabled (asymmetric is already
    # fast via the cache, and prebatch would bypass it and waste RAM).
    prebatched: dict[str, tuple[list[FactItem], torch.Tensor]] = {}
    if bool(cfg.evaluate.get("prebatch_facts", False)) and not cache.enabled:
        pb_facts: dict[str, list[FactItem]] = {}
        order: list[str] = []
        for example_index, example in enumerate(examples):
            for qa_index, _ in enumerate(example["qa"]["questions"]):
                fact_row = fact_rows.get(example_index, qa_index)
                if fact_row is None:
                    continue  # the per-example loop raises the precise error below
                key = fact_row_cache_key(example_index, qa_index, fact_row)
                if key in pb_facts:
                    continue
                pb_facts[key] = normalize_fact_items(fact_row)
                order.append(key)
        flat_texts: list[str] = []
        spans: dict[str, tuple[int, int]] = {}
        for key in order:
            start = len(flat_texts)
            flat_texts.extend(fact.fact for fact in pb_facts[key])
            spans[key] = (start, len(flat_texts))
        print(f"[prebatch] encoding {len(flat_texts)} facts from {len(order)} fact rows in one "
              f"batched pass (batch_size={embedder.batch_size}) ...", flush=True)
        all_fact_emb = embedder.encode(flat_texts, is_query=False, desc="Embedding facts (prebatched)")
        for key in order:
            start, end = spans[key]
            prebatched[key] = (pb_facts[key], all_fact_emb[start:end])

    results: list[dict[str, Any]] = []
    show_progress = bool(cfg.evaluate.get("show_progress", True))
    progress = tqdm(
        enumerate(examples),
        total=len(examples),
        desc="Evaluating fact retrieval",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for example_index, example in progress:
        questions = example["qa"]["questions"]
        answers = example["qa"].get("answers", [])
        encoded_fact_rows: dict[str, tuple[list[FactItem], torch.Tensor]] = {}
        progress.set_postfix(
            example_id=example["id"],
            fact_rows=fact_rows.granularity,
            refresh=False,
        )

        for qa_index, question in enumerate(questions):
            fact_row = fact_rows.get(example_index, qa_index)
            if fact_row is None:
                raise KeyError(
                    "Fact bag row not found for "
                    f"row_index={example_index}, qa_index={qa_index}."
                )
            fact_cache_key = fact_row_cache_key(example_index, qa_index, fact_row)
            prebatched_payload = prebatched.get(fact_cache_key)
            cached_fact_payload = encoded_fact_rows.get(fact_cache_key)
            if prebatched_payload is not None:
                facts, fact_embeddings = prebatched_payload  # encoded up front in the big pass
            elif cached_fact_payload is None:
                facts = normalize_fact_items(fact_row)
                fact_embeddings = cache.get_or_encode(
                    example_index=example_index,
                    qa_index=qa_index,
                    fact_row=fact_row,
                    facts=facts,
                    embedder=embedder,
                )
                encoded_fact_rows[fact_cache_key] = (facts, fact_embeddings)
            else:
                facts, fact_embeddings = cached_fact_payload

            progress.set_postfix(
                example_id=example["id"],
                qa_index=qa_index,
                facts=len(facts),
                refresh=False,
            )
            query_embedding = query_embeddings[query_slot[(example_index, qa_index)]]
            ranked = rank_facts(facts, fact_embeddings, query_embedding)
            useful_indices = get_useful_indices(example, qa_index)
            gold_fact_indices = [
                fact.index for fact in facts if fact.session_index in set(useful_indices)
            ]
            per_k = {
                str(k): compute_fact_metrics_at_k(
                    ranked=ranked,
                    gold_sessions=useful_indices,
                    gold_fact_indices=gold_fact_indices,
                    k=k,
                )
                for k in top_ks
            }
            results.append(
                {
                    "example_id": example["id"],
                    "example_index": example_index,
                    "qa_index": qa_index,
                    "question": question,
                    "answer": answers[qa_index] if qa_index < len(answers) else None,
                    "useful_session_indices": useful_indices,
                    "gold_fact_indices": gold_fact_indices,
                    "num_facts": len(facts),
                    "num_gold_facts": len(gold_fact_indices),
                    "metrics": per_k,
                    "retrieved_facts": [
                        {
                            "rank": item["rank"],
                            "fact_index": item["fact_index"],
                            "session_index": item["session_index"],
                            "score": item["score"],
                            "fact": facts[item["fact_index"]].fact,
                        }
                        for item in ranked[: max(top_ks)]
                    ],
                }
            )

    cache.save_if_dirty()
    summary = summarize_fact_results(results, top_ks)
    summary["fact_bag_granularity"] = fact_rows.granularity
    summary["num_row_level_fact_rows"] = len(fact_rows.by_row)
    summary["num_question_level_fact_rows"] = len(fact_rows.by_question)
    train_eval_split = summarize_training_splits(results, top_ks, summarize_fact_results, cfg)
    if train_eval_split is not None:
        summary["train_eval_split"] = train_eval_split

    payload = {
        "run_metadata": run_metadata(cfg),
        "summary": summary,
        "results": results,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "fact_embedding_cache": cache.summary(),
        "notes": {
            "retrieval_unit": "facts",
            "fact_embedding_policy": "Fact embeddings use the base model with adapter disabled.",
            "query_embedding_policy": "Query embeddings use instruction conditioning and adapter when configured. All queries are encoded in one batched pass before ranking.",
            "session_metrics": "Session metrics are credited when a retrieved fact's session_index is a gold session.",
            "fact_metrics": "Fact metrics are credited when a retrieved fact itself comes from a gold session.",
        },
    }
    atomic_write_json(output_path(cfg), payload)
    return payload


def load_fact_rows(path: Path) -> FactRows:
    if not path.exists():
        raise FileNotFoundError(f"Fact bag JSONL file not found: {path}")
    by_row: dict[int, dict[str, Any]] = {}
    by_question: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_row_index = fact_row_source_row_index(row, line_number - 1)
            source_qa_index = fact_row_source_qa_index(row)
            if source_qa_index is None:
                if source_row_index in by_row:
                    raise ValueError(f"Duplicate fact row_index={source_row_index} in {path}")
                by_row[source_row_index] = row
                continue

            question_key = (source_row_index, source_qa_index)
            if question_key in by_question:
                raise ValueError(
                    "Duplicate fact question row "
                    f"row_index={source_row_index}, qa_index={source_qa_index} in {path}"
                )
            by_question[question_key] = row

    if by_row and by_question:
        granularity = "mixed"
    elif by_question:
        granularity = "question"
    else:
        granularity = "row"
    return FactRows(by_row=by_row, by_question=by_question, granularity=granularity)


def fact_row_source_row_index(row: dict[str, Any], fallback: int) -> int:
    for key in SOURCE_ROW_INDEX_KEYS:
        value = row.get(key)
        if value is not None:
            return int(value)
    return fallback


def fact_row_source_qa_index(row: dict[str, Any]) -> int | None:
    for key in SOURCE_QA_INDEX_KEYS:
        value = row.get(key)
        if value is not None:
            return int(value)

    qa_item = row.get("qa")
    if isinstance(qa_item, dict):
        return fact_row_source_qa_index(qa_item)
    if isinstance(qa_item, list) and len(qa_item) == 1 and isinstance(qa_item[0], dict):
        return fact_row_source_qa_index(qa_item[0])
    return None


def fact_row_cache_key(
    example_index: int,
    qa_index: int | None,
    fact_row: dict[str, Any],
) -> str:
    source_row_index = fact_row_source_row_index(fact_row, example_index)
    source_qa_index = fact_row_source_qa_index(fact_row)
    if source_qa_index is not None:
        return f"question:{source_row_index}:{source_qa_index}"
    return f"row:{source_row_index}"


def normalize_fact_items(row: dict[str, Any]) -> list[FactItem]:
    facts = []
    for fact_index, item in enumerate(row.get("facts", [])):
        fact_text = str(item.get("fact", "")).strip()
        if not fact_text:
            continue
        facts.append(
            FactItem(
                index=len(facts),
                session_index=int(item["session_index"]),
                fact=fact_text,
                evidence=str(item.get("evidence")) if item.get("evidence") is not None else None,
            )
        )
    return facts


def rank_facts(
    facts: list[FactItem],
    fact_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
) -> list[dict[str, Any]]:
    if fact_embeddings.numel() == 0:
        return []
    scores = fact_embeddings.to(dtype=query_embedding.dtype) @ query_embedding
    ranked_indices = torch.argsort(scores, descending=True).tolist()
    return [
        {
            "rank": rank,
            "fact_index": facts[index].index,
            "session_index": facts[index].session_index,
            "score": float(scores[index].item()),
        }
        for rank, index in enumerate(ranked_indices, start=1)
    ]


def compute_fact_metrics_at_k(
    *,
    ranked: list[dict[str, Any]],
    gold_sessions: list[int],
    gold_fact_indices: list[int],
    k: int,
) -> dict[str, Any]:
    retrieved = ranked[:k]
    retrieved_fact_indices = [int(item["fact_index"]) for item in retrieved]
    retrieved_sessions = [int(item["session_index"]) for item in retrieved]
    gold_session_set = set(int(item) for item in gold_sessions)
    gold_fact_set = set(int(item) for item in gold_fact_indices)

    hit_fact_indices = [idx for idx in retrieved_fact_indices if idx in gold_fact_set]
    hit_sessions = gold_session_set.intersection(retrieved_sessions)
    unique_retrieved_gold_sessions = sorted(hit_sessions)

    return {
        "fact_recall": len(set(hit_fact_indices)) / len(gold_fact_set) if gold_fact_set else 0.0,
        "fact_precision": len(hit_fact_indices) / len(retrieved) if retrieved else 0.0,
        "session_recall": len(hit_sessions) / len(gold_session_set) if gold_session_set else 0.0,
        "session_hit": bool(hit_sessions),
        "all_gold_sessions_hit": bool(gold_session_set)
        and hit_sessions == gold_session_set,
        "num_fact_hits": len(set(hit_fact_indices)),
        "num_session_hits": len(hit_sessions),
        "fact_ndcg": fact_ndcg_at_k(retrieved_fact_indices, gold_fact_set, k),
        "session_ndcg": session_ndcg_at_k(retrieved_sessions, gold_session_set, k),
        "retrieved_fact_indices": retrieved_fact_indices,
        "retrieved_session_indices": retrieved_sessions,
        "unique_retrieved_gold_sessions": unique_retrieved_gold_sessions,
    }


def fact_ndcg_at_k(retrieved_fact_indices: list[int], gold_fact_indices: set[int], k: int) -> float:
    if not gold_fact_indices:
        return 0.0
    dcg = 0.0
    for rank, fact_index in enumerate(retrieved_fact_indices[:k], start=1):
        if fact_index in gold_fact_indices:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(k, len(gold_fact_indices))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def session_ndcg_at_k(retrieved_sessions: list[int], gold_sessions: set[int], k: int) -> float:
    if not gold_sessions:
        return 0.0
    seen_gold_sessions: set[int] = set()
    dcg = 0.0
    for rank, session_index in enumerate(retrieved_sessions[:k], start=1):
        if session_index in gold_sessions and session_index not in seen_gold_sessions:
            seen_gold_sessions.add(session_index)
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(k, len(gold_sessions))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def summarize_fact_results(results: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_questions": len(results),
        "num_examples": len({item["example_index"] for item in results}),
        "num_questions_without_gold_facts": sum(
            1 for item in results if item["num_gold_facts"] == 0
        ),
        "mean_num_facts": (
            sum(item["num_facts"] for item in results) / len(results)
            if results
            else 0.0
        ),
        "mean_num_gold_facts": (
            sum(item["num_gold_facts"] for item in results) / len(results)
            if results
            else 0.0
        ),
    }
    for k in top_ks:
        key = str(k)
        metrics = [item["metrics"][key] for item in results]
        if not metrics:
            summary.update(
                {
                    f"fact_recall@{k}": 0.0,
                    f"fact_precision@{k}": 0.0,
                    f"session_recall@{k}": 0.0,
                    f"session_hit_rate@{k}": 0.0,
                    f"all_gold_sessions_hit_rate@{k}": 0.0,
                    f"fact_ndcg@{k}": 0.0,
                    f"session_ndcg@{k}": 0.0,
                }
            )
            continue
        summary[f"fact_recall@{k}"] = average(metrics, "fact_recall")
        summary[f"fact_precision@{k}"] = average(metrics, "fact_precision")
        summary[f"session_recall@{k}"] = average(metrics, "session_recall")
        summary[f"session_hit_rate@{k}"] = sum(
            1 for item in metrics if item["session_hit"]
        ) / len(metrics)
        summary[f"all_gold_sessions_hit_rate@{k}"] = sum(
            1 for item in metrics if item["all_gold_sessions_hit"]
        ) / len(metrics)
        summary[f"fact_ndcg@{k}"] = average(metrics, "fact_ndcg")
        summary[f"session_ndcg@{k}"] = average(metrics, "session_ndcg")
    return summary


def average(items: list[dict[str, Any]], key: str) -> float:
    return sum(float(item[key]) for item in items) / len(items) if items else 0.0


def resolve_fact_cache_path(cfg: DictConfig) -> Path | None:
    """The auto fact-embedding cache path for this run, or None if the cache is disabled.

    Pure: needs only cfg (no model load), so callers can check whether a cache already exists
    before deciding to do expensive work (e.g. the sweep skipping its prewarm). This is the ONE
    source of truth for the path -- FactEmbeddingCache._resolve_path delegates here so the two
    can never drift.
    """
    cache_cfg = cfg.evaluate.get("fact_embedding_cache", {}) or {}
    if not bool(cache_cfg.get("enabled", False)):
        return None
    configured = cache_cfg.get("path", None)
    if configured:
        return Path(str(configured)).expanduser()
    cache_root = Path(str(cache_cfg.get("root_dir", "embedding_cache"))).expanduser()
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "data": OmegaConf.to_container(cfg.data, resolve=True) if "data" in cfg else None,
                "fact_bag_path": str(cfg.evaluate.fact_bag_path),
                "embedding": {
                    "model": str(cfg.evaluate.embedding.get("base_model", "")),
                    "max_length": int(cfg.evaluate.embedding.get("max_length", 8192)),
                    "normalize": bool(cfg.evaluate.embedding.get("normalize", True)),
                    "dtype": str(cfg.evaluate.embedding.get("dtype", "auto")),
                },
                "max_rows": cfg.run.get("max_rows", None) if "run" in cfg else None,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return (
        cache_root
        / cache_model_slug(str(cfg.evaluate.embedding.get("base_model", "")))
        / cache_dataset_slug(cfg)
        / "facts"
        / f"{cache_key}.pt"
    )


class FactEmbeddingCache:
    def __init__(
        self,
        cfg: DictConfig,
        embedder: LocalEmbeddingModel,
        examples: list[dict[str, Any]],
        fact_rows: FactRows,
    ) -> None:
        cache_cfg = cfg.evaluate.get("fact_embedding_cache", {})
        self.enabled = bool(cache_cfg.get("enabled", False))
        self.model_slug = cache_model_slug(str(cfg.evaluate.embedding.get("base_model", "")))
        self.dataset_slug = cache_dataset_slug(cfg)
        self.explicit_path = bool(cache_cfg.get("path", None))
        self.storage_dtype = str(cache_cfg.get("storage_dtype", "float16"))
        self.path = self._resolve_path(cfg, cache_cfg)
        self.metadata = build_fact_cache_metadata(cfg, embedder, examples, fact_rows)
        self.records: dict[str, dict[str, Any]] = {}
        self.loaded = False
        self.dirty = False
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self._load()

    def get_or_encode(
        self,
        *,
        example_index: int,
        qa_index: int | None,
        fact_row: dict[str, Any],
        facts: list[FactItem],
        embedder: LocalEmbeddingModel,
    ) -> torch.Tensor:
        if not self.enabled:
            self.misses += 1
            return embedder.encode(
                [fact.fact for fact in facts],
                is_query=False,
                desc="Embedding facts",
            )

        key = self._key(example_index, qa_index, fact_row)
        fact_texts = [fact.fact for fact in facts]
        cached = self.records.get(key)
        if cached is not None and cached.get("fact_texts") == fact_texts:
            self.hits += 1
            return cached["embeddings"].clone()

        self.misses += 1
        embeddings = embedder.encode(fact_texts, is_query=False, desc="Embedding facts")
        self.records[key] = {
            "row_index": fact_row_source_row_index(fact_row, example_index),
            "qa_index": fact_row_source_qa_index(fact_row),
            "fact_texts": fact_texts,
            "session_indices": [fact.session_index for fact in facts],
            "embeddings": cast_cache_tensor(embeddings.cpu(), self.storage_dtype),
        }
        self.dirty = True
        return embeddings

    def save_if_dirty(self) -> None:
        if self.enabled and self.dirty:
            self._save()
            self.dirty = False

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "path": str(self.path) if self.path else None,
            "loaded": self.loaded,
            "hits": self.hits,
            "misses": self.misses,
            "num_records": len(self.records),
            "model_slug": self.model_slug,
            "dataset_slug": self.dataset_slug,
            "explicit_path": self.explicit_path,
            "storage_dtype": self.storage_dtype,
            "metadata": self.metadata,
        }

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        self._validate_path_layout()
        payload = torch.load(self.path, map_location="cpu")
        if payload.get("metadata") != self.metadata:
            raise ValueError(
                "Fact embedding cache metadata does not match this evaluation run. "
                f"Cache: {self.path}"
            )
        self.records = dict(payload.get("records", {}))
        self.loaded = True

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save({"metadata": self.metadata, "records": self.records}, tmp_path)
        tmp_path.replace(self.path)

    def _resolve_path(self, cfg: DictConfig, cache_cfg: Any) -> Path | None:
        if not self.enabled:
            return None
        return resolve_fact_cache_path(cfg)

    def _validate_path_layout(self) -> None:
        if self.path is None or self.explicit_path:
            return
        parts = self.path.parts
        expected = [self.model_slug, self.dataset_slug, "facts"]
        if not all(part in parts for part in expected):
            raise ValueError(
                "Fact embedding cache path does not match this run's model/dataset "
                f"layout. Expected path to include '{'/'.join(expected)}', got: {self.path}"
            )

    def _key(
        self,
        example_index: int,
        qa_index: int | None,
        fact_row: dict[str, Any],
    ) -> str:
        return fact_row_cache_key(example_index, qa_index, fact_row)


def cast_cache_tensor(tensor: torch.Tensor, dtype_name: str) -> torch.Tensor:
    if dtype_name in {"float16", "fp16", "half"}:
        return tensor.to(torch.float16)
    if dtype_name in {"bfloat16", "bf16"}:
        return tensor.to(torch.bfloat16)
    if dtype_name in {"float32", "fp32"}:
        return tensor.to(torch.float32)
    raise ValueError(f"Unsupported fact cache storage_dtype: {dtype_name}")


def build_fact_cache_metadata(
    cfg: DictConfig,
    embedder: LocalEmbeddingModel,
    examples: list[dict[str, Any]],
    fact_rows: FactRows,
) -> dict[str, Any]:
    fingerprint = hashlib.sha256()
    seen_fact_rows: set[str] = set()
    for example_index, example in enumerate(examples):
        questions = example.get("qa", {}).get("questions", []) or [None]
        for qa_index, _ in enumerate(questions):
            fact_row = fact_rows.get(example_index, qa_index)
            if fact_row is None:
                continue
            key = fact_row_cache_key(example_index, qa_index, fact_row)
            if key in seen_fact_rows:
                continue
            seen_fact_rows.add(key)
            fingerprint.update(key.encode("utf-8"))
            for fact in normalize_fact_items(fact_row):
                fingerprint.update(str(fact.session_index).encode("utf-8"))
                fingerprint.update(fact.fact.encode("utf-8"))
    return {
        "data": OmegaConf.to_container(cfg.data, resolve=True) if "data" in cfg else None,
        "fact_bag_path": str(cfg.evaluate.fact_bag_path),
        "num_examples": len(examples),
        "num_fact_rows": len(fact_rows),
        "num_row_level_fact_rows": len(fact_rows.by_row),
        "num_question_level_fact_rows": len(fact_rows.by_question),
        "fact_bag_granularity": fact_rows.granularity,
        "fact_fingerprint": fingerprint.hexdigest(),
        "base_model": str(embedder.cfg.get("base_model", "") or embedder.cfg.get("model", "")),
        "max_length": int(embedder.max_length),
        "normalize": bool(embedder.normalize),
        "dtype": str(embedder.cfg.get("dtype", "auto")),
        "trust_remote_code": bool(embedder.cfg.get("trust_remote_code", False)),
        "adapter_policy": "adapter disabled for fact embeddings",
        "fact_text_policy": "fact-only",
    }


def top_ks_from_cfg(cfg: DictConfig) -> list[int]:
    return sorted({int(value) for value in cfg.evaluate.get("top_ks", [1, 3, 5, 10, 20]) if int(value) > 0})


def output_path(cfg: DictConfig) -> Path:
    return Path(str(cfg.evaluate.output_dir)) / str(cfg.evaluate.output_file)


def run_metadata(cfg: DictConfig) -> dict[str, Any]:
    metadata = cfg.evaluate.get("run_metadata", {})
    return OmegaConf.to_container(metadata, resolve=True) if metadata else {}


def _set_max_rows(cfg: DictConfig, max_rows: int) -> None:
    if "run" not in cfg:
        cfg.run = {}
    cfg.run.max_rows = int(max_rows)


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    path = Path(path).resolve()
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        return compose(config_name=path.name.removesuffix(path.suffix), overrides=overrides or [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fact-level retrieval.")
    parser.add_argument("--config", default="config/evaluate_fact_retrieval.yaml")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--print-fact-cache-path",
        action="store_true",
        help="Resolve and print the fact-embedding cache path for this config, then exit 0 if the "
        "file exists / 1 if not (no model load, no eval). Used to skip the sweep prewarm.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


if __name__ == "__main__":
    main()
