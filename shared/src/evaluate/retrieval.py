from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.data.load_data import load_data
from src.evaluate.split_summary import summarize_training_splits
from src.train.train_embedding import last_token_pool
from src.train.utils import add_instructions
from src.utils import atomic_write_json


@dataclass
class SessionItem:
    index: int
    text: str


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.max_rows is not None:
        _set_max_rows(cfg, args.max_rows)
    payload = evaluate_retrieval(cfg)
    summary = payload["summary"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def evaluate_retrieval(cfg: DictConfig) -> dict[str, Any]:
    examples = load_data(cfg)
    embedder = LocalEmbeddingModel(_embedding_cfg(cfg))
    top_ks = _top_ks(cfg)
    session_cache = SessionEmbeddingCache(cfg, embedder, examples)

    results: list[dict[str, Any]] = []
    show_progress = bool(cfg.get("evaluate", {}).get("show_progress", True))
    progress = tqdm(
        enumerate(examples),
        total=len(examples),
        desc="Evaluating retrieval",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for example_index, example in progress:
        sessions = flatten_sessions(example)
        progress.set_postfix(
            example_id=example["id"],
            sessions=len(sessions),
            refresh=False,
        )
        session_embeddings = session_cache.get_or_encode(
            example_index=example_index,
            example=example,
            sessions=sessions,
            embedder=embedder,
        )

        questions = example["qa"]["questions"]
        answers = example["qa"].get("answers", [])
        for qa_index, question in enumerate(questions):
            query_embedding = embedder.encode([question], is_query=True)
            ranked = rank_sessions(sessions, session_embeddings, query_embedding[0])
            useful_indices = get_useful_indices(example, qa_index)
            per_k = {
                str(k): compute_at_k(ranked, useful_indices, k)
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
                    "metrics": per_k,
                    "retrieved_sessions": [
                        {
                            "rank": item["rank"],
                            "session_index": item["session_index"],
                            "score": item["score"],
                            "text": sessions[item["session_index"]].text,
                        }
                        for item in ranked[: max(top_ks)]
                    ],
                }
            )

    session_cache.save_if_dirty()
    summary = summarize(results, top_ks)
    train_eval_split = summarize_training_splits(results, top_ks, summarize, cfg)
    if train_eval_split is not None:
        summary["train_eval_split"] = train_eval_split

    payload = {
        "summary": summary,
        "results": results,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "session_embedding_cache": session_cache.summary(),
    }
    output_path = _output_path(cfg)
    atomic_write_json(output_path, payload)
    return payload


class LocalEmbeddingModel:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.batch_size = int(cfg.get("batch_size", 8))
        self.max_length = int(cfg.get("max_length", 8192))
        self.normalize = bool(cfg.get("normalize", True))
        self.query_instruction = str(cfg.get("query_instruction", "") or "")
        self.show_progress = bool(cfg.get("show_progress", True))
        self.apply_adapter_to_query = bool(cfg.get("apply_adapter_to_query", True))
        self.apply_adapter_to_memory = bool(cfg.get("apply_adapter_to_memory", False))

        model_name = str(cfg.get("model", "") or cfg.get("base_model", ""))
        adapter_path = _configured_adapter_path(cfg.get("adapter_path", None))
        model_path = _optional_path(model_name)
        if adapter_path is None and model_path and (model_path / "adapter_config.json").exists():
            adapter_path = model_path
            model_name = _base_model_from_adapter(adapter_path, fallback=model_name)
        if not model_name:
            raise ValueError("Set evaluate.embedding.model or evaluate.embedding.base_model.")

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
        }
        dtype = resolve_torch_dtype(cfg.get("dtype", "auto"))
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        device = cfg.get("device", None)
        if device and str(device) == "auto":
            model_kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        )
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.has_adapter = False
        if adapter_path is not None:
            self.model = load_peft_adapter(self.model, adapter_path)
            self.has_adapter = True

        if not (device and str(device) == "auto"):
            target_device = str(device) if device else ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(target_device)
        self.model.eval()

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool,
        desc: str | None = None,
    ) -> torch.Tensor:
        if not texts:
            return torch.empty(0)
        formatted = self._format_texts(texts, is_query=is_query)
        batches: list[torch.Tensor] = []
        device = next(self.model.parameters()).device
        use_adapter = self._should_use_adapter(is_query=is_query)
        adapter_context = self._adapter_context(use_adapter=use_adapter)
        with torch.no_grad(), adapter_context:
            batch_starts = range(0, len(formatted), self.batch_size)
            for start in tqdm(
                batch_starts,
                total=(len(formatted) + self.batch_size - 1) // self.batch_size,
                desc=desc or ("Embedding queries" if is_query else "Embedding sessions"),
                leave=False,
                dynamic_ncols=True,
                disable=not self.show_progress,
            ):
                batch = formatted[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)
                outputs = self.model(**encoded)
                embeddings = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
                if self.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                batches.append(embeddings.detach().float().cpu())
        return torch.cat(batches, dim=0)

    def _format_texts(self, texts: list[str], *, is_query: bool) -> list[str]:
        if not is_query or not self.query_instruction:
            return texts
        return add_instructions(texts, self.query_instruction)

    def _should_use_adapter(self, *, is_query: bool) -> bool:
        if not self.has_adapter:
            return False
        if is_query:
            return self.apply_adapter_to_query and bool(self.query_instruction)
        return self.apply_adapter_to_memory

    def _adapter_context(self, *, use_adapter: bool) -> Any:
        if use_adapter or not self.has_adapter:
            return nullcontext()
        disable_adapter = getattr(self.model, "disable_adapter", None)
        return disable_adapter() if callable(disable_adapter) else nullcontext()


class SessionEmbeddingCache:
    def __init__(
        self,
        cfg: DictConfig,
        embedder: LocalEmbeddingModel,
        examples: list[dict[str, Any]],
    ) -> None:
        cache_cfg = cfg.evaluate.get("session_embedding_cache", {})
        self.enabled = bool(cache_cfg.get("enabled", False))
        self.model_slug = cache_model_slug(str(cfg.evaluate.embedding.get("base_model", "")))
        self.dataset_slug = cache_dataset_slug(cfg)
        self.explicit_path = bool(cache_cfg.get("path", None))
        self.path = self._resolve_path(cfg, cache_cfg)
        self.metadata = build_session_cache_metadata(cfg, embedder, examples)
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
        example: dict[str, Any],
        sessions: list[SessionItem],
        embedder: LocalEmbeddingModel,
    ) -> torch.Tensor:
        if not self.enabled:
            self.misses += 1
            return embedder.encode([session.text for session in sessions], is_query=False)

        key = self._key(example_index, example)
        cached = self.records.get(key)
        session_texts = [session.text for session in sessions]
        if cached is not None and cached.get("session_texts") == session_texts:
            self.hits += 1
            return cached["embeddings"].clone()

        self.misses += 1
        embeddings = embedder.encode(session_texts, is_query=False)
        self.records[key] = {
            "example_id": str(example.get("id", example_index)),
            "example_index": int(example_index),
            "session_texts": session_texts,
            "embeddings": embeddings.cpu(),
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
            "metadata": self.metadata,
        }

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        self._validate_path_layout()
        payload = torch.load(self.path, map_location="cpu")
        if payload.get("metadata") != self.metadata:
            raise ValueError(
                "Session embedding cache metadata does not match this evaluation run. "
                f"Cache: {self.path}"
            )
        self.records = dict(payload.get("records", {}))
        self.loaded = True

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(
            {
                "metadata": self.metadata,
                "records": self.records,
            },
            tmp_path,
        )
        tmp_path.replace(self.path)

    def _resolve_path(self, cfg: DictConfig, cache_cfg: Any) -> Path | None:
        if not self.enabled:
            return None
        configured = cache_cfg.get("path", None)
        if configured:
            return Path(str(configured)).expanduser()
        cache_root = Path(str(cache_cfg.get("root_dir", "embedding_cache"))).expanduser()
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "data": OmegaConf.to_container(cfg.data, resolve=True)
                    if "data" in cfg
                    else None,
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
            / self.model_slug
            / self.dataset_slug
            / f"{cache_key}.pt"
        )

    def _key(self, example_index: int, example: dict[str, Any]) -> str:
        return f"{example_index}:{example.get('id', example_index)}"

    def _validate_path_layout(self) -> None:
        if self.path is None or self.explicit_path:
            return
        parts = self.path.parts
        if self.model_slug not in parts or self.dataset_slug not in parts:
            raise ValueError(
                "Session embedding cache path does not match this run's "
                f"model/dataset layout. Expected path to include "
                f"'{self.model_slug}/{self.dataset_slug}', got: {self.path}"
            )


def flatten_sessions(example: dict[str, Any]) -> list[SessionItem]:
    sessions: list[SessionItem] = []
    for session_index, session in enumerate(example["conversations"]):
        turns: list[str] = []
        for interaction_index, interaction in enumerate(session):
            user = str(interaction.get("user", "")).strip()
            assistant = str(interaction.get("assistant", "")).strip()
            turns.append(
                f"Interaction {interaction_index}\n"
                f"User: {user}\n"
                f"Assistant: {assistant}"
            )
        sessions.append(
            SessionItem(
                index=session_index,
                text=f"Session {session_index}\n" + "\n\n".join(turns),
            )
        )
    return sessions


def rank_sessions(
    sessions: list[SessionItem],
    session_embeddings: torch.Tensor,
    query_embedding: torch.Tensor,
) -> list[dict[str, Any]]:
    if session_embeddings.numel() == 0:
        return []
    scores = session_embeddings @ query_embedding
    ranked_indices = torch.argsort(scores, descending=True).tolist()
    return [
        {
            "rank": rank,
            "session_index": sessions[index].index,
            "score": float(scores[index].item()),
        }
        for rank, index in enumerate(ranked_indices, start=1)
    ]


def compute_at_k(ranked: list[dict[str, Any]], useful_indices: list[int], k: int) -> dict[str, Any]:
    retrieved = [int(item["session_index"]) for item in ranked[:k]]
    gold = set(useful_indices)
    hits = gold.intersection(retrieved)
    precision = len(hits) / len(retrieved) if retrieved else 0.0
    recall = len(hits) / len(gold) if gold else 0.0
    normalized_precision = len(hits) / min(k, len(gold)) if gold else 0.0
    return {
        "precision": precision,
        "normalized_precision": normalized_precision,
        "recall": recall,
        "hit": bool(hits),
        "num_hits": len(hits),
        "ndcg": ndcg_at_k(retrieved, gold, k),
        "retrieved_session_indices": retrieved,
    }


def summarize(results: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    first_gold_ranks = [
        rank_of_first_gold(item["retrieved_sessions"], item["useful_session_indices"])
        for item in results
    ]
    finite_ranks = [rank for rank in first_gold_ranks if rank is not None]
    summary: dict[str, Any] = {
        "num_questions": len(results),
        "num_first_gold_found": len(finite_ranks),
        "num_first_gold_missing": len(results) - len(finite_ranks),
        "mrr": (
            sum(1.0 / rank for rank in finite_ranks) / len(results)
            if results
            else 0.0
        ),
        "mean_rank_of_first_gold": (
            sum(finite_ranks) / len(finite_ranks)
            if finite_ranks
            else None
        ),
        "median_rank_of_first_gold": median(finite_ranks),
    }
    for k in top_ks:
        key = str(k)
        if not results:
            summary[f"precision@{k}"] = 0.0
            summary[f"normalized_precision@{k}"] = 0.0
            summary[f"recall@{k}"] = 0.0
            summary[f"hit_rate@{k}"] = 0.0
            summary[f"ndcg@{k}"] = 0.0
            continue
        metrics = [item["metrics"][key] for item in results]
        summary[f"precision@{k}"] = sum(item["precision"] for item in metrics) / len(metrics)
        summary[f"normalized_precision@{k}"] = (
            sum(item["normalized_precision"] for item in metrics) / len(metrics)
        )
        summary[f"recall@{k}"] = sum(item["recall"] for item in metrics) / len(metrics)
        summary[f"hit_rate@{k}"] = sum(1 for item in metrics if item["hit"]) / len(metrics)
        summary[f"ndcg@{k}"] = sum(item["ndcg"] for item in metrics) / len(metrics)
    return summary


def rank_of_first_gold(
    ranked_sessions: list[dict[str, Any]],
    useful_indices: list[int],
) -> int | None:
    gold = set(useful_indices)
    if not gold:
        return None
    for item in ranked_sessions:
        if int(item["session_index"]) in gold:
            return int(item["rank"])
    return None


def ndcg_at_k(retrieved: list[int], gold: set[int], k: int) -> float:
    if not gold:
        return 0.0
    dcg = 0.0
    for rank, session_index in enumerate(retrieved[:k], start=1):
        if session_index in gold:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(k, len(gold))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def median(values: list[int]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[middle])
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def get_useful_indices(example: dict[str, Any], qa_index: int) -> list[int]:
    useful_indices = example.get("useful_indices", [])
    if not useful_indices:
        return []
    item = useful_indices[qa_index] if qa_index < len(useful_indices) else useful_indices[0]
    if isinstance(item, list):
        return [int(value) for value in item]
    return [int(item)]


def load_peft_adapter(model: torch.nn.Module, adapter_path: Path) -> torch.nn.Module:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Install peft to evaluate a LoRA/PEFT adapter.") from exc
    return PeftModel.from_pretrained(model, str(adapter_path))


def resolve_torch_dtype(value: Any) -> torch.dtype | None:
    if value is None:
        return None
    value_str = str(value).lower()
    if value_str in {"", "auto", "none", "null"}:
        return None
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
    }
    return getattr(torch, aliases.get(value_str, value_str))


def _base_model_from_adapter(adapter_path: Path, fallback: str) -> str:
    config_path = adapter_path / "adapter_config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return str(payload.get("base_model_name_or_path") or fallback)


def _embedding_cfg(cfg: DictConfig) -> DictConfig:
    if "evaluate" in cfg and "embedding" in cfg.evaluate:
        return cfg.evaluate.embedding
    if "rag" in cfg and "embedding" in cfg.rag:
        embedding_cfg = OmegaConf.create(OmegaConf.to_container(cfg.rag.embedding, resolve=True))
        if "rag" in cfg and "retrieval" in cfg.rag:
            embedding_cfg.query_instruction = cfg.rag.retrieval.get("query_instruction", "")
            embedding_cfg.batch_size = cfg.rag.retrieval.get("embedding_batch_size", 8)
        return embedding_cfg
    raise ValueError("Config must contain evaluate.embedding or rag.embedding.")


def _top_ks(cfg: DictConfig) -> list[int]:
    if "evaluate" in cfg and cfg.evaluate.get("top_ks", None):
        values = [int(value) for value in cfg.evaluate.top_ks]
    elif "rag" in cfg and "retrieval" in cfg.rag:
        values = [1, 3, int(cfg.rag.retrieval.get("top_k", 5))]
    else:
        values = [1, 3, 5, 10, 20]
    return sorted({value for value in values if value > 0})


def _output_path(cfg: DictConfig) -> Path:
    if "evaluate" in cfg:
        return Path(str(cfg.evaluate.output_dir)) / str(cfg.evaluate.output_file)
    return Path(str(cfg.run.output_dir)) / "retrieval_eval.json"


def _set_max_rows(cfg: DictConfig, max_rows: int) -> None:
    if "run" not in cfg:
        cfg.run = {}
    cfg.run.max_rows = int(max_rows)


def build_session_cache_metadata(
    cfg: DictConfig,
    embedder: LocalEmbeddingModel,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    example_fingerprint = hashlib.sha256()
    for example_index, example in enumerate(examples):
        sessions = flatten_sessions(example)
        example_fingerprint.update(str(example_index).encode("utf-8"))
        example_fingerprint.update(str(example.get("id", example_index)).encode("utf-8"))
        for session in sessions:
            example_fingerprint.update(str(session.index).encode("utf-8"))
            example_fingerprint.update(session.text.encode("utf-8"))
    return {
        "data": OmegaConf.to_container(cfg.data, resolve=True) if "data" in cfg else None,
        "num_examples": len(examples),
        "example_session_fingerprint": example_fingerprint.hexdigest(),
        "base_model": str(embedder.cfg.get("base_model", "") or embedder.cfg.get("model", "")),
        "max_length": int(embedder.max_length),
        "normalize": bool(embedder.normalize),
        "dtype": str(embedder.cfg.get("dtype", "auto")),
        "trust_remote_code": bool(embedder.cfg.get("trust_remote_code", False)),
        "adapter_policy": "adapter disabled for session embeddings",
        "session_format": "flatten_sessions:v1",
    }


def cache_model_slug(model_name: str) -> str:
    value = model_name.split("/")[-1] if model_name else "unknown-model"
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "unknown-model"


def cache_dataset_slug(cfg: DictConfig) -> str:
    if "data" not in cfg:
        return "unknown-data"
    data_name = str(cfg.data.get("name", "unknown-data") or "unknown-data")
    data_type = cfg.data.get("type", None)
    data_split = cfg.data.get("split", None)
    parts = [data_name]
    if data_type not in {None, "", "null"}:
        parts.append(str(data_type))
    if data_split not in {None, "", "null"}:
        parts.append(str(data_split))
    slug = "-".join(parts).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug or "unknown-data"


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.exists() else None


def _configured_adapter_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"LoRA adapter path does not exist: {path}")
    if not (path / "adapter_config.json").exists():
        raise FileNotFoundError(f"LoRA adapter_config.json not found under: {path}")
    return path


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    path = Path(path).resolve()
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        return compose(config_name=path.name.removesuffix(path.suffix), overrides=overrides or [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate session-level embedding retrieval.")
    parser.add_argument("--config", default="config/evaluate_retrieval.yaml")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. evaluate.embedding.adapter_path=output/embedding_model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
