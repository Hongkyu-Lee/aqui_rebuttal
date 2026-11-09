"""Synthetic-validation utilities (E1, E2, E3). Benchmark data is NEVER read here.

Subcommands
-----------
overlap  : rebuild the exact train/val split used by training (same loader + split_validation,
           seed 42, 10% of memory_ids) and report how much text crosses the split
           (answers 2KMq-Q2b: "were queries sharing the same synthetic memory grouped together?").
eval     : synthetic-validation RETRIEVAL score of one or more adapters.
           Pool    = every unique fact text (gold_session_fact + general_fact) of all VAL rows.
           Query   = each val row's query (with the training instruction; adapter on the query).
           Gold    = the gold_session_fact texts of the row's memory (union over val rows that
                     share the memory_id -- they are the same user memory, and training treats
                     same-memory samples as positives).
           Metrics = hit@K (>=1 gold in top-K; the synthetic analogue of session recall, since a
                     memory is one "session") and recall@K (fraction of gold facts in top-K).
           Selection score (pre-registered) = val_mean_hit = mean(hit@10, hit@20, hit@50, hit@100).
           For the SYMMETRIC arm pass --arm sym: facts are then encoded THROUGH the adapter.

Usage
-----
  python -m tools.synthetic_val overlap --data v11con_2400.jsonl --out overlap.json
  python -m tools.synthetic_val eval --data v11con_2400.jsonl --out val.csv \
        --adapter aqui:/path/adapterA --adapter sym:/path/adapterB --adapter base:-
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

KS = [1, 5, 10, 20, 50, 100]


def _cfg(data_path: str, ratio: float = 0.1, seed: int = 42):
    from omegaconf import OmegaConf
    return OmegaConf.create({"train": {"data": {
        "name": "oppose_gen_grouped", "path": data_path, "file": None,
        "query_field": "query", "positive_field": "gold_session_fact", "negative_field": "general_fact",
        "negative_pairing": "cycle_negatives", "group_field": "group_id", "group_source_field": "memory_id",
        "val_split_ratio": ratio, "val_split_seed": seed, "val_split_by_group": True}}})


def split_rows(data_path: str):
    """Return (train_rows, val_rows) of SOURCE rows, using the training code's own split."""
    from src.data.load_train_data import load_train_data, read_jsonl_rows
    from src.entry_points.train_embedding_model import split_validation
    cfg = _cfg(data_path)
    samples = load_train_data(cfg)
    train_s, val_s = split_validation(samples, cfg)
    raw = read_jsonl_rows(Path(data_path))
    val_idx = sorted({s["source_row_index"] for s in val_s})
    train_idx = sorted({s["source_row_index"] for s in train_s})
    assert not set(val_idx) & set(train_idx), "a source row landed on both sides of the split"
    return [raw[i] for i in train_idx], [raw[i] for i in val_idx], len(train_s), len(val_s)


def _facts(row, field):
    from src.data.load_train_data import normalize_fact_list
    return normalize_fact_list(row.get(field), field, 0, 0)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def cmd_overlap(args) -> None:
    train, val, n_train_s, n_val_s = split_rows(args.data)
    tm = {r["memory_id"] for r in train}
    vm = {r["memory_id"] for r in val}
    tgold = {_norm(f) for r in train for f in _facts(r, "gold_session_fact")}
    tgen = {_norm(f) for r in train for f in _facts(r, "general_fact")}
    tq = {_norm(r["query"]) for r in train}
    vgold = [_norm(f) for r in val for f in _facts(r, "gold_session_fact")]
    vgen = [_norm(f) for r in val for f in _facts(r, "general_fact")]
    vq = [_norm(r["query"]) for r in val]
    out = {
        "split": {"train_rows": len(train), "val_rows": len(val), "train_samples": n_train_s,
                  "val_samples": n_val_s, "train_memories": len(tm), "val_memories": len(vm),
                  "val_split_ratio": 0.1, "seed": 42, "grouped_by": "memory_id"},
        "memory_ids_on_both_sides": len(tm & vm),
        "val_queries_identical_to_a_train_query": sum(q in tq for q in vq),
        "val_gold_facts": len(vgold),
        "val_gold_facts_whose_text_appears_as_train_gold": sum(f in tgold for f in vgold),
        "val_gold_facts_whose_text_appears_anywhere_in_train": sum((f in tgold) or (f in tgen) for f in vgold),
        "val_general_facts_whose_text_appears_in_train": sum((f in tgold) or (f in tgen) for f in vgen),
        "unique_val_gold_texts": len(set(vgold)),
        "unique_val_gold_texts_in_train": len({f for f in vgold if f in tgold}),
    }
    out["frac_val_gold_facts_in_train"] = out["val_gold_facts_whose_text_appears_as_train_gold"] / max(1, len(vgold))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if args.clean_val_out:
        # val rows whose gold facts NEVER appear in train -> an overlap-free validation subset
        clean = [r for r in val if not any(_norm(f) in tgold for f in _facts(r, "gold_session_fact"))]
        with open(args.clean_val_out, "w", encoding="utf-8") as fh:
            for r in clean:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"overlap-free val subset: {len(clean)}/{len(val)} rows -> {args.clean_val_out}")


def _build_val_task(val_rows):
    pool: list[str] = []
    index: dict[str, int] = {}

    def add(t):
        k = _norm(t)
        if k not in index:
            index[k] = len(pool)
            pool.append(t.strip())
        return index[k]

    mem_gold = defaultdict(set)
    for r in val_rows:
        for f in _facts(r, "gold_session_fact"):
            mem_gold[r["memory_id"]].add(add(f))
        for f in _facts(r, "general_fact"):
            add(f)
    queries = [r["query"] for r in val_rows]
    golds = [sorted(mem_gold[r["memory_id"]]) for r in val_rows]
    return pool, queries, golds


def cmd_eval(args) -> None:
    import torch
    from omegaconf import OmegaConf
    from src.evaluate.retrieval import LocalEmbeddingModel

    if args.val_rows:
        from src.data.load_train_data import read_jsonl_rows
        val = read_jsonl_rows(Path(args.val_rows))
    else:
        _, val, _, _ = split_rows(args.data)
    pool, queries, golds = _build_val_task(val)
    print(f"[synthetic-val] {len(queries)} val queries, pool of {len(pool)} unique facts")

    import hashlib
    pool_key = hashlib.sha256(("\n".join(pool) + args.model).encode("utf-8")).hexdigest()[:12]
    base_cache = Path(args.cache_dir) / f"synthetic_val_base_facts_{pool_key}.pt" if args.cache_dir else None
    rows_out = []
    for spec in args.adapter:
        arm, path = spec.split(":", 1)
        emb_cfg = OmegaConf.create({
            "model": args.model, "base_model": args.model,
            "adapter_path": None if path in ("-", "", "null") else path,
            "max_length": 2048, "batch_size": args.batch_size, "normalize": True, "dtype": "auto",
            "show_progress": False, "apply_adapter_to_query": True,
            "apply_adapter_to_memory": arm == "sym",
            "query_instruction": args.instruction})
        emb = LocalEmbeddingModel(emb_cfg)
        if arm != "sym" and base_cache is not None and base_cache.exists():
            fact_emb = torch.load(base_cache)
        else:
            fact_emb = emb.encode(pool, is_query=False)
            if arm != "sym" and base_cache is not None:
                base_cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(fact_emb, base_cache)
        q_emb = emb.encode(queries, is_query=True)
        scores = q_emb @ fact_emb.T
        order = torch.argsort(scores, dim=1, descending=True)[:, : max(KS)].tolist()
        res = {"arm": arm, "adapter": path, "config": Path(path).name if path not in ("-", "") else "base"}
        for k in KS:
            hits, recs = [], []
            for ranked, gold in zip(order, golds):
                g = set(gold)
                top = set(ranked[:k])
                hits.append(1.0 if g & top else 0.0)
                recs.append(len(g & top) / len(g) if g else 0.0)
            res[f"val_hit@{k}"] = sum(hits) / len(hits)
            res[f"val_recall@{k}"] = sum(recs) / len(recs)
        res["val_mean_hit"] = sum(res[f"val_hit@{k}"] for k in (10, 20, 50, 100)) / 4
        ms = Path(path) / "model_selection.json"
        res["best_eval_loss"] = json.loads(ms.read_text())["best_eval_loss"] if ms.exists() else ""
        print(json.dumps(res))
        rows_out.append(res)
        del emb
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists() or args.overwrite
    with out.open("w" if args.overwrite else "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {len(rows_out)} row(s) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overlap")
    o.add_argument("--data", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--clean-val-out", default=None)
    e = sub.add_parser("eval")
    e.add_argument("--data", required=True)
    e.add_argument("--val-rows", default=None, help="optional jsonl of val rows (e.g. overlap-free subset)")
    e.add_argument("--adapter", action="append", required=True, help="arm:path  (arm = base|aqui|sym)")
    e.add_argument("--out", required=True)
    e.add_argument("--overwrite", action="store_true")
    e.add_argument("--cache-dir", default=None)
    e.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--instruction", default="Given a user query, retrieve target knowledge or reasoning that is useful for answering it.")
    args = ap.parse_args()
    {"overlap": cmd_overlap, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
