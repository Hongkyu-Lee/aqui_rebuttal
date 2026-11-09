"""E2b: how much does SYMMETRIC LoRA move the fact (memory) space?

AQUI never changes fact embeddings (facts are encoded by the frozen base), so its drift is 0 by
construction. For the symmetric adapter we measure, on a fixed random sample of real memory facts:
  * cos_to_base     : mean cosine between a fact's base embedding and its adapted embedding
  * knn_overlap@10  : mean |kNN_base(f) intersect kNN_adapted(f)| / 10 within the sample
                      (does the neighbourhood structure of the memory index survive?)
  * rank_corr       : mean Spearman correlation of each fact's similarity profile (base vs adapted)

  python -m tools.drift --fact-bag user_facts.jsonl --adapter <sym_adapter> --n 20000 --out drift.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def sample_facts(path: str, n: int, seed: int) -> list[str]:
    facts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                for f in json.loads(line).get("facts", []):
                    t = str(f.get("fact", "")).strip()
                    if t:
                        facts.append(t)
    facts = sorted(set(facts))
    random.Random(seed).shuffle(facts)
    return facts[:n]


def main() -> None:
    import torch
    from omegaconf import OmegaConf
    from src.evaluate.retrieval import LocalEmbeddingModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--fact-bag", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--adapter", action="append", required=True, help="name=path of an adapter whose MEMORY side is measured")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = {}
    for ad in args.adapter:
        aname, apath = ad.split("=", 1)
        emb = LocalEmbeddingModel(OmegaConf.create({
            "model": args.model, "base_model": args.model, "adapter_path": apath, "max_length": 2048,
            "batch_size": args.batch_size, "normalize": True, "dtype": "auto", "show_progress": False,
            "apply_adapter_to_query": True, "apply_adapter_to_memory": False,
            "query_instruction": ""}))
        for fb in args.fact_bag:
            fname, fpath = fb.split("=", 1)
            texts = sample_facts(fpath, args.n, args.seed)
            emb.apply_adapter_to_memory = False
            base = emb.encode(texts, is_query=False)
            emb.apply_adapter_to_memory = True
            adapted = emb.encode(texts, is_query=False)
            cos = (base * adapted).sum(1)
            res = {"n_facts": len(texts), "cos_to_base_mean": float(cos.mean()),
                   "cos_to_base_p05": float(cos.quantile(0.05)), "cos_to_base_min": float(cos.min())}
            # neighbourhoods within the sample (chunked to bound memory)
            k = 10
            overl, rc = [], []
            sub = min(len(texts), 5000)          # query facts for kNN / rank stats
            for s in range(0, sub, 512):
                qb, qa = base[s:s + 512], adapted[s:s + 512]
                sb, sa = qb @ base.T, qa @ adapted.T
                idx = torch.arange(s, min(s + 512, sub))
                sb[torch.arange(len(idx)), idx] = -2
                sa[torch.arange(len(idx)), idx] = -2
                nb = sb.topk(k, dim=1).indices
                na = sa.topk(k, dim=1).indices
                for i in range(len(idx)):
                    overl.append(len(set(nb[i].tolist()) & set(na[i].tolist())) / k)
                rb = sb.argsort(1).argsort(1).float()
                ra = sa.argsort(1).argsort(1).float()
                rb = rb - rb.mean(1, keepdim=True)
                ra = ra - ra.mean(1, keepdim=True)
                rc.extend(((rb * ra).sum(1) / (rb.norm(dim=1) * ra.norm(dim=1))).tolist())
            res[f"knn_overlap@{k}"] = sum(overl) / len(overl)
            res["similarity_rank_corr"] = sum(rc) / len(rc)
            report[f"{aname}::{fname}"] = res
            print(aname, fname, json.dumps(res))
        del emb
        torch.cuda.empty_cache()
    report["AQUI (query-only LoRA)"] = {"cos_to_base_mean": 1.0, "knn_overlap@10": 1.0,
                                       "note": "facts encoded by the frozen base -> identical by construction"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
