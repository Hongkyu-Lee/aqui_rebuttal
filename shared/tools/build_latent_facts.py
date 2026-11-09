"""E6 helper: build the judge's "latent memory" file (facts.json) from a dataset + fact bag.

The judge (shared/judge/evaluate_answer_quality.py --facts) expects
  {"rows": [{"sessions": [{"session_index": i, "facts": [{"text": ...}]}],
             "qa": {"questions": [...], "gold_session_ids": [[...]]}}]}
The paper's judge runs used ActMem's extracted facts.json (${ACTMEM_FACTS_ROOT}/<ds>/facts.json).
ALWAYS prefer that file when it exists (so oracle and paper rows see the same latent memory);
this builder is the fallback.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="implexconv | actmemeval:<path.json>")
    ap.add_argument("--fact-bag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from omegaconf import OmegaConf
    from src.data.load_data import load_data
    from src.evaluate.fact_retrieval import load_fact_rows, normalize_fact_items

    if args.data == "implexconv":
        cfg = OmegaConf.create({"data": {"name": "implexconv", "type": "opposed", "split": None, "path": None}, "run": {}})
    else:
        cfg = OmegaConf.create({"data": {"name": "actmemeval", "type": None, "split": None,
                                         "path": args.data.split(":", 1)[1]}, "run": {}})
    examples = load_data(cfg)
    bag = load_fact_rows(Path(args.fact_bag))
    rows = []
    for ei, ex in enumerate(examples):
        fr = bag.get(ei, None)
        by_s = defaultdict(list)
        for f in (normalize_fact_items(fr) if fr else []):
            by_s[f.session_index].append({"text": f.fact})
        rows.append({"sessions": [{"session_index": s, "facts": fs} for s, fs in sorted(by_s.items())],
                     "qa": {"questions": ex["qa"]["questions"], "gold_session_ids": ex["qa"]["gold_session_ids"]}})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    print(f"latent facts for {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
