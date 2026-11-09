"""E5 step 2: score retrieval results under three recall definitions.

  session_recall@K  : fraction of gold sessions with >=1 retrieved fact (the paper's metric)
  fact_recall@K     : fraction of the gold-session facts retrieved
  evidence_hit@K    : >=1 of the LLM-labelled EVIDENCE facts retrieved ("the specific fact")
  evidence_recall@K : fraction of evidence facts retrieved
Questions with an empty evidence set are excluded from the evidence metrics (count reported).

  python -m tools.recall_definitions --evidence ev.jsonl --ks 10,20,50,100 \
        --result base=base.json --result aqui=aqui.json --result sym=sym.json --out table
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--result", action="append", required=True, help="name=fact_retrieval_result.json")
    ap.add_argument("--ks", default="10,20,30,50,100")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ev = {}
    for l in open(args.evidence, encoding="utf-8"):
        if l.strip():
            d = json.loads(l)
            ev[(d["example_index"], d["qa_index"])] = set(d["evidence_fact_indices"])
    ks = [int(k) for k in args.ks.split(",")]
    rows = []
    for spec in args.result:
        name, path = spec.split("=", 1)
        res = json.load(open(path, encoding="utf-8"))["results"]
        row = {"method": name, "n_questions": len(res)}
        with_ev = [r for r in res if ev.get((r["example_index"], r["qa_index"]))]
        row["n_with_evidence"] = len(with_ev)
        for k in ks:
            if str(k) in res[0]["metrics"]:
                row[f"session_recall@{k}"] = sum(r["metrics"][str(k)]["session_recall"] for r in res) / len(res)
                row[f"fact_recall@{k}"] = sum(r["metrics"][str(k)]["fact_recall"] for r in res) / len(res)
            hits, recs = [], []
            for r in with_ev:
                e = ev[(r["example_index"], r["qa_index"])]
                top = {int(x["fact_index"]) for x in r["retrieved_facts"][:k]}
                hits.append(1.0 if e & top else 0.0)
                recs.append(len(e & top) / len(e))
            row[f"evidence_hit@{k}"] = sum(hits) / max(1, len(hits))
            row[f"evidence_recall@{k}"] = sum(recs) / max(1, len(recs))
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
