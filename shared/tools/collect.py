"""Collect fact_retrieval result JSONs into one table (CSV + markdown).

  python -m tools.collect --out table <file1.json> <file2.json> ...      # or --glob 'dir/*.json'
Columns: file, n_questions, session_recall@K for K in --ks.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--glob", default=None)
    ap.add_argument("--ks", default="10,20,30,50,100")
    ap.add_argument("--metric", default="session_recall")
    ap.add_argument("--out", required=True, help="output prefix (writes .csv and .md)")
    args = ap.parse_args()
    files = list(args.files) + (sorted(glob.glob(args.glob)) if args.glob else [])
    ks = [k.strip() for k in args.ks.split(",") if k.strip()]
    rows = []
    for f in files:
        try:
            s = json.load(open(f, encoding="utf-8"))["summary"]
        except Exception as exc:  # noqa: BLE001
            print(f"skip {f}: {exc}")
            continue
        r = {"file": Path(f).name, "n_questions": s.get("num_questions")}
        for k in ks:
            v = s.get(f"{args.metric}@{k}")
            r[f"{args.metric}@{k}"] = round(v, 4) if isinstance(v, (int, float)) else ""
        rows.append(r)
    if not rows:
        raise SystemExit("no result files")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    Path(f"{args.out}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
