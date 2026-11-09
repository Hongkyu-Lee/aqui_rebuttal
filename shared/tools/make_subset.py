"""Subset an ActMemEval-style dataset JSON + its fact bag to a list of question_ids (E4 paired eval).

Fact-bag rows are keyed by row_index = position in the dataset JSON. The subset re-numbers rows
0..n-1 in the ORDER OF --ids-from (so original / lexical / hard subsets line up row by row).

  python -m tools.make_subset --dataset ActMemEval.json --fact-bag user_facts.jsonl \
        --ids-from ActMemEval_implicit_hard.json --out-dataset orig72.json --out-fact-bag orig72_facts.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fact-bag", required=True)
    ap.add_argument("--ids-from", required=True, help="dataset JSON whose question_id order defines the subset")
    ap.add_argument("--out-dataset", required=True)
    ap.add_argument("--out-fact-bag", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, list))
    want = [r["question_id"] for r in json.loads(Path(args.ids_from).read_text(encoding="utf-8"))]
    pos = {r["question_id"]: i for i, r in enumerate(data)}
    missing = [q for q in want if q not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} question_ids not found in {args.dataset}: {missing[:5]}")
    bag = {}
    with open(args.fact_bag, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh):
            if line.strip():
                r = json.loads(line)
                bag[int(r.get("row_index", line_no))] = r
    out_rows, out_bag = [], []
    for new_i, q in enumerate(want):
        old = pos[q]
        if old not in bag:
            raise SystemExit(f"fact bag has no row_index={old} (question_id={q})")
        out_rows.append(data[old])
        fb = dict(bag[old])
        fb["row_index"] = new_i
        fb["original_row_index"] = old
        fb["example_id"] = str(new_i)
        out_bag.append(fb)
    Path(args.out_dataset).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_dataset).write_text(json.dumps(out_rows, ensure_ascii=False), encoding="utf-8")
    with open(args.out_fact_bag, "w", encoding="utf-8") as fh:
        for r in out_bag:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"subset {len(out_rows)} rows -> {args.out_dataset} / {args.out_fact_bag}")


if __name__ == "__main__":
    main()
