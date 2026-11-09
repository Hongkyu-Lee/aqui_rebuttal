"""Rebuild ActMemEval_implicit_hard.json (72 rows) from bundled, local artifacts.

Why: the copy of ActMemEval_implicit_hard.json in the old repo is 0 bytes. The benchmark is
fully determined by
  * ActMemEval_lexical_controlled.json  (224 rows; the input to the implicit rewrite step),
  * implicit_hard_rewrites.jsonl        (cache of rewritten GOLD SESSIONS, key implicit_hard:<row>:<session>),
  * ActMemEval_implicit_hard_metadata.json (per-session validation records -> which rows were kept).
This mirrors dataset_analysis/create_actmemeval_variants.py::build_implicit_hard:
a row is kept iff every gold-session rewrite passed (rewrite_passed AND lexical_validation.passed
AND implicit_validation.passed) and no record carries a failed row_validation.

The rebuilt file is checked against the fact bag that the paper evaluated on (72 rows, same
session ids per row). Exit code 1 if the check fails.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
from collections import defaultdict
from pathlib import Path


def lit(v):
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return ast.literal_eval(v)
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexical", required=True)
    ap.add_argument("--rewrites", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--fact-bag", default=None, help="paper fact bag, used to verify the rebuild")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lexical = json.loads(Path(args.lexical).read_text(encoding="utf-8"))
    by_qid = {r["question_id"]: r for r in lexical}
    cache = {}
    with open(args.rewrites, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                cache[rec["cache_key"]] = rec
    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))

    recs_by_row: dict[int, list[dict]] = defaultdict(list)
    for rec in meta["records"]:
        recs_by_row[int(rec["row_index"])].append(rec)

    kept = []
    for row_index in sorted(recs_by_row):          # original-row order == build order
        recs = recs_by_row[row_index]
        qid = recs[0]["question_id"]
        ok = all(r["rewrite_passed"] and r["lexical_validation"].get("passed")
                 and r["implicit_validation"].get("passed") for r in recs)
        ok = ok and not any(r.get("row_validation") and not r["row_validation"].get("passed") for r in recs)
        if not ok:
            continue
        if qid not in by_qid:
            raise SystemExit(f"question_id {qid} not in lexical file")
        row = copy.deepcopy(by_qid[qid])
        for key in ("haystack_sessions", "haystack_session_ids", "answer_session_ids", "haystack_dates"):
            if key in row:
                row[key] = lit(row[key])
        for r in recs:
            ck = f"implicit_hard:{row_index}:{r['session_index']}"
            if ck not in cache:
                raise SystemExit(f"missing rewrite cache entry {ck}")
            row["haystack_sessions"][int(r["session_index"])] = [
                {"role": m["role"], "content": m["content"]} for m in cache[ck]["session"]]
        row["_source_row_index"] = row_index
        kept.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"rebuilt {len(kept)} rows -> {args.out}")

    if args.fact_bag:
        bag = [json.loads(l) for l in open(args.fact_bag, encoding="utf-8") if l.strip()]
        bag.sort(key=lambda r: int(r["row_index"]))
        problems = []
        if len(bag) != len(kept):
            problems.append(f"row count {len(kept)} != fact bag {len(bag)}")
        for i, (row, fb) in enumerate(zip(kept, bag)):
            if int(fb.get("num_sessions", len(row["haystack_sessions"]))) != len(row["haystack_sessions"]):
                problems.append(f"row {i}: num_sessions mismatch")
            ids = row["haystack_session_ids"]
            for f in fb.get("facts", [])[:50]:
                si = int(f["session_index"])
                if f.get("session_id") and (si >= len(ids) or ids[si] != f["session_id"]):
                    problems.append(f"row {i}: session id mismatch at session {si}")
                    break
        if problems:
            print("VERIFY FAILED:\n  " + "\n  ".join(problems[:20]))
            raise SystemExit(1)
        print("verified against fact bag: row order, session counts and session ids all match")


if __name__ == "__main__":
    main()
