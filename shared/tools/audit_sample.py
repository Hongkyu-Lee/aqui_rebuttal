"""E4(c): draw a fixed random sample of kept ActMemEval-hard gold-session rewrites for a HUMAN check.

Writes <out>.csv (one row per sampled rewrite; empty rating columns for 2 annotators) and
<out>_instructions.md. After annotation, `--score <filled.csv>` prints pass rates and the
inter-annotator agreement (Cohen's kappa on the overall pass decision).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from pathlib import Path

COLS = ["meaning_preserved_0to5", "answer_still_supported_yn", "requires_bridge_yn", "overall_pass_yn"]

INSTR = """# Human check of ActMemEval-hard rewrites

For each row, read the QUESTION and GOLD ANSWER, then compare the REFERENCE session (before
rewriting) with the REWRITTEN session. Fill the columns for your annotator id (a1_ or a2_):

- meaning_preserved_0to5 : 5 = every user-specific fact is still there (paraphrase is fine); 0 = lost/changed.
- answer_still_supported_yn : y if, using only the rewritten session as memory, the gold answer is still correct and justified.
- requires_bridge_yn : y if connecting the question to the rewritten session needs an inference/world-knowledge bridge (not just the same surface topic).
- overall_pass_yn : y if the rewrite is a valid benchmark item (meaning >= 4, answer supported, needs a bridge).

Annotate independently; do not discuss rows before both annotators finish.
"""


def lit(v):
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return ast.literal_eval(v)
    return v


def fmt(s):
    return "\n".join(f"{m['role']}: {m['content']}" for m in s)


def yn(x):
    return str(x).strip().lower() in ("y", "yes", "1", "true")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata")
    ap.add_argument("--rewrites")
    ap.add_argument("--lexical")
    ap.add_argument("--hard")
    ap.add_argument("--original", default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", default=None)
    ap.add_argument("--score", default=None, help="filled CSV -> print agreement and pass rates")
    args = ap.parse_args()

    if args.score:
        rows = list(csv.DictReader(open(args.score, encoding="utf-8")))
        a1 = [yn(r["a1_overall_pass_yn"]) for r in rows]
        a2 = [yn(r["a2_overall_pass_yn"]) for r in rows]
        n = len(rows)
        po = sum(x == y for x, y in zip(a1, a2)) / n
        pe = (sum(a1) / n) * (sum(a2) / n) + (1 - sum(a1) / n) * (1 - sum(a2) / n)
        print(json.dumps({"n": n, "a1_pass_rate": sum(a1) / n, "a2_pass_rate": sum(a2) / n,
                          "both_pass_rate": sum(x and y for x, y in zip(a1, a2)) / n,
                          "agreement": po, "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else 1.0}, indent=2))
        return

    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    cache = {json.loads(l)["cache_key"]: json.loads(l) for l in open(args.rewrites, encoding="utf-8") if l.strip()}
    lexical = {r["question_id"]: r for r in json.loads(Path(args.lexical).read_text(encoding="utf-8"))}
    kept = {r["question_id"] for r in json.loads(Path(args.hard).read_text(encoding="utf-8"))}
    ref_src, ref_kind = lexical, "lexical_controlled"
    if args.original and Path(args.original).exists():
        od = json.loads(Path(args.original).read_text(encoding="utf-8"))
        od = od if isinstance(od, list) else next(v for v in od.values() if isinstance(v, list))
        ref_src, ref_kind = {r["question_id"]: r for r in od}, "original"
    pool = [r for r in meta["records"] if r["question_id"] in kept]
    random.Random(args.seed).shuffle(pool)
    sample = pool[: args.n]
    out = Path(f"{args.out}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fields = ["item", "question_id", "session_index", "question", "gold_answer", f"reference_session ({ref_kind})",
                  "rewritten_session"] + [f"a{i}_{c}" for i in (1, 2) for c in COLS]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(sample):
            ck = f"implicit_hard:{r['row_index']}:{r['session_index']}"
            w.writerow({"item": i, "question_id": r["question_id"], "session_index": r["session_index"],
                        "question": r["question"], "gold_answer": r["answer"],
                        f"reference_session ({ref_kind})": fmt(lit(ref_src[r["question_id"]]["haystack_sessions"])[int(r["session_index"])]),
                        "rewritten_session": fmt(cache[ck]["session"])})
    Path(f"{args.out}_instructions.md").write_text(INSTR, encoding="utf-8")
    print(f"wrote {len(sample)} items -> {out}")


if __name__ == "__main__":
    main()
