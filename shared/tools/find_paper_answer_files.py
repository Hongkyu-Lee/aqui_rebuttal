"""E6 helper: identify which saved answer files are the paper's AQUI rows (Table 2).

Nobody recorded which files in gepa_answergen/ produced Table 2. This tool reconstructs it:

1. Scan the answer folder (gepa_answergen/answers__*.json). Each file's "summary" records the
   answer model, the reranker model and the number of facts K.
2. Scan the old judge outputs (judge_outputs/*.json). Each judged file's summary records the
   "input" answer file it scored, the judge model and the rigorous/lenient scores.
3. For every Table 2 cell (answer model x dataset x K), list the candidate answer files. A
   candidate is CONFIRMED when its old Qwen3.5-122B judge scores equal the Table 2 numbers
   (within --tol).

Writes <out>.md (table for a human) and <out>.txt (one confirmed file per line; feed it to E6:
    REJUDGE_FILES="$(tr '\n' ' ' < <out>.txt)" bash E6/run.sh ).

Usage:
  python -m tools.find_paper_answer_files \
      --answers /group-volume/users/h.lee4/prototypes/version_03_artifacts/gepa_answergen \
      --judged  /group-volume/users/h.lee4/judge_outputs \
      --out     $OUT_BASE/E6/paper_aqui_answer_files
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

# Table 2 of the submission: (answer model, dataset, K) -> (rigorous, lenient), AQUI rows.
TABLE2 = {
    ("r1_70b", "actmemeval_hard", 10): (0.528, 0.583), ("r1_70b", "actmemeval_hard", 20): (0.528, 0.576),
    ("r1_70b", "actmemeval_hard", 30): (0.583, 0.645), ("r1_70b", "actmemeval_hard", 50): (0.583, 0.632),
    ("r1_70b", "actmemeval_hard", 68): (0.569, 0.611),
    ("r1_70b", "implexconv", 10): (0.088, 0.155), ("r1_70b", "implexconv", 20): (0.088, 0.167),
    ("r1_70b", "implexconv", 30): (0.090, 0.124), ("r1_70b", "implexconv", 50): (0.094, 0.134),
    ("r1_70b", "implexconv", 151): (0.182, 0.217),
    ("qwen80b", "actmemeval_hard", 10): (0.514, 0.556), ("qwen80b", "actmemeval_hard", 14): (0.528, 0.576),
    ("qwen80b", "actmemeval_hard", 20): (0.528, 0.576), ("qwen80b", "actmemeval_hard", 30): (0.583, 0.625),
    ("qwen80b", "actmemeval_hard", 40): (0.521, 0.545),
    ("qwen80b", "implexconv", 10): (0.100, 0.168), ("qwen80b", "implexconv", 20): (0.112, 0.175),
    ("qwen80b", "implexconv", 30): (0.098, 0.162), ("qwen80b", "implexconv", 50): (0.102, 0.167),
    ("qwen80b", "implexconv", 92): (0.096, 0.156),
}


def short_model(name: str) -> str:
    n = str(name).lower()
    if "deepseek" in n or "r1" in n:
        return "r1_70b"
    if "80b" in n or "next" in n:
        return "qwen80b"
    if "4b" in n:
        return "qwen4b"
    return n.split("/")[-1] or "?"


def dataset_of(path: str, summary: dict) -> str:
    blob = (str(summary.get("fact_bag_path", "")) + " " + os.path.basename(path)).lower()
    if "implexconv" in blob or "opposed" in blob:
        return "implexconv"
    if "actmemeval" in blob:
        return "actmemeval_hard"
    return "unknown"


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", required=True, help="gepa_answergen folder (searched recursively)")
    ap.add_argument("--judged", default="", help="old judge_outputs folder (optional but needed to CONFIRM)")
    ap.add_argument("--out", required=True, help="output prefix (writes .md and .txt)")
    ap.add_argument("--tol", type=float, default=0.0015, help="max |score - Table 2| to count as a match")
    args = ap.parse_args()

    # 1. answer files
    answers = {}
    for p in glob.glob(os.path.join(args.answers, "**", "*.json"), recursive=True):
        d = load(p)
        if not isinstance(d, dict) or "samples" not in d:
            continue
        s = d.get("summary") or {}
        rr = str(s.get("reranker_model", "") or "").strip()
        bienc = (not rr) or any(t in p.lower() for t in ("norerank", "bi_encoder", "bienc"))
        answers[os.path.realpath(p)] = {
            "path": p, "answer": short_model(s.get("answer_model", "")),
            "reranker": "none" if bienc else short_model(rr), "k": s.get("answer_top_k_facts"),
            "dataset": dataset_of(p, s), "n": len(d.get("samples") or []),
            "adapter": next((str(v) for k, v in s.items() if "adapter" in k.lower() or "ckpt" in k.lower()), ""),
            "mtime": os.path.getmtime(p),
        }

    # 2. old judge outputs -> scores per input answer file (keep the 122B judge only)
    judged = defaultdict(list)
    if args.judged and os.path.isdir(args.judged):
        for p in glob.glob(os.path.join(args.judged, "**", "*.json"), recursive=True):
            d = load(p)
            s = d.get("summary") if isinstance(d, dict) else None
            if not isinstance(s, dict) or "rigorous_accuracy" not in s or not s.get("input"):
                continue
            is122 = "122b" in os.path.basename(p).lower() or "j120b" in os.path.basename(p).lower() \
                or "122b" in str(s.get("judge_model", "")).lower()
            judged[os.path.realpath(s["input"])].append(
                {"judged_file": p, "judge122b": is122, "rig": s["rigorous_accuracy"], "len": s["lenient_score"]})
            # the input path may be recorded relative to another mount: also index by basename
            judged["base:" + os.path.basename(s["input"])].append(judged[os.path.realpath(s["input"])][-1])

    # 3. per Table 2 cell
    lines = ["# Paper AQUI answer files (Table 2) -- reconstruction", "",
             f"answers: `{args.answers}` ({len(answers)} answer files) | judged: `{args.judged or '-'}`", "",
             "| answer model | dataset | K | Table 2 (rig / len) | status | answer file | reranker | old judge (rig / len) |",
             "|---|---|---|---|---|---|---|---|"]
    confirmed = []
    for (am, ds, k), (rig, len_) in TABLE2.items():
        cands = [a for a in answers.values() if a["answer"] == am and a["dataset"] == ds and str(a["k"]) == str(k)]
        best = None
        rows = []
        for a in sorted(cands, key=lambda a: (a["reranker"] == "none", -a["mtime"])):
            js = judged.get(os.path.realpath(a["path"])) or judged.get("base:" + os.path.basename(a["path"])) or []
            js122 = [j for j in js if j["judge122b"]] or js
            match = next((j for j in js122 if abs(j["rig"] - rig) <= args.tol and abs(j["len"] - len_) <= args.tol), None)
            sc = " ; ".join(f"{j['rig']:.3f} / {j['len']:.3f}" for j in js122) or "not judged / judge file missing"
            status = "CONFIRMED" if match else ("candidate" if a["reranker"] != "none" else "no (no reranker)")
            if match and best is None:
                best = a["path"]
            rows.append(f"| {am} | {ds} | {k} | {rig:.3f} / {len_:.3f} | {status} | `{os.path.basename(a['path'])}` | {a['reranker']} | {sc} |")
        if not rows:
            rows.append(f"| {am} | {ds} | {k} | {rig:.3f} / {len_:.3f} | **NOT FOUND** | - | - | - |")
        lines += rows
        if best:
            confirmed.append(best)
    n_cells = len(TABLE2)
    lines += ["", f"**Confirmed {len(confirmed)} of {n_cells} Table 2 cells.**", "",
              "* CONFIRMED: the old 122B-judge scores of this file equal Table 2 -> it is the paper's file.",
              "* candidate: right model / dataset / K and uses a reranker, but no old judge score to confirm.",
              "  If a cell has several candidates, the paper used a reranker (2 LLM calls per question); the",
              "  4B reranker (qwen4b) is the paper's main setting. Ask h.lee4 before picking.",
              "* NOT FOUND: no answer file for that cell in the folder."]
    with open(args.out + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(args.out + ".txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(confirmed) + ("\n" if confirmed else ""))
    print("\n".join(lines))
    print(f"\nwrote {args.out}.md and {args.out}.txt ({len(confirmed)} confirmed files)")


if __name__ == "__main__":
    main()
