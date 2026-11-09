#!/usr/bin/env python3
"""Unified LLM-as-judge answer-quality eval for BOTH TaciTree and ActMem, with explicit
LATENT-REASONING (grounding) evaluation for this implicit-reasoning dataset.

The judge scores TWO independent booleans per answer:
  - conclusion_aligned : does the answer reach the gold answer's conclusion? (meaning, not wording)
  - grounded           : does the answer justify itself using the GOLD SESSION MEMORY (the
                         latent fact), rather than a lucky unrelated reason?

From those two we derive BOTH a rigorous and a lenient score (no double-judging):

  aligned grounded  case  rigorous  lenient
     T       T        1      1.0      1.0     (right answer, right reason)
     T       F        2      0.0      0.5     (LUCKY: right answer, wrong/absent reason)
     F       T        3      0.0      0.5
     F       F        4      0.0      0.0

The four case counts are recorded so any statistic (luck gap = case2, grounding rate =
cases1+3, plain conclusion accuracy = cases1+2, ...) can be computed later.

Latent memory: pass --facts <dataset facts.json>. For each question we join its
gold_session_ids to that session's fact texts and hand them to the judge as the reasoning
the answer MUST be based on. Both systems use the SAME dataset facts.json -> same latent
ground truth -> fair.

What counts as "the answer":
  ActMem   : answers[*].answer         vs answers[*].gold_answer
  TaciTree : <qa>.select_fact          vs <qa>.ground_truth   (--method to change)

Judge model: the local chat server (e.g. gpt-4o-mini alias). SAME model for all runs.

    python evaluate_answer_quality.py --system tacitree \
        --input ../tacitree/datasets/actmemeval_hard/actmemeval_hard_response.json \
        --facts $ROOT/actmemeval_implicit_hard/facts.json --model gpt-4o-mini --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from judgelib.clients import ChatClient
from judgelib.utils import atomic_write_json, load_env_file, read_json


# ----------------------------------------------------------------------------- loaders

def load_actmem(path: Path) -> list[dict]:
    if path.is_dir():
        results = path / "results.json"
        files = [results] if results.exists() else sorted(path.glob("row_*/answers.json"))
    else:
        files = [path]
    raw: list[dict] = []
    for fp in files:
        payload = read_json(fp)
        if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "answers" in payload[0]:
            for row in payload:
                raw.extend(a for a in (row.get("answers") or []) if isinstance(a, dict))
        elif isinstance(payload, list):
            raw.extend(a for a in payload if isinstance(a, dict))
    return [{"question": a.get("question"), "gold_answer": a.get("gold_answer"),
             "answer": a.get("answer")} for a in raw]


def load_tacitree(path: Path, method: str = "select_fact") -> list[dict]:
    payload = read_json(path)
    out = []
    for conv in payload:
        items = conv if isinstance(conv, list) else [conv]
        for qa in items:
            if isinstance(qa, dict):
                out.append({"question": qa.get("question"), "gold_answer": qa.get("ground_truth"),
                            "answer": qa.get(method)})
    return out


def load_iris(path: Path) -> list[dict]:
    """iris method: {"summary": ..., "samples": [{question, generated_answer, gold_answer, ...}]}."""
    payload = read_json(path)
    samples = payload.get("samples", []) if isinstance(payload, dict) else payload
    return [{"question": r.get("question"), "gold_answer": r.get("gold_answer"),
             "answer": r.get("generated_answer")}
            for r in samples if isinstance(r, dict)]


# -------------------------------------------------------------- latent (gold) memory

def _norm_q(q) -> str:
    return " ".join(str(q or "").split()).lower()


def build_latent_map(facts_path: Path, max_facts: int = 20) -> dict[str, list[str]]:
    """question(normalized) -> gold-session fact texts, from the dataset facts.json."""
    payload = read_json(facts_path)
    rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
    qmap: dict[str, list[str]] = {}
    for row in rows:
        sess_facts: dict[int, list[str]] = {}
        for s in row.get("sessions", []):
            idx = s.get("session_index")
            if idx is None:
                continue
            texts = [(f.get("text") or "").strip() for f in s.get("facts", []) if (f.get("text") or "").strip()]
            sess_facts[int(idx)] = texts
        qa = row.get("qa") or {}
        questions = qa.get("questions") or []
        golds = qa.get("gold_session_ids") or []
        for i, q in enumerate(questions):
            if not q:
                continue
            latent: list[str] = []
            for gsid in (golds[i] if i < len(golds) else []) or []:
                latent.extend(sess_facts.get(int(gsid), []))
            qmap[_norm_q(q)] = latent[:max_facts]
    return qmap


# ------------------------------------------------------------------------- the judge

JUDGE_SYSTEM = """
You rigorously evaluate whether a model's answer reflects the correct LATENT REASONING for
an implicit-reasoning task -- not just a lucky-correct conclusion.

For each item you get:
- question: the user's query
- latent_memory: facts from the user's PAST sessions that the correct answer must be based
  on (the hidden reasoning). May be empty -- then fall back to the gold answer's own reason.
- gold_answer: the reference answer (its conclusion, and usually its justification)
- answer: the model's answer to judge

Judge TWO things INDEPENDENTLY:

1. conclusion_aligned (true/false): does the answer reach the same core conclusion /
   recommendation as the gold answer? Judge meaning, not wording.

2. grounded (true/false): does the answer's JUSTIFICATION actually use the latent_memory
   (or, if empty, the gold answer's stated reason)? A correct conclusion supported by an
   UNRELATED or WRONG reason is NOT grounded -- the model was lucky. Only true if the
   answer's reasoning genuinely reflects the latent fact.

Worked example -- query "suggest outdoor activities", latent_memory "the user injured his knee":
- "Try some indoor yoga, since your knee is still healing" -> conclusion_aligned=true, grounded=true
- "Try some indoor yoga, it's rainy today"                 -> conclusion_aligned=true, grounded=FALSE (lucky)
- "How about a mountain hike!"                              -> conclusion_aligned=false, grounded=false

Return JSON with EXACTLY this shape:
{"items": [{"eval_id": "copied from input", "conclusion_aligned": true|false,
            "grounded": true|false, "reason": "short explanation"}]}
""".strip()


def judge_batch(batch: list[dict], client: ChatClient) -> dict[str, dict]:
    user = json.dumps({"items": [
        {"eval_id": r["eval_id"], "question": r["question"],
         "latent_memory": r.get("latent_facts") or [],
         "gold_answer": r["gold_answer"], "answer": r["answer"]}
        for r in batch
    ]}, ensure_ascii=False, indent=2)
    response = client.complete_json(system=JUDGE_SYSTEM, user=user)
    out = {}
    for item in response.get("items", []):
        if isinstance(item, dict) and str(item.get("eval_id", "")).strip():
            out[str(item["eval_id"]).strip()] = item
    return out


def _case(aligned: bool, grounded: bool) -> int:
    return 1 if (aligned and grounded) else 2 if aligned else 3 if grounded else 4


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=["actmem", "tacitree", "iris"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--facts", default=None, help="dataset facts.json -> latent gold-session memory")
    ap.add_argument("--method", default="select_fact", help="tacitree: which method's answer")
    ap.add_argument("--model", default="gpt-4o-mini", help="judge model alias (SAME for all runs)")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--output", default=None)
    return ap.parse_args()


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    args = parse_args()
    load_env_file(args.env_file)

    in_path = Path(args.input)
    if args.system == "actmem":
        records = load_actmem(in_path)
        tag = "actmem"
    elif args.system == "iris":
        records = load_iris(in_path)
        tag = "iris"
    else:
        records = load_tacitree(in_path, args.method)
        tag = f"tacitree_{args.method}"

    latent_map = build_latent_map(Path(args.facts)) if args.facts else {}
    if args.facts:
        print(f"latent memory: {len(latent_map)} questions mapped from {args.facts}")

    evaluable, skipped, with_latent = [], 0, 0
    for i, r in enumerate(records):
        gold = (r.get("gold_answer") or "").strip() or None
        ans = (r.get("answer") or "").strip() or None
        if not gold or not ans:
            skipped += 1
            continue
        latent = latent_map.get(_norm_q(r.get("question")), [])
        if latent:
            with_latent += 1
        evaluable.append({"eval_id": f"{tag}:{i}", "question": r.get("question") or "",
                          "gold_answer": gold, "answer": ans, "latent_facts": latent})
    print(f"{args.system}: {len(records)} answers | {len(evaluable)} judgeable | "
          f"{with_latent} with latent memory | {skipped} skipped")
    if not evaluable:
        raise SystemExit("nothing to judge")
    if args.facts and with_latent == 0:
        print("WARNING: 0 questions matched the facts.json -- grounding falls back to gold_answer's reason. "
              "Check that --facts is the same dataset as --input.")

    base_dir = in_path if in_path.is_dir() else in_path.parent
    out_path = Path(args.output) if args.output else base_dir / f"answer_quality_{tag}.json"
    jsonl_path = out_path.with_suffix(".jsonl")

    judged: dict[str, dict] = {}
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    judged[rec["eval_id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"resuming: {len(judged)} already judged")
    todo = [r for r in evaluable if r["eval_id"] not in judged]

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    write_lock = threading.Lock()
    handle = open(jsonl_path, "a", encoding="utf-8")

    def run_batch(batch):
        client = ChatClient(model=args.model, api_key_env=args.api_key_env, base_url=base_url,
                            temperature=args.temperature, max_tokens=args.max_tokens)
        judgments = judge_batch(batch, client)
        out = []
        for item in batch:
            j = judgments.get(item["eval_id"], {})
            aligned = bool(j.get("conclusion_aligned"))
            grounded = bool(j.get("grounded"))
            out.append({"eval_id": item["eval_id"], "question": item["question"],
                        "gold_answer": item["gold_answer"], "answer": item["answer"],
                        "latent_facts": item["latent_facts"],
                        "conclusion_aligned": aligned, "grounded": grounded,
                        "case": _case(aligned, grounded), "reason": j.get("reason", "")})
        return out

    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_batch, b) for b in batches]
            for fut in as_completed(futures):
                try:
                    results = fut.result()
                except Exception as exc:
                    print(f"  batch failed: {exc}", flush=True)
                    continue
                with write_lock:
                    for rec in results:
                        judged[rec["eval_id"]] = rec
                        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    handle.flush()
                    n = len(judged)
                    if n % 50 == 0 or n == len(evaluable):
                        print(f"  judged {n}/{len(evaluable)}", flush=True)
    finally:
        handle.close()

    # ----- aggregate: four cases -> both scorings + slice-able stats
    recs = list(judged.values())
    N = len(recs)
    counts = {c: sum(1 for r in recs if r.get("case") == c) for c in (1, 2, 3, 4)}
    frac = (lambda n: n / N if N else None)
    summary = {
        "system": args.system,
        "method": args.method if args.system == "tacitree" else "final_answer",
        "judge_model": args.model, "input": str(in_path), "facts": args.facts,
        "num_answers": len(records), "num_judged": N, "num_skipped": skipped,
        "num_with_latent_memory": with_latent,
        "case_counts": {
            "1_aligned_grounded": counts[1], "2_aligned_lucky": counts[2],
            "3_misaligned_grounded": counts[3], "4_wrong": counts[4],
        },
        # rigorous: only case 1 earns credit
        "rigorous_accuracy": frac(counts[1]),
        # lenient: case1=1.0, case2=case3=0.5, case4=0.0
        "lenient_score": ((counts[1] * 1.0 + counts[2] * 0.5 + counts[3] * 0.5) / N) if N else None,
        # slice-able views
        "conclusion_accuracy": frac(counts[1] + counts[2]),   # right conclusion (grounded or not)
        "grounding_rate": frac(counts[1] + counts[3]),        # reasoning used the latent fact
        "luck_gap": frac(counts[2]),                          # right answer, WRONG reason
    }
    atomic_write_json(out_path, {"summary": summary, "results": recs})

    print("\n=== answer quality (rigorous + lenient) ===")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
