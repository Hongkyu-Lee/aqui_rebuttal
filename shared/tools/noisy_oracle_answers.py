"""E7: NOISY-ORACLE answer generation (gold evidence hidden among realistic distractors).

E6 gives the answer model only the gold facts (clean oracle). E7 asks the next question: can the
answer model still answer when the gold evidence is present but surrounded by K-n other facts, as
in a real retrieval list? Three context modes, all on the same questions:

  evidence  : only the evidence fact(s) of the question (E5 labels); no distractors
  noisy     : the evidence fact(s) + (K - n) DISTRACTORS, shuffled into a random order.
              Distractors = AQUI's highest-ranked facts that are NOT from any gold session
              (hard, realistic negatives, not random ones).
  retrieved : AQUI's own top-K facts, in rank order (what a real AQUI run gives the model)

So noisy differs from retrieved only in that the evidence is guaranteed to be present, and from
evidence only in the presence of distractors.

Questions whose evidence label is empty (no evidence fact found by E5) are skipped in ALL modes,
so every mode is scored on the same questions; the count is reported in the summary.

The prompt and decoding are imported from tools.oracle_answers (the paper's answer_gen.py prompt,
copied verbatim there), so they are identical to E6 and to the paper.

Output: the judge's input format (--system iris), as in E6:
  {"summary": {...}, "samples": [{question, gold_answer, generated_answer, top_k_facts, ...}]}
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.oracle_answers import ANSWER_SYSTEM, answer_messages, split_reasoning


def load_evidence(path: str) -> dict[tuple[int, int], list[int]]:
    ev = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            d = json.loads(line)
            ev[(d["example_index"], d["qa_index"])] = list(d.get("evidence_fact_indices") or [])
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="implexconv | actmemeval:<path.json>")
    ap.add_argument("--fact-bag", required=True)
    ap.add_argument("--evidence", required=True, help="E5 evidence labels (evidence_<ds>.jsonl)")
    ap.add_argument("--retrieval", required=True, help="AQUI fact_retrieval result JSON (ranked facts per question)")
    ap.add_argument("--mode", choices=["evidence", "noisy", "retrieved"], required=True)
    ap.add_argument("--k", type=int, default=0, help="list length for noisy / retrieved")
    ap.add_argument("--seed", type=int, default=0, help="seed for the position of the evidence in the noisy list")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--no-system", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tag", required=True, help="dataset tag written to the summary (implexconv_opposed | actmemeval_implicit_hard)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode != "evidence" and args.k <= 0:
        ap.error("--k is required for noisy / retrieved")

    from omegaconf import OmegaConf
    from openai import OpenAI
    from src.data.load_data import load_data
    from src.evaluate.fact_retrieval import load_fact_rows, normalize_fact_items
    from src.evaluate.retrieval import get_useful_indices

    if args.data == "implexconv":
        cfg = OmegaConf.create({"data": {"name": "implexconv", "type": "opposed", "split": None, "path": None}, "run": {}})
    else:
        cfg = OmegaConf.create({"data": {"name": "actmemeval", "type": None, "split": None,
                                         "path": args.data.split(":", 1)[1]}, "run": {}})
    examples = load_data(cfg)
    rows = load_fact_rows(Path(args.fact_bag))
    evidence = load_evidence(args.evidence)
    ranked = {(r["example_index"], r["qa_index"]): r["retrieved_facts"]
              for r in json.load(open(args.retrieval, encoding="utf-8"))["results"]}

    jobs, skipped = [], {"no_evidence": 0, "no_ranking": 0, "short_list": 0}
    for ei, ex in enumerate(examples):
        for qi, q in enumerate(ex["qa"]["questions"]):
            key = (ei, qi)
            fr = rows.get(ei, qi)
            if not evidence.get(key) or fr is None:
                skipped["no_evidence"] += 1
                continue
            if key not in ranked:
                skipped["no_ranking"] += 1
                continue
            facts = {f.index: f for f in normalize_fact_items(fr)}
            gold_sessions = set(get_useful_indices(ex, qi))
            ev_idx = [i for i in evidence[key] if i in facts]
            if not ev_idx:
                skipped["no_evidence"] += 1
                continue
            if args.mode == "evidence":
                ctx = ev_idx
            elif args.mode == "retrieved":
                ctx = [int(x["fact_index"]) for x in ranked[key][: args.k]]
            else:  # noisy
                ev_idx = ev_idx[: args.k]
                need = args.k - len(ev_idx)
                distract = [int(x["fact_index"]) for x in ranked[key]
                            if int(x["session_index"]) not in gold_sessions][:need]
                if len(distract) < need:
                    skipped["short_list"] += 1
                ctx = ev_idx + distract
                random.Random(f"{args.seed}:{ei}:{qi}").shuffle(ctx)
            jobs.append({"example_index": ei, "qa_index": qi, "question": q,
                         "gold_answer": ex["qa"]["answers"][qi] if qi < len(ex["qa"]["answers"]) else "",
                         "top_k_facts": [facts[i].fact for i in ctx if i in facts],
                         "context_fact_indices": ctx, "evidence_fact_indices": ev_idx,
                         "evidence_positions": [n + 1 for n, i in enumerate(ctx) if i in set(ev_idx)]})

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY", base_url=args.base_url, timeout=900)
    out = Path(args.out)
    partial = out.with_suffix(".partial.jsonl")
    done = {}
    if partial.exists():
        for line in open(partial, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                done[(d["example_index"], d["qa_index"])] = d
    todo = [j for j in jobs if (j["example_index"], j["qa_index"]) not in done]
    print(f"{args.mode} K={args.k or '-'}: {len(jobs)} questions, {len(todo)} to answer, skipped {skipped}")

    def run(j):
        kw = dict(model=args.model, messages=answer_messages(j["question"], j["top_k_facts"], args.no_system),
                  max_tokens=args.max_tokens, temperature=args.temperature)
        if args.top_p is not None:
            kw["top_p"] = args.top_p
        try:
            resp = client.chat.completions.create(**kw)
            ans, reasoning = split_reasoning(resp.choices[0].message)
            truncated = bool(reasoning) and not ans
        except Exception as exc:  # noqa: BLE001
            ans, truncated = f"<answergen error: {exc}>", False
        return dict(j, generated_answer=ans, reasoning_truncated=truncated,
                    answer_top_k=args.k if args.mode != "evidence" else "evidence")

    partial.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(run, j) for j in todo]):
            r = fut.result()
            done[(r["example_index"], r["qa_index"])] = r
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
    samples = [done[(j["example_index"], j["qa_index"])] for j in jobs]
    summary = {"answer_model": args.model, "answer_system_prompt": ANSWER_SYSTEM, "reranker_model": "",
               "retrieval": {"evidence": "EVIDENCE ONLY (E5 labels)",
                             "noisy": "NOISY ORACLE (evidence + AQUI-ranked non-gold distractors, shuffled)",
                             "retrieved": "AQUI top-K (bi-encoder, rank order)"}[args.mode],
               "mode": args.mode, "k": args.k, "seed": args.seed,
               "fact_bag_path": f"{args.tag} {args.fact_bag}", "retrieval_file": args.retrieval,
               "evidence_file": args.evidence, "answer_top_k_facts": args.k or "evidence",
               "num_samples": len(samples), "skipped": skipped,
               "mean_num_context_facts": sum(len(s["top_k_facts"]) for s in samples) / max(1, len(samples)),
               "mean_num_evidence_facts": sum(len(s["evidence_fact_indices"]) for s in samples) / max(1, len(samples)),
               "evidence_in_context_rate": sum(bool(s["evidence_positions"]) for s in samples) / max(1, len(samples)),
               "num_reasoning_truncated": sum(s["reasoning_truncated"] for s in samples),
               "num_errors": sum(str(s["generated_answer"]).startswith("<answergen error") for s in samples)}
    out.write_text(json.dumps({"summary": summary, "samples": samples}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
