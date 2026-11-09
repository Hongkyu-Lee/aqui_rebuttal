"""E6: ORACLE-MEMORY answer generation.

Same answer model, same prompt and same decoding as the paper's answer-generation step
(implicit_version_03/src/evaluate/answer_gen.py; prompt copied verbatim below), but the retrieved
facts are replaced by the GOLD facts: every fact from the question's gold session(s) in the same
fact bag the retrievers search. This is the upper bound for "perfect retrieval".

Output is in the format the judge reads (--system iris):
  {"summary": {...}, "samples": [{question, gold_answer, generated_answer, top_k_facts, ...}]}
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---- copied verbatim from implicit_version_03/src/evaluate/answer_gen.py ----
ANSWER_SYSTEM = (
    "You are an active reasoning module. You are given something the user is about to do or ask, "
    "together with a set of memories retrieved about that user.\n\n"
    "Reason in three steps:\n"
    "1. Think about the potential negative consequences of the user's intended action or question "
    "-- hidden risks, conflicts, unmet prerequisites, and constraints.\n"
    "2. Do not treat every memory as relevant. Assess each memory on its own and keep only the few "
    "that carry the most significant negative implications; disregard the rest.\n"
    "3. Craft your response grounded in those selected memories: state the key risks and why they "
    "matter for the user.\n\n"
    "Be specific and concrete. Return concise prose."
)


def answer_messages(question: str, facts: list[str], no_system: bool) -> list[dict[str, str]]:
    listing = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(facts))
    user = (f"The user is about to act or ask:\n{question}\n\n"
            f"Retrieved memories about the user:\n{listing}")
    if no_system:
        return [{"role": "user", "content": f"{ANSWER_SYSTEM}\n\n{user}"}]
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]


def split_reasoning(msg) -> tuple[str, str]:
    content = getattr(msg, "content", None) or ""
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning is None:
        reasoning = (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
    if reasoning is None and "</think>" in content:
        pre, _, post = content.rpartition("</think>")
        reasoning, content = pre.split("<think>", 1)[-1], post
    return content.strip(), (reasoning or "").strip()
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="implexconv | actmemeval:<path.json>")
    ap.add_argument("--fact-bag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--no-system", action="store_true")
    ap.add_argument("--max-facts", type=int, default=0, help="0 = all gold facts")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tag", required=True, help="dataset tag written to the summary (implexconv_opposed | actmemeval_implicit_hard)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

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
    jobs = []
    for ei, ex in enumerate(examples):
        for qi, q in enumerate(ex["qa"]["questions"]):
            fr = rows.get(ei, qi)
            gold = set(get_useful_indices(ex, qi))
            facts = [f.fact for f in (normalize_fact_items(fr) if fr else []) if f.session_index in gold]
            if args.max_facts:
                facts = facts[: args.max_facts]
            jobs.append({"example_index": ei, "qa_index": qi, "question": q,
                         "gold_answer": ex["qa"]["answers"][qi] if qi < len(ex["qa"]["answers"]) else "",
                         "top_k_facts": facts})
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY", base_url=args.base_url, timeout=900)

    out = Path(args.out)
    partial = out.with_suffix(".partial.jsonl")
    done = {}
    if partial.exists():
        for l in open(partial, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                done[(d["example_index"], d["qa_index"])] = d
    todo = [j for j in jobs if (j["example_index"], j["qa_index"]) not in done]
    print(f"{len(jobs)} questions ({sum(not j['top_k_facts'] for j in jobs)} without gold facts), {len(todo)} to answer")

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
        return dict(j, generated_answer=ans, reasoning_truncated=truncated, answer_top_k="oracle")

    partial.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(run, j) for j in todo]):
            r = fut.result()
            done[(r["example_index"], r["qa_index"])] = r
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
    samples = [done[(j["example_index"], j["qa_index"])] for j in jobs]
    summary = {"answer_model": args.model, "answer_system_prompt": ANSWER_SYSTEM, "reranker_model": "",
               "retrieval": "ORACLE (gold-session facts)", "fact_bag_path": f"{args.tag} {args.fact_bag}",
               "answer_top_k_facts": "oracle", "num_samples": len(samples),
               "mean_num_oracle_facts": sum(len(s["top_k_facts"]) for s in samples) / max(1, len(samples)),
               "num_reasoning_truncated": sum(s["reasoning_truncated"] for s in samples),
               "num_errors": sum(str(s["generated_answer"]).startswith("<answergen error") for s in samples)}
    out.write_text(json.dumps({"summary": summary, "samples": samples}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
