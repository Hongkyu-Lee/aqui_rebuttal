"""E4(b): INDEPENDENT re-validation of the ActMemEval-hard session rewrites.

The benchmark was built (and validated) with gpt-5.4-mini. Here a model from a DIFFERENT family
(default: a local vLLM Llama-3.3-70B-Instruct; or any OpenAI-compatible endpoint such as the
Thinking Machines Lab API) re-judges every rewritten gold session against its reference session.

Note on "query meaning": the construction rewrote the GOLD SESSIONS only; questions and gold
answers are copied verbatim from ActMemEval (checked below), so query meaning is preserved by
construction. What needs checking is that each rewritten session still carries the same user
memory, still supports the gold answer, and is genuinely implicit.

Reference session = the ORIGINAL ActMemEval session if --original is given (preferred), else the
lexical-controlled session (stage-1 output, which already passed a token-overlap filter).

Outputs <out>.jsonl (one verdict per rewrite, resumable) and <out>_summary.json:
  * kept-72 benchmark: % of its gold sessions / rows that pass the independent judge
  * agreement with the original judge's pass/fail over all rewrites (accuracy, Cohen's kappa)
"""
from __future__ import annotations

import argparse
import ast
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.llm import Chat

SYSTEM = (
    "You are an expert annotator auditing a benchmark for implicit memory retrieval. "
    "A conversation session from a user's chat history was REWRITTEN. You must decide whether the "
    "rewrite is a faithful, still-useful, and genuinely implicit version of the reference session. "
    "Be strict and return only JSON."
)

USER_TMPL = """Question the user later asks:
{question}

Gold answer (the correct response, which relies on what the user said in this session):
{answer}

REFERENCE session (before rewriting):
{reference}

REWRITTEN session:
{candidate}

Return JSON with exactly these keys:
- "meaning_preserved_score": integer 0-5. 5 = every user-specific fact in the reference is still stated (possibly in other words); 0 = the user facts are lost or changed.
- "invented_facts": boolean. true if the rewrite adds user facts that are not in the reference.
- "answer_still_supported": boolean. Given ONLY the rewritten session as the user's memory, is the gold answer still correct and justified?
- "gold_session_relevant": boolean. Is the rewritten session still the piece of memory one needs to answer the question well?
- "direct_topic_score": integer 0-5. 0 = the rewritten session is not about the question's surface topic at all; 5 = it is explicitly about the same topic/object/task.
- "requires_external_bridge": boolean. Does connecting the question to the rewritten session require an inference or world-knowledge bridge (rather than surface topic match)?
- "rationale": one short sentence."""


def lit(v):
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return ast.literal_eval(v)
    return v


def fmt(session) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in session)


def passes(v: dict, topic_max=2, meaning_min=4) -> bool:
    try:
        return (int(v["meaning_preserved_score"]) >= meaning_min and int(v["direct_topic_score"]) <= topic_max
                and bool(v["requires_external_bridge"]) and bool(v["answer_still_supported"])
                and bool(v["gold_session_relevant"]) and not bool(v["invented_facts"]))
    except (KeyError, TypeError, ValueError):
        return False


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--rewrites", required=True)
    ap.add_argument("--lexical", required=True)
    ap.add_argument("--hard", required=True, help="the 72-row ActMemEval-hard JSON (defines the kept rows)")
    ap.add_argument("--original", default=None, help="original ActMemEval.json (preferred reference)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--scope", choices=["kept", "all"], default="all")
    ap.add_argument("--out", required=True, help="output prefix")
    args = ap.parse_args()

    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    cache = {json.loads(l)["cache_key"]: json.loads(l) for l in open(args.rewrites, encoding="utf-8") if l.strip()}
    lexical = {r["question_id"]: r for r in json.loads(Path(args.lexical).read_text(encoding="utf-8"))}
    kept_q = {r["question_id"] for r in json.loads(Path(args.hard).read_text(encoding="utf-8"))}
    original = None
    if args.original and Path(args.original).exists():
        od = json.loads(Path(args.original).read_text(encoding="utf-8"))
        od = od if isinstance(od, list) else next(v for v in od.values() if isinstance(v, list))
        original = {r["question_id"]: r for r in od}
    ref_kind = "original" if original else "lexical_controlled"

    items = []
    for r in meta["records"]:
        if args.scope == "kept" and r["question_id"] not in kept_q:
            continue
        ck = f"implicit_hard:{r['row_index']}:{r['session_index']}"
        if ck not in cache or not cache[ck].get("session"):
            continue
        src = (original or lexical)[r["question_id"]]
        ref_sess = lit(src["haystack_sessions"])[int(r["session_index"])]
        # questions/answers must be identical across versions (query meaning preserved by construction)
        same_q = (lexical[r["question_id"]]["question"] == r["question"]) and (
            original is None or original[r["question_id"]]["question"] == r["question"])
        items.append({"key": ck, "question_id": r["question_id"], "question": r["question"], "answer": r["answer"],
                      "reference": fmt(ref_sess), "candidate": fmt(cache[ck]["session"]),
                      "kept_row": r["question_id"] in kept_q,
                      "original_pass": bool(r["rewrite_passed"] and (r.get("implicit_validation") or {}).get("passed")),
                      "question_unchanged": same_q})

    out_jsonl = Path(f"{args.out}.jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_jsonl.exists():
        for l in open(out_jsonl, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                done[d["key"]] = d
    todo = [it for it in items if it["key"] not in done]
    print(f"{len(items)} rewrites ({ref_kind} reference), {len(done)} already judged, {len(todo)} to go")
    chat = Chat(args.model, args.base_url, args.api_key_env, max_tokens=args.max_tokens)

    def run(it):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(**it)}]
        try:
            v = chat.json(msgs)
        except Exception as exc:  # noqa: BLE001
            v = {"error": str(exc)[:200]}
        rec = {k: it[k] for k in ("key", "question_id", "kept_row", "original_pass", "question_unchanged")}
        rec["verdict"] = v
        rec["independent_pass"] = passes(v)
        return rec

    with open(out_jsonl, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(run, it) for it in todo]):
            rec = fut.result()
            done[rec["key"]] = rec
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    recs = [done[it["key"]] for it in items if it["key"] in done]
    valid = [r for r in recs if "error" not in r["verdict"]]
    kept = [r for r in valid if r["kept_row"]]
    kept_rows = {}
    for r in kept:
        kept_rows.setdefault(r["question_id"], []).append(r["independent_pass"])
    v = lambda key: [r["verdict"].get(key) for r in kept]
    summary = {
        "judge_model": args.model, "reference": ref_kind,
        "n_rewrites_judged": len(valid), "n_errors": len(recs) - len(valid),
        "questions_unchanged_rate": sum(r["question_unchanged"] for r in recs) / max(1, len(recs)),
        "kept72": {
            "n_gold_sessions": len(kept),
            "session_pass_rate": sum(r["independent_pass"] for r in kept) / max(1, len(kept)),
            "row_pass_rate (all gold sessions pass)": sum(all(x) for x in kept_rows.values()) / max(1, len(kept_rows)),
            "mean_meaning_preserved_score": sum(int(x or 0) for x in v("meaning_preserved_score")) / max(1, len(kept)),
            "answer_still_supported_rate": sum(bool(x) for x in v("answer_still_supported")) / max(1, len(kept)),
            "gold_session_relevant_rate": sum(bool(x) for x in v("gold_session_relevant")) / max(1, len(kept)),
            "invented_facts_rate": sum(bool(x) for x in v("invented_facts")) / max(1, len(kept)),
            "mean_direct_topic_score": sum(int(x or 0) for x in v("direct_topic_score")) / max(1, len(kept)),
            "requires_bridge_rate": sum(bool(x) for x in v("requires_external_bridge")) / max(1, len(kept)),
        },
    }
    if args.scope == "all" and valid:
        a = [r["original_pass"] for r in valid]
        b = [r["independent_pass"] for r in valid]
        summary["agreement_with_original_judge"] = {
            "n": len(valid), "accuracy": sum(x == y for x, y in zip(a, b)) / len(valid), "cohen_kappa": kappa(a, b),
            "original_pass_rate": sum(a) / len(a), "independent_pass_rate": sum(b) / len(b)}
    Path(f"{args.out}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
