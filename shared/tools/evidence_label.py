"""E5 step 1: label, per question, WHICH gold-session facts are the evidence needed to answer it.

Session recall credits a hit when ANY fact from a gold session is retrieved. 2KMq asks whether we
instead recover "the specific fact required to answer the query". Neither benchmark annotates that
fact, so an LLM picks it from the gold-session facts (given question + gold answer). Fact indices
use EXACTLY the indexing of src.evaluate.fact_retrieval (normalize_fact_items), so the labels can be
joined with any fact_retrieval result JSON.

Output: <out>.jsonl  {"example_index", "qa_index", "evidence_fact_indices": [...], "n_gold_facts"}
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.llm import Chat

SYSTEM = ("You annotate evidence for a memory-retrieval benchmark. Given a user's question, the "
          "reference answer, and numbered facts extracted from the user's relevant past conversation, "
          "select the MINIMAL set of facts that the reference answer actually relies on. Return only JSON.")
USER = """Question: {q}

Reference answer: {a}

Facts from the relevant past session(s):
{facts}

Return JSON: {{"evidence": [list of fact numbers the reference answer relies on; at least 1 if any fact is relevant, [] if none]}}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="implexconv | actmemeval:<path.json>")
    ap.add_argument("--fact-bag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-facts", type=int, default=80)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from omegaconf import OmegaConf
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
    items = []
    for ei, ex in enumerate(examples):
        for qi, q in enumerate(ex["qa"]["questions"]):
            fr = rows.get(ei, qi)
            if fr is None:
                continue
            gold = set(get_useful_indices(ex, qi))
            facts = [f for f in normalize_fact_items(fr) if f.session_index in gold]
            items.append({"example_index": ei, "qa_index": qi, "q": q,
                          "a": ex["qa"]["answers"][qi] if qi < len(ex["qa"]["answers"]) else "",
                          "facts": facts[: args.max_facts]})
    out = Path(f"{args.out}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                done.add((d["example_index"], d["qa_index"]))
    todo = [it for it in items if (it["example_index"], it["qa_index"]) not in done]
    print(f"{len(items)} questions, {len(todo)} to label")
    chat = Chat(args.model, args.base_url, args.api_key_env, max_tokens=512)

    def run(it):
        if not it["facts"]:
            return {"example_index": it["example_index"], "qa_index": it["qa_index"],
                    "evidence_fact_indices": [], "n_gold_facts": 0}
        listing = "\n".join(f"[{n + 1}] {f.fact}" for n, f in enumerate(it["facts"]))
        try:
            v = chat.json([{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": USER.format(q=it["q"], a=it["a"], facts=listing)}])
            nums = [int(x) for x in v.get("evidence", []) if 1 <= int(x) <= len(it["facts"])]
        except Exception as exc:  # noqa: BLE001
            return {"example_index": it["example_index"], "qa_index": it["qa_index"], "error": str(exc)[:200]}
        return {"example_index": it["example_index"], "qa_index": it["qa_index"],
                "evidence_fact_indices": sorted({it["facts"][n - 1].index for n in nums}),
                "n_gold_facts": len(it["facts"])}

    with open(out, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        for fut in as_completed([pool.submit(run, it) for it in todo]):
            r = fut.result()
            if "error" not in r:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
    print(f"labels -> {out}")


if __name__ == "__main__":
    main()
