"""E4(a): the 247 -> 224 -> 72 exclusion funnel of ActMemEval-hard, from the construction metadata.

No GPU, no LLM. Reads the two metadata files written by create_actmemeval_variants.py:
  stage 1 (lexical_controlled): every GOLD session of every one of the 247 rows is rewritten so it
          shares no query tokens (query coverage 0, BM25 0); a row is dropped if any rewrite fails.
  stage 2 (implicit_hard)     : every gold session of the 224 survivors is rewritten again so that
          relevance needs an external bridge; an LLM judge must give direct_topic<=2,
          meaning_preserved>=4, requires_external_bridge, useful_for_answer, no fatal issues;
          a row is kept only if ALL its gold sessions pass (up to 3 attempts each).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexical-meta", required=True)
    ap.add_argument("--hard-meta", required=True)
    ap.add_argument("--out", required=True, help="output prefix (.json and .md)")
    args = ap.parse_args()
    lex = json.loads(Path(args.lexical_meta).read_text(encoding="utf-8"))
    hard = json.loads(Path(args.hard_meta).read_text(encoding="utf-8"))

    # ---- stage 1
    lex_rows = defaultdict(list)
    for r in lex["records"]:
        lex_rows[r["row_index"]].append(r)
    lex_fail_reasons = Counter()
    for recs in lex_rows.values():
        for r in recs:
            if not (r["rewrite_passed"] and (r.get("after_validation") or {}).get("passed", r["rewrite_passed"])):
                err = str(r.get("error") or "")
                lex_fail_reasons["gold session still shares query tokens (coverage/BM25 > 0) after 3 attempts"
                                 if err in ("", "lexical_validation_failed") else f"rewriter error: {err[:60]}"] += 1
    # ---- stage 2
    rows = defaultdict(list)
    for r in hard["records"]:
        rows[r["row_index"]].append(r)
    sess_reason = Counter()
    row_reason = Counter()
    meaning_scores = Counter()
    kept = 0
    for recs in rows.values():
        failed = []
        for r in recs:
            iv = r.get("implicit_validation") or {}
            ok = r["rewrite_passed"] and (r.get("lexical_validation") or {}).get("passed") and iv.get("passed")
            if ok:
                meaning_scores[iv.get("meaning_preserved_score")] += 1
            else:
                # The construction script did not store the per-criterion verdicts of FAILED
                # attempts (judge={} , error="implicit_validation_failed"), so we can only say
                # that the judge rejected all 3 attempts. Re-judging is done in E4(b).
                err = str(r.get("error") or "")
                if err == "implicit_validation_failed":
                    reason = "rejected by the implicit-validity judge on all 3 attempts"
                elif err:
                    reason = "rewriter output was not valid JSON"
                elif not (r.get("lexical_validation") or {}).get("passed"):
                    reason = "rewrite re-introduced query tokens"
                else:
                    reason = "rewrite not usable"
                sess_reason[reason] += 1
                failed.append(r)
        if failed or any(r.get("row_validation") and not r["row_validation"].get("passed") for r in recs):
            row_reason[f"{len(failed)}/{len(recs)} gold sessions failed"] += 1
        else:
            kept += 1
    sess_per_row = Counter(len(v) for v in rows.values())
    out = {
        "stage0_original_rows": lex["summary"]["selected_rows"],
        "stage1_lexical": {"kept_rows": lex["summary"]["output_rows"], "dropped_rows": lex["summary"]["failed_rows"],
                           "session_rewrites": lex["summary"]["session_rewrites"],
                           "passed_session_rewrites": lex["summary"]["passed_session_rewrites"],
                           "session_failure_reasons": dict(lex_fail_reasons)},
        "stage2_implicit": {"input_rows": hard["summary"]["selected_rows"], "kept_rows": kept,
                            "dropped_rows": hard["summary"]["selected_rows"] - kept,
                            "session_rewrites": hard["summary"]["session_rewrites"],
                            "passed_session_rewrites": hard["summary"]["passed_session_rewrites"],
                            "gold_sessions_per_row": dict(sorted(sess_per_row.items())),
                            "session_failure_reasons (a session can have several)": dict(sess_reason.most_common()),
                            "dropped_rows_by_number_of_failed_sessions": dict(row_reason.most_common()),
                            "meaning_preserved_score_of_passed_rewrites": dict(sorted(meaning_scores.items(), key=lambda x: str(x[0])))},
        "thresholds": {k: hard["config"]["implicit_hard"][k] for k in
                       ("direct_topic_score_max", "meaning_preserved_score_min", "require_external_bridge", "max_rewrite_attempts")},
        "rewriter_and_judge_model": hard["config"]["llm"]["model"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    s1, s2 = out["stage1_lexical"], out["stage2_implicit"]
    md = [
        "| Stage | Rows in | Rows out | Session rewrites (passed/total) | Main reason rows are dropped |",
        "|---|---|---|---|---|",
        f"| 0. Original ActMemEval | - | {out['stage0_original_rows']} | - | - |",
        f"| 1. Lexical control (no query tokens in gold sessions) | {out['stage0_original_rows']} | {s1['kept_rows']} | {s1['passed_session_rewrites']}/{s1['session_rewrites']} | rewrite could not remove query overlap |",
        f"| 2. Implicit rewrite + LLM judge (all gold sessions must pass) | {s2['input_rows']} | {s2['kept_rows']} | {s2['passed_session_rewrites']}/{s2['session_rewrites']} | {next(iter(s2['session_failure_reasons (a session can have several)']), '-')} |",
    ]
    Path(f"{args.out}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
