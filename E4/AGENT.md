# E4 -- notes for the agent running / fixing this experiment

## Intent (do not change)
Give an auditable account of how ActMemEval-hard was built, and check it with a judge that did
not build it. The benchmark itself must **not** change: don't drop, add or re-rewrite items based
on the new judge. We only *report* the new judge's verdicts.

## Key facts (already verified locally)
* `ActMemEval_implicit_hard.json` on the old repo is 0 bytes. `tools/rebuild_actmemeval_hard.py`
  rebuilds it from:
  * `ActMemEval_lexical_controlled.json` (224 rows),
  * the rewrite cache `implicit_hard_rewrites.jsonl` (keys `implicit_hard:<original_row>:<session>`),
  * `ActMemEval_implicit_hard_metadata.json`.

  The result is verified against the paper's fact bag: 72 rows, same order, same session ids.
  A server copy, if found, is compared to the rebuild.
* The metadata did **not** store the original judge's per-criterion verdicts for failed rewrites
  (`judge={}`, `error="implicit_validation_failed"`). The funnel can therefore only say "rejected on
  all 3 attempts" for those, and the independent judge provides the per-criterion picture.
* Questions and answers are never rewritten; `rewrite_judge` reports `questions_unchanged_rate`,
  which should be 1.0.

## Judge
* Default: local vLLM `meta-llama/Llama-3.3-70B-Instruct`, TP=4 on GPUs 0-3, port 8100. It needs HF
  access to the gated Llama repo (`HF_TOKEN` or `hf auth login` under `${HF_HOME}`). If gated
  access fails, use another non-OpenAI family, e.g. `Qwen/Qwen2.5-72B-Instruct` or
  `mistralai/Mistral-Large-Instruct-2411`, and record which one.
* `JUDGE_BACKEND=api` uses any OpenAI-compatible endpoint. The user has access to a **Thinking
  Machines Lab** API. If its interface is not OpenAI-compatible, add a small adapter in
  `shared/tools/llm.py` (`Chat.complete` is the only call site) instead of changing
  `rewrite_judge.py`.
* Pass rule of the independent judge: meaning ≥4, direct_topic ≤2, requires_bridge, answer still
  supported, session still relevant, no invented facts. Keep the first three identical to the
  original thresholds for comparability.

## Paired retrieval
* `tools/make_subset.py` cuts each version down to the 72 hard `question_id`s, in the same order,
  and renumbers the fact-bag `row_index`.
* The lexical-controlled version and its fact bag are bundled.
* **Original version:** needs `ACTMEM_ORIG_JSON` (247 rows; the builder read
  `data/actmem/ActMemEval.json` in `/users/h.lee4/codes/prototype/version_03` on AWS) and
  `ACTMEM_ORIG_FACT_BAG` (extracted by gpt-5.4-mini; probably
  `/groups/AIC-MV/h.lee4/version_03/output_json/user_facts/actmemeval/user_facts.jsonl`).
  If the fact bag is missing, it must be extracted with the same extractor. Ask the user first,
  since it costs API calls. Without it, report lexical vs hard only and say so.
* AQUI adapter: `AQUI_ADAPTER` env, else the paper adapter. Symmetric adapter: `SYM_ADAPTER`,
  else E2's pick. It is skipped if neither exists.
* Baselines (ActMem, TaciTree) are **not** rerun here; their pipelines live in the baselines repo.
  If the user wants them in the paired table, recompute recall on the 72 ids from their existing
  per-question outputs (`/group-volume/users/h.lee4/baseline_outputs/...`).

## What to report back
`E4_summary.md`: the funnel table, the independent-judge summary (pass rates on the kept 72,
κ vs the original judge), and the paired table. (The human-check sheet is off by default; do not run it unless asked.)
