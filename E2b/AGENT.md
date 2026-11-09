# E2b -- notes for the agent running / fixing this experiment

## Intent (do not change)
Measure the memory-side effect of symmetric training, using the **E2-selected** adapters. Do not
train anything new, and do not pick adapters by these results. Report whatever comes out; the
user has agreed in advance to soften the paper's claim if the effect is small.

## Details
* `tools/drift.py` loads the symmetric adapter once and encodes the same fact sample twice:
  adapter disabled (= frozen base) and adapter enabled on the memory side. Facts are encoded
  without an instruction, exactly like at retrieval time. The sample comprises 20k unique fact
  texts per fact bag, shuffled with seed 0.
* Explicit retrieval uses the standard `eval_retrieval` helper: `base`, `aqui`, `sym` (sym
  re-encodes facts, cache off).

## Missing inputs (likely) and how to get them
* **ImplexConv-Supportive fact bag** (`IMPLEX_SUPPORTIVE_FACT_BAG`). It may not exist on SPACE.
  Look for `.../version_03_artifacts/supportive_reasoning_hf/user_facts.jsonl` or search for
  `supportive` under `/group-volume/users/h.lee4`. If it doesn't exist, it can be produced with
  the baselines repo's fact extractor (`tacitree/run_local/run_extract_user_facts.sh`, dataset
  `supportive_reasoning_hf`, gpt-5.4-mini, the same extractor as the opposed bag). That needs the
  OpenAI API and the user's approval, so ask first.
* **Original ActMemEval + its fact bag** (`ACTMEM_ORIG_JSON`, `ACTMEM_ORIG_FACT_BAG`). The variant
  builder read `data/actmem/ActMemEval.json` on AWS (`/users/h.lee4/codes/prototype/version_03/`)
  and the fact bag was written under `/groups/AIC-MV/h.lee4/version_03/output_json/user_facts/actmemeval/`.
  Copy both to the server you run on and export the two variables. The JSON has 247 rows; the fact
  bag must have 247 rows with `row_index` 0..246.
* If neither dataset is available, run step 1 only (`STEPS="1 3"`) and report drift alone.

## What to report back
`E2b_summary.md`: the drift numbers per benchmark, plus the explicit-retrieval table
(pretrained / AQUI / symmetric).
