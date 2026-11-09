# E1 -- Clean model selection

## What this experiment is about
The paper's final AQUI configuration was ranked by **ImplexConv-Opposed test recall**
(`HPO_RESULTS.md`: "Selection metric: implexconv-opposed session_recall"), which is why the top row
of Table 7 reproduces Table 1 exactly. Early stopping inside each run already used only the
synthetic validation split. E1 removes the benchmark from the *configuration choice* as well:

1. **Split audit.** Rebuild the exact train/validation split (10% of synthetic memories, grouped by
   `memory_id`, seed 42) and measure what crosses it (shared memories, duplicate queries, gold-fact
   text that appears on both sides). Also writes an overlap-free validation subset.
2. **Loader check + reproduction.** Confirm that the rebuilt data loader reproduces the original
   training split, and that the paper adapter reproduces Table 1 (sanity gate).
3. **Synthetic-validation retrieval** for all 216 HPO adapters. Pool = all facts of the validation
   memories, query = validation query, gold = that memory's facts. The primary criterion is
   pre-registered as `val_mean_hit` = mean hit@{10,20,50,100}. Validation eval_loss is reported too,
   but it is not comparable across temperatures.
4. **Re-select** the configuration with synthetic data only, and report its **test** scores on
   ImplexConv-Opposed and ActMemEval-hard next to the paper's pick.

## Reviewer points addressed
| ID | Point |
|---|---|
| Ap7K-W2, Ap7K-Q2 | Which data determine Tables 7-8 and the final configuration; how they relate to Table 1 |
| Ap7K-Q1b | How hyperparameters were selected (same protocol is then used for the baseline in E2) |
| 2KMq-W2a, 2KMq-Q2a | Which partition produced Tables 7-8; why the scores match Table 1 |
| 2KMq-Q2b | The train/validation split; whether queries sharing a synthetic memory were grouped |
| HeEX-W2, HeEX-Q2 | Examples/split used for early stopping and ranking; whether benchmark recall influenced selection |

## Expected outcome (goal)
* **Split:** no memory on both sides (grouping by `memory_id` holds). Local preview with the rebuilt
  loader: 0 shared memories; 73/2,015 validation gold facts (3.6%) and 8 queries also occur verbatim
  in training. We will report this, plus results on the 205/255-row overlap-free subset.
* **Selection:** the config picked by synthetic validation is the paper's config, or one whose test
  recall is close to it (goal: within ~0.02 R@10 of 0.3322), and still far above the pretrained
  encoder (0.0862) and the symmetric baseline. If it is lower, the rebuttal reports the
  synthetic-selected numbers as the main result and keeps the old ones as "test-selected (oracle)".
* **Deliverable:** revised Tables 7-8 ranked by synthetic validation; a corrected Section 3.1 statement.

## Resources
* **Server: SPACE (required).** The 216 HPO adapters, `HPO_ranking.csv` and the paper adapter exist only under `/group-volume/.../version_03_artifacts/oppose_gen_hpo`.
* **GPUs:** 8 x A100-80GB (works with 1; step 4 is ~216 short jobs). CPU-only steps: 1, 2, 5, 7.
* **Wall-clock (rough):** step 3 ~1-2 h (one pass over the ~0.86M ImplexConv facts fills the fact
  cache); step 4 ~2-3 h on 8 GPUs; step 6 ~30 min per config.
* **Inputs (SPACE):** `${HPO_ROOT}/adapters/hpo_*` (216 adapters), `${HPO_ROOT}/HPO_ranking.csv`,
  ImplexConv fact bag, v11con_2400 (bundled copy as fallback).

## Run
```bash
bash E1/run.sh                 # all steps
STEPS="1" bash E1/run.sh       # split audit only (CPU)
STRICT_REPRO=0 bash E1/run.sh  # do not stop if Table 1 is not reproduced (not recommended)
```
Outputs: `${OUT_BASE}/E1/` -> `split_audit.json`, `loader_check.json`, `synthetic_val_all.csv`,
`selection_table.csv`, `selection.json`, `test/*.json`, `E1_summary.md`.
