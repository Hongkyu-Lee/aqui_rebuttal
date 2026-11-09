# E1 -- notes for the agent running / fixing this experiment

## Intent (do not change)
Show reviewers that the reported AQUI model can be chosen **without any benchmark data**.
Selection may only use the synthetic data (`v11con_2400.jsonl`) and its fixed 10% validation split
(grouped by `memory_id`, seed 42). Benchmark (ImplexConv-Opposed, ActMemEval-hard) numbers are
computed only for the configurations that synthetic selection picks, plus the paper's pick for
reference. Never rank, filter or tune anything by a benchmark number. That would reproduce the
exact problem the reviewers raised.

The primary criterion is pre-registered: `val_mean_hit` (mean of synthetic-val hit@10/20/50/100).
Do not switch criteria after seeing test results. If you think another criterion is needed, report
it as an additional row and keep the primary one.

## How the pieces fit
* `run.sh` sources `shared/env.sh`, which detects the server (SPACE vs AWS) and sets all paths.
  E1 must run on **SPACE**, where the 216 adapters are
  (`/group-volume/users/h.lee4/prototypes/version_03_artifacts/oppose_gen_hpo/adapters/`).
* Step 3 is a **gate**. It evaluates the paper adapter with the rebuilt eval loader and must
  reproduce Table 1 (ImplexConv R@10/20/100 = 0.3322/0.4588/0.8412, tolerance 0.01).
  `GEPA_RERANK_RESULTS.md` reported 0.3328/0.4602/0.8439 for the same adapter, so differences of
  around 0.003 are expected; larger ones mean a bug.
* `tools/synthetic_val.py eval` loads the 8B embedder once per adapter. It caches the base-model
  embeddings of the validation fact pool in `${OUT}/cache`, written by the `base:-` run, which must
  finish before the pool starts.
* `tools/hpo_reselect.py` reads `best_eval_loss` from each adapter's `model_selection.json`.
  The original HPO runner had a bug: it read the key `best_metric`, so the `best_val_loss` column in
  `HPO_ranking.csv` is probably empty. Don't rely on that column.

## If something goes wrong
* **Table 1 not reproduced (step 3 dies):** the most likely cause is the rewritten
  `shared/src/data/load_data.py`, which was reconstructed because the original was never committed.
  Check, in order:
  1. Row order of the HF dataset `Kaylee0501/ImplexConv-opposed` vs `row_index` in the fact bag.
     `num_questions` in the result JSON should be 3633 and `num_examples` 1550.
  2. Gold-session ids. `retrieved_conv_ids` are used as 0-based session positions, matching the
     baselines repo's `build_qa_gold_session_facts.py`. If recall is close to 0 or far too high,
     test 1-based ids.
  3. `num_questions_without_gold_facts` in the summary should be about 0.
  4. dtype: the original eval used `dtype: auto` (fp32 load), which is kept here.
  Fix the loader. Do not change the metric code, which was copied unchanged.
* **Loader check mismatch (step 2):** compare `num_training_samples`. The original expands one
  sample per gold fact with `cycle_negatives`, giving 19,100 samples in total and 17,085 in train
  per our reconstruction. If the original manifest differs, inspect its `sample_row_preview` and
  `question_groups` and adapt `shared/src/data/load_train_data.py`. E2 and E3 depend on this.
* **Fewer than 216 adapters:** run steps 4-7 on what exists and state the count in the summary.
  If the adapters are gone entirely, E1 needs the full grid retrained with `train_lora` (216 runs,
  ~1-2 days on 8 GPUs). Ask the user before doing that.
* **OOM in the synthetic eval:** lower `--batch-size` in `val_chunk` (default 64).
* **Fact-cache metadata mismatch error:** the cache key includes the data config. Delete
  `${OUT_BASE}/fact_embedding_cache` or point `FACT_CACHE_ROOT` elsewhere.

## What to report back
`E1_summary.md`, plus one sentence each on:
* (a) whether Table 1 was reproduced,
* (b) the synthetic-selected config and its test R@10/20/100 vs the paper pick,
* (c) the split-audit numbers.
