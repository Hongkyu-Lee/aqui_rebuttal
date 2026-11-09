# E3 -- Seeds and paired significance

## What this experiment is about
A single training run per arm can't show that the AQUI-vs-symmetric gap is real. E3 retrains each
arm's **E2-selected** configuration with extra training seeds, so there are 3 seeds in total
(decision D3: 42 from E2, plus 43 and 44). The data split stays fixed (validation seed 42); only
LoRA initialization, data order and dropout change. Every seed is evaluated on both benchmarks.
We then report:
* mean ± std over seeds;
* a **paired bootstrap** over questions (10k resamples) of the seed-averaged per-question
  difference AQUI − symmetric, with a 95% CI and a two-sided p-value.

## Reviewer points addressed
| ID | Point |
|---|---|
| HeEX-Q1 | "Does the claimed advantage of asymmetry remain?" -- needs variance + a test, not one run |
| 2KMq-W1 | Attribution of the gain to freezing the fact space |

## Expected outcome (goal)
Small seed variance relative to the gap (std well below the AQUI − symmetric difference) and a
95% CI that excludes 0 at K = 10, 20 and 50 on ImplexConv-Opposed. ActMemEval-hard has only 72
questions, so its CI will be wide; report it as is.

## Resources
* **Server: SPACE** (same server as E2; it reuses E2's seed-42 adapters and results).
* **GPUs:** 8 x A100-80GB. Four trainings in parallel (2 arms × 2 extra seeds), then evaluation.
  The symmetric evaluations dominate the wall-clock (~1-3 h each on ImplexConv).
* **Inputs:** `${OUT_BASE}/E2/e2_selection.json` and the E2 seed-42 test results (reused).

## Run
```bash
bash E3/run.sh
EXTRA_SEEDS="43 44 45 46" bash E3/run.sh    # 5 seeds
```
Outputs: `${OUT_BASE}/E3/test/*.json`, `stats_implexconv.json`, `stats_actmem_hard.json`, `E3_summary.md`.
