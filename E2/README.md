# E2 -- Matched symmetric-LoRA control

## What this experiment is about
All three reviewers read Table 1 as "full-encoder InfoNCE vs. query-only LoRA", so the gain could
come from parameterization rather than asymmetry. The caption is wrong: the naive baseline was
already LoRA on both sides. But its hyperparameters and tuning budget were not matched to AQUI's.
E2 runs a clean, matched comparison:

| | AQUI (asymmetric) | Symmetric LoRA |
|---|---|---|
| base model | Qwen3-Embedding-8B | same |
| LoRA rank / alpha / modules | r, 2r, all-linear | **same** |
| data / negatives | v11con_2400 / general facts | **same** |
| loss | multi-positive InfoNCE + 0.25 alignment | **same** |
| val split / early stopping / seed | memory-grouped 10%, patience 2, 42 | **same** |
| what the LoRA touches | query only (facts = frozen base) | query **and** facts |
| fact embeddings at test time | base model (index reused) | **re-encoded through the adapter** |
| tuning budget | the same 2-3 configs | the same 2-3 configs |
| config selection | synthetic validation retrieval | synthetic validation retrieval |

Default configs (`lr|temp|clip|warmup|batch|r`): the paper's AQUI config, the best-@10 config of the
original grid, and the original symmetric script's default. E1's synthetic pick replaces the third
if it differs.

## Reviewer points addressed
| ID | Point |
|---|---|
| Ap7K-W1, Ap7K-Q1a/b | Baseline with the same LoRA rank and target modules on both branches; how its HPs were selected |
| 2KMq-W1, 2KMq-Q1a/b | Symmetric LoRA with the same rank, data, negatives and tuning budget; were fact embeddings recomputed |
| HeEX-W1, HeEX-Q1 | Under matched conditions and comparable tuning, does query-only still beat symmetric? |

## Expected outcome (goal)
AQUI's synthetic-selected adapter beats the synthetic-selected symmetric adapter on both
benchmarks at every K. The paper's gap (R@10 0.33 vs 0.15 on ImplexConv) should largely survive
the matched tuning. If the gap shrinks a lot, the rebuttal reports it honestly and relies on the
deployment argument: the fact index is never re-encoded. Deliverable: matched rows for Table 1,
plus the sentence "facts were re-encoded through the adapter for the symmetric model".

## Resources
* **Server: SPACE.** It reads E1's `selection.json` from `${OUT_BASE}/E1` and the ImplexConv fact bag, which is confirmed to exist only on SPACE. E2b, E3, E4 and E5 then read E2's outputs from the same `${OUT_BASE}`, so they must run on the same server.
* **GPUs:** 8 x A100-**80GB**. Six trainings run in parallel, one per GPU, then evaluations run.
  The symmetric arm encodes positives and negatives with gradients, which takes about 2-3x the
  activation memory of AQUI. Batch 128 is the main OOM risk (see AGENT.md).
* **Wall-clock (rough):** training 1-4 h per run (early stopping), all in parallel. Symmetric test
  evaluation re-encodes about 0.86M ImplexConv facts per adapter: ~1-3 h each.
* **Inputs:** v11con_2400 (bundled), ImplexConv fact bag, ActMemEval-hard (bundled rebuild).

## Run
```bash
bash E2/run.sh
CONFIGS="5e-5|0.02|1.0|0.1|128|16 2e-5|0.1|1.0|0.1|128|32" bash E2/run.sh   # 2 configs
EVAL_ALL=1 bash E2/run.sh            # also evaluate non-selected adapters on the benchmarks
```
Outputs: `${OUT_BASE}/E2/` -> `adapters/`, `val/*.csv`, `e2_selection.json` (used by E2b/E3/E4/E5),
`test/*.json`, `E2_summary.md`.
