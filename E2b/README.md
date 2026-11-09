# E2b -- Why does freezing the fact space help?

## What this experiment is about
Ap7K argues that query-only LoRA might win simply because it changes fewer parameters
(regularization on a small dataset), not because it "preserves world knowledge" in the memory
space. E2 shows *whether* asymmetry wins at matched capacity; E2b looks at *why*, using E2's
adapters:

1. **Fact-space drift.** On 20k real memory facts from each benchmark, measure how far the
   symmetric adapter moves each fact embedding from the frozen base:
   * cosine to base;
   * kNN-overlap@10, i.e. does each fact keep its neighbours?
   * similarity-rank correlation.

   AQUI's drift is zero by construction.
2. **Explicit retrieval.** On benchmarks where the query and the memory are directly related,
   check whether symmetric training hurts ordinary retrieval while AQUI does not:
   * ImplexConv-**Supportive** (the explicit sibling of ImplexConv-Opposed);
   * the **original** ActMemEval (before the implicit rewrite).

## Reviewer points addressed
| ID | Point |
|---|---|
| Ap7K-W1 (second half) | "The gains could partly reflect regularization on a small training set, rather than the claimed preservation of world knowledge" |

## Expected outcome (goal)
Symmetric LoRA measurably moves the memory space: cosine to base clearly below 1 and kNN-overlap
well below 1. It also loses explicit-retrieval recall relative to the pretrained encoder, while
AQUI stays at or above pretrained. That supports "freezing protects the memory index".
**Pre-agreed fallback (decision D8):** if drift is small and explicit retrieval is unchanged, we
soften the paper's wording to "keeps the fact index fixed (no re-indexing) and regularizes
adaptation" instead of "preserves world knowledge".

## Resources
* **Server: SPACE** (same server as E2, whose adapters it reads). The original ActMemEval JSON, its fact bag and the Supportive fact bag exist only on **AWS**. Stage them once with `shared/tools/stage_from_aws.sh` (see `../SERVERS.md` §3); otherwise those parts are skipped.
* **GPUs:** 4 x A100-80GB (step 1 uses 1 GPU for ~30-60 min; step 2 runs about 6 evals in a pool).
* **Inputs:** `${OUT_BASE}/E2/e2_selection.json` (or `SYM_ADAPTER`/`AQUI_ADAPTER`), the ImplexConv
  fact bag, and for step 2 the ImplexConv-Supportive fact bag and the original ActMemEval JSON +
  fact bag. Those last files are marked (VERIFY) in `shared/env.sh`; each dataset is skipped with a
  warning if missing.

## Run
```bash
bash E2b/run.sh
SYM_ADAPTER=/path/sym AQUI_ADAPTER=/path/aqui bash E2b/run.sh    # without E2
```
Outputs: `${OUT_BASE}/E2b/drift.json`, `explicit/*.json`, `E2b_summary.md`.
