# E7 -- Noisy-oracle answer baseline (follow-up to E6)

## What this experiment is about
2KMq asks whether low ImplexConv answer accuracy comes from evidence retrieval, answer generation,
or evaluation. E6 gave the answer model **only** the gold facts and it scored 0.867 / 0.932
(rigorous / lenient). That shows the model can answer from clean evidence, but not that it can
**find** the evidence inside a K-fact list like the one a retriever returns.

E7 keeps E6's answer model, prompt, decoding, judge and judge memory, and changes only the facts
given to the answer model. All rows of a dataset use the **same questions** (those with an E5
evidence label):

| Row | Facts given to the answer model |
|---|---|
| `evidence` | Only the evidence fact(s) of the question (E5 labels). Clean reference. |
| `noisy_k<K>` | The evidence fact(s) + (K − n) **distractors**, shuffled into a random order. Distractors are AQUI's highest-ranked facts that are **not** from any gold session. |
| `retrieved_k<K>` | AQUI's own top-K facts, in rank order. |

How to read it:
* `noisy` ≈ `evidence`: the answer model copes with distractors, so the gap to `retrieved` is
  retrieval.
* `noisy` well below `evidence`: the answer model also struggles to pick the evidence out of K
  facts, so generation is part of the bottleneck.
* `noisy` vs `retrieved`: the only difference is that the evidence is guaranteed to be present.

Default K: ImplexConv 10 and 50; ActMemEval-hard 10 and 30. Answer model: Qwen3-Next-80B only.

## Reviewer points addressed
| ID | Point |
|---|---|
| 2KMq-Q4c | "An oracle-memory answer baseline would clarify whether the bottleneck is evidence retrieval, answer generation, or evaluation" |

## Depends on
* **E5** (run first): `${OUT_BASE}/E5/evidence_{implexconv,actmem_hard}.jsonl` (evidence labels)
  and `${OUT_BASE}/E5/results/aqui__{implexconv,actmem_hard}.json` (AQUI ranking with the paper
  adapter; `E1/test/paper_adapter__*.json` is used if the E5 copy is missing).
* **E6** (recommended): E7 reuses E6's judge memory so the rows are comparable, and step 3 lists
  E6's judged files (clean oracle, re-judged paper AQUI) next to E7's.

## Resources
* **Server: SPACE** (same as E6).
* **GPUs:** 8 x A100-80GB. Step 1: Qwen-80B on GPUs 0-3. Step 2: the 122B judge on all 8.
* **Wall-clock (rough):** 5 answer files per dataset. ImplexConv ≈ 3.6k questions each; ~2-3 h
  for answers and ~2-3 h for judging in total. ActMemEval-hard (72 questions) takes minutes.
* **Ports:** 8001 (answer model), 8000 (judge), the same as E6. Do not run E6 and E7 at the same time.

## Run
```bash
CHECK_ONLY=1 bash E7/run.sh                 # checks E5 outputs and paths; runs nothing
bash E7/run.sh
DS_LIST="implexconv" bash E7/run.sh         # ImplexConv only
KS_implexconv="10 20 30 50" bash E7/run.sh  # more K
```
Outputs: `${OUT_BASE}/E7/answers/*.json`, `judged/*__judged.json`, `E7_summary.md`.
