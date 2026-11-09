# E5 -- Recall definitions: any gold-session fact vs. the specific fact needed

## What this experiment is about
The paper's session recall@K credits a question when **any** retrieved fact comes from a gold
session. 2KMq asks whether we recover "the specific fact required to answer the query". E5
reports three stricter definitions next to the paper's:

| Metric | Credits a hit when |
|---|---|
| session_recall@K (paper) | ≥1 retrieved fact comes from a gold session |
| fact_recall@K | fraction of all gold-session facts retrieved |
| evidence_hit@K | ≥1 of the **evidence facts** (the specific facts the gold answer relies on) is retrieved |
| evidence_recall@K | fraction of the evidence facts retrieved |

Neither benchmark annotates the evidence fact. An LLM (Qwen3-Next-80B-A3B-Instruct, local vLLM)
selects the minimal set of gold-session facts the reference answer relies on. The same labels are
applied to every method, so the comparison is fair.

## Reviewer points addressed
| ID | Point |
|---|---|
| 2KMq-Q4a | Whether session recall measures any fact from the gold session or the specific fact needed |
| 2KMq-Q4b | (text only) K=151 = TaciTree's average output size; AQUI's pool is extended to 151 at that K |

## Expected outcome (goal)
Absolute numbers drop under the stricter definitions, but the **ranking of methods is unchanged**:
AQUI > symmetric > pretrained on both benchmarks at every K, and AQUI's margin on evidence_hit is
similar to its margin on session recall.

## Resources
* **Server: SPACE** (it reuses E1/E2 retrieval results from `${OUT_BASE}`; Qwen3-Next-80B is known to serve in SPACE's `anaconda/vllm_2`).
* **GPUs:** 8 x A100-80GB. GPUs 0-3 serve the labeling LLM (TP=4); GPUs 4-7 compute any retrieval
  results not already available from E1/E2.
* **Inputs:** the ImplexConv fact bag, ActMemEval-hard (bundled), adapters (the paper adapter or
  `AQUI_ADAPTER`; the E2 symmetric pick or `SYM_ADAPTER`).

## Run
```bash
bash E5/run.sh
LABEL_BACKEND=api LABEL_BASE_URL=<url>/v1 LABEL_MODEL=<id> bash E5/run.sh
```
Outputs: `${OUT_BASE}/E5/evidence_*.jsonl`, `results/*.json`, `table_*.csv`, `E5_summary.md`.
