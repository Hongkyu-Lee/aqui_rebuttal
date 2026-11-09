# E6 -- Oracle-memory answer baseline

## What this experiment is about
On ImplexConv the judged answer accuracy is low even though AQUI's session recall is much higher.
2KMq asks whether the bottleneck is evidence retrieval, answer generation, or evaluation. E6 swaps
the retriever for a perfect one: the answer model receives the **gold facts** (all facts extracted
from the question's gold session(s), from the same fact bag the retrievers search). It uses the
paper's answer models, prompt, decoding and judge:

* **Answer models:** Qwen3-Next-80B-A3B-Instruct (temperature 0, 512 tokens) and
  DeepSeek-R1-Distill-Llama-70B (temperature 0.6, top-p 0.95, 3072 tokens, no system prompt,
  reasoning stripped).
* **Prompt:** copied verbatim from `answer_gen.py` (the "active reasoning module" instruction).
* **Judge:** Qwen3.5-122B-A10B, served as `gpt-4o-mini`, with the paper's rubric. It produces the
  rigorous score (conclusion aligned AND grounded) and the lenient score.

## Reviewer points addressed
| ID | Point |
|---|---|
| 2KMq-Q4c | "An oracle-memory answer baseline would clarify whether the bottleneck is evidence retrieval, answer generation, or evaluation" |

## Expected outcome (goal)
Oracle accuracy sets the ceiling. If it is also low on ImplexConv, the low numbers come from
answer generation and the strict judge, not from AQUI's retrieval, and we report how much of the
oracle accuracy AQUI recovers. If it is high, the gap measures the remaining headroom for
retrieval. Either way the reviewer's question is answered.

## Resources
* **Server: SPACE.** The judge's latent memory (`baseline_outputs/actmem/outputs/*/facts.json`), the paper's answer files and the old judge outputs are all on SPACE, and `anaconda/vllm_2` already served Qwen3-Next and Qwen3.5-122B. E6 needs no adapters, so it is the one experiment that could run on AWS: copy those files and set `VLLM_ENV` to an env with a recent vLLM.
* **GPUs:** 8 x A100-80GB.
  * Step 1: both answer models at once, Qwen-80B on GPUs 0-3 and R1-70B on GPUs 4-7 (TP=4 each).
  * Step 2: the 122B judge on all 8 GPUs (TP=8; `JUDGE_TP=4 JUDGE_GPUS=0,1,2,3` also fits at
    max_len 8192).
* **Wall-clock (rough):** ImplexConv has 3,633 questions. R1 is the slowest at ~1-3 h; judging takes ~1 h.
* **Inputs:** the ImplexConv fact bag and ActMemEval-hard (bundled). For identical latent memory in
  the judge, use ActMem's `facts.json` under `${ACTMEM_FACTS_ROOT}/<dataset>/facts.json`. It is
  built from the fact bag if missing.

## Run
```bash
bash E6/run.sh
ANSWER_MODELS="qwen80b" bash E6/run.sh                         # one answer model
REJUDGE_FILES="/path/answers__...k20.json" bash E6/run.sh      # also re-judge paper AQUI answers in the same run
```
Outputs: `${OUT_BASE}/E6/answers/oracle__*.json`, `judged/*__judged.json`, `E6_summary.md`.
