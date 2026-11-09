# E6 -- notes for the agent running / fixing this experiment

## Intent (do not change)
Keep everything identical to the paper's answer-generation and judging setup **except** the facts
given to the answer model, which become the gold facts. That makes the oracle row directly
comparable with the paper's AQUI rows.

Specifically, keep identical:
* answer models and decoding,
* the prompt (copied verbatim into `tools/oracle_answers.py`),
* the judge model and rubric (`shared/judge/evaluate_answer_quality.py`, copied unchanged except
  for the import path),
* the latent-memory file.

## Details
* **Oracle facts.** Every fact in the fact bag whose `session_index` is one of the question's gold
  sessions, in fact-bag order, uncapped (`--max-facts 0`). The paper's AQUI runs used K = 10…100
  facts; the oracle set is usually smaller, which is fine because it is the ideal evidence.
  `summary.mean_num_oracle_facts` records the size.
* **Latent memory for the judge.** The paper used ActMem's extracted `facts.json`
  (`/group-volume/users/h.lee4/baseline_outputs/actmem/outputs/{implexconv_opposed,actmemeval_implicit_hard}/facts.json`).
  Use it if present (run.sh does). Otherwise `tools/build_latent_facts.py` builds the same schema
  from our fact bag. In that case, also re-judge one paper AQUI answer file in the same run
  (`REJUDGE_FILES=...`), so the comparison uses one latent memory.
  The paper's answer files live under `/group-volume/users/h.lee4/prototypes/version_03_artifacts/gepa_answergen/`.
* **Judge alias.** It is served as `gpt-4o-mini` because the judge code and the old tags expect
  that name. It is *not* OpenAI. `OPENAI_BASE_URL` points to the local server.
* **Rigorous** = case 1 / N; **lenient** = (case1 + 0.5·case2 + 0.5·case3) / N. Empty answers are
  skipped; this matches the paper.

## If something goes wrong
* Qwen3-Next and Qwen3.5 need a recent vLLM. SPACE's `anaconda/vllm_2` (0.25.1) served both
  before. On AWS, set `VLLM_ENV` to an env with vLLM ≥ 0.10.
* R1 truncation: check `summary.num_reasoning_truncated`. If it is above a few %, raise
  `--max-tokens` to 4096 in the R1 spec. Do this only for the oracle run if the paper runs used
  3072, and note it.
* The judge OOMs at TP=4: use TP=8 (default) or lower `--max-num-seqs`.
* Every step is resumable: answers go to `*.partial.jsonl` and judge verdicts to `*.jsonl`.

## What to report back
`E6_summary.md`: the oracle rigorous/lenient accuracy per answer model and benchmark, next to the
paper's AQUI rows for the same answer model.
