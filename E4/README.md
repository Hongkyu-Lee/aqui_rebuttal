# E4 -- ActMemEval-hard: exclusion funnel, independent validation, paired results

## What this experiment is about
ActMemEval-hard keeps 72 of the 247 ActMemEval cases, and its rewrites were validated only by the
same model that produced them (gpt-5.4-mini). Important fact for the rebuttal: **the construction
rewrote the gold SESSIONS, not the queries**. Questions and gold answers are copied verbatim (step 2
checks this), so query meaning is preserved by construction. E4 provides:

1. **Funnel (CPU).**
   * 247 original rows.
   * Stage 1, lexical control: every gold session is rewritten to share no query tokens. 224 rows
     remain; 622/650 session rewrites pass.
   * Stage 2, implicit rewrite plus an LLM judge requiring direct-topic ≤2, meaning ≥4, an external
     bridge, usefulness and no fatal issues; a row is kept only if **all** its gold sessions pass.
     72 rows remain; 373/591 session rewrites pass.
   * Of the 152 dropped rows, 67 failed on only 1 of 3 gold sessions.

   Also re-verifies the 72-row file. The rebuilt copy matches the paper's fact bag row for row.
2. **Independent judge.** A model from a different family re-judges every session rewrite against
   its reference session. Default: local Llama-3.3-70B-Instruct; alternatively the Thinking Machines
   Lab API. It checks:
   * meaning preserved,
   * no invented facts,
   * **gold answer still supported**,
   * **session still relevant**,
   * topic directness and the need for a bridge.

   It also reports agreement with the original judge (accuracy, Cohen's κ).
3. **Human check sheet (optional, OFF by default).** 50 random rewritten gold sessions from the kept
   72 rows, with columns for 2 annotators and a scoring command (κ). Not run unless
   `HUMAN_CHECK=1` (or `STEPS="3"`); the code is kept in case it is needed later.
4. **Paired results on the same 72 questions.** Original / lexical-controlled / implicit-hard
   versions × {pretrained, AQUI, symmetric}, so the drop caused by the rewrite is measured on
   identical questions.

## Reviewer points addressed
| ID | Point |
|---|---|
| 2KMq-W2b | 247 → 72 without the full exclusion process; no independent verification of the rewrites |
| 2KMq-Q3a | Why only 72 of 247 remain |
| 2KMq-Q3b | How query meaning, answer validity and gold-session relevance were checked |
| 2KMq-Q3c | Paired before/after results on the same 72 cases |

## Expected outcome (goal)
* The independent judge passes most kept sessions; goal: ≥85% of sessions with meaning ≥4 and the
  answer still supported. Agreement with the original judge is moderate or better (κ ≥ 0.4).
* (If the optional human check is run: high pass rate with reasonable inter-annotator agreement.)
* Paired retrieval: every method loses recall from original → lexical → hard (the rewrite removes
  the shortcuts), and AQUI loses the least, i.e. keeps the largest margin on hard.

## Resources
* **Server: SPACE** (the paired-72 retrieval uses the paper adapter and E2's symmetric adapter). Steps 1-2 (funnel, judge) need no local GPU when `JUDGE_BACKEND=api` and can run anywhere. The original ActMemEval JSON and fact bag (for the "original" judge reference and the original72 row) exist only on AWS; stage them once (`../SERVERS.md` §3).
* **GPUs:** 8 x A100-80GB. GPUs 0-3 serve the judge LLM (70B, TP=4); GPUs 4-7 run the retrieval
  evaluations at the same time. With `JUDGE_BACKEND=api`, 4 GPUs are enough.
* **Human time:** none by default (the optional human check takes ~2-3 h for two annotators).
* **Inputs:** bundled construction files (no download needed). For the "original" column you also
  need the original ActMemEval JSON and its fact bag (see AGENT.md).

## Run
```bash
bash E4/run.sh                                     # local judge (vLLM)
JUDGE_BACKEND=api JUDGE_BASE_URL=<url>/v1 JUDGE_MODEL=<id> JUDGE_API_KEY_ENV=TML_API_KEY bash E4/run.sh
STEPS="1" bash E4/run.sh                           # CPU-only funnel
HUMAN_CHECK=1 bash E4/run.sh                       # also write the optional human-check sheet
python -m tools.audit_sample --score <filled.csv>  # after annotation (run from shared/)
```
Outputs: `${OUT_BASE}/E4/` -> `funnel.{md,json}`, `independent_judge*.json(l)`, `human_check.csv` (only with `HUMAN_CHECK=1`),
`paired/*.json`, `E4_summary.md`.
