# E7 -- notes for the agent running / fixing this experiment

## Intent (do not change)
Identical to E6 (answer model, prompt, decoding, judge, rubric, **judge memory**) except the facts
given to the answer model. The prompt and decoding are imported from `tools/oracle_answers.py`,
so they cannot drift from E6.

* **Same questions in every row.** Questions without an E5 evidence label are skipped in all modes.
  Do not "fix" this by falling back to all gold-session facts for some questions: rows would then
  be on different questions.
* **Distractors are hard, not random.** They are AQUI's top-ranked facts whose `session_index` is
  not a gold session. Random facts would make the task much easier and overstate the answer model.
* **Evidence position is random** (seeded by `SEED`, `example_index`, `qa_index`). Always putting it
  first would let the model rely on position. `evidence_positions` in each sample records where it went.
* **Judge memory = E6's.** `latent_for` uses `${OUT_BASE}/E6/latent_<ds>.json` if E6 built one,
  otherwise ActMem's `facts.json` (what E6 used when it existed). Check that the path logged in
  step 2 matches the one in `E6/logs/run.log`. If they differ, set `LATENT_implexconv=...` /
  `LATENT_actmem_hard=...`. Otherwise the E7 and E6 numbers are not comparable.

## Details
* AQUI ranking: the bi-encoder ranking of the **paper adapter** (E5 step 1 output). Table 2's AQUI
  rows used this adapter plus a 4B reranker; E7's `retrieved` row has no reranker. That is fine:
  `noisy` vs `retrieved` is compared within E7.
* If a question has more evidence facts than K, only the first K are used (rare).
* `skipped.short_list` counts questions where the ranking had too few non-gold facts to fill K
  (should be 0; rankings hold 151 facts).

## If something goes wrong
* Same vLLM / OOM notes as `E6/AGENT.md`.
* Every step is resumable: answers go to `*.partial.jsonl`; finished answer and judged files are
  skipped on re-run (`FORCE=1` to redo).

## What to report back
`E7_summary.md`, `logs/run.log`, and the judged files `judged/*__judged.json`.
