# E5 -- notes for the agent running / fixing this experiment

## Intent (do not change)
Re-score the **same** retrieval outputs under stricter recall definitions. The evidence labels are
produced once per question, independently of any method, and shared by all methods. Don't tune the
labeling prompt by looking at method results.

## Details
* Fact indices: `tools/evidence_label.py` lists gold-session facts using
  `src.evaluate.fact_retrieval.normalize_fact_items`, the same indexing as the `fact_index` field
  in result JSONs. Keep it that way, or the evidence join silently breaks.
* Result JSONs contain the top-`max(top_ks)` retrieved facts per question, i.e. 151 with the
  shared eval config. Evidence metrics are computed from `retrieved_facts`.
* Results are reused from `${OUT_BASE}/E2/test` and `${OUT_BASE}/E1/test` when present; otherwise
  they are computed here.
* Questions with an empty evidence set are excluded from the evidence metrics, and their count is
  reported (`n_with_evidence`). If more than ~10% are empty, inspect a few labels: the labeler may
  be failing to parse.

## If something goes wrong
* The labeler server is Qwen3-Next-80B, which needs a recent vLLM (SPACE: `anaconda/vllm_2`).
  If it does not load, use `Qwen/Qwen2.5-72B-Instruct` (set `LABEL_MODEL`) and record it.
* Labeling is resumable (`evidence_*.jsonl` is appended to); rerun to fill gaps.
* Optional quality check if time allows: hand-check 30 labels and report agreement.

## What to report back
`E5_summary.md`: per benchmark, for each method: session_recall, fact_recall, evidence_hit and
evidence_recall at each K, plus `n_with_evidence`.
