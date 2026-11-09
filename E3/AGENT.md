# E3 -- notes for the agent running / fixing this experiment

## Intent (do not change)
Same configs as E2's per-arm picks; only `train.optimization.seed` changes. The validation split
seed (`train.data.val_split_seed=42`) must **not** change, so every seed sees the same train/val
data. Don't re-select configs per seed and don't drop seeds after seeing results. Every trained
seed goes into the statistics.

## Details
* HPs are recovered from E2's selected adapter directory name
  (`<arm>__lr.._t.._gc.._wu.._bs.._r..__s42`) by `hp_of` in run.sh.
* Seed-42 test JSONs are copied from E2 (`selected_<arm>__*`); if absent, they are recomputed.
* `tools/bootstrap.py` pairs questions by `(example_index, qa_index)`. Per question it averages
  session_recall@K over seeds, then bootstraps the mean difference. The p-value is two-sided:
  2 × min(fraction of resamples ≤ 0, fraction ≥ 0).

## If something goes wrong
* The OOM / loader issues are the same as in E2 (see E2/AGENT.md). Apply any workaround to both
  arms and all seeds.
* If a seed fails to train, rerun it (`FORCE_TRAIN=1` deletes nothing; it retrains into the same
  dir). Don't silently drop it; if it can't be fixed, report "n seeds" explicitly.

## What to report back
`E3_summary.md`: per benchmark and K, the mean ± std of each arm, the paired difference, the 95%
CI and the p-value.
