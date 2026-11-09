# Path checklist — validate BEFORE running any experiment

Every file or directory the experiments read from **outside this folder** is listed below.
The code itself is self-contained: nothing is imported from the old repos.

## How to validate (one command, no GPU, reads only)

```bash
cd iclr_2027_rebuttal
# put any overrides you will use for the runs first, e.g. export OUT_BASE=/group-volume/users/<you>/rebuttal_out
bash shared/tools/check_paths.sh            # writes PATH_CHECK_REPORT_<server>.md and prints it
```
* **FAIL**: blocks the experiment(s) in the "Used by" column. Fix the path (override the variable),
  the permission, or the environment before running anything.
* **WARN**: an optional part is skipped or rebuilt from bundled data, or a row count differs from
  what we expect (check it).
* **Send `PATH_CHECK_REPORT_space.md` back to h.lee4** before starting E1.

If a file exists somewhere else, do not edit the scripts. Set the variable instead, e.g.
`export IMPLEX_FACT_BAG=/path/to/user_facts.jsonl`, and re-run the check.
Each experiment also re-checks its own inputs at start (`CHECK_ONLY=1 bash E1/run.sh`).

---

## SPACE (`U = /group-volume/users/h.lee4`). All experiments run here.

### Environments and writable locations
| ✓ | Variable | Path | Expect | Used by |
|---|---|---|---|---|
| ☐ | `PY` | `U/anaconda/v03/bin/python` | runs; imports torch, transformers, peft, hydra, omegaconf, datasets, openai, wandb; CUDA visible | all |
| ☐ | `VLLM_ENV` | `U/anaconda/vllm_2` | `bin/vllm` exists (vLLM 0.25.1) | E4 (local judge), E5, E6 |
| ☐ | `HF_HOME` | `U/cache` | readable (writable if models must be downloaded) | all |
| ☐ | `CUDA_COMPAT_CACHE` | `U/cache/cuda-compat-13/**/libcuda.so.*` | present (old driver + CUDA-13 builds) | all GPU steps |
| ☐ | `OUT_BASE` | `U/iclr_2027_rebuttal_outputs` | **writable** by the person running (else set your own, same for all experiments) | all |

### Models and datasets in `HF_HOME/hub` (downloaded on first use if absent; needs internet)
| ✓ | Repo | Used by |
|---|---|---|
| ☐ | `Qwen/Qwen3-Embedding-8B` | E1–E5 |
| ☐ | `Kaylee0501/ImplexConv-opposed` (dataset) | E1, E2, E3, E5, E6 |
| ☐ | `Kaylee0501/ImplexConv-supportive` (dataset) | E2b |
| ☐ | `Qwen/Qwen3-Next-80B-A3B-Instruct` | E5, E6 |
| ☐ | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | E6 |
| ☐ | `Qwen/Qwen3.5-122B-A10B` | E6 (judge) |
| ☐ | `meta-llama/Llama-3.3-70B-Instruct` (gated, needs `HF_TOKEN`) | E4, only if not using the API judge |

### Data
| ✓ | Variable | Path | Expect | Used by | If missing |
|---|---|---|---|---|---|
| ☐ | `HPO_ROOT` | `U/prototypes/version_03_artifacts/oppose_gen_hpo` | dir | E1 | **E1 cannot run** |
| ☐ | (adapters) | `HPO_ROOT/adapters/hpo_*` | **216** dirs, each with `adapter_config.json` (+ `model_selection.json`) | E1 | **E1 cannot run** |
| ☐ | (ranking) | `HPO_ROOT/HPO_ranking.csv` | exists | E1 | reference column blank (selection unaffected) |
| ☐ | `PAPER_ADAPTER` | `HPO_ROOT/adapters/hpo_v11con_2400_8B__lr5e-5_t0p02_gc1p0_wu0p1_bs128_r16/` | `adapter_config.json`, `adapter_model.safetensors`, `model_selection.json` | E1, E2, E4, E5 | **blocks** |
| ☐ | (manifest) | `PAPER_ADAPTER/train_data_manifest.json` | exists | E1 step 2 | loader check skipped (weaker validation) |
| ☐ | `V11CON_2400_JSONL` | `U/prototypes/version_05/data/generated/con_test/space_gen_v11con_2400/v11con_2400.jsonl` | 2400 lines | E1, E2, E3 | bundled copy used (identical) |
| ☐ | `IMPLEX_FACT_BAG` | `U/prototypes/version_03_artifacts/opposed_reasoning_hf/user_facts.jsonl` | **1550** lines | E1, E2, E2b, E3, E5, E6 | **blocks** |
| ☐ | `ACTMEM_HARD_FACT_BAG` | `U/prototypes/version_03_artifacts/actmemeval_implicit_hard/user_facts.jsonl` | 72 lines | E1–E6 | bundled copy used |
| ☐ | `ACTMEM_HARD_JSON` | `U/implicit-baselines/datasets/actmemeval_variants/ActMemEval_implicit_hard.json` | 72 rows, non-empty | E1–E6 | bundled rebuilt copy used (verified) |
| ☐ | `ACTMEM_FACTS_ROOT` | `U/baseline_outputs/actmem/outputs/implexconv_opposed/facts.json` and `.../actmemeval_implicit_hard/facts.json` | both exist | E6 (judge latent memory) | rebuilt from fact bag; then re-judge one paper file too |
| ☐ | `JUDGE_ROOT_OLD` | `U/judge_outputs` | dir | E6 summary | comparison column blank |
| ☐ | `GEPA_ANSWERGEN_ROOT` | `U/prototypes/version_03_artifacts/gepa_answergen` | dir | E6 (`REJUDGE_FILES`) | only needed if facts.json is missing |
| ☐ | `ACTMEM_ORIG_JSON` | `U/iclr_2027_rebuttal_inputs/from_aws/ActMemEval.json` | 247 rows | E2b, E4 | that part skipped (stage from AWS) |
| ☐ | `ACTMEM_ORIG_FACT_BAG` | `U/iclr_2027_rebuttal_inputs/from_aws/actmemeval/user_facts.jsonl` | 247 lines | E2b, E4 | that part skipped (stage from AWS) |
| ☐ | `IMPLEX_SUPPORTIVE_FACT_BAG` | `U/iclr_2027_rebuttal_inputs/from_aws/supportive_reasoning_hf/user_facts.jsonl` | non-empty | E2b | that part skipped (stage from AWS) |

### Secrets (presence only)
| ✓ | Name | Needed for |
|---|---|---|
| ☐ | `TML_API_KEY` (+ `JUDGE_BASE_URL`, `JUDGE_MODEL`) | E4 independent judge via Thinking Machines API (`JUDGE_BACKEND=api`) |
| ☐ | `HF_TOKEN` | E4 local Llama judge (only if not using the API) |

---

## AWS. Source of three files only (optional; for E2b/E4 "original" and "Supportive" parts)

| ✓ | Variable | Path | Note |
|---|---|---|---|
| ☐ | `ACTMEM_ORIG_JSON` | `/users/h.lee4/codes/prototype/version_03/data/actmem/ActMemEval.json` | **home dir: probably only h.lee4 can read it** |
| ☐ | `ACTMEM_ORIG_FACT_BAG` | `/groups/AIC-MV/h.lee4/version_03/output_json/user_facts/actmemeval/user_facts.jsonl` | group path |
| ☐ | `IMPLEX_SUPPORTIVE_FACT_BAG` | `/users/h.lee4/codes/TaciTree/outputs/user_facts/supportive_reasoning_hf/user_facts.jsonl` | home dir; **unconfirmed** that it was ever generated |

Pack on AWS with `bash shared/tools/stage_from_aws.sh`, unpack on SPACE into
`U/iclr_2027_rebuttal_inputs/from_aws/` (`SERVERS.md` §3). Then re-run `check_paths.sh` on SPACE.

---

## Bundled inside the package (`shared/data/`; check_paths.sh verifies they arrived)
`v11con_2400.jsonl`, `ActMemEval_lexical_controlled.json`, `ActMemEval_lexical_controlled_metadata.json`,
`ActMemEval_implicit_hard_metadata.json`, `implicit_hard_rewrites.jsonl`,
`ActMemEval_implicit_hard.rebuilt.json`, `actmemeval_implicit_hard_user_facts.jsonl`,
`actmemeval_lexical_controlled_user_facts.jsonl`, `implexconv_opposed_qa_gold_session_facts.jsonl`.

## Paths created by the experiments (no need to exist beforehand)
`OUT_BASE/E1 … E6/`, with `logs/`, `test/`, `val_chunks/`, `cache/` (synthetic-val fact cache),
`OUT_BASE/fact_embedding_cache/` (base-model ImplexConv fact embeddings, shared by E1–E5),
and `OUT_BASE/E2/adapters/`, `OUT_BASE/E3/adapters/` (new LoRAs, ~100–300 MB each).
Experiments pass results to each other only through `OUT_BASE` (E1 → E2 → E2b/E3/E4/E5),
so it must be the same value for every run.
