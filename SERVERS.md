# Which experiment runs on which server

Two old servers. They do **not** share a filesystem.

| | SPACE | AWS |
|---|---|---|
| Path prefix | `/group-volume/users/h.lee4` | `/groups/AIC-MV/h.lee4`, home `/users/h.lee4` |
| How to get GPUs | 8-GPU box, run directly | SLURM: `srun -p gpu-aic-mv-01 --gres=gpu:8 ... --pty bash` first |
| Detected by `shared/env.sh` | `/group-volume/users/h.lee4` exists | `/groups/AIC-MV/h.lee4` exists |

## 1. Assignment

| Exp | Server | Why it must be there |
|---|---|---|
| E1 | **SPACE** | The 216 HPO adapters and the paper adapter exist only on SPACE |
| E2 | **SPACE** | Reads E1's output; ImplexConv fact bag |
| E2b | **SPACE** + staged AWS files | Reads E2's adapters; original ActMemEval and Supportive fact bag come from AWS (section 3) |
| E3 | **SPACE** | Reuses E2's seed-42 adapters and results |
| E4 | **SPACE** + staged AWS files | Paper adapter and E2's symmetric adapter; original ActMemEval comes from AWS (section 3) |
| E5 | **SPACE** | Reuses E1/E2 results |
| E6 | **SPACE** (AWS possible) | Judge latent memory and paper answer/judge outputs are on SPACE |

**Nothing needs to run on AWS.** AWS is only the source of three data files, packed by one
command (section 3). Each `run.sh` refuses to start on the wrong server (`require_server`).
Override with `ALLOW_ANY_SERVER=1` only if you have copied the inputs.

Why one server: the experiments find each other's outputs under `${OUT_BASE}/E*`, and
`OUT_BASE` is a different folder on each server.

## 2. Every input, where it lives, and who uses it

The "Evidence" column names the old script that read or wrote the file, i.e. why we believe it is there.
Every path can be overridden with the environment variable of the same name.

### SPACE (`U = /group-volume/users/h.lee4`)
| Variable | Path | Used by | Evidence |
|---|---|---|---|
| `PY` | `U/anaconda/v03/bin/python` | all | v03 SPACE launchers |
| `VLLM_ENV` | `U/anaconda/vllm_2` | E4 (local judge), E5, E6 | gepa_answergen / judge launchers |
| `HF_HOME` | `U/cache` (also the cuda-compat-13 shim) | all | tacitree_output_paths.sh |
| `HPO_ROOT` | `U/prototypes/version_03_artifacts/oppose_gen_hpo` | E1 | HPO runner, gepa_answergen.sh |
| `PAPER_ADAPTER` | `HPO_ROOT/adapters/hpo_v11con_2400_8B__lr5e-5_t0p02_gc1p0_wu0p1_bs128_r16` | E1, E2, E4, E5 | gepa_answergen.sh |
| `V11CON_2400_JSONL` | `U/prototypes/version_05/data/generated/con_test/space_gen_v11con_2400/v11con_2400.jsonl` (bundled copy if missing) | E1, E2, E3 | run_ablation_negatives.sh |
| `IMPLEX_FACT_BAG` | `U/prototypes/version_03_artifacts/opposed_reasoning_hf/user_facts.jsonl` | E1, E2, E2b, E3, E5, E6 | run_gepa_localbox.sh, run_ablation_negatives.sh |
| `ACTMEM_HARD_FACT_BAG` | `U/prototypes/version_03_artifacts/actmemeval_implicit_hard/user_facts.jsonl` (bundled copy if missing) | E1–E6 | baseline_symmetric/run_*.sh, gepa_answergen.sh |
| `ACTMEM_HARD_JSON` | `U/implicit-baselines/datasets/actmemeval_variants/ActMemEval_implicit_hard.json` (bundled rebuilt copy if missing/empty) | E1–E6 | baseline_symmetric/run_*.sh |
| `ACTMEM_FACTS_ROOT` | `U/baseline_outputs/actmem/outputs/{implexconv_opposed,actmemeval_implicit_hard}/facts.json` | E6 | actmem/sweep_topk_retrieval.py, run_judge_*.sh |
| `JUDGE_ROOT_OLD` | `U/judge_outputs` | E6 (summary only) | run_judge_*.sh |
| `ACTMEM_ORIG_JSON` | `U/iclr_2027_rebuttal_inputs/from_aws/ActMemEval.json` | E2b, E4 | **staged from AWS** |
| `ACTMEM_ORIG_FACT_BAG` | `U/iclr_2027_rebuttal_inputs/from_aws/actmemeval/user_facts.jsonl` | E2b, E4 | **staged from AWS** |
| `IMPLEX_SUPPORTIVE_FACT_BAG` | `U/iclr_2027_rebuttal_inputs/from_aws/supportive_reasoning_hf/user_facts.jsonl` | E2b | **staged from AWS** |
| `OUT_BASE` | `U/iclr_2027_rebuttal_outputs` (created) | all | new |

### AWS (`G = /groups/AIC-MV/h.lee4`)
Only these are read, by `shared/tools/stage_from_aws.sh`:
| Variable | Path | Evidence |
|---|---|---|
| `ACTMEM_ORIG_JSON` | `/users/h.lee4/codes/prototype/version_03/data/actmem/ActMemEval.json` | `source_path` of config/create_actmemeval_variants.yaml (run from that repo) |
| `ACTMEM_ORIG_FACT_BAG` | `G/version_03/output_json/user_facts/actmemeval/user_facts.jsonl` | `output_dir` of config/extract_user_facts.yaml |
| `IMPLEX_SUPPORTIVE_FACT_BAG` | `/users/h.lee4/codes/TaciTree/outputs/user_facts/supportive_reasoning_hf/user_facts.jsonl` | **(VERIFY)** slurm/eval_all_topk.sh reads `.../${DATASET_TYPE}_reasoning_hf/`; only `opposed` is confirmed |

Not needed on any server (bundled in `shared/data/`): the lexical-controlled ActMemEval and its
metadata, the implicit-hard metadata and rewrite cache, both ActMemEval fact bags, the rebuilt
ActMemEval-hard JSON, `v11con_2400.jsonl`.
Downloaded into `HF_HOME` on first use: `Qwen/Qwen3-Embedding-8B`, `Kaylee0501/ImplexConv-{opposed,supportive}`,
and the vLLM models (E4–E6).

## 3. One-time staging: AWS → SPACE

```bash
# on AWS (login node is fine, no GPU)
cd iclr_2027_rebuttal && bash shared/tools/stage_from_aws.sh       # -> aws_inputs_for_space.tar.gz + MANIFEST
# move the tarball to SPACE by whatever route you have (laptop, object storage, ...)
# on SPACE
mkdir -p /group-volume/users/h.lee4/iclr_2027_rebuttal_inputs/from_aws
tar -xzf aws_inputs_for_space.tar.gz -C /group-volume/users/h.lee4/iclr_2027_rebuttal_inputs/from_aws
```
Without staging, E2b and E4 still run; they skip the "original ActMemEval" and "Supportive" parts with a warning.

## 4. Check before running

Each `run.sh` prints an input table and stops if a required input is missing:
```bash
CHECK_ONLY=1 bash E1/run.sh      # print [ok]/[MISSING]/[absent] for E1's inputs and exit; runs nothing
for e in E1 E2 E2b E3 E4 E5 E6; do CHECK_ONLY=1 bash $e/run.sh; done
```
`[MISSING]` = required, the run will not start. `[absent]` = optional, that part is skipped
(or rebuilt from bundled data). If a file is somewhere else, set the variable:
`IMPLEX_FACT_BAG=/path/user_facts.jsonl bash E2/run.sh`.
