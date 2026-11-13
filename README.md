# ICLR 2027 rebuttal experiments (AQUI, submission 34198)

Self-contained package: every experiment runs from this folder alone, with one command
(`bash E*/run.sh`). Nothing imports from the old repos; the code they need was copied into
`shared/`, and the parts that were missing were rewritten (see "What was rewritten").

**Running this for someone else? Start with [HANDOFF.md](HANDOFF.md), then [SERVERS.md](SERVERS.md).**
**Before any run: validate paths with [PATH_CHECKLIST.md](PATH_CHECKLIST.md) (`bash shared/tools/check_paths.sh`).**

## Layout

```
iclr_2027_rebuttal/
  README.md                 <- this file
  shared/                   <- everything shared by the experiments
    env.sh                  server profile (SPACE / AWS), paths, python/vLLM envs  (sourced by every run.sh)
    lib.sh                  bash helpers: GPU pool, train_lora, eval_retrieval, vLLM start/wait/stop
    config/                 train.yaml, evaluate_fact_retrieval.yaml, data/{implexconv,actmemeval}.yaml
    src/                    training + retrieval-eval code (copied from implicit_version_03)
      data/                 REWRITTEN loaders (the originals were never committed)
    baseline_symmetric/     symmetric-LoRA trainer (copied; overrides one method)
    judge/                  LLM-as-judge used in the paper (copied from implicit_baselines/actmem)
    tools/                  experiment-specific python tools (all new)
    data/                   bundled small data: v11con_2400 training set, ActMemEval-hard construction
                            files + fact bags, rebuilt ActMemEval_implicit_hard (72 rows)
  E1/ ... E7/   one folder per experiment: run.sh, README.md, AGENT.md
```

## Experiments

| Exp | What | Answers | A100-80GB GPUs | Server | Depends on |
|---|---|---|---|---|---|
| E1 | Clean model selection (synthetic validation only) + split audit | Ap7K-W2/Q1b/Q2, 2KMq-W2a/Q2a/Q2b, HeEX-W2/Q2 | 8 (min 1) | **SPACE** (required) | HPO adapters on SPACE |
| E2 | Matched symmetric-LoRA control, equal tuning | Ap7K-W1/Q1, 2KMq-W1/Q1, HeEX-W1/Q1 | 8 | SPACE | (E1 optional) |
| E2b | Fact-space drift + explicit retrieval | Ap7K-W1 (regularization vs knowledge) | 4 | SPACE (+ copy original ActMemEval from AWS) | E2 |
| E3 | 3 seeds + paired bootstrap | HeEX-Q1, 2KMq-W1 | 8 | SPACE | E2 |
| E4 | ActMemEval-hard funnel, independent judge, paired 72 (human sheet opt-in) | 2KMq-W2b/Q3 | 8 (4 with API judge) | SPACE (+ copy original ActMemEval from AWS) | (E2 optional) |
| E5 | Session vs fact vs evidence recall | 2KMq-Q4a | 8 | SPACE | (E1/E2 results reused) |
| E6 | Oracle-memory answer baseline | 2KMq-Q4c | 8 | SPACE (AWS possible) | - |
| E7 | Noisy-oracle answer baseline (evidence + AQUI-ranked distractors) | 2KMq-Q4c | 8 | SPACE | E5 (E6 recommended) |

Recommended order: **E1 -> E2 -> (E2b, E3)**; E4, E5, E6 are independent and can run any time; E7 after E5 (and E6).
Not GPU experiments (answered in text): K=151 (2KMq-Q4b), related work, typos.

## Running

```bash
cd iclr_2027_rebuttal
bash E1/run.sh                          # auto-detects the server
SERVER=space GPUS="0 1 2 3" bash E1/run.sh
STEPS="5 6 7" bash E1/run.sh            # re-run only some steps (each run.sh documents its steps)
```
* Outputs go to `${OUT_BASE}/<Exp>/` (SPACE default: `/group-volume/users/h.lee4/iclr_2027_rebuttal_outputs`).
  Every run writes `logs/run.log` and a `<Exp>_summary.md`.
* Steps skip work whose output already exists (`FORCE_TRAIN=1`, `FORCE_EVAL=1` to redo).
* **AWS is a SLURM cluster**: get a node first, then run the same command, e.g.
  `srun -p gpu-aic-mv-01 --gres=gpu:8 --cpus-per-task=32 --mem=400G --time=24:00:00 --pty bash`
  and then `SERVER=aws bash E2/run.sh`.

## Servers

| | SPACE | AWS |
|---|---|---|
| path prefix | `/group-volume/users/h.lee4` | `/groups/AIC-MV/h.lee4` (home `/users/h.lee4`) |
| python (torch/peft) | `anaconda/v03/bin/python` | `miniconda3/envs/actmem/bin/python` |
| vLLM env | `anaconda/vllm_2` (vLLM 0.25.1, CUDA-13 + compat shim) | not confirmed (set `VLLM_ENV`) |
| what lives there | paper HPO adapters (`prototypes/version_03_artifacts/oppose_gen_hpo`), v11con data, ImplexConv fact bag, GEPA / answer-gen / judge outputs, baseline outputs | older June-July runs (size sweep, early grids), original ActMemEval data |

**Full per-server file map, evidence and AWS→SPACE staging: [SERVERS.md](SERVERS.md).** `CHECK_ONLY=1 bash E*/run.sh` prints which inputs are found.

**Run every experiment on SPACE.** The 216 HPO adapters, the paper adapter, the ImplexConv fact bag and the paper's answer and judge outputs live there. The experiments also chain through `${OUT_BASE}` (E1 → E2 → E2b/E3/E4/E5), and `OUT_BASE` differs per server, so splitting the chain across servers breaks it. Two things come from AWS: the original `ActMemEval.json` (and its fact bag) used by E2b and E4. Stage them once with `shared/tools/stage_from_aws.sh`. Paths marked
`(VERIFY)` in `shared/env.sh` were inferred from old scripts; override any of them with an env var.

## What was rewritten (and how it is verified)

1. `shared/src/data/load_data.py`, `load_train_data.py` -- missing from the old repo. Verified by
   (a) `tools/check_loader_against_manifest.py` (training split vs the paper adapter's own
   `train_data_manifest.json`; E1 step 2 / E2 step 1) and (b) E1 step 3, which must reproduce the
   paper's ImplexConv recall of the paper adapter (0.3322 / 0.4588 / 0.8412 at K=10/20/100).
2. `shared/data/ActMemEval_implicit_hard.rebuilt.json` -- the old copy is 0 bytes; rebuilt from the
   construction cache and **verified** (72 rows, row order, session counts and session ids match the
   paper's fact bag; 96% of fact evidence quotes found verbatim in the rebuilt sessions).
