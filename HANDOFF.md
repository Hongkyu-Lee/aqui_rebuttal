# Handoff: running the rebuttal experiments for someone else

You are running these experiments on behalf of h.lee4. Everything is in this folder; nothing else
needs to be cloned. Read this page first, then `SERVERS.md` (which server and which files).
Each experiment folder has a `README.md` (what and why) and an `AGENT.md` (details, known risks,
how to fix things). If something breaks, `AGENT.md` explains what must not change.

## 0. Before anything: what the package assumes about you

The scripts default to **h.lee4's** environments and caches. Check each item:

| Item | Default | Needs | If not OK |
|---|---|---|---|
| Python (torch/peft) | `/group-volume/users/h.lee4/anaconda/v03/bin/python` | read + execute | `PY=/your/python` (torch, transformers, peft, omegaconf, hydra; see `shared/requirements.txt`) |
| vLLM | `/group-volume/users/h.lee4/anaconda/vllm_2` | read + execute | `VLLM_ENV=/your/env` with vLLM >= 0.10 |
| HF cache (models, ImplexConv) | `/group-volume/users/h.lee4/cache` | read; write for new downloads | Keep it if readable (models already cached there); otherwise `HF_HOME=/your/cache` |
| Outputs | `/group-volume/users/h.lee4/iclr_2027_rebuttal_outputs` | **write** | `OUT_BASE=/your/dir`, and use the **same** `OUT_BASE` for every experiment (E2-E5 read E1/E2 outputs from it) |
| Inputs under `/group-volume/users/h.lee4/prototypes/...` etc. | see `SERVERS.md` §2 | read | override the variable named in `SERVERS.md` |
| AWS files under `/users/h.lee4/...` (home dir) | see `SERVERS.md` §3 | read | home dirs are usually private: ask h.lee4 to run the staging, or skip them (E2b/E4 drop those parts) |

Easiest: put your overrides in one file and source it before every run:
```bash
# my_env.sh
export OUT_BASE=/group-volume/users/<you>/iclr_2027_rebuttal_outputs
# export HF_HOME=...   PY=...   VLLM_ENV=...
```

## 1. Secrets / accounts (ask h.lee4)
* **E4 independent judge via Thinking Machines Lab API**: `JUDGE_BACKEND=api`, `JUDGE_BASE_URL=<endpoint>`,
  `JUDGE_MODEL=<model id>`, and the key in `TML_API_KEY`. Without them E4 uses a local
  Llama-3.3-70B judge, which needs **Hugging Face access to the gated Llama repo** (`HF_TOKEN`).
* Nothing else needs a key: all other models are served locally with vLLM. W&B is off.

## 2. Validate paths first (required)

```bash
source my_env.sh
bash shared/tools/check_paths.sh     # -> PATH_CHECK_REPORT_space.md; send it to h.lee4 before running
```
The full list with expected values is in `PATH_CHECKLIST.md`. Do not start E1 while any row says FAIL.

## 3. Check, then run (SPACE)

```bash
source my_env.sh
for e in E1 E2 E2b E3 E4 E5 E6 E7; do CHECK_ONLY=1 bash $e/run.sh; done   # prints [ok]/[MISSING]/[absent]; runs nothing
```
Fix every `[MISSING]` before starting. `[absent]` is fine (that part is skipped).

Runs take hours. Use `tmux`/`screen` (or `nohup ... &`) so a dropped SSH session does not kill them.
Logs: `${OUT_BASE}/<Exp>/logs/run.log` (+ one log per vLLM server / job).

| Order | Command | GPUs | Notes |
|---|---|---|---|
| 1 | `bash E1/run.sh` | 8 | **Gate.** Step 3 must reproduce the paper's ImplexConv recall (R@10 = 0.3322). If it stops there, do not continue: the rewritten data loaders are wrong; see `E1/AGENT.md`. |
| 2 | `bash E2/run.sh` | 8 | Longest (trains 6 LoRAs). Symmetric arm may OOM at batch 128; see `E2/AGENT.md`. |
| 3 | `bash E2b/run.sh`, `bash E3/run.sh` | 4, 8 | After E2. |
| any | `bash E4/run.sh`, `bash E5/run.sh`, `bash E6/run.sh` | 8 each | Independent of E1-E3 (E4/E5 reuse E2 outputs if present). |
| after E5 | `bash E7/run.sh` | 8 | Needs E5's evidence labels and AQUI ranking; uses E6's judge memory. Not at the same time as E6 (same ports). |

Useful knobs: `GPUS="0 1 2 3"` (restrict GPUs), `STEPS="5 6"` (re-run only some steps; each
`run.sh` header lists its steps). Finished outputs are skipped on re-run (`FORCE_TRAIN=1`, `FORCE_EVAL=1` to redo).

**Shared machine:** the vLLM servers use ports 8000, 8001, 8002 (E6), 8000/8001 (E7), 8100 (E4), 8200 (E5).
If a port is taken the script stops; set `JUDGE_PORT` / `LABEL_PORT` or edit the port in the
experiment's `run.sh`. Do not run two experiments on the same GPUs at once.

## 4. What to send back
For each experiment: `${OUT_BASE}/<Exp>/<Exp>_summary.md` and `logs/run.log`.
Also send E1's `split_audit.json`, `loader_check.json`, `selection.json`, E2's `e2_selection.json`,
E4's `funnel.md`, and the `MANIFEST.txt` of the
AWS staging, if done. Small JSON/CSV only; no adapters or caches.

Report any change you made to the code or to a default path, and why. Results must stay
comparable with the paper (see each `AGENT.md`, "Intent (do not change)").
