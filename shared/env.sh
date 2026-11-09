#!/bin/bash
# =============================================================================
# Server profile for the rebuttal package. SOURCE this (every run.sh does):
#     source "$(dirname "$0")/../shared/env.sh"
#
# Two old servers:
#   SPACE : paths under /group-volume/...      (8-GPU box; the paper's HPO, GEPA, answer-gen,
#                                               judge and baseline outputs live here)
#   AWS   : paths under /groups/AIC-MV/...     (SLURM cluster; older June-July runs;
#                                               home = /users/h.lee4)
# Auto-detected from which root exists; force with SERVER=space|aws.
#
# EVERY variable below can be overridden from the environment before calling run.sh, e.g.
#     IMPLEX_FACT_BAG=/my/copy/user_facts.jsonl bash E1/run.sh
# Each default cites the old script it was taken from; (VERIFY) = no old script ever read that exact file.
# Which experiment runs where, and what to copy between servers: ../SERVERS.md
# =============================================================================

REBUTTAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED="${REBUTTAL_ROOT}/shared"
export REBUTTAL_ROOT SHARED

if [[ -z "${SERVER:-}" ]]; then
  if [[ -d /group-volume/users/h.lee4 ]]; then SERVER=space
  elif [[ -d /groups/AIC-MV/h.lee4 ]]; then SERVER=aws
  else SERVER=unknown; fi
fi
export SERVER

case "$SERVER" in
  space)
    _U=/group-volume/users/h.lee4
    : "${PY:=${_U}/anaconda/v03/bin/python}"                       # torch/transformers/peft env
    : "${VLLM_ENV:=${_U}/anaconda/vllm_2}"                          # vLLM 0.25.1 (CUDA 13 build)
    : "${HF_HOME:=${_U}/cache}"
    : "${CUDA_COMPAT_CACHE:=${_U}/cache}"                           # cuda-compat-13 shim (old driver)
    : "${V03_ART:=${_U}/prototypes/version_03_artifacts}"
    : "${V11CON_2400_JSONL:=${_U}/prototypes/version_05/data/generated/con_test/space_gen_v11con_2400/v11con_2400.jsonl}"
    : "${HPO_ROOT:=${V03_ART}/oppose_gen_hpo}"                      # adapters/ results/ HPO_ranking.csv
    : "${IMPLEX_FACT_BAG:=${V03_ART}/opposed_reasoning_hf/user_facts.jsonl}"
    : "${STAGE_FROM_AWS:=${_U}/iclr_2027_rebuttal_inputs/from_aws}"  # files copied from AWS (see SERVERS.md)
    : "${IMPLEX_SUPPORTIVE_FACT_BAG:=${STAGE_FROM_AWS}/supportive_reasoning_hf/user_facts.jsonl}"   # AWS-only; staged
    : "${ACTMEM_HARD_JSON:=${_U}/implicit-baselines/datasets/actmemeval_variants/ActMemEval_implicit_hard.json}"
    : "${ACTMEM_HARD_FACT_BAG:=${V03_ART}/actmemeval_implicit_hard/user_facts.jsonl}"
    : "${ACTMEM_ORIG_JSON:=${STAGE_FROM_AWS}/ActMemEval.json}"                              # AWS-only; staged
    : "${ACTMEM_ORIG_FACT_BAG:=${STAGE_FROM_AWS}/actmemeval/user_facts.jsonl}"              # AWS-only; staged
    : "${SYM_ROOT:=${V03_ART}/baseline_symmetric}"                  # Table-1 symmetric run (if any)
    : "${ACTMEM_FACTS_ROOT:=${_U}/baseline_outputs/actmem/outputs}" # <ds>/facts.json (judge latent memory)
    : "${GEPA_ANSWERGEN_ROOT:=${V03_ART}/gepa_answergen}"           # paper answer-gen outputs
    : "${JUDGE_ROOT_OLD:=${_U}/judge_outputs}"                      # paper judge outputs
    : "${OUT_BASE:=${_U}/iclr_2027_rebuttal_outputs}"
    ;;
  aws)
    _G=/groups/AIC-MV/h.lee4
    : "${PY:=${_G}/miniconda3/envs/actmem/bin/python}"
    : "${VLLM_ENV:=${_G}/miniconda3/envs/actmem}"                   # (VERIFY) vLLM may live in another env
    : "${HF_HOME:=${_G}/.cache/huggingface}"
    : "${CUDA_COMPAT_CACHE:=}"
    : "${V03_ART:=${_G}/version_03/output}"
    : "${V11CON_2400_JSONL:=${SHARED}/data/v11con_2400.jsonl}"      # bundled copy (identical file)
    : "${HPO_ROOT:=}"                                                # HPO ran on SPACE only
    : "${TACITREE_FACTS:=/users/h.lee4/codes/TaciTree/outputs/user_facts}"   # slurm/eval_all_topk.sh, SESSION_HANDOFF_datagen.md
    : "${IMPLEX_FACT_BAG:=${TACITREE_FACTS}/opposed_reasoning_hf/user_facts.jsonl}"
    : "${IMPLEX_SUPPORTIVE_FACT_BAG:=${TACITREE_FACTS}/supportive_reasoning_hf/user_facts.jsonl}"  # (VERIFY) same dir, DATASET_TYPE=supportive
    : "${ACTMEM_HARD_JSON:=${_G}/version_03/output_json/actmemeval_variants/ActMemEval_implicit_hard.json}"  # config/create_actmemeval_variants.yaml
    : "${ACTMEM_HARD_FACT_BAG:=${SHARED}/data/actmemeval_implicit_hard_user_facts.jsonl}"
    : "${ACTMEM_ORIG_JSON:=/users/h.lee4/codes/prototype/version_03/data/actmem/ActMemEval.json}"  # source_path of create_actmemeval_variants.yaml
    : "${ACTMEM_ORIG_FACT_BAG:=${_G}/version_03/output_json/user_facts/actmemeval/user_facts.jsonl}"  # config/extract_user_facts.yaml
    : "${SYM_ROOT:=}"
    : "${ACTMEM_FACTS_ROOT:=}"
    : "${GEPA_ANSWERGEN_ROOT:=}"
    : "${JUDGE_ROOT_OLD:=}"
    : "${OUT_BASE:=${_G}/iclr_2027_rebuttal_outputs}"
    ;;
  *)
    : "${PY:=python}"
    : "${VLLM_ENV:=}"
    : "${OUT_BASE:=${REBUTTAL_ROOT}/outputs}"
    : "${V11CON_2400_JSONL:=${SHARED}/data/v11con_2400.jsonl}"
    : "${ACTMEM_HARD_FACT_BAG:=${SHARED}/data/actmemeval_implicit_hard_user_facts.jsonl}"
    ;;
esac

# The paper's selected AQUI adapter (Table 1 / Table 7 row 1).
: "${PAPER_ADAPTER:=${HPO_ROOT:-/nonexistent-on-${SERVER}}/adapters/hpo_v11con_2400_8B__lr5e-5_t0p02_gc1p0_wu0p1_bs128_r16}"
: "${EMBED_MODEL:=Qwen/Qwen3-Embedding-8B}"
# Bundled fallbacks (shipped inside this folder) if the server copy is missing.
: "${BUNDLED_HARD_JSON:=${SHARED}/data/ActMemEval_implicit_hard.rebuilt.json}"
[[ -f "${ACTMEM_HARD_FACT_BAG:-/nonexistent}" ]] || ACTMEM_HARD_FACT_BAG="${SHARED}/data/actmemeval_implicit_hard_user_facts.jsonl"
[[ -f "${V11CON_2400_JSONL:-/nonexistent}" ]] || V11CON_2400_JSONL="${SHARED}/data/v11con_2400.jsonl"

export PY VLLM_ENV HF_HOME CUDA_COMPAT_CACHE V03_ART V11CON_2400_JSONL HPO_ROOT IMPLEX_FACT_BAG \
  IMPLEX_SUPPORTIVE_FACT_BAG ACTMEM_HARD_JSON ACTMEM_HARD_FACT_BAG ACTMEM_ORIG_JSON ACTMEM_ORIG_FACT_BAG \
  SYM_ROOT ACTMEM_FACTS_ROOT GEPA_ANSWERGEN_ROOT JUDGE_ROOT_OLD OUT_BASE PAPER_ADAPTER EMBED_MODEL \
  BUNDLED_HARD_JSON

export PYTHONPATH="${SHARED}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

# CUDA forward-compat shim (SPACE: torch/vLLM are CUDA-13 builds on a CUDA-12.2 driver).
if [[ -n "${CUDA_COMPAT_CACHE:-}" ]]; then
  _compat_lib="$(find "${CUDA_COMPAT_CACHE}/cuda-compat-13" -name 'libcuda.so.*' 2>/dev/null | head -1 || true)"
  [[ -n "${_compat_lib}" ]] && export LD_LIBRARY_PATH="$(dirname "${_compat_lib}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  unset _compat_lib
fi

source "${SHARED}/lib.sh"

# Another person may run this package: fail early if the output / HF cache dirs are not writable.
if [[ "$SERVER" != unknown && "${CHECK_ONLY:-0}" != 1 ]]; then
  mkdir -p "$OUT_BASE" 2>/dev/null && [[ -w "$OUT_BASE" ]] || die "OUT_BASE not writable: $OUT_BASE -- set OUT_BASE=/a/dir/you/own (see HANDOFF.md)"
  [[ -w "$HF_HOME" ]] || warn "HF_HOME not writable ($HF_HOME): models already cached there still load, new downloads will fail -- set HF_HOME to your own cache if needed"
fi
