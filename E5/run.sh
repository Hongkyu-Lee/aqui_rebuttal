#!/bin/bash
# =============================================================================
# E5 -- Recall definitions: "any fact from the gold session" vs "the specific fact needed".
# GPUs: 8 x A100-80GB (4 for the labeling LLM + 4 for retrieval runs that are not already available)
#
#   1 [GPU] retrieval result JSONs for pretrained / AQUI / symmetric on both benchmarks
#           (reused from E2 or E1 if they exist; otherwise computed here)
#   2 [GPU] LLM labels the EVIDENCE fact(s) among the gold-session facts of every question
#   3 [CPU] session recall vs fact recall vs evidence hit/recall table, per method and benchmark
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../shared/env.sh"
OUT="${OUT:-${OUT_BASE}/E5}"; LOG_DIR="${OUT}/logs"; mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1
STEPS="${STEPS:-1 2 3}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
cd "$SHARED"
require_server space          # see ../SERVERS.md
check_inputs PY:req PAPER_ADAPTER:req IMPLEX_FACT_BAG:req ACTMEM_HARD_FACT_BAG:req VLLM_ENV:opt
trap vllm_stop_all EXIT
LABEL_BACKEND="${LABEL_BACKEND:-local}"
LABEL_MODEL="${LABEL_MODEL:-Qwen/Qwen3-Next-80B-A3B-Instruct}"
LABEL_PORT="${LABEL_PORT:-8200}"; LABEL_GPUS="${LABEL_GPUS:-0,1,2,3}"; LABEL_TP="${LABEL_TP:-4}"
HARD_JSON="$(resolve_hard_json)"
mkdir -p "${OUT}/results"

AQUI_AD="${AQUI_ADAPTER:-$PAPER_ADAPTER}"
SYM_AD="${SYM_ADAPTER:-}"
SEL="${OUT_BASE}/E2/e2_selection.json"
[[ -z "$SYM_AD" && -f "$SEL" ]] && SYM_AD="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['arms']['sym']['adapter'])" "$SEL")"

# find an existing result: E2 selected > E1 paper adapter; else compute
find_result() {  # $1=method $2=ds
  case "$1" in
    pretrained) ls "${OUT_BASE}/E2/test/pretrained__$2.json" "${OUT_BASE}/E1/test/pretrained__$2.json" 2>/dev/null | head -1 ;;
    aqui) [[ -z "${AQUI_ADAPTER:-}" ]] && ls "${OUT_BASE}/E1/test/paper_adapter__$2.json" 2>/dev/null | head -1 ;;
    sym)  ls "${OUT_BASE}/E2/test/selected_sym__"*"__$2.json" 2>/dev/null | head -1 ;;
  esac
}
retr_job() {  # $1=gpu $2=method $3=adapter $4=ds
  local arm="$2"; [[ "$arm" == pretrained ]] && arm=base
  eval_retrieval "$arm" "$1" "$3" "$4" "${OUT}/results" "$2__$4.json" > "${LOG_DIR}/retr_$2_$4.log" 2>&1
}

label_pid=""
if want 2; then
  if [[ "$LABEL_BACKEND" == local ]]; then
    vllm_start labeler "$LABEL_MODEL" "$LABEL_PORT" "$LABEL_GPUS" "$LABEL_TP" 16384 --served-model-name "$LABEL_MODEL"
    wait_vllm "$LABEL_PORT" "$LABEL_MODEL" "${LOG_DIR}/vllm_labeler.pid"
    LABEL_BASE_URL="http://localhost:${LABEL_PORT}/v1"; LABEL_API_KEY_ENV="${LABEL_API_KEY_ENV:-OPENAI_API_KEY}"
  else
    : "${LABEL_BASE_URL:?set LABEL_BASE_URL for LABEL_BACKEND=api}"; LABEL_API_KEY_ENV="${LABEL_API_KEY_ENV:-TML_API_KEY}"
  fi
  ( "$PY" -m tools.evidence_label --data implexconv --fact-bag "$IMPLEX_FACT_BAG" --model "$LABEL_MODEL" \
        --base-url "$LABEL_BASE_URL" --api-key-env "$LABEL_API_KEY_ENV" --out "${OUT}/evidence_implexconv" \
    && "$PY" -m tools.evidence_label --data "actmemeval:${HARD_JSON}" --fact-bag "$ACTMEM_HARD_FACT_BAG" \
        --model "$LABEL_MODEL" --base-url "$LABEL_BASE_URL" --api-key-env "$LABEL_API_KEY_ENV" --out "${OUT}/evidence_actmem_hard" \
  ) > "${LOG_DIR}/label.log" 2>&1 &
  label_pid=$!
fi

if want 1; then
  log "STEP 1: retrieval results"
  [[ "$LABEL_BACKEND" == local && " $STEPS " == *" 2 "* ]] && GPUS="${RETR_GPUS:-4 5 6 7}"
  pool_init
  for ds in implexconv actmem_hard; do
    for m in pretrained aqui sym; do
      r="$(find_result "$m" "$ds")"
      if [[ -n "$r" ]]; then cp -n "$r" "${OUT}/results/${m}__${ds}.json"; log "reuse ${r}"; continue; fi
      case "$m" in
        pretrained) pool_run "retr:$m:$ds" retr_job pretrained - "$ds" ;;
        aqui) pool_run "retr:$m:$ds" retr_job aqui "$AQUI_AD" "$ds" ;;
        sym) [[ -n "$SYM_AD" ]] && pool_run "retr:$m:$ds" retr_job sym "$SYM_AD" "$ds" || warn "no symmetric adapter -> skipped" ;;
      esac
    done
  done
  pool_wait
fi

if [[ -n "$label_pid" ]]; then log "waiting for evidence labeling"; wait "$label_pid" || die "labeling failed (${LOG_DIR}/label.log)"; fi
vllm_stop_all

if want 3; then
  for ds in implexconv actmem_hard; do
    args=(); for m in pretrained aqui sym; do [[ -f "${OUT}/results/${m}__${ds}.json" ]] && args+=(--result "${m}=${OUT}/results/${m}__${ds}.json"); done
    ks="10,20,30,50,100"; [[ "$ds" == actmem_hard ]] && ks="10,14,20,30,50,68"
    "$PY" -m tools.recall_definitions --evidence "${OUT}/evidence_${ds}.jsonl" "${args[@]}" --ks "$ks" --out "${OUT}/table_${ds}" \
      || warn "recall table failed for ${ds}"
  done
  { echo "# E5 summary ($(stamp))"; for ds in implexconv actmem_hard; do echo "## ${ds}"; echo '```'; cat "${OUT}/table_${ds}.csv" 2>/dev/null; echo '```'; done; } > "${OUT}/E5_summary.md"
  log "wrote ${OUT}/E5_summary.md"
fi
log "E5 done"
