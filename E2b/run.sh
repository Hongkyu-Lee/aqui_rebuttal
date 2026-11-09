#!/bin/bash
# =============================================================================
# E2b -- Why does freezing the fact space help? (Ap7K-W1: regularization vs. preserving knowledge)
# GPUs: 4 x A100-80GB (1 is enough for step 1; step 2 uses a pool)
#
# Needs E2 (reads ${OUT_BASE}/E2/e2_selection.json), or set SYM_ADAPTER / AQUI_ADAPTER.
#   1 [GPU] fact-space drift of the symmetric adapter on real memory facts (both benchmarks)
#   2 [GPU] EXPLICIT retrieval (no implicit reasoning needed) for pretrained / AQUI / symmetric:
#           ImplexConv-Supportive and the ORIGINAL ActMemEval (each only if its files exist)
#   3 [CPU] summary
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../shared/env.sh"
OUT="${OUT:-${OUT_BASE}/E2b}"; LOG_DIR="${OUT}/logs"; mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1
STEPS="${STEPS:-1 2 3}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
cd "$SHARED"
require_server space          # see ../SERVERS.md
check_inputs PY:req IMPLEX_FACT_BAG:req IMPLEX_SUPPORTIVE_FACT_BAG:opt ACTMEM_ORIG_JSON:opt ACTMEM_ORIG_FACT_BAG:opt

SEL="${OUT_BASE}/E2/e2_selection.json"
if [[ -z "${SYM_ADAPTER:-}" || -z "${AQUI_ADAPTER:-}" ]]; then
  require_file "$SEL" "run E2 first or set SYM_ADAPTER and AQUI_ADAPTER"
  SYM_ADAPTER="${SYM_ADAPTER:-$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['arms']['sym']['adapter'])" "$SEL")}"
  AQUI_ADAPTER="${AQUI_ADAPTER:-$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['arms']['aqui']['adapter'])" "$SEL")}"
fi
require_dir "$SYM_ADAPTER"; require_dir "$AQUI_ADAPTER"
log "E2b start | sym=${SYM_ADAPTER} | aqui=${AQUI_ADAPTER}"
HARD_JSON="$(resolve_hard_json)"

if want 1; then
  log "STEP 1: drift"
  FB=(--fact-bag "actmem_hard=${ACTMEM_HARD_FACT_BAG}")
  have_file "$IMPLEX_FACT_BAG" && FB+=(--fact-bag "implexconv=${IMPLEX_FACT_BAG}")
  CUDA_VISIBLE_DEVICES="$(gpu_list | awk '{print $1}')" "$PY" -m tools.drift "${FB[@]}" \
    --adapter "symmetric=${SYM_ADAPTER}" --n "${DRIFT_N:-20000}" --out "${OUT}/drift.json" || die "drift failed"
fi

explicit_job() {  # $1=gpu $2=arm $3=adapter $4=dataset-spec $5=name
  eval_retrieval "$2" "$1" "$3" "$4" "${OUT}/explicit" "$5.json" > "${LOG_DIR}/$5.log" 2>&1
}
if want 2; then
  log "STEP 2: explicit retrieval"
  declare -a DS=()
  have_file "$IMPLEX_SUPPORTIVE_FACT_BAG" && DS+=("implexconv_supportive|implexconv_supportive") \
    || warn "no ImplexConv-Supportive fact bag (${IMPLEX_SUPPORTIVE_FACT_BAG}) -> skipped (see AGENT.md)"
  if have_file "$ACTMEM_ORIG_JSON" && have_file "$ACTMEM_ORIG_FACT_BAG"; then
    DS+=("actmem_json:${ACTMEM_ORIG_JSON}:${ACTMEM_ORIG_FACT_BAG}|actmem_original")
  else warn "original ActMemEval JSON/fact bag missing -> skipped (see AGENT.md)"; fi
  [[ ${#DS[@]} -gt 0 ]] || die "no explicit-retrieval dataset available"
  pool_init
  for d in "${DS[@]}"; do
    IFS='|' read -r spec tag <<<"$d"
    pool_run "base:${tag}" explicit_job base - "$spec" "pretrained__${tag}"
  done
  pool_wait   # base first (fills the fact cache), then the adapters in parallel
  pool_init
  for d in "${DS[@]}"; do
    IFS='|' read -r spec tag <<<"$d"
    pool_run "aqui:${tag}" explicit_job aqui "$AQUI_ADAPTER" "$spec" "aqui__${tag}"
    pool_run "sym:${tag}" explicit_job sym "$SYM_ADAPTER" "$spec" "symmetric__${tag}"
  done
  pool_wait
fi

if want 3; then
  "$PY" -m tools.collect --glob "${OUT}/explicit/*.json" --ks 1,5,10,20,50 --out "${OUT}/table_explicit"
  { echo "# E2b summary ($(stamp))"; echo; echo "## Fact-space drift (symmetric adapter vs frozen base)"; echo '```'
    cat "${OUT}/drift.json" 2>/dev/null; echo '```'; echo "## Explicit retrieval (session recall)"; cat "${OUT}/table_explicit.md" 2>/dev/null; } > "${OUT}/E2b_summary.md"
  log "wrote ${OUT}/E2b_summary.md"
fi
log "E2b done"
