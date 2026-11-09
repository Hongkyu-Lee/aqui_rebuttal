#!/bin/bash
# =============================================================================
# E3 -- Seeds + paired significance for the E2 comparison.      GPUs: 8 x A100-80GB
#
# Re-trains the E2-selected config of EACH arm with extra training seeds (seed 42 is reused from
# E2), evaluates every seed on both benchmarks, then runs a paired bootstrap AQUI vs symmetric.
# The data split (val seed 42) is fixed; only the TRAINING seed changes (init of LoRA, shuffling,
# dropout).
#
#   bash E3/run.sh                      # seeds 43 44 (+42 from E2) = 3 seeds (decision D3)
#   EXTRA_SEEDS="43 44 45 46" bash E3/run.sh
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../shared/env.sh"
OUT="${OUT:-${OUT_BASE}/E3}"; LOG_DIR="${OUT}/logs"; mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1
STEPS="${STEPS:-1 2 3}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
cd "$SHARED"
require_server space          # see ../SERVERS.md
check_inputs PY:req V11CON_2400_JSONL:req IMPLEX_FACT_BAG:req ACTMEM_HARD_FACT_BAG:req ACTMEM_HARD_JSON:opt
EXTRA_SEEDS="${EXTRA_SEEDS:-43 44}"
SEL="${OUT_BASE}/E2/e2_selection.json"
require_file "$SEL" "run E2 first"
HARD_JSON="$(resolve_hard_json)"

# selected adapter dir name: <arm>__lr.._t.._gc.._wu.._bs.._r..__s42  -> recover HPs
hp_of() {  # $1=adapter path -> "lr t gc wu bs r"
  "$PY" - "$1" <<'PY'
import re, sys
m = re.search(r"lr(?P<lr>[0-9ep\-]+)_t(?P<t>[0-9p]+)_gc(?P<gc>[0-9p]+)_wu(?P<wu>[0-9p]+)_bs(?P<bs>\d+)_r(?P<r>\d+)", sys.argv[1])
print(*(m[k].replace("p", ".") for k in ("lr", "t", "gc", "wu", "bs", "r")))
PY
}
declare -A AD
for arm in aqui sym; do AD[$arm]="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['arms'][sys.argv[2]]['adapter'])" "$SEL" "$arm")"; done
log "E3 start | aqui=${AD[aqui]} | sym=${AD[sym]} | extra seeds=${EXTRA_SEEDS}"

seed_dir() { local base; base="$(basename "${AD[$1]}")"; echo "${OUT}/adapters/${base%__s*}__s${2}"; }

train_job() {  # $1=gpu $2=arm $3=seed
  local hp; hp=($(hp_of "${AD[$2]}"))
  train_lora "$2" "$1" "$(seed_dir "$2" "$3")" "${hp[@]}" "$3" > "${LOG_DIR}/train_${2}_s${3}.log" 2>&1
}
if want 1; then
  log "STEP 1: training extra seeds"
  pool_init
  for s in $EXTRA_SEEDS; do for arm in aqui sym; do pool_run "train:${arm}:s${s}" train_job "$arm" "$s"; done; done
  pool_wait
fi

test_job() {  # $1=gpu $2=arm $3=adapter $4=name
  eval_retrieval "$2" "$1" "$3" implexconv "${OUT}/test" "$4__implexconv.json" > "${LOG_DIR}/test_$4_implex.log" 2>&1 || return 1
  eval_retrieval "$2" "$1" "$3" actmem_hard "${OUT}/test" "$4__actmem_hard.json" > "${LOG_DIR}/test_$4_hard.log" 2>&1
}
if want 2; then
  log "STEP 2: evaluation (seed 42 results are copied from E2 when present)"
  mkdir -p "${OUT}/test"
  for arm in aqui sym; do for ds in implexconv actmem_hard; do
    src="$(ls "${OUT_BASE}/E2/test/selected_${arm}__"*"__${ds}.json" 2>/dev/null | head -1)"
    [[ -n "$src" ]] && cp -n "$src" "${OUT}/test/${arm}_s42__${ds}.json"
  done; done
  pool_init
  for arm in aqui sym; do
    [[ -f "${OUT}/test/${arm}_s42__implexconv.json" && -f "${OUT}/test/${arm}_s42__actmem_hard.json" ]] || \
      pool_run "test:${arm}:s42" test_job "$arm" "${AD[$arm]}" "${arm}_s42"
    for s in $EXTRA_SEEDS; do pool_run "test:${arm}:s${s}" test_job "$arm" "$(seed_dir "$arm" "$s")" "${arm}_s${s}"; done
  done
  pool_wait
fi

if want 3; then
  log "STEP 3: statistics"
  for ds in implexconv actmem_hard; do
    "$PY" -m tools.bootstrap --a "${OUT}"/test/aqui_s*__${ds}.json --b "${OUT}"/test/sym_s*__${ds}.json \
      --name-a AQUI --name-b symmetric --ks 10,20,30,50,100 --out "${OUT}/stats_${ds}.json" || warn "bootstrap failed for ${ds}"
  done
  { echo "# E3 summary ($(stamp))"; for ds in implexconv actmem_hard; do echo "## ${ds}"; echo '```'; cat "${OUT}/stats_${ds}.json" 2>/dev/null; echo '```'; done; } > "${OUT}/E3_summary.md"
  log "wrote ${OUT}/E3_summary.md"
fi
log "E3 done"
