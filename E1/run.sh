#!/bin/bash
# =============================================================================
# E1 -- Clean model selection (synthetic validation only).   GPUs: 8 x A100-80GB (works with 1, slower)
#
#   bash E1/run.sh                 # everything
#   STEPS="1 2" bash E1/run.sh     # only some steps
#   GPUS="0 1 2 3" bash E1/run.sh  # restrict GPUs
#
# Steps
#   1  [CPU] train/val split audit: memory grouping + text overlap across the split (2KMq-Q2b)
#   2  [CPU] verify the rebuilt training loader against the paper adapter's train_data_manifest.json
#   3  [GPU] SANITY: re-evaluate the paper adapter on ImplexConv-Opposed (must give R@10 ~= 0.3322)
#            and on ActMemEval-hard; this also fills the shared base-model fact-embedding cache
#   4  [GPU] synthetic-validation retrieval score of every HPO adapter (216) + the base model
#   5  [CPU] re-select the config by synthetic validation only (hpo_reselect)
#   6  [GPU] evaluate the re-selected config(s) on both test benchmarks (test numbers reported, never used)
#   7  [CPU] summary table -> ${OUT}/E1_summary.md
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../shared/env.sh"
OUT="${OUT:-${OUT_BASE}/E1}"; LOG_DIR="${OUT}/logs"; mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1
STEPS="${STEPS:-1 2 3 4 5 6 7}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
cd "$SHARED"
require_server space          # see ../SERVERS.md
check_inputs PY:req HPO_ROOT:req PAPER_ADAPTER:req V11CON_2400_JSONL:req IMPLEX_FACT_BAG:req ACTMEM_HARD_FACT_BAG:req ACTMEM_HARD_JSON:opt
log "E1 start | server=${SERVER} | out=${OUT} | gpus=$(gpu_list)"

# ---------------- preflight
[[ -x "$PY" ]] || die "python not found: PY=$PY"
require_file "$V11CON_2400_JSONL" "synthetic training data"
require_dir "${HPO_ROOT}/adapters" "HPO adapters (set HPO_ROOT)"
N_AD=$(ls -d "${HPO_ROOT}"/adapters/hpo_* 2>/dev/null | wc -l)
log "found ${N_AD} HPO adapter dirs under ${HPO_ROOT}/adapters (expected 216)"
[[ -f "${HPO_ROOT}/HPO_ranking.csv" ]] || warn "HPO_ranking.csv missing -> test recalls of non-selected configs will be blank"
HARD_JSON="$(resolve_hard_json)"; log "ActMemEval-hard JSON: ${HARD_JSON}"

# ---------------- 1. split audit
if want 1; then
  log "STEP 1: split audit"
  "$PY" -m tools.synthetic_val overlap --data "$V11CON_2400_JSONL" --out "${OUT}/split_audit.json" \
      --clean-val-out "${OUT}/val_rows_overlap_free.jsonl" || die "split audit failed"
fi

# ---------------- 2. loader check
if want 2; then
  log "STEP 2: loader vs original manifest"
  if [[ -f "${PAPER_ADAPTER}/train_data_manifest.json" ]]; then
    "$PY" -m tools.check_loader_against_manifest --data "$V11CON_2400_JSONL" \
        --manifest "${PAPER_ADAPTER}/train_data_manifest.json" | tee "${OUT}/loader_check.json" \
      || warn "loader does NOT match the original manifest -- see AGENT.md before trusting E2/E3"
  else
    warn "no ${PAPER_ADAPTER}/train_data_manifest.json -> cannot verify the rebuilt loader"
  fi
fi

# ---------------- 3. sanity reproduction of Table 1
if want 3; then
  log "STEP 3: reproduce paper adapter (Table 1: ImplexConv R@10=0.3322 R@20=0.4588 R@100=0.8412)"
  require_dir "$PAPER_ADAPTER" "paper adapter"
  G="$(gpu_list | awk '{print $1}')"
  eval_retrieval aqui "$G" "$PAPER_ADAPTER" implexconv "${OUT}/test" paper_adapter__implexconv.json || die "eval failed"
  eval_retrieval aqui "$G" "$PAPER_ADAPTER" actmem_hard "${OUT}/test" paper_adapter__actmem_hard.json || die "eval failed"
  eval_retrieval base "$G" - implexconv "${OUT}/test" pretrained__implexconv.json || warn "base eval failed"
  eval_retrieval base "$G" - actmem_hard "${OUT}/test" pretrained__actmem_hard.json || warn "base eval failed"
  "$PY" - "${OUT}/test/paper_adapter__implexconv.json" <<'PY' || { [[ "${STRICT_REPRO:-1}" == 1 ]] && die "paper number NOT reproduced (see AGENT.md); STRICT_REPRO=0 to continue anyway"; }
import json, sys
s = json.load(open(sys.argv[1]))["summary"]
got = {k: round(s[f"session_recall@{k}"], 4) for k in (10, 20, 100)}
ref = {10: 0.3322, 20: 0.4588, 100: 0.8412}
print("reproduction:", got, "paper:", ref)
sys.exit(0 if all(abs(got[k] - ref[k]) <= 0.01 for k in ref) else 1)
PY
fi

# ---------------- 4. synthetic validation retrieval for all adapters
val_chunk() {  # $1=gpu $2=list file $3=out csv
  local gpu="$1" list="$2" out="$3" a; local -a args=()
  while read -r a; do [[ -n "$a" ]] && args+=(--adapter "aqui:${a}"); done < "$list"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m tools.synthetic_val eval --data "$V11CON_2400_JSONL" \
      --out "$out" --overwrite --cache-dir "${OUT}/cache" "${args[@]}" > "${LOG_DIR}/$(basename "$out" .csv).log" 2>&1
}
if want 4; then
  log "STEP 4: synthetic-validation retrieval for ${N_AD} adapters"
  mkdir -p "${OUT}/val_chunks"
  G0="$(gpu_list | awk '{print $1}')"
  # base model first (also writes the base fact cache that every chunk reuses)
  CUDA_VISIBLE_DEVICES="$G0" "$PY" -m tools.synthetic_val eval --data "$V11CON_2400_JSONL" \
      --out "${OUT}/val_chunks/base.csv" --overwrite --cache-dir "${OUT}/cache" --adapter "base:-" || die "base val eval failed"
  ls -d "${HPO_ROOT}"/adapters/hpo_* > "${OUT}/val_chunks/all_adapters.txt"
  NG=$(gpu_count)
  split -n "l/${NG}" -d "${OUT}/val_chunks/all_adapters.txt" "${OUT}/val_chunks/list_"
  pool_init
  for lst in "${OUT}"/val_chunks/list_*; do
    [[ -s "$lst" ]] && pool_run "val:$(basename "$lst")" val_chunk "$lst" "${OUT}/val_chunks/$(basename "$lst").csv"
  done
  pool_wait
  "$PY" - "${OUT}/val_chunks" "${OUT}/synthetic_val_all.csv" <<'PY'
import csv, glob, sys
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/list_*.csv")) + [sys.argv[1] + "/base.csv"]:
    rows += list(csv.DictReader(open(f)))
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"merged {len(rows)} rows -> {sys.argv[2]}")
PY
fi

# ---------------- 5. re-selection
if want 5; then
  log "STEP 5: re-select by synthetic validation"
  "$PY" -m tools.hpo_reselect --hpo-root "$HPO_ROOT" --val-csv "${OUT}/synthetic_val_all.csv" --out-dir "${OUT}" \
    || die "hpo_reselect failed"
fi

# ---------------- 6. test evaluation of selected configs
test_one() {  # $1=gpu $2=run name
  local gpu="$1" run="$2"
  eval_retrieval aqui "$gpu" "${HPO_ROOT}/adapters/${run}" implexconv "${OUT}/test" "${run}__implexconv.json" \
    > "${LOG_DIR}/test_${run}_implex.log" 2>&1 || return 1
  eval_retrieval aqui "$gpu" "${HPO_ROOT}/adapters/${run}" actmem_hard "${OUT}/test" "${run}__actmem_hard.json" \
    > "${LOG_DIR}/test_${run}_hard.log" 2>&1
}
if want 6; then
  log "STEP 6: test-benchmark evaluation of the re-selected config(s)"
  require_file "${OUT}/configs_to_test.txt"
  pool_init
  while read -r run; do [[ -n "$run" ]] && pool_run "test:${run}" test_one "$run"; done < "${OUT}/configs_to_test.txt"
  pool_wait
fi

# ---------------- 7. summary
if want 7; then
  log "STEP 7: summary"
  "$PY" -m tools.collect --glob "${OUT}/test/*__implexconv.json" --ks 10,20,30,50,92,100,151 --out "${OUT}/table_implexconv"
  "$PY" -m tools.collect --glob "${OUT}/test/*__actmem_hard.json" --ks 10,14,20,30,50,68,100 --out "${OUT}/table_actmem_hard"
  {
    echo "# E1 summary ($(stamp))"; echo
    echo "## Split audit"; echo '```'; cat "${OUT}/split_audit.json" 2>/dev/null; echo '```'
    echo "## Selection (synthetic validation only)"; echo '```'; cat "${OUT}/selection.json" 2>/dev/null; echo '```'
    echo "## Test: ImplexConv-Opposed"; cat "${OUT}/table_implexconv.md" 2>/dev/null
    echo "## Test: ActMemEval-hard"; cat "${OUT}/table_actmem_hard.md" 2>/dev/null
  } > "${OUT}/E1_summary.md"
  log "wrote ${OUT}/E1_summary.md"
fi
log "E1 done"
