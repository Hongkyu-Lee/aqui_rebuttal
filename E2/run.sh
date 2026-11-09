#!/bin/bash
# =============================================================================
# E2 -- Matched symmetric-LoRA control.            GPUs: 8 x A100-80GB (6 training jobs run in parallel)
#
# Two arms, IDENTICAL except which side the LoRA touches:
#   aqui : query-only LoRA, facts encoded by the frozen base           (src.entry_points.train_embedding_model)
#   sym  : LoRA on query AND facts (positives/negatives re-encoded)    (baseline_symmetric.train_symmetric)
# Same base model, rank, alpha, target modules (all-linear), data (v11con_2400), negatives
# (general_fact), loss (multi-positive InfoNCE + 0.25 alignment), val split, early stopping, seed.
# Tuning budget (decision D2): the SAME 2-3 configs for both arms; each arm's config is picked by
# SYNTHETIC validation retrieval (val_mean_hit), never by benchmark data.
#
#   bash E2/run.sh
#   CONFIGS="5e-5|0.02|1.0|0.1|128|16 2e-5|0.1|1.0|0.1|128|32" bash E2/run.sh
#   STEPS="4 5" bash E2/run.sh
#
# Steps
#   1 [CPU] loader check against the paper adapter's manifest (warn only)
#   2 [GPU] train len(CONFIGS) x 2 arms (seed 42)
#   3 [GPU] synthetic-validation retrieval of every trained adapter (sym: facts through the adapter)
#   4 [CPU] pick the best config per arm -> ${OUT}/e2_selection.json (read by E2b / E3 / E4 / E5)
#   5 [GPU] test evaluation of the picked adapters + pretrained on ImplexConv-Opposed and ActMemEval-hard
#           (EVAL_ALL=1 evaluates every trained adapter, not only the picked ones)
#   6 [CPU] summary
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../shared/env.sh"
OUT="${OUT:-${OUT_BASE}/E2}"; LOG_DIR="${OUT}/logs"; mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1
STEPS="${STEPS:-1 2 3 4 5 6}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
cd "$SHARED"
require_server space          # see ../SERVERS.md
check_inputs PY:req V11CON_2400_JSONL:req IMPLEX_FACT_BAG:req ACTMEM_HARD_FACT_BAG:req PAPER_ADAPTER:opt ACTMEM_HARD_JSON:opt

# lr|temperature|max_grad_norm|warmup|batch|lora_r
#  C1 = the paper's AQUI config (Table 7 row 1)
#  C2 = best-@10 config of the original HPO grid
#  C3 = the default config of the original symmetric-baseline script
CONFIGS="${CONFIGS:-5e-5|0.02|1.0|0.1|128|16 2e-5|0.1|1.0|0.1|128|32 1e-4|0.02|1.0|0.05|64|32}"
# If E1 picked a different config by synthetic validation, it replaces C3 (keeps the budget at 3).
E1_SEL="${OUT_BASE}/E1/selection.json"
if [[ -f "$E1_SEL" && "${USE_E1_PICK:-1}" == 1 ]]; then
  PICK="$("$PY" - "$E1_SEL" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["criteria"]
p = next((v for k, v in s.items() if k.startswith("PRIMARY")), None)
if p: print(f"{p['lr'].replace('e-0','e-')}|{p['temperature']}|{p['max_grad_norm']}|{p['warmup']}|{p['batch']}|{p['lora_r']}")
PY
)"
  if [[ -n "$PICK" && " $CONFIGS " != *" $PICK "* ]]; then
    read -r -a _C <<<"$CONFIGS"; _C[2]="$PICK"; CONFIGS="${_C[*]}"
    log "E1 synthetic-validation pick ${PICK} replaces C3"
  fi
fi
SEED="${SEED:-42}"
ARMS="${ARMS:-aqui sym}"
log "E2 start | server=${SERVER} | out=${OUT} | configs=${CONFIGS} | arms=${ARMS} | gpus=$(gpu_list)"
[[ -x "$PY" ]] || die "python not found: PY=$PY"
require_file "$V11CON_2400_JSONL"
HARD_JSON="$(resolve_hard_json)"

run_dir() { echo "${OUT}/adapters/${1}__$(cfg_tag $(tr '|' ' ' <<<"$2"))__s${3}"; }   # arm cfg seed

if want 1; then
  if [[ -f "${PAPER_ADAPTER}/train_data_manifest.json" ]]; then
    "$PY" -m tools.check_loader_against_manifest --data "$V11CON_2400_JSONL" \
      --manifest "${PAPER_ADAPTER}/train_data_manifest.json" > "${OUT}/loader_check.json" 2>&1 \
      && log "loader check: MATCH" || warn "loader check: MISMATCH (see ${OUT}/loader_check.json and AGENT.md)"
  else warn "cannot run loader check (no paper manifest)"; fi
fi

train_job() {  # $1=gpu $2=arm $3=cfg
  local gpu="$1" arm="$2" cfg="$3" lr t gc wu bs r
  IFS='|' read -r lr t gc wu bs r <<<"$cfg"
  train_lora "$arm" "$gpu" "$(run_dir "$arm" "$cfg" "$SEED")" "$lr" "$t" "$gc" "$wu" "$bs" "$r" "$SEED" \
    > "${LOG_DIR}/train_$(basename "$(run_dir "$arm" "$cfg" "$SEED")").log" 2>&1
}
if want 2; then
  log "STEP 2: training"
  pool_init
  for arm in $ARMS; do for cfg in $CONFIGS; do pool_run "train:${arm}:${cfg}" train_job "$arm" "$cfg"; done; done
  pool_wait
fi

val_job() {  # $1=gpu $2=arm $3=dir
  CUDA_VISIBLE_DEVICES="$1" "$PY" -m tools.synthetic_val eval --data "$V11CON_2400_JSONL" --overwrite \
    --cache-dir "${OUT}/cache" --adapter "${2}:${3}" --out "${OUT}/val/$(basename "$3").csv" \
    > "${LOG_DIR}/val_$(basename "$3").log" 2>&1
}
if want 3; then
  log "STEP 3: synthetic validation"
  mkdir -p "${OUT}/val"
  pool_init
  for arm in $ARMS; do for cfg in $CONFIGS; do
    d="$(run_dir "$arm" "$cfg" "$SEED")"
    [[ -f "${d}/adapter_config.json" ]] && pool_run "val:$(basename "$d")" val_job "$arm" "$d" || warn "no adapter: $d"
  done; done
  pool_wait
fi

if want 4; then
  log "STEP 4: per-arm selection"
  "$PY" - "${OUT}/val" "${OUT}/e2_selection.json" <<'PY' || die "selection failed"
import csv, glob, json, sys
rows = [r for f in sorted(glob.glob(sys.argv[1] + "/*.csv")) for r in csv.DictReader(open(f))]
sel = {"criterion": "val_mean_hit (synthetic validation retrieval)", "arms": {}, "all": rows}
for arm in sorted({r["arm"] for r in rows}):
    best = max((r for r in rows if r["arm"] == arm), key=lambda r: float(r["val_mean_hit"]))
    sel["arms"][arm] = {"adapter": best["adapter"], "val_mean_hit": float(best["val_mean_hit"]),
                        "best_eval_loss": best["best_eval_loss"]}
json.dump(sel, open(sys.argv[2], "w"), indent=2)
print(json.dumps(sel["arms"], indent=2))
PY
fi

test_job() {  # $1=gpu $2=arm $3=adapter $4=name
  local gpu="$1" arm="$2" ad="$3" name="$4"
  eval_retrieval "$arm" "$gpu" "$ad" implexconv "${OUT}/test" "${name}__implexconv.json" > "${LOG_DIR}/test_${name}_implex.log" 2>&1 || return 1
  eval_retrieval "$arm" "$gpu" "$ad" actmem_hard "${OUT}/test" "${name}__actmem_hard.json" > "${LOG_DIR}/test_${name}_hard.log" 2>&1
}
if want 5; then
  log "STEP 5: test evaluation"
  require_file "${OUT}/e2_selection.json"
  # pretrained first, alone: it fills the shared base fact cache used by every AQUI eval
  test_job "$(gpu_list | awk '{print $1}')" base - pretrained || warn "pretrained eval failed"
  pool_init
  if [[ "${EVAL_ALL:-0}" == 1 ]]; then
    for arm in $ARMS; do for cfg in $CONFIGS; do d="$(run_dir "$arm" "$cfg" "$SEED")"
      [[ -f "${d}/adapter_config.json" ]] && pool_run "test:$(basename "$d")" test_job "$arm" "$d" "$(basename "$d")"; done; done
  else
    while IFS='|' read -r arm ad; do
      pool_run "test:${arm}" test_job "$arm" "$ad" "selected_${arm}__$(basename "$ad")"
    done < <("$PY" -c "import json,sys;[print(f'{a}|{v[\"adapter\"]}') for a,v in json.load(open(sys.argv[1]))['arms'].items()]" "${OUT}/e2_selection.json")
  fi
  pool_wait
fi

if want 6; then
  "$PY" -m tools.collect --glob "${OUT}/test/*__implexconv.json" --ks 10,20,30,50,92,100,151 --out "${OUT}/table_implexconv"
  "$PY" -m tools.collect --glob "${OUT}/test/*__actmem_hard.json" --ks 10,14,20,30,50,68,100 --out "${OUT}/table_actmem_hard"
  "$PY" -m tools.collect --glob "${OUT}/test/*__implexconv.json" --metric fact_recall --ks 10,20,50,100 --out "${OUT}/table_implexconv_factrecall"
  { echo "# E2 summary ($(stamp))"; echo; echo "configs: ${CONFIGS}"; echo
    echo "## Selection (synthetic validation)"; echo '```'; "$PY" -c "import json;print(json.dumps(json.load(open('${OUT}/e2_selection.json'))['arms'],indent=2))"; echo '```'
    echo "## ImplexConv-Opposed (session recall)"; cat "${OUT}/table_implexconv.md"
    echo "## ActMemEval-hard (session recall)"; cat "${OUT}/table_actmem_hard.md"; } > "${OUT}/E2_summary.md"
  log "wrote ${OUT}/E2_summary.md"
fi
log "E2 done"
