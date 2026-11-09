#!/bin/bash
# =============================================================================
# E4 -- ActMemEval-hard: exclusion funnel, independent validation, human check, paired results.
# GPUs: 8 x A100-80GB (4 for the local judge LLM + 4 for retrieval). With JUDGE_BACKEND=api: 4.
#
#   bash E4/run.sh
#   JUDGE_BACKEND=api JUDGE_BASE_URL=https://<tml-endpoint>/v1 JUDGE_MODEL=<model-id> \
#       JUDGE_API_KEY_ENV=TML_API_KEY bash E4/run.sh          # Thinking Machines Lab (or any OpenAI-compatible API)
#
# Steps
#   1 [CPU] rebuild + verify ActMemEval-hard (72 rows) and write the 247 -> 224 -> 72 funnel table
#   2 [GPU/API] independent judge (different model family than gpt-5.4-mini) re-checks every session rewrite
#   3 [CPU] human-check sheet: 50 random gold-session rewrites from the kept 72 rows.
#           OFF by default (not run unless asked): HUMAN_CHECK=1 bash E4/run.sh   or   STEPS="3" bash E4/run.sh
#   4 [GPU] paired retrieval on the SAME 72 questions: original / lexical-controlled / implicit-hard
#           x {pretrained, AQUI, symmetric(E2, if available)}
#   5 [CPU] summary
#   (After annotation:  python -m tools.audit_sample --score E4/out/human_check_filled.csv)
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../shared/env.sh"
OUT="${OUT:-${OUT_BASE}/E4}"; LOG_DIR="${OUT}/logs"; mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1
STEPS="${STEPS:-1 2 4 5}"             # step 3 (human-check sheet) is opt-in
[[ "${HUMAN_CHECK:-0}" == 1 && " $STEPS " != *" 3 "* ]] && STEPS="$STEPS 3"
want() { [[ " $STEPS " == *" $1 "* ]]; }
cd "$SHARED"
require_server space          # see ../SERVERS.md
check_inputs PY:req PAPER_ADAPTER:req ACTMEM_HARD_FACT_BAG:req ACTMEM_HARD_JSON:opt ACTMEM_ORIG_JSON:opt ACTMEM_ORIG_FACT_BAG:opt VLLM_ENV:opt
trap vllm_stop_all EXIT
D="${SHARED}/data"

JUDGE_BACKEND="${JUDGE_BACKEND:-local}"                 # local (vLLM) | api
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
JUDGE_PORT="${JUDGE_PORT:-8100}"
JUDGE_GPUS="${JUDGE_GPUS:-0,1,2,3}"; JUDGE_TP="${JUDGE_TP:-4}"
RETR_GPUS="${RETR_GPUS:-4 5 6 7}"
ORIG_ARG=(); have_file "$ACTMEM_ORIG_JSON" && ORIG_ARG=(--original "$ACTMEM_ORIG_JSON")
[[ ${#ORIG_ARG[@]} -eq 0 ]] && warn "original ActMemEval.json not found (${ACTMEM_ORIG_JSON}); judge (and human sheet, if enabled) use the lexical-controlled session as reference"

# ---------------- 1
if want 1; then
  log "STEP 1: rebuild + funnel"
  "$PY" tools/rebuild_actmemeval_hard.py --lexical "$D/ActMemEval_lexical_controlled.json" \
    --rewrites "$D/implicit_hard_rewrites.jsonl" --metadata "$D/ActMemEval_implicit_hard_metadata.json" \
    --fact-bag "$D/actmemeval_implicit_hard_user_facts.jsonl" --out "${OUT}/ActMemEval_implicit_hard.rebuilt.json" \
    || die "rebuild/verification failed"
  if have_file "$ACTMEM_HARD_JSON" && [[ -s "$ACTMEM_HARD_JSON" ]]; then
    "$PY" - "$ACTMEM_HARD_JSON" "${OUT}/ActMemEval_implicit_hard.rebuilt.json" <<'PY' || warn "server copy differs from rebuild (see AGENT.md)"
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
same = len(a) == len(b) and all(x["question_id"] == y["question_id"] and x["haystack_sessions"] == y["haystack_sessions"] for x, y in zip(a, b))
print("server ActMemEval_implicit_hard.json == rebuild:", same); sys.exit(0 if same else 1)
PY
  fi
  "$PY" tools/funnel.py --lexical-meta "$D/ActMemEval_lexical_controlled_metadata.json" \
    --hard-meta "$D/ActMemEval_implicit_hard_metadata.json" --out "${OUT}/funnel"
fi
HARD_JSON="${OUT}/ActMemEval_implicit_hard.rebuilt.json"; [[ -f "$HARD_JSON" ]] || HARD_JSON="$(resolve_hard_json)"

# ---------------- 2 (judge runs in the background while step 4 uses the other GPUs)
judge_pid=""
if want 2; then
  log "STEP 2: independent judge (${JUDGE_BACKEND}: ${JUDGE_MODEL})"
  if [[ "$JUDGE_BACKEND" == local ]]; then
    vllm_start judge "$JUDGE_MODEL" "$JUDGE_PORT" "$JUDGE_GPUS" "$JUDGE_TP" 16384 --served-model-name "$JUDGE_MODEL"
    wait_vllm "$JUDGE_PORT" "$JUDGE_MODEL" "${LOG_DIR}/vllm_judge.pid"
    chat_ok "$JUDGE_PORT" "$JUDGE_MODEL" || die "judge endpoint does not answer"
    JUDGE_BASE_URL="http://localhost:${JUDGE_PORT}/v1"; JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-OPENAI_API_KEY}"
  else
    : "${JUDGE_BASE_URL:?set JUDGE_BASE_URL for JUDGE_BACKEND=api}"
    JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-TML_API_KEY}"
  fi
  ( "$PY" -m tools.rewrite_judge --metadata "$D/ActMemEval_implicit_hard_metadata.json" \
      --rewrites "$D/implicit_hard_rewrites.jsonl" --lexical "$D/ActMemEval_lexical_controlled.json" \
      --hard "$HARD_JSON" "${ORIG_ARG[@]}" --model "$JUDGE_MODEL" --base-url "$JUDGE_BASE_URL" \
      --api-key-env "$JUDGE_API_KEY_ENV" --scope all --workers "${JUDGE_WORKERS:-16}" \
      --out "${OUT}/independent_judge" > "${LOG_DIR}/judge.log" 2>&1 ) &
  judge_pid=$!
fi

# ---------------- 3
if want 3; then
  log "STEP 3: human-check sheet"
  "$PY" -m tools.audit_sample --metadata "$D/ActMemEval_implicit_hard_metadata.json" --rewrites "$D/implicit_hard_rewrites.jsonl" \
    --lexical "$D/ActMemEval_lexical_controlled.json" --hard "$HARD_JSON" "${ORIG_ARG[@]}" --n 50 --out "${OUT}/human_check"
fi

# ---------------- 4
paired_job() {  # $1=gpu $2=arm $3=adapter $4=dataset-spec $5=name
  eval_retrieval "$2" "$1" "$3" "$4" "${OUT}/paired" "$5.json" > "${LOG_DIR}/paired_$5.log" 2>&1
}
if want 4; then
  log "STEP 4: paired retrieval on the same 72 questions"
  mkdir -p "${OUT}/subsets"
  declare -a VERS=("hard|actmem_json:${HARD_JSON}:${ACTMEM_HARD_FACT_BAG}")
  "$PY" -m tools.make_subset --dataset "$D/ActMemEval_lexical_controlled.json" \
    --fact-bag "$D/actmemeval_lexical_controlled_user_facts.jsonl" --ids-from "$HARD_JSON" \
    --out-dataset "${OUT}/subsets/lexical72.json" --out-fact-bag "${OUT}/subsets/lexical72_facts.jsonl" \
    && VERS+=("lexical|actmem_json:${OUT}/subsets/lexical72.json:${OUT}/subsets/lexical72_facts.jsonl")
  if have_file "$ACTMEM_ORIG_JSON" && have_file "$ACTMEM_ORIG_FACT_BAG"; then
    "$PY" -m tools.make_subset --dataset "$ACTMEM_ORIG_JSON" --fact-bag "$ACTMEM_ORIG_FACT_BAG" --ids-from "$HARD_JSON" \
      --out-dataset "${OUT}/subsets/original72.json" --out-fact-bag "${OUT}/subsets/original72_facts.jsonl" \
      && VERS+=("original|actmem_json:${OUT}/subsets/original72.json:${OUT}/subsets/original72_facts.jsonl")
  else
    warn "original ActMemEval + fact bag not found -> 'original' column skipped (see AGENT.md)"
  fi
  AQUI_AD="${AQUI_ADAPTER:-$PAPER_ADAPTER}"
  SYM_AD="${SYM_ADAPTER:-}"
  [[ -z "$SYM_AD" && -f "${OUT_BASE}/E2/e2_selection.json" ]] && \
    SYM_AD="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['arms']['sym']['adapter'])" "${OUT_BASE}/E2/e2_selection.json")"
  OLD_GPUS="${GPUS:-}"; GPUS="$RETR_GPUS"
  [[ " $STEPS " == *" 2 "* && "$JUDGE_BACKEND" == local ]] || GPUS="${OLD_GPUS:-$(gpu_list)}"
  pool_init
  for v in "${VERS[@]}"; do
    IFS='|' read -r tag spec <<<"$v"
    pool_run "base:${tag}" paired_job base - "$spec" "pretrained__${tag}"
    have_file "${AQUI_AD}/adapter_config.json" && pool_run "aqui:${tag}" paired_job aqui "$AQUI_AD" "$spec" "aqui__${tag}"
    [[ -n "$SYM_AD" ]] && pool_run "sym:${tag}" paired_job sym "$SYM_AD" "$spec" "symmetric__${tag}"
  done
  pool_wait
  GPUS="$OLD_GPUS"
fi

if [[ -n "$judge_pid" ]]; then log "waiting for the judge to finish"; wait "$judge_pid" || warn "judge failed (see ${LOG_DIR}/judge.log)"; fi
vllm_stop_all

# ---------------- 5
if want 5; then
  "$PY" -m tools.collect --glob "${OUT}/paired/*.json" --ks 10,14,20,30,50,68 --out "${OUT}/table_paired"
  { echo "# E4 summary ($(stamp))"; echo; echo "## Funnel"; cat "${OUT}/funnel.md" 2>/dev/null
    echo; echo "## Independent judge"; echo '```'; cat "${OUT}/independent_judge_summary.json" 2>/dev/null; echo '```'
    echo; echo "## Paired retrieval on the same 72 questions (session recall)"; cat "${OUT}/table_paired.md" 2>/dev/null
    [[ -f "${OUT}/human_check.csv" ]] && { echo; echo "Human check sheet: ${OUT}/human_check.csv (fill a1_/a2_ columns, then score with tools.audit_sample --score)"; }; } > "${OUT}/E4_summary.md"
  log "wrote ${OUT}/E4_summary.md"
fi
log "E4 done"
