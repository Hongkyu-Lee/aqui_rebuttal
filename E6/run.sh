#!/bin/bash
# =============================================================================
# E6 -- Oracle-memory answer baseline (2KMq-Q4c).                 GPUs: 8 x A100-80GB
#
# Same answer models, prompt and judge as the paper; the retrieved facts are replaced by the GOLD
# facts. Answer models (paper, Sec. 4): Qwen3-Next-80B-A3B-Instruct and DeepSeek-R1-Distill-Llama-70B.
# Judge (paper): Qwen3.5-122B-A10B, served under the alias gpt-4o-mini, rigorous + lenient scores.
#
#   1 [GPU] serve both answer models (GPUs 0-3 and 4-7), generate oracle answers on both benchmarks
#   2 [GPU] serve the judge, judge every oracle answer file (+ any files in REJUDGE_FILES)
#   3 [CPU] summary next to the paper's judged AQUI numbers (if found under JUDGE_ROOT_OLD)
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../shared/env.sh"
OUT="${OUT:-${OUT_BASE}/E6}"; LOG_DIR="${OUT}/logs"; mkdir -p "$LOG_DIR" "${OUT}/answers" "${OUT}/judged"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1
STEPS="${STEPS:-1 2 3}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
cd "$SHARED"
require_server space aws          # see ../SERVERS.md
check_inputs PY:req VLLM_ENV:opt IMPLEX_FACT_BAG:req ACTMEM_HARD_FACT_BAG:req ACTMEM_FACTS_ROOT:opt JUDGE_ROOT_OLD:opt
trap vllm_stop_all EXIT
HARD_JSON="$(resolve_hard_json)"

# name|hf model|port|gpus|tp|max_len|answer args
QWEN="qwen80b|Qwen/Qwen3-Next-80B-A3B-Instruct|8001|0,1,2,3|4|32768|--max-tokens 512 --temperature 0.0"
R1="r1_70b|deepseek-ai/DeepSeek-R1-Distill-Llama-70B|8002|4,5,6,7|4|32768|--max-tokens 3072 --temperature 0.6 --top-p 0.95 --no-system"
ANSWER_MODELS="${ANSWER_MODELS:-qwen80b r1_70b}"
JUDGE_HF="${JUDGE_HF:-Qwen/Qwen3.5-122B-A10B}"; JUDGE_ALIAS="${JUDGE_ALIAS:-gpt-4o-mini}"
JUDGE_PORT="${JUDGE_PORT:-8000}"; JUDGE_GPUS="${JUDGE_GPUS:-0,1,2,3,4,5,6,7}"; JUDGE_TP="${JUDGE_TP:-8}"
DATASETS="implexconv|implexconv|${IMPLEX_FACT_BAG}|implexconv_opposed actmemeval:${HARD_JSON}|actmem_hard|${ACTMEM_HARD_FACT_BAG}|actmemeval_implicit_hard"

gen_for() {  # $1 = spec line
  local name model port gpus tp maxlen aargs
  IFS='|' read -r name model port gpus tp maxlen aargs <<<"$1"
  local extra=(); [[ "$name" == r1_70b ]] && extra=(--reasoning-parser deepseek_r1)
  vllm_start "$name" "$model" "$port" "$gpus" "$tp" "$maxlen" --served-model-name "$model" "${extra[@]}"
}
if want 1; then
  log "STEP 1: oracle answers"
  for n in $ANSWER_MODELS; do case "$n" in qwen80b) gen_for "$QWEN";; r1_70b) gen_for "$R1";; esac; done
  pids=()
  for n in $ANSWER_MODELS; do
    spec="$([[ $n == qwen80b ]] && echo "$QWEN" || echo "$R1")"
    IFS='|' read -r name model port gpus tp maxlen aargs <<<"$spec"
    wait_vllm "$port" "$model" "${LOG_DIR}/vllm_${name}.pid"
    for d in $DATASETS; do
      IFS='|' read -r data tag fb dtag <<<"$d"
      require_file "$fb" "fact bag for ${tag}"
      ( "$PY" -m tools.oracle_answers --data "$data" --fact-bag "$fb" --model "$model" \
          --base-url "http://localhost:${port}/v1" $aargs --workers "${ANSWER_WORKERS:-32}" --tag "$dtag" \
          --out "${OUT}/answers/oracle__${name}__${tag}.json" > "${LOG_DIR}/answers_${name}_${tag}.log" 2>&1 ) &
      pids+=($!)
    done
  done
  for p in "${pids[@]}"; do wait "$p" || warn "an answer job failed (see ${LOG_DIR}/answers_*.log)"; done
  vllm_stop_all
fi

latent_for() {  # $1=tag $2=data $3=fact bag $4=actmem dataset tag  -> path of facts.json for the judge
  local actmem="${ACTMEM_FACTS_ROOT:-/nonexistent}/$4/facts.json"
  if have_file "$actmem"; then echo "$actmem"; return; fi
  local built="${OUT}/latent_${1}.json"
  [[ -f "$built" ]] || "$PY" -m tools.build_latent_facts --data "$2" --fact-bag "$3" --out "$built" >&2
  echo "$built"
}
if want 2; then
  log "STEP 2: judge (${JUDGE_HF} as ${JUDGE_ALIAS})"
  vllm_start judge "$JUDGE_HF" "$JUDGE_PORT" "$JUDGE_GPUS" "$JUDGE_TP" 8192 --served-model-name "$JUDGE_ALIAS" "$JUDGE_HF" --max-num-seqs 256
  wait_vllm "$JUDGE_PORT" "$JUDGE_ALIAS" "${LOG_DIR}/vllm_judge.pid"
  chat_ok "$JUDGE_PORT" "$JUDGE_ALIAS" || die "judge endpoint does not answer"
  for d in $DATASETS; do
    IFS='|' read -r data tag fb dtag <<<"$d"
    facts="$(latent_for "$tag" "$data" "$fb" "$dtag")"
    log "latent memory for ${tag}: ${facts}"
    for f in "${OUT}"/answers/oracle__*__"${tag}".json ${REJUDGE_FILES:-}; do
      [[ -f "$f" ]] || continue
      [[ "$f" == *"__${tag}.json" || " ${REJUDGE_FILES:-} " == *" $f "* ]] || continue
      (cd "${SHARED}/judge" && OPENAI_BASE_URL="http://localhost:${JUDGE_PORT}/v1" "$PY" evaluate_answer_quality.py \
         --system iris --input "$f" --facts "$facts" --model "$JUDGE_ALIAS" --workers 16 --batch-size 2 \
         --max-tokens 4096 --output "${OUT}/judged/$(basename "$f" .json)__judged.json") > "${LOG_DIR}/judge_$(basename "$f" .json).log" 2>&1 \
        || warn "judge failed for $f"
    done
  done
  vllm_stop_all
fi

if want 3; then
  "$PY" - "${OUT}/judged" "${JUDGE_ROOT_OLD:-}" "${OUT}/E6_summary.md" <<'PY'
import glob, json, os, sys
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    s = json.load(open(f))["summary"]
    rows.append((os.path.basename(f), s.get("num_judged"), s.get("rigorous_accuracy"), s.get("lenient_score")))
old = []
if sys.argv[2] and os.path.isdir(sys.argv[2]):
    for f in sorted(glob.glob(sys.argv[2] + "/iris__*.json")):
        try:
            s = json.load(open(f))["summary"]
            old.append((os.path.basename(f), s.get("num_judged"), s.get("rigorous_accuracy"), s.get("lenient_score")))
        except Exception:
            pass
fmt = lambda r: f"| {r[0]} | {r[1]} | {r[2] if r[2] is None else round(r[2], 4)} | {r[3] if r[3] is None else round(r[3], 4)} |"
md = ["# E6 summary", "", "## Oracle memory (gold facts)", "| file | n | rigorous | lenient |", "|---|---|---|---|"] + [fmt(r) for r in rows]
md += ["", "## Paper runs found under JUDGE_ROOT_OLD (same judge; for comparison)", "| file | n | rigorous | lenient |", "|---|---|---|---|"] + [fmt(r) for r in old]
open(sys.argv[3], "w").write("\n".join(md) + "\n")
print("\n".join(md))
PY
fi
log "E6 done"
