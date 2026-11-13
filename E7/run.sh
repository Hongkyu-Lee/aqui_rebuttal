#!/bin/bash
# =============================================================================
# E7 -- Noisy-oracle answer baseline (2KMq-Q4c, follow-up to E6).   GPUs: 8 x A100-80GB
#
# E6 gave the answer model ONLY the gold facts. E7 hides the gold evidence among realistic
# distractors, so we can tell "retrieval misses the evidence" apart from "the answer model cannot
# use the evidence when it is in a K-fact list". Same answer model, prompt, decoding, judge and
# judge memory as E6. Per dataset, on the same questions:
#   evidence      evidence fact(s) only                                  (clean reference)
#   noisy_k<K>    evidence + (K-n) AQUI-ranked non-gold facts, shuffled  (the new row)
#   retrieved_k<K> AQUI's own top-K, rank order                          (what AQUI really gives)
#
# Needs E5 outputs: ${OUT_BASE}/E5/evidence_<ds>.jsonl and ${OUT_BASE}/E5/results/aqui__<ds>.json
# (run E5 first, or set EVIDENCE_<ds> / AQUI_RESULT_<ds>).
#
#   1 [GPU] serve the answer model (GPUs 0-3), generate answers for every mode x K x dataset
#   2 [GPU] serve the judge (all 8 GPUs), judge every answer file with E6's judge memory
#   3 [CPU] summary, next to E6's clean oracle and E6's re-judged paper AQUI rows (if present)
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../shared/env.sh"
OUT="${OUT:-${OUT_BASE}/E7}"; LOG_DIR="${OUT}/logs"; mkdir -p "$LOG_DIR" "${OUT}/answers" "${OUT}/judged"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1
STEPS="${STEPS:-1 2 3}"
want() { [[ " $STEPS " == *" $1 "* ]]; }
cd "$SHARED"
require_server space aws          # see ../SERVERS.md

# answer model: identical to E6's Qwen spec (name|hf model|port|gpus|tp|max_len|answer args)
QWEN="qwen80b|Qwen/Qwen3-Next-80B-A3B-Instruct|8001|0,1,2,3|4|32768|--max-tokens 512 --temperature 0.0"
JUDGE_HF="${JUDGE_HF:-Qwen/Qwen3.5-122B-A10B}"; JUDGE_ALIAS="${JUDGE_ALIAS:-gpt-4o-mini}"
JUDGE_PORT="${JUDGE_PORT:-8000}"; JUDGE_GPUS="${JUDGE_GPUS:-0,1,2,3,4,5,6,7}"; JUDGE_TP="${JUDGE_TP:-8}"
SEED="${SEED:-0}"
KS_implexconv="${KS_implexconv:-10 20 30 50}"
KS_actmem_hard="${KS_actmem_hard:-10 14 20 30 40}"
DS_LIST="${DS_LIST:-implexconv actmem_hard}"

# tag -> data spec | fact bag | judge dataset tag
ds_spec() {
  case "$1" in
    implexconv)  echo "implexconv|${IMPLEX_FACT_BAG}|implexconv_opposed" ;;
    actmem_hard) echo "actmemeval:${HARD_JSON}|${ACTMEM_HARD_FACT_BAG}|actmemeval_implicit_hard" ;;
    *) die "unknown dataset $1" ;;
  esac
}
evidence_for() { local v="EVIDENCE_$1"; echo "${!v:-${OUT_BASE}/E5/evidence_$1.jsonl}"; }
aqui_result_for() {
  local v="AQUI_RESULT_$1"; [[ -n "${!v:-}" ]] && { echo "${!v}"; return; }
  ls "${OUT_BASE}/E5/results/aqui__$1.json" "${OUT_BASE}/E1/test/paper_adapter__$1.json" 2>/dev/null | head -1
}

# E5 outputs (checked first, because check_inputs exits when CHECK_ONLY=1)
e5_missing=0
for ds in $DS_LIST; do
  ev="$(evidence_for "$ds")"; r="$(aqui_result_for "$ds")"
  if [[ -f "$ev" && -n "$r" ]]; then log "[ok] ${ds}: evidence=${ev} | aqui ranking=${r}"
  else warn "[MISSING] ${ds}: E5 evidence labels (${ev}) or AQUI ranking (E5/results/aqui__${ds}.json). Run E5, or set EVIDENCE_${ds} / AQUI_RESULT_${ds}"; e5_missing=1; fi
done
[[ "${CHECK_ONLY:-0}" != 1 && $e5_missing -ne 0 ]] && die "E5 outputs missing (see above)"
check_inputs PY:req VLLM_ENV:opt IMPLEX_FACT_BAG:req ACTMEM_HARD_FACT_BAG:req ACTMEM_FACTS_ROOT:opt
trap vllm_stop_all EXIT
HARD_JSON="$(resolve_hard_json)"

if want 1; then
  log "STEP 1: answers (seed ${SEED})"
  IFS='|' read -r name model port gpus tp maxlen aargs <<<"$QWEN"
  vllm_start "$name" "$model" "$port" "$gpus" "$tp" "$maxlen" --served-model-name "$model"
  wait_vllm "$port" "$model" "${LOG_DIR}/vllm_${name}.pid"
  pids=()
  for ds in $DS_LIST; do
    IFS='|' read -r data fb dtag <<<"$(ds_spec "$ds")"
    ksv="KS_${ds}"; runs="evidence|0"
    for k in ${!ksv}; do runs+=" noisy|${k} retrieved|${k}"; done
    for rk in $runs; do
      IFS='|' read -r mode k <<<"$rk"
      label="$mode"; [[ "$mode" != evidence ]] && label="${mode}_k${k}"
      f="${OUT}/answers/${label}__${name}__${ds}.json"
      [[ -f "$f" && "${FORCE:-0}" != 1 ]] && { log "skip (exists): $f"; continue; }
      ( "$PY" -m tools.noisy_oracle_answers --data "$data" --fact-bag "$fb" --evidence "$(evidence_for "$ds")" \
          --retrieval "$(aqui_result_for "$ds")" --mode "$mode" --k "$k" --seed "$SEED" \
          --model "$model" --base-url "http://localhost:${port}/v1" $aargs --workers "${ANSWER_WORKERS:-32}" \
          --tag "$dtag" --out "$f" > "${LOG_DIR}/answers_${label}_${ds}.log" 2>&1 ) &
      pids+=($!)
    done
  done
  for p in "${pids[@]}"; do wait "$p" || warn "an answer job failed (see ${LOG_DIR}/answers_*.log)"; done
  vllm_stop_all
fi

# Judge memory: MUST be the same file E6 used, so E7 rows compare with E6's rows.
latent_for() {  # $1=ds $2=data $3=fact bag $4=judge dataset tag
  local v="LATENT_$1"; [[ -n "${!v:-}" ]] && { echo "${!v}"; return; }
  local e6="${OUT_BASE}/E6/latent_$1.json"
  local actmem="${ACTMEM_FACTS_ROOT:-/nonexistent}/$4/facts.json"
  if have_file "$e6"; then echo "$e6"; return; fi            # E6 rebuilt it -> reuse the same file
  if have_file "$actmem"; then echo "$actmem"; return; fi     # E6 used ActMem's facts.json
  local built="${OUT}/latent_$1.json"
  [[ -f "$built" ]] || "$PY" -m tools.build_latent_facts --data "$2" --fact-bag "$3" --out "$built" >&2
  echo "$built"
}
if want 2; then
  log "STEP 2: judge (${JUDGE_HF} as ${JUDGE_ALIAS})"
  vllm_start judge "$JUDGE_HF" "$JUDGE_PORT" "$JUDGE_GPUS" "$JUDGE_TP" 8192 --served-model-name "$JUDGE_ALIAS" "$JUDGE_HF" --max-num-seqs 256
  wait_vllm "$JUDGE_PORT" "$JUDGE_ALIAS" "${LOG_DIR}/vllm_judge.pid"
  chat_ok "$JUDGE_PORT" "$JUDGE_ALIAS" || die "judge endpoint does not answer"
  for ds in $DS_LIST; do
    IFS='|' read -r data fb dtag <<<"$(ds_spec "$ds")"
    facts="$(latent_for "$ds" "$data" "$fb" "$dtag")"
    log "judge memory for ${ds}: ${facts}   <- must be the same file as in E6 (see E6 logs/run.log)"
    for f in "${OUT}"/answers/*__"${ds}".json; do
      [[ -f "$f" ]] || continue
      o="${OUT}/judged/$(basename "$f" .json)__judged.json"
      [[ -f "$o" && "${FORCE:-0}" != 1 ]] && { log "skip (judged): $o"; continue; }
      (cd "${SHARED}/judge" && OPENAI_BASE_URL="http://localhost:${JUDGE_PORT}/v1" "$PY" evaluate_answer_quality.py \
         --system iris --input "$f" --facts "$facts" --model "$JUDGE_ALIAS" --workers 16 --batch-size 2 \
         --max-tokens 4096 --output "$o") > "${LOG_DIR}/judge_$(basename "$f" .json).log" 2>&1 \
        || warn "judge failed for $f"
    done
  done
  vllm_stop_all
fi

if want 3; then
  "$PY" - "${OUT}" "${OUT_BASE}/E6/judged" "${OUT}/E7_summary.md" <<'PY'
import glob, json, os, sys
out, e6, dst = sys.argv[1:4]
def row(f):
    try:
        s = json.load(open(f))["summary"]
    except Exception:
        return None
    r, l = s.get("rigorous_accuracy"), s.get("lenient_score")
    return f"| {os.path.basename(f)} | {s.get('num_judged')} | {r if r is None else round(r, 4)} | {l if l is None else round(l, 4)} |"
md = ["# E7 summary (noisy oracle)", "",
      "Same answer model (Qwen3-Next-80B), prompt, judge and judge memory as E6.", "",
      "## E7 rows (same questions in every row of a dataset: those with an E5 evidence label)",
      "| file | n judged | rigorous | lenient |", "|---|---|---|---|"]
md += [r for r in map(row, sorted(glob.glob(out + "/judged/*.json"))) if r]
md += ["", "## E6 rows for reference (clean oracle = all gold-session facts; re-judged paper AQUI answers)",
       "| file | n judged | rigorous | lenient |", "|---|---|---|---|"]
md += [r for r in map(row, sorted(glob.glob(e6 + "/*.json"))) if r]
md += ["", "## Context statistics (from the answer files)", "| file | n | mean facts | mean evidence facts | evidence in context | skipped |", "|---|---|---|---|---|---|"]
for f in sorted(glob.glob(out + "/answers/*.json")):
    if f.endswith(".partial.jsonl"):
        continue
    try:
        s = json.load(open(f))["summary"]
    except Exception:
        continue
    md.append(f"| {os.path.basename(f)} | {s['num_samples']} | {s['mean_num_context_facts']:.1f} | "
              f"{s['mean_num_evidence_facts']:.2f} | {s['evidence_in_context_rate']:.3f} | {s['skipped']} |")
open(dst, "w").write("\n".join(md) + "\n")
print("\n".join(md))
PY
fi
log "E7 done"
