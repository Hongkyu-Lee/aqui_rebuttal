#!/bin/bash
# =============================================================================
# Validate every external path the experiments use, BEFORE running anything.
# Writes a markdown report (send it back to h.lee4) and prints it.
#
#   bash shared/tools/check_paths.sh                 # auto-detect server
#   SERVER=space bash shared/tools/check_paths.sh
#   OUT_BASE=/my/out bash shared/tools/check_paths.sh   # same overrides you will use for the runs
#
# Report: ./PATH_CHECK_REPORT_<server>.md (or $REPORT). No GPU needed; reads files only.
# Exit code = number of FAIL rows that block an experiment.
# =============================================================================
set -u
CHECK_ONLY=1   # makes env.sh skip its writability die (we report it instead)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
REPORT="${REPORT:-$PWD/PATH_CHECK_REPORT_${SERVER}.md}"
PYC="$PY"; command -v "$PYC" >/dev/null 2>&1 || PYC=python3
FAILS=0
ROWS=()

row() {  # row <status> <variable/item> <path> <detail> <used by>
  ROWS+=("| $1 | \`$2\` | \`${3:-<unset>}\` | $4 | $5 |")
  [[ "$1" == FAIL ]] && FAILS=$((FAILS+1)); return 0
}
nrows() {  # number of records in a .jsonl (lines) or .json (top-level list / dict)
  case "$1" in
    *.jsonl) wc -l < "$1" | tr -d ' ' ;;
    *.json)  "$PYC" -c "import json,sys;d=json.load(open(sys.argv[1]));print(len(d) if isinstance(d,(list,dict)) else '?')" "$1" 2>/dev/null || echo "?" ;;
  esac
}
# check_file <status-if-missing> <var> <used by> [expected rows]
check_file() {
  local miss="$1" var="$2" used="$3" exp="${4:-}" p="${!2:-}" n
  if [[ -z "$p" ]]; then row "$miss" "$var" "" "not set for this server" "$used"; return; fi
  if [[ ! -e "$p" ]]; then row "$miss" "$var" "$p" "does not exist" "$used"; return; fi
  if [[ ! -r "$p" ]]; then row FAIL "$var" "$p" "exists but NOT readable by $(whoami)" "$used"; return; fi
  if [[ ! -s "$p" ]]; then row "$miss" "$var" "$p" "exists but is EMPTY (0 bytes)" "$used"; return; fi
  n="$(nrows "$p")"
  if [[ -n "$exp" && "$n" != "$exp" ]]; then row WARN "$var" "$p" "rows=$n, expected $exp" "$used"
  else row OK "$var" "$p" "rows=${n:-?}$(du -h "$p" 2>/dev/null | awk '{print ", "$1}')" "$used"; fi
}
check_dir() {  # check_dir <status-if-missing> <var> <used by>
  local miss="$1" var="$2" used="$3" p="${!2:-}"
  if [[ -z "$p" ]]; then row "$miss" "$var" "" "not set for this server" "$used"
  elif [[ ! -d "$p" ]]; then row "$miss" "$var" "$p" "directory does not exist" "$used"
  elif [[ ! -r "$p" || ! -x "$p" ]]; then row FAIL "$var" "$p" "exists but NOT readable by $(whoami)" "$used"
  else row OK "$var" "$p" "directory" "$used"; fi
}

# ---------------- environments
if [[ -x "${PY:-}" ]] || command -v "${PY:-}" >/dev/null 2>&1; then
  pyinfo="$("$PY" - <<'PY' 2>&1
import importlib
mods = ["torch", "transformers", "peft", "hydra", "omegaconf", "datasets", "openai", "tqdm", "wandb"]
missing = []
for m in mods:
    try: importlib.import_module(m)
    except Exception: missing.append(m)
import torch
print(f"torch {torch.__version__}, cuda_available={torch.cuda.is_available()}, gpus={torch.cuda.device_count()}"
      + (f"; MISSING: {', '.join(missing)}" if missing else "; all imports ok"))
PY
)"
  if [[ "$pyinfo" == *MISSING* || "$pyinfo" == *Error* ]]; then row FAIL PY "$PY" "${pyinfo//$'\n'/ }" all
  else row OK PY "$PY" "$pyinfo" all; fi
else
  row FAIL PY "${PY:-}" "python not found / not executable by $(whoami)" all
fi
VB="$(vllm_bin)"
if [[ -x "$VB" ]] || command -v "$VB" >/dev/null 2>&1; then
  vv="$("${VLLM_ENV:+${VLLM_ENV}/bin/}python" -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo '?')"
  row OK VLLM_ENV "${VLLM_ENV:-$VB}" "vllm ${vv}" "E4 (local judge), E5, E6"
else
  row FAIL VLLM_ENV "${VLLM_ENV:-}" "no vllm executable" "E4 (local judge), E5, E6"
fi

# ---------------- writable locations
if mkdir -p "$OUT_BASE" 2>/dev/null && [[ -w "$OUT_BASE" ]]; then row OK OUT_BASE "$OUT_BASE" "writable" all
else row FAIL OUT_BASE "$OUT_BASE" "NOT writable by $(whoami) -- set OUT_BASE" all; fi
if [[ -d "${HF_HOME:-/nonexistent}" ]]; then
  [[ -w "$HF_HOME" ]] && row OK HF_HOME "$HF_HOME" "readable + writable" all \
                      || row WARN HF_HOME "$HF_HOME" "read-only: cached models load, new downloads fail" all
else row WARN HF_HOME "${HF_HOME:-}" "does not exist (models will be downloaded on first use)" all; fi

# ---------------- models / datasets in the HF cache (downloaded on first use if absent)
hf_item() {  # hf_item <repo id> <models|datasets> <used by>
  local d="${HF_HOME:-/nonexistent}/hub/$2--${1//\//--}"
  [[ -d "$d" ]] && row OK "HF $2" "$1" "cached at $d" "$3" || row WARN "HF $2" "$1" "not in cache: will download (needs internet${4:+, $4})" "$3"
}
hf_item Qwen/Qwen3-Embedding-8B models "E1-E5"
hf_item Kaylee0501/ImplexConv-opposed datasets "E1-E3, E5, E6"
hf_item Kaylee0501/ImplexConv-supportive datasets "E2b"
hf_item Qwen/Qwen3-Next-80B-A3B-Instruct models "E5, E6"
hf_item deepseek-ai/DeepSeek-R1-Distill-Llama-70B models "E6"
hf_item Qwen/Qwen3.5-122B-A10B models "E6 (judge)"
hf_item meta-llama/Llama-3.3-70B-Instruct models "E4 (local judge only)" "gated: HF_TOKEN"

# ---------------- data inputs
check_dir  FAIL HPO_ROOT "E1"
if [[ -d "${HPO_ROOT:-/nonexistent}/adapters" ]]; then
  n=$(ls -d "${HPO_ROOT}"/adapters/hpo_* 2>/dev/null | wc -l)
  nc=$(ls "${HPO_ROOT}"/adapters/hpo_*/adapter_config.json 2>/dev/null | wc -l)
  [[ "$n" -eq 216 && "$nc" -eq 216 ]] && row OK "HPO adapters" "${HPO_ROOT}/adapters/hpo_*" "216 dirs, all with adapter_config.json" E1 \
    || row FAIL "HPO adapters" "${HPO_ROOT}/adapters/hpo_*" "dirs=$n, with adapter_config.json=$nc (expected 216)" E1
else row FAIL "HPO adapters" "${HPO_ROOT:-}/adapters" "missing" E1; fi
HPO_RANKING="${HPO_ROOT:-/nonexistent}/HPO_ranking.csv"; check_file WARN HPO_RANKING "E1 (reference column only; never used to select)"
for f in adapter_config.json adapter_model.safetensors model_selection.json train_data_manifest.json; do
  P="${PAPER_ADAPTER}/${f}"
  if [[ -s "$P" ]]; then row OK "PAPER_ADAPTER/$f" "$P" "present" "E1-E5"
  else
    st=FAIL; [[ "$f" == train_data_manifest.json ]] && st=WARN
    row "$st" "PAPER_ADAPTER/$f" "$P" "missing$([[ $st == WARN ]] && echo ' (E1 step 2 loader check is skipped)')" "E1-E5"
  fi
done
check_file FAIL V11CON_2400_JSONL "E1, E2, E3" 2400
check_file FAIL IMPLEX_FACT_BAG "E1, E2, E2b, E3, E5, E6" 1550
check_file WARN IMPLEX_SUPPORTIVE_FACT_BAG "E2b (optional part)"
check_file FAIL ACTMEM_HARD_FACT_BAG "E1-E6" 72
if [[ -s "${ACTMEM_HARD_JSON:-/nonexistent}" ]]; then check_file WARN ACTMEM_HARD_JSON "E1-E6" 72
else row OK ACTMEM_HARD_JSON "${ACTMEM_HARD_JSON:-}" "missing/empty on server -> bundled rebuilt copy is used (verified)" "E1-E6"; fi
check_file WARN ACTMEM_ORIG_JSON "E2b, E4 (optional part; staged from AWS)" 247
check_file WARN ACTMEM_ORIG_FACT_BAG "E2b, E4 (optional part; staged from AWS)" 247
for ds in implexconv_opposed actmemeval_implicit_hard; do
  F="${ACTMEM_FACTS_ROOT:-/nonexistent}/${ds}/facts.json"
  [[ -s "$F" ]] && row OK "ACTMEM_FACTS_ROOT/$ds" "$F" "present" "E6 (judge latent memory)" \
                || row WARN "ACTMEM_FACTS_ROOT/$ds" "$F" "missing -> rebuilt from the fact bag (then also re-judge one paper file, see E6/AGENT.md)" "E6"
done
check_dir WARN JUDGE_ROOT_OLD "E6 (summary comparison only)"
check_dir WARN GEPA_ANSWERGEN_ROOT "E6 (paper answer files for REJUDGE_FILES)"

# ---------------- bundled data (inside this folder)
for f in v11con_2400.jsonl ActMemEval_lexical_controlled.json ActMemEval_lexical_controlled_metadata.json \
         ActMemEval_implicit_hard_metadata.json implicit_hard_rewrites.jsonl ActMemEval_implicit_hard.rebuilt.json \
         actmemeval_implicit_hard_user_facts.jsonl actmemeval_lexical_controlled_user_facts.jsonl \
         implexconv_opposed_qa_gold_session_facts.jsonl; do
  [[ -s "${SHARED}/data/$f" ]] && row OK "bundled" "shared/data/$f" "present" "-" || row FAIL "bundled" "shared/data/$f" "MISSING from the package (copy incomplete?)" "-"
done

# ---------------- secrets (only presence, never the value)
[[ -n "${TML_API_KEY:-}" ]] && row OK TML_API_KEY "(env)" "set" "E4 with JUDGE_BACKEND=api" \
                            || row WARN TML_API_KEY "(env)" "not set (only needed for JUDGE_BACKEND=api)" "E4"
[[ -n "${HF_TOKEN:-}" || -s "${HF_HOME:-/nonexistent}/token" ]] && row OK HF_TOKEN "(env or HF_HOME/token)" "present" "E4 local Llama judge" \
                            || row WARN HF_TOKEN "(env or HF_HOME/token)" "absent (only needed for the local Llama judge)" "E4"

# ---------------- write report
{
  echo "# Path check report"
  echo
  echo "* server: **${SERVER}** | host: \`$(hostname)\` | user: \`$(whoami)\` | $(stamp)"
  echo "* package: \`${REBUTTAL_ROOT}\`"
  echo "* **FAIL rows: ${FAILS}** (FAIL = blocks the experiment(s) listed; WARN = an optional part is skipped/rebuilt, or a count differs from what we expect)"
  echo
  echo "| Status | Item | Path | Detail | Used by |"
  echo "|---|---|---|---|---|"
  printf '%s\n' "${ROWS[@]}"
} > "$REPORT"
cat "$REPORT"
echo; log "report written: $REPORT"
exit "$FAILS"
