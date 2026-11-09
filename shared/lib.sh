#!/bin/bash
# Shared bash helpers. Sourced by shared/env.sh (do not run directly).

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log()   { echo "[$(stamp)] $*"; }
warn()  { echo "[$(stamp)] WARN: $*" >&2; }
die()   { echo "[$(stamp)] ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1 ${2:+($2)}"; }
require_dir()  { [[ -d "$1" ]] || die "missing dir: $1 ${2:+($2)}"; }
have_file()    { [[ -n "${1:-}" && -f "$1" ]]; }

# GPU list helpers: GPUS="0 1 2 3" (space separated)
gpu_list()  { echo "${GPUS:-0 1 2 3 4 5 6 7}"; }
gpu_count() { wc -w <<<"$(gpu_list)"; }
gpu_csv()   { local n="$1"; gpu_list | tr ' ' '\n' | head -n "$n" | paste -sd, -; }

# ---------------------------------------------------------------------------
# ActMemEval-hard JSON: prefer the server copy, else rebuild it from the bundled
# lexical-controlled file + session rewrites (shared/tools/rebuild_actmemeval_hard.py).
resolve_hard_json() {
  if have_file "${ACTMEM_HARD_JSON:-}" && [[ -s "${ACTMEM_HARD_JSON}" ]]; then
    echo "${ACTMEM_HARD_JSON}"; return
  fi
  if [[ ! -s "${BUNDLED_HARD_JSON}" ]]; then
    "$PY" "${SHARED}/tools/rebuild_actmemeval_hard.py" \
      --lexical "${SHARED}/data/ActMemEval_lexical_controlled.json" \
      --rewrites "${SHARED}/data/implicit_hard_rewrites.jsonl" \
      --metadata "${SHARED}/data/ActMemEval_implicit_hard_metadata.json" \
      --fact-bag "${SHARED}/data/actmemeval_implicit_hard_user_facts.jsonl" \
      --out "${BUNDLED_HARD_JSON}" >&2 || die "could not rebuild ActMemEval-hard"
  fi
  echo "${BUNDLED_HARD_JSON}"
}

# ---------------------------------------------------------------------------
# Training one LoRA.  train_lora <arm:aqui|sym> <gpu> <outdir> <lr> <temp> <clip> <warmup> <bs> <r> <seed>
train_lora() {
  local arm="$1" gpu="$2" outdir="$3" lr="$4" temp="$5" clip="$6" wu="$7" bs="$8" r="$9" seed="${10}"
  local module=src.entry_points.train_embedding_model
  [[ "$arm" == "sym" ]] && module=baseline_symmetric.train_symmetric
  if [[ -f "${outdir}/adapter_config.json" && -f "${outdir}/model_selection.json" && "${FORCE_TRAIN:-0}" != 1 ]]; then
    log "skip train (exists): ${outdir}"; return 0
  fi
  mkdir -p "$outdir"
  (cd "$SHARED" && CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m "$module" --config "${SHARED}/config/train.yaml" \
    train.model.name="${EMBED_MODEL}" \
    train.model.output_dir="$outdir" \
    train.model.lora.r="$r" train.model.lora.alpha="$((2 * r))" \
    train.data.path="${V11CON_2400_JSONL}" \
    train.optimization.learning_rate="$lr" \
    train.optimization.temperature="$temp" \
    train.optimization.max_grad_norm="$clip" \
    train.optimization.warmup_ratio="$wu" \
    train.optimization.batch_size="$bs" \
    train.optimization.seed="$seed" \
    train.logging.run_name="$(basename "$outdir")" train.logging.dir="${outdir}/tb")
}

# ---------------------------------------------------------------------------
# Fact-retrieval eval.  eval_retrieval <arm:base|aqui|sym> <gpu> <adapter|-> <dataset:implexconv|implexconv_supportive|actmem_hard|actmem_json:<path>:<factbag>> <outdir> <outfile>
eval_retrieval() {
  local arm="$1" gpu="$2" adapter="$3" ds="$4" outdir="$5" outfile="$6"
  local data_args fact_bag mem=false cache=true prebatch=false adapter_arg="null"
  case "$ds" in
    implexconv)            data_args="data=implexconv data.type=opposed"; fact_bag="$IMPLEX_FACT_BAG" ;;
    implexconv_supportive) data_args="data=implexconv data.type=supported"; fact_bag="$IMPLEX_SUPPORTIVE_FACT_BAG" ;;
    actmem_hard)           data_args="data=actmemeval data.path=$(resolve_hard_json)"; fact_bag="$ACTMEM_HARD_FACT_BAG" ;;
    actmem_json:*)         IFS=: read -r _ _p _fb <<<"$ds"; data_args="data=actmemeval data.path=${_p}"; fact_bag="${_fb}" ;;
    *) die "unknown dataset spec: $ds" ;;
  esac
  require_file "$fact_bag" "fact bag for $ds"
  [[ "$arm" != base && "$adapter" != "-" ]] && adapter_arg="$adapter"
  if [[ "$arm" == sym ]]; then mem=true; cache=false; prebatch=true; fi   # symmetric: re-encode facts
  if [[ -f "${outdir}/${outfile}" && "${FORCE_EVAL:-0}" != 1 ]]; then log "skip eval (exists): ${outdir}/${outfile}"; return 0; fi
  mkdir -p "$outdir"
  (cd "$SHARED" && CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m src.evaluate.fact_retrieval \
    --config "${SHARED}/config/evaluate_fact_retrieval.yaml" $data_args \
    evaluate.embedding.model="${EMBED_MODEL}" evaluate.embedding.base_model="${EMBED_MODEL}" \
    evaluate.embedding.adapter_path="$adapter_arg" \
    evaluate.embedding.apply_adapter_to_query=true \
    evaluate.embedding.apply_adapter_to_memory="$mem" \
    evaluate.embedding.batch_size="${EVAL_BS:-64}" \
    evaluate.prebatch_facts="$prebatch" \
    evaluate.fact_bag_path="$fact_bag" \
    evaluate.fact_embedding_cache.enabled="$cache" \
    evaluate.fact_embedding_cache.root_dir="${FACT_CACHE_ROOT:-${OUT_BASE}/fact_embedding_cache}" \
    evaluate.show_progress=false evaluate.embedding.show_progress=false \
    evaluate.output_dir="$outdir" evaluate.output_file="$outfile")
}

# ---------------------------------------------------------------------------
# Simple GPU job pool. Usage:
#   pool_init; for job in ...; do pool_run "<label>" <function> <args...>; done; pool_wait
# The function receives the GPU id as its FIRST argument.
declare -A _POOL_PID=() _POOL_LABEL=()
pool_init() { _POOL_PID=(); _POOL_LABEL=(); _POOL_FAIL=0; }
_pool_free() { for g in $(gpu_list); do [[ -z "${_POOL_PID[$g]:-}" ]] && { echo "$g"; return; }; done; }
_pool_reap() {
  for g in $(gpu_list); do
    local p="${_POOL_PID[$g]:-}"
    if [[ -n "$p" ]] && ! kill -0 "$p" 2>/dev/null; then
      wait "$p"; local rc=$?
      [[ $rc -ne 0 ]] && { warn "job '${_POOL_LABEL[$g]}' on gpu $g failed rc=$rc"; _POOL_FAIL=$((_POOL_FAIL+1)); }
      [[ $rc -eq 0 ]] && log "job '${_POOL_LABEL[$g]}' on gpu $g done"
      unset "_POOL_PID[$g]"; unset "_POOL_LABEL[$g]"
    fi
  done
}
pool_run() {
  local label="$1"; shift; local fn="$1"; shift; local g
  while :; do g="$(_pool_free)"; [[ -n "$g" ]] && break; _pool_reap; [[ -z "$(_pool_free)" ]] && sleep 15; done
  "$fn" "$g" "$@" &
  _POOL_PID[$g]=$!; _POOL_LABEL[$g]="$label"
  log "launched '${label}' on gpu ${g} (pid $!)"
}
pool_wait() {
  while :; do _pool_reap; local busy=0; for g in $(gpu_list); do [[ -n "${_POOL_PID[$g]:-}" ]] && busy=1; done; [[ $busy == 0 ]] && break; sleep 15; done
  [[ "${_POOL_FAIL:-0}" -eq 0 ]] || warn "${_POOL_FAIL} pool job(s) failed -- check logs"
}

# ---------------------------------------------------------------------------
# vLLM lifecycle.  vllm_start <name> <model> <port> <gpus_csv> <tp> <max_len> [extra vllm args...]
# Writes pid to ${LOG_DIR}/vllm_<name>.pid ; log to ${LOG_DIR}/vllm_<name>.log
declare -a _VLLM_PIDS=()
vllm_bin() {
  if [[ -n "${VLLM_ENV:-}" && -x "${VLLM_ENV}/bin/vllm" ]]; then echo "${VLLM_ENV}/bin/vllm"; else command -v vllm || echo vllm; fi
}
vllm_start() {
  local name="$1" model="$2" port="$3" gpus="$4" tp="$5" maxlen="$6"; shift 6
  local logf="${LOG_DIR}/vllm_${name}.log"
  if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
    die "port ${port} already serving something -- stop it or set another port"
  fi
  log "starting vLLM '${name}': ${model} gpus=${gpus} tp=${tp} port=${port} (log ${logf})"
  ( export CUDA_VISIBLE_DEVICES="$gpus" PATH="${VLLM_ENV:+${VLLM_ENV}/bin:}${PATH}"
    exec "$(vllm_bin)" serve "$model" --port "$port" --host 127.0.0.1 \
      --tensor-parallel-size "$tp" --max-model-len "$maxlen" \
      --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.90}" "$@" ) >"$logf" 2>&1 &
  local pid=$!
  echo "$pid" > "${LOG_DIR}/vllm_${name}.pid"
  _VLLM_PIDS+=("$pid")
}
# wait_vllm <port> <served_model_id> <pid_file> [timeout_s]
wait_vllm() {
  local port="$1" want="$2" pidf="$3" timeout="${4:-${VLLM_TIMEOUT:-3600}}" t0; t0=$(date +%s)
  log "waiting for vLLM on :${port} to serve '${want}' (timeout ${timeout}s)"
  while :; do
    if curl -sf "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "\"${want}\""; then
      log "vLLM on :${port} is UP"; return 0
    fi
    if [[ -f "$pidf" ]] && ! kill -0 "$(cat "$pidf")" 2>/dev/null; then
      tail -40 "${pidf%.pid}.log" >&2; die "vLLM process died before becoming ready (see ${pidf%.pid}.log)"
    fi
    (( $(date +%s) - t0 > timeout )) && die "vLLM on :${port} not ready after ${timeout}s"
    sleep 20
  done
}
vllm_stop_all() {
  for p in "${_VLLM_PIDS[@]:-}"; do
    [[ -n "$p" ]] && kill "$p" 2>/dev/null && log "stopped vLLM pid $p"
  done
  sleep 5
  for p in "${_VLLM_PIDS[@]:-}"; do [[ -n "$p" ]] && kill -9 "$p" 2>/dev/null; done
  _VLLM_PIDS=()
}
# smoke-test a chat endpoint: chat_ok <port> <model>
chat_ok() {
  curl -sf "http://localhost:$1/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"reply OK\"}],\"max_tokens\":8}" \
    | "$PY" -c "import sys,json;print('chat ok:', json.load(sys.stdin)['choices'][0]['message']['content'][:40])"
}

# Config tag used for adapter dir names (matches the original HPO naming).
cfg_tag() { local t="lr${1}_t${2}_gc${3}_wu${4}_bs${5}_r${6}"; echo "${t//./p}"; }

# ---------------------------------------------------------------------------
# Server + input checks (called at the top of every run.sh). See SERVERS.md.
# require_server <allowed...>   e.g. require_server space   |   require_server space aws
require_server() {
  local s; for s in "$@"; do [[ "$SERVER" == "$s" ]] && { log "server check: ${SERVER} (allowed: $*)"; return 0; }; done
  [[ "${ALLOW_ANY_SERVER:-0}" == 1 ]] && { warn "server ${SERVER} not in allowed list ($*) -- continuing because ALLOW_ANY_SERVER=1"; return 0; }
  die "this experiment runs on: $* (detected SERVER=${SERVER}). See SERVERS.md; override with ALLOW_ANY_SERVER=1."
}
# check_inputs VAR:req|opt ...   prints a found/missing table; dies if a required one is missing.
# CHECK_ONLY=1 bash E*/run.sh  -> print the table and exit (exit code = number of missing required inputs).
check_inputs() {
  local spec v kind p miss=0
  log "input check (server=${SERVER}):"
  for spec in "$@"; do
    v="${spec%%:*}"; kind="${spec#*:}"; p="${!v:-}"
    [[ "$v" == PY && -n "$p" ]] && p="$(command -v "$p" 2>/dev/null || echo "$p")"
    if [[ -n "$p" && -e "$p" ]]; then printf '    [ok]      %-27s %s\n' "$v" "$p"
    elif [[ "$kind" == req ]]; then printf '    [MISSING] %-27s %s\n' "$v" "${p:-<unset>}"; miss=$((miss+1))
    else printf '    [absent]  %-27s %s  (optional: that part is skipped or rebuilt)\n' "$v" "${p:-<unset>}"; fi
  done
  if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
    [[ $miss -eq 0 ]] && log "CHECK_ONLY: all required inputs present" || warn "CHECK_ONLY: ${miss} required input(s) missing"
    exit "$miss"
  fi
  [[ $miss -eq 0 ]] || die "${miss} required input(s) missing -- see SERVERS.md (override paths with env vars)"
}
