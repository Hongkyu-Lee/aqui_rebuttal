#!/bin/bash
# Run ON AWS. Packs the AWS-only inputs that SPACE experiments (E2b, E4) need into one tarball.
#   bash shared/tools/stage_from_aws.sh            -> ./aws_inputs_for_space.tar.gz
# Then on SPACE:
#   mkdir -p /group-volume/users/h.lee4/iclr_2027_rebuttal_inputs/from_aws
#   tar -xzf aws_inputs_for_space.tar.gz -C /group-volume/users/h.lee4/iclr_2027_rebuttal_inputs/from_aws
# Resulting layout (what shared/env.sh expects on SPACE via STAGE_FROM_AWS):
#   from_aws/ActMemEval.json
#   from_aws/actmemeval/user_facts.jsonl
#   from_aws/supportive_reasoning_hf/user_facts.jsonl     (if it exists on AWS)
#   from_aws/MANIFEST.txt                                 (source path, size, sha256)
set -uo pipefail
SERVER=aws source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
OUT_TAR="${1:-$PWD/aws_inputs_for_space.tar.gz}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
put() {  # put <src> <dest-relative> <required|optional>
  if [[ -s "$1" ]]; then mkdir -p "$TMP/$(dirname "$2")"; cp "$1" "$TMP/$2"
    echo "$2  <-  $1  ($(stat -c %s "$1") bytes, sha256 $(sha256sum "$1" | cut -c1-16))" >> "$TMP/MANIFEST.txt"; log "packed $2"
  elif [[ "$3" == required ]]; then die "missing on AWS: $1"
  else warn "not found on AWS (skipped): $1"; fi
}
put "$ACTMEM_ORIG_JSON"           ActMemEval.json                               required
put "$ACTMEM_ORIG_FACT_BAG"       actmemeval/user_facts.jsonl                   required
put "$IMPLEX_SUPPORTIVE_FACT_BAG" supportive_reasoning_hf/user_facts.jsonl      optional
tar -czf "$OUT_TAR" -C "$TMP" . && log "wrote $OUT_TAR"; cat "$TMP/MANIFEST.txt"
