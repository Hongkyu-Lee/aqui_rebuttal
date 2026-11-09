"""Verify the REWRITTEN training loader against an ORIGINAL adapter's train_data_manifest.json.

Every original training run wrote <adapter>/train_data_manifest.json with the TRAIN split's
num_training_samples, source_row_indices and question groups. If our loader + the unchanged
split_validation() produce the same numbers, the reconstruction is faithful.

  python -m tools.check_loader_against_manifest --data v11con_2400.jsonl --manifest <adapter>/train_data_manifest.json
Exit 0 = match, 1 = mismatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()
    from src.data.load_train_data import load_train_data
    from src.entry_points.train_embedding_model import split_validation
    from tools.synthetic_val import _cfg

    cfg = _cfg(args.data)
    train, val = split_validation(load_train_data(cfg), cfg)
    mine = {"num_training_samples": len(train),
            "source_row_indices": sorted({int(r["source_row_index"]) for r in train})}
    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    ok = True
    if man.get("num_training_samples") != mine["num_training_samples"]:
        print(f"MISMATCH num_training_samples: original={man.get('num_training_samples')} ours={mine['num_training_samples']}")
        ok = False
    if sorted(man.get("source_row_indices", [])) != mine["source_row_indices"]:
        a, b = set(man.get("source_row_indices", [])), set(mine["source_row_indices"])
        print(f"MISMATCH source_row_indices: only-original={len(a-b)} only-ours={len(b-a)}")
        ok = False
    print(json.dumps({"ours": {"train_samples": len(train), "val_samples": len(val),
                               "train_rows": len(mine["source_row_indices"])},
                      "original": {"train_samples": man.get("num_training_samples"),
                                   "train_rows": man.get("num_source_rows")},
                      "match": ok}, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
