"""E1: re-select the HPO configuration using SYNTHETIC validation only.

Inputs
  --hpo-root   <HPO_ROOT>  with adapters/<run>/model_selection.json and HPO_ranking.csv
  --val-csv    output of tools.synthetic_val eval over the adapters (val_mean_hit, val_hit@K, ...)
Outputs
  selection_table.csv : one row per config: HPs, best_eval_loss (synthetic), synthetic val retrieval
                        metrics, and the ImplexConv test recalls already stored by the HPO
                        (recall@10/20/100) -- the latter are NEVER used for selection.
  selection.json      : the config chosen by each synthetic criterion + the paper's choice.

NOTE (bug in the original runner): run_oppose_gen_hpo.sh read `best_metric` from
model_selection.json, but the file stores `best_eval_loss`, so the `best_val_loss` column of
HPO_ranking.csv is probably EMPTY. We therefore read best_eval_loss from each adapter directory
(fallback: trainer_state.json best_metric inside the checkpoints).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

PAPER_RUN = "hpo_v11con_2400_8B__lr5e-5_t0p02_gc1p0_wu0p1_bs128_r16"
TAG_RE = re.compile(r"lr(?P<lr>[0-9ep\-]+)_t(?P<t>[0-9p]+)_gc(?P<gc>[0-9p]+)_wu(?P<wu>[0-9p]+)_bs(?P<bs>\d+)_r(?P<r>\d+)")


def best_eval_loss(adapter_dir: Path):
    ms = adapter_dir / "model_selection.json"
    if ms.exists():
        v = json.loads(ms.read_text()).get("best_eval_loss")
        if v is not None:
            return float(v)
    best = None
    for ts in glob.glob(str(adapter_dir / "checkpoint-*" / "trainer_state.json")):
        st = json.loads(Path(ts).read_text())
        if st.get("best_metric") is not None:
            best = float(st["best_metric"])
        for h in st.get("log_history", []):
            if "eval_loss" in h:
                best = h["eval_loss"] if best is None else min(best, h["eval_loss"])
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpo-root", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    root = Path(args.hpo_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    test = {}
    rank_csv = root / "HPO_ranking.csv"
    if rank_csv.exists():
        for row in csv.DictReader(open(rank_csv, encoding="utf-8")):
            test[row["config"]] = row
    val = {}
    for row in csv.DictReader(open(args.val_csv, encoding="utf-8")):
        val[Path(row["adapter"]).name] = row

    table = []
    for adir in sorted((root / "adapters").glob("hpo_*")):
        run = adir.name
        m = TAG_RE.search(run)
        rec = {"config": run}
        if m:
            rec.update({"lr": m["lr"].replace("p", "."), "temperature": m["t"].replace("p", "."),
                        "max_grad_norm": m["gc"].replace("p", "."), "warmup": m["wu"].replace("p", "."),
                        "batch": m["bs"], "lora_r": m["r"]})
        rec["best_eval_loss"] = best_eval_loss(adir)
        v = val.get(run, {})
        for k in ("val_mean_hit", "val_hit@10", "val_hit@20", "val_hit@50", "val_hit@100", "val_recall@20"):
            rec[k] = float(v[k]) if v.get(k) not in (None, "") else None
        t = test.get(run, {})
        for k in ("recall@10", "recall@20", "recall@100"):
            rec[f"implex_test_{k}"] = float(t[k]) if t.get(k) not in (None, "") else None
        table.append(rec)
    if not table:
        raise SystemExit(f"no adapters under {root/'adapters'}")

    with open(out / "selection_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    def pick(key, lower_better=False):
        cands = [r for r in table if r.get(key) is not None]
        if not cands:
            return None
        return (min if lower_better else max)(cands, key=lambda r: r[key])

    sel = {
        "n_configs": len(table),
        "n_with_val_loss": sum(r["best_eval_loss"] is not None for r in table),
        "n_with_val_retrieval": sum(r["val_mean_hit"] is not None for r in table),
        "criteria": {
            "PRIMARY synthetic val retrieval (val_mean_hit)": pick("val_mean_hit"),
            "synthetic val eval_loss (lowest)": pick("best_eval_loss", lower_better=True),
            "synthetic val hit@20": pick("val_hit@20"),
            "synthetic val hit@100": pick("val_hit@100"),
        },
        "paper_choice (was ranked by ImplexConv test recall@100)": next((r for r in table if r["config"] == PAPER_RUN), None),
        "note": "eval_loss is not comparable across temperatures (InfoNCE scale depends on tau); "
                "val_mean_hit is the pre-registered primary criterion.",
    }
    (out / "selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    print(json.dumps(sel, indent=2))
    chosen = sorted({r["config"] for r in sel["criteria"].values() if r} | {PAPER_RUN})
    (out / "configs_to_test.txt").write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print(f"configs to evaluate on the test benchmarks: {chosen}")


if __name__ == "__main__":
    main()
