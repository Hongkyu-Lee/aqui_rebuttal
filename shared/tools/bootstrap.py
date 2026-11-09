"""E3: seed statistics + paired bootstrap between two methods (e.g. AQUI vs symmetric LoRA).

Inputs are fact_retrieval result JSONs (one per training seed) for each method, on the SAME
benchmark. Per question, the per-seed session_recall@K values are averaged over seeds, then the
difference A-B is bootstrapped over questions (paired: same question in both arms).

  python -m tools.bootstrap --a aqui_s42.json aqui_s43.json aqui_s44.json \
        --b sym_s42.json sym_s43.json sym_s44.json --ks 10,20,50,100 --out stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def per_question(path: str, k: str) -> dict[tuple[int, int], float]:
    d = json.load(open(path, encoding="utf-8"))
    return {(int(r["example_index"]), int(r["qa_index"])): float(r["metrics"][k]["session_recall"])
            for r in d["results"]}


def summary_value(path: str, k: str) -> float:
    return float(json.load(open(path, encoding="utf-8"))["summary"][f"session_recall@{k}"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="method A result JSONs (one per seed)")
    ap.add_argument("--b", nargs="+", required=True, help="method B result JSONs (one per seed)")
    ap.add_argument("--name-a", default="AQUI")
    ap.add_argument("--name-b", default="symmetric")
    ap.add_argument("--ks", default="10,20,50,100")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    report = {"A": args.name_a, "B": args.name_b, "seeds_A": len(args.a), "seeds_B": len(args.b), "per_k": {}}
    for k in [x.strip() for x in args.ks.split(",") if x.strip()]:
        a_seeds = [per_question(p, k) for p in args.a]
        b_seeds = [per_question(p, k) for p in args.b]
        keys = sorted(set.intersection(*(set(s) for s in a_seeds + b_seeds)))
        a = [statistics.mean(s[q] for s in a_seeds) for q in keys]
        b = [statistics.mean(s[q] for s in b_seeds) for q in keys]
        diff = [x - y for x, y in zip(a, b)]
        n = len(diff)
        boots = []
        for _ in range(args.n_boot):
            idx = [rng.randrange(n) for _ in range(n)]
            boots.append(sum(diff[i] for i in idx) / n)
        boots.sort()
        mean_diff = sum(diff) / n
        p_two = 2 * min(sum(x <= 0 for x in boots), sum(x >= 0 for x in boots)) / args.n_boot
        sa = [summary_value(p, k) for p in args.a]
        sb = [summary_value(p, k) for p in args.b]
        report["per_k"][k] = {
            "n_questions": n,
            f"{args.name_a}_mean": statistics.mean(sa), f"{args.name_a}_std": statistics.pstdev(sa) if len(sa) > 1 else 0.0,
            f"{args.name_b}_mean": statistics.mean(sb), f"{args.name_b}_std": statistics.pstdev(sb) if len(sb) > 1 else 0.0,
            "diff_mean (seed-averaged, paired)": mean_diff,
            "diff_95ci": [boots[int(0.025 * args.n_boot)], boots[int(0.975 * args.n_boot) - 1]],
            "p_value_two_sided": min(1.0, p_two),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
