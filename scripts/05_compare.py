#!/usr/bin/env python
"""Compare two baseline/experiment prediction sets (paired diff + TOST).

Loads two ``<exp>_preds.csv`` files (from 04_simple_baseline.py or any source
with columns participant_id,label,prob,fold,seed), computes per-(fold,seed) AUC
for each, and reports:

* mean AUC difference (A - B) with a paired summary,
* a paired t-test p-value (is A different from B?),
* a TOST equivalence p-value within +/- eps (are A and B equivalent?).

Usage:
    python scripts/05_compare.py --a baseline_fusion --b baseline_text --eps 0.03
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import roc_auc  # noqa: E402
from src.eval.stats import tost_equivalence  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("compare")


def _per_run_auc(preds: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    for (fold, seed), g in preds.groupby(["fold", "seed"]):
        out[(int(fold), int(seed))] = roc_auc(g["label"].to_numpy(), g["prob"].to_numpy())
    return out


def _load(exp: str, out_dir: str) -> pd.DataFrame:
    p = Path(out_dir) / f"{exp}_preds.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing predictions: {p}")
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="Experiment A name (e.g. baseline_fusion).")
    ap.add_argument("--b", required=True, help="Experiment B name (e.g. baseline_text).")
    ap.add_argument("--eps", type=float, default=0.03, help="AUC equivalence margin.")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()

    a, b = _per_run_auc(_load(args.a, args.out_dir)), _per_run_auc(_load(args.b, args.out_dir))
    keys = sorted(set(a) & set(b))
    diffs = np.array([a[k] - b[k] for k in keys if not (np.isnan(a[k]) or np.isnan(b[k]))])
    if len(diffs) < 2:
        logger.error("Not enough paired (fold,seed) runs to compare.")
        return

    mean_a = np.nanmean([a[k] for k in keys])
    mean_b = np.nanmean([b[k] for k in keys])
    mdiff = diffs.mean()

    # Paired t-test (difference != 0?).
    from scipy.stats import ttest_rel

    paired = [(a[k], b[k]) for k in keys if not (np.isnan(a[k]) or np.isnan(b[k]))]
    t_p = ttest_rel([x for x, _ in paired], [y for _, y in paired]).pvalue
    tost_p = tost_equivalence(diffs, args.eps)

    logger.info("=== Compare %s vs %s (n=%d paired runs) ===", args.a, args.b, len(diffs))
    logger.info("AUC  A=%.3f  B=%.3f  mean(A-B)=%+.3f", mean_a, mean_b, mdiff)
    logger.info("Paired t-test p=%.4f (%s)", t_p,
                "A!=B" if t_p < 0.05 else "no significant difference")
    logger.info("TOST eps=%.3f p=%.4f -> %s", args.eps, tost_p,
                "EQUIVALENT" if (not np.isnan(tost_p) and tost_p < 0.05) else "not equivalent")

    verdict = "A superior" if (t_p < 0.05 and mdiff > 0) else (
        "B superior" if (t_p < 0.05 and mdiff < 0) else (
            "equivalent" if (not np.isnan(tost_p) and tost_p < 0.05) else "inconclusive"))
    logger.info("Verdict: %s", verdict)


if __name__ == "__main__":
    main()
