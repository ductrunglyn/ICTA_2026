#!/usr/bin/env python
"""Aggregate predictions into an EvaluationCard + ablation/confound report.

Loads per-fold/seed prediction files for one experiment, computes the primary
metric suite with participant-level bootstrap CIs, the confound-conditioned
report, and (optionally) a TOST equivalence test against a second experiment.

Usage:
    python scripts/03_make_report.py --exp E2_corpus_adv \
        --baseline E0_acoustic_only --manifest data/manifests/all.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.confounds import ConfoundExtractor, length_bins  # noqa: E402
from src.eval.confound_eval import ConfoundEvaluator  # noqa: E402
from src.eval.metrics import compute_all_metrics, roc_auc  # noqa: E402
from src.eval.stats import bootstrap_ci, tost_equivalence  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("make_report")


def load_preds(exp: str, out_dir: str) -> pd.DataFrame:
    """Load and concatenate all prediction files for an experiment."""
    pred_dir = Path(out_dir) / "preds" / exp
    frames: List[pd.DataFrame] = []
    for f in sorted(pred_dir.glob("*.parquet")):
        frames.append(pd.read_parquet(f))
    for f in sorted(pred_dir.glob("*.csv")):
        frames.append(pd.read_csv(f))
    if not frames:
        raise FileNotFoundError(f"No predictions found under {pred_dir}")
    return pd.concat(frames, ignore_index=True)


def per_run_auc(preds: pd.DataFrame) -> Dict[tuple, float]:
    """AUC per (fold, seed) run, for TOST paired differences."""
    out = {}
    for (fold, seed), grp in preds.groupby(["fold", "seed"]):
        out[(int(fold), int(seed))] = roc_auc(grp["label"].to_numpy(), grp["prob"].to_numpy())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--baseline", default=None, help="Experiment to TOST against.")
    ap.add_argument("--manifest", default="data/manifests/all.csv")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--report_out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    preds = load_preds(args.exp, args.out_dir)
    manifest = pd.read_csv(args.manifest)

    # Pool predictions across seeds by averaging per participant (then evaluate).
    pooled = preds.groupby("participant_id").agg(
        label=("label", "first"),
        prob=("prob", "mean"),
        logit=("logit", "mean"),
        gate=("gate", "mean"),
    ).reset_index()
    merged = pooled.merge(manifest, on="participant_id", how="left", suffixes=("", "_m"))

    y = merged["label"].to_numpy()
    p = merged["prob"].to_numpy()
    logit = merged["logit"].to_numpy()

    metrics = compute_all_metrics(y, p, threshold=0.5)
    auc_mean, (auc_lo, auc_hi) = bootstrap_ci(y, p, n=cfg.stats.bootstrap_n)

    # Confound report.
    conf = ConfoundEvaluator()
    C = ConfoundExtractor().fit_transform(merged)
    groups: Dict[str, np.ndarray] = {}
    if "gender" in merged:
        groups["gender"] = merged["gender"].fillna(0).astype(int).to_numpy()
    if "corpus_id" in merged:
        groups["corpus"] = merged["corpus_id"].fillna(0).astype(int).to_numpy()
    elif "corpus" in merged:
        groups["corpus"] = pd.factorize(merged["corpus"])[0]
    if "interview_len_s" in merged:
        groups["length"] = length_bins(merged["interview_len_s"].to_numpy())
    confound_report = conf.evaluate(y, p, logit, C, groups)

    # TOST equivalence vs baseline (paired per fold x seed AUC differences).
    tost_p: Optional[float] = None
    if args.baseline:
        base_preds = load_preds(args.baseline, args.out_dir)
        a, b = per_run_auc(preds), per_run_auc(base_preds)
        diffs = [a[k] - b[k] for k in a if k in b and not (np.isnan(a[k]) or np.isnan(b[k]))]
        if diffs:
            tost_p = tost_equivalence(diffs, eps=cfg.stats.eps_auc)

    card = {
        "experiment": args.exp,
        "n_participants": int(len(merged)),
        "metrics": metrics,
        "auc_bootstrap": {"mean": auc_mean, "ci95": [auc_lo, auc_hi]},
        "confound_report": confound_report,
        "tost_vs_baseline": (
            {"baseline": args.baseline, "eps_auc": cfg.stats.eps_auc, "p": tost_p}
            if args.baseline else None
        ),
    }

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"{args.exp}_card.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2, default=float)

    md_path = Path(args.report_out) if args.report_out else out_dir / f"{args.exp}_card.md"
    _write_markdown(card, md_path)
    logger.info("EvaluationCard written: %s / %s", json_path, md_path)


def _write_markdown(card: Dict, path: Path) -> None:
    lines = [f"# EvaluationCard — {card['experiment']}", ""]
    lines.append(f"- Participants: **{card['n_participants']}**")
    m = card["metrics"]
    ab = card["auc_bootstrap"]
    lines.append(
        f"- AUC: **{m['auc']:.3f}** "
        f"(bootstrap mean {ab['mean']:.3f}, 95% CI [{ab['ci95'][0]:.3f}, {ab['ci95'][1]:.3f}])"
    )
    lines.append(f"- Partial AUC: {m['partial_auc']:.3f} | ECE: {m['ece']:.3f} | Brier: {m['brier']:.3f}")
    lines.append(
        f"- Sensitivity: {m['sensitivity']:.3f} | Specificity: {m['specificity']:.3f} | F1: {m['f1']:.3f}"
    )
    cr = card["confound_report"]
    lines += ["", "## Confound report", ""]
    lines.append(f"- Residualized AUC: {cr.get('residualized_auc', float('nan')):.3f}")
    for key, val in cr.items():
        if key.startswith("auc_by_"):
            pretty = ", ".join(f"{g}:{v:.3f}" for g, v in val.items())
            lines.append(f"- {key}: {pretty}")
    if card.get("tost_vs_baseline"):
        t = card["tost_vs_baseline"]
        verdict = "EQUIVALENT" if (t["p"] is not None and t["p"] < 0.05) else "not equivalent"
        lines += ["", "## TOST equivalence", ""]
        lines.append(
            f"- vs `{t['baseline']}` (eps={t['eps_auc']}): p={t['p']} -> **{verdict}**"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
