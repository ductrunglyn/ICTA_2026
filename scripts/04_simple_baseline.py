#!/usr/bin/env python
"""Simple participant-level baseline: pooled features + Logistic Regression.

This is the mandatory reference (E0-style) that the deep model must beat, and a
*diagnostic*: on very small datasets a regularised linear model over
participant-level pooled features is often stronger than a deep MIL network. If
this baseline is also near chance, the bottleneck is the data, not the model.

Per participant, every modality is pooled to a fixed vector:
  segment frames --(mean over time)--> per-segment vector
  per-segment vectors --(mean & std over segments)--> participant vector
Available modality vectors are concatenated (missing modalities -> zeros).
Evaluation uses the same :class:`LeakageFreeSplitter` (5-fold x multi-seed),
participant-level, with bootstrap CIs.

Usage:
    python scripts/04_simple_baseline.py --manifest data/manifests/all.csv \
        --segments data/manifests/segments.csv --modalities audio,text
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

from src.data.dataset import MODALITY_ORDER  # noqa: E402
from src.data.features import FeatureCache  # noqa: E402
from src.data.splitter import LeakageFreeSplitter, add_corpus_id  # noqa: E402
from src.eval.metrics import compute_all_metrics  # noqa: E402
from src.eval.stats import aggregate_seeds, bootstrap_ci  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("simple_baseline")


def _pool_segment(arr: np.ndarray) -> np.ndarray:
    """Pool one segment's array to a vector (mean over time if 2-D)."""
    a = np.asarray(arr, dtype=np.float64)
    return a.mean(axis=0) if a.ndim == 2 else a


def _participant_vector(
    seg_ids: List[str], cache: FeatureCache, dims: Dict[str, int], modalities: List[str]
) -> np.ndarray:
    """Build a fixed-length pooled vector for one participant.

    Args:
        seg_ids: Segment ids belonging to the participant.
        cache: Feature cache.
        dims: Per-modality per-segment pooled dim (global, for zero-fill).
        modalities: Modalities to include.

    Returns:
        Concatenated ``[mean || std]`` vector across modalities.
    """
    per_mod: Dict[str, List[np.ndarray]] = {m: [] for m in modalities}
    for sid in seg_ids:
        feat = cache.load(sid)
        for m in modalities:
            val = feat.get(m)
            if val is not None:
                per_mod[m].append(_pool_segment(val))
    parts: List[np.ndarray] = []
    for m in modalities:
        d = dims[m]
        if per_mod[m]:
            stack = np.stack(per_mod[m], axis=0)        # (n_seg, d)
            parts.append(stack.mean(axis=0))
            parts.append(stack.std(axis=0))
        else:
            parts.append(np.zeros(d))
            parts.append(np.zeros(d))
    return np.concatenate(parts)


def _infer_dims(
    segments: pd.DataFrame, cache: FeatureCache, modalities: List[str]
) -> Dict[str, int]:
    """Infer each modality's per-segment pooled dim from the cache."""
    dims: Dict[str, int] = {}
    for sid in segments["seg_id"]:
        feat = cache.load(sid)
        for m in modalities:
            if m not in dims and feat.get(m) is not None:
                dims[m] = int(_pool_segment(feat[m]).shape[-1])
        if len(dims) == len(modalities):
            break
    for m in modalities:  # fallback for entirely-absent modalities
        dims.setdefault(m, 1)
    return dims


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifests/all.csv")
    ap.add_argument("--segments", default="data/manifests/segments.csv")
    ap.add_argument("--cache_dir", default="data/interim/features")
    ap.add_argument("--modalities", default="audio,text")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--C", type=float, default=1.0, help="Inverse L2 strength.")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--exp_name", default="baseline_logreg")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    modalities = [m.strip() for m in args.modalities.split(",")]
    manifest = pd.read_csv(args.manifest)
    if "corpus_id" not in manifest.columns:
        manifest = add_corpus_id(manifest)
    segments = pd.read_csv(args.segments)
    cache = FeatureCache(args.cache_dir)

    dims = _infer_dims(segments, cache, modalities)
    logger.info("Per-segment pooled dims: %s", dims)

    segs_by_pid = {pid: g["seg_id"].tolist() for pid, g in segments.groupby("participant_id")}
    pids = [p for p in manifest["participant_id"] if p in segs_by_pid]
    logger.info("Building pooled vectors for %d participants...", len(pids))

    X = np.stack([_participant_vector(segs_by_pid[p], cache, dims, modalities) for p in pids])
    label_by_pid = dict(zip(manifest["participant_id"], manifest["label"]))
    y = np.array([int(label_by_pid[p]) for p in pids])
    pid_index = {p: i for i, p in enumerate(pids)}
    man_used = manifest[manifest["participant_id"].isin(pids)].reset_index(drop=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    per_run: List[Dict[str, float]] = []
    all_pred = {"participant_id": [], "label": [], "prob": [], "fold": [], "seed": []}

    for seed in seeds:
        set_seed(seed)
        splitter = LeakageFreeSplitter(man_used, n_folds=args.n_folds, seed=seed, mode="pooled")
        for fold, (tr, te) in enumerate(splitter.folds()):
            tr_idx = [pid_index[p] for p in tr if p in pid_index]
            te_idx = [pid_index[p] for p in te if p in pid_index]
            scaler = StandardScaler().fit(X[tr_idx])
            clf = LogisticRegression(
                C=args.C, class_weight="balanced", max_iter=2000
            ).fit(scaler.transform(X[tr_idx]), y[tr_idx])
            prob = clf.predict_proba(scaler.transform(X[te_idx]))[:, 1]
            per_run.append(compute_all_metrics(y[te_idx], prob, threshold=0.5))
            for j, p in enumerate([pids[i] for i in te_idx]):
                all_pred["participant_id"].append(p)
                all_pred["label"].append(int(y[te_idx][j]))
                all_pred["prob"].append(float(prob[j]))
                all_pred["fold"].append(fold)
                all_pred["seed"].append(seed)

    pred_df = pd.DataFrame(all_pred)
    pooled = pred_df.groupby("participant_id").agg(
        label=("label", "first"), prob=("prob", "mean")
    ).reset_index()
    auc_mean, (lo, hi) = bootstrap_ci(pooled["label"].to_numpy(), pooled["prob"].to_numpy())

    agg = aggregate_seeds(per_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": args.exp_name,
        "n_participants": len(pooled),
        "modalities": modalities,
        "feature_dim": int(X.shape[1]),
        "aggregate": agg,
        "auc_bootstrap": {"mean": auc_mean, "ci95": [lo, hi]},
    }
    with open(out_dir / f"{args.exp_name}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    pred_df.to_csv(out_dir / f"{args.exp_name}_preds.csv", index=False)

    logger.info("=== Simple baseline (%s) ===", args.exp_name)
    logger.info("feature_dim=%d | n=%d", X.shape[1], len(pooled))
    logger.info("AUC per-run: %.3f +/- %.3f (n=%d)",
                agg["auc"]["mean"], agg["auc"]["std"], agg["auc"]["n"])
    logger.info("AUC pooled bootstrap: %.3f  CI95 [%.3f, %.3f]", auc_mean, lo, hi)
    logger.info("F1 %.3f | ECE %.3f | Brier %.3f",
                agg["f1"]["mean"], agg["ece"]["mean"], agg["brier"]["mean"])


if __name__ == "__main__":
    main()
