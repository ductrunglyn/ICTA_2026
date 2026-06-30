#!/usr/bin/env python
"""Simple participant-level baseline: pooled features + regularised classifier.

The mandatory reference (E0-style) the deep model must beat, and a diagnostic.
At n~=56 with thousands of pooled features the bottleneck is the curse of
dimensionality, so this baseline adds the levers that actually matter for small
data:

* **Dimensionality reduction** (``--pca K``) before the classifier.
* **Automatic L2 strength** selection (``--C auto``) via inner stratified CV.
* **L1 / L2** penalties (``--penalty``).
* **Probability calibration** on an inner split (``--calibrate``).
* **Late fusion** (``--fusion late``): train one classifier per modality and
  average their (optionally calibrated) probabilities — robust when one modality
  (audio) is noisy and would otherwise swamp a strong one (text) in early
  concatenation.

Per participant each modality is pooled to ``[mean || std]`` over segments
(after mean-over-time per segment). Evaluation uses the leakage-free
:class:`LeakageFreeSplitter` (K-fold x multi-seed) with bootstrap CIs.

Usage:
    python scripts/04_simple_baseline.py --modalities text --C auto --pca 30 \
        --calibrate --exp_name baseline_text
    python scripts/04_simple_baseline.py --modalities audio,text --fusion late \
        --C auto --pca 30 --calibrate --exp_name baseline_latefusion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import FeatureCache  # noqa: E402
from src.data.splitter import LeakageFreeSplitter, add_corpus_id  # noqa: E402
from src.eval.metrics import compute_all_metrics  # noqa: E402
from src.eval.stats import aggregate_seeds, bootstrap_ci  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = get_logger("simple_baseline")

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


def _pool_segment(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    return a.mean(axis=0) if a.ndim == 2 else a


def _modality_vectors(
    seg_ids: List[str], cache: FeatureCache, dims: Dict[str, int], modalities: List[str]
) -> Dict[str, np.ndarray]:
    """Per-modality ``[mean || std]`` participant vector (zeros if absent)."""
    per_mod: Dict[str, List[np.ndarray]] = {m: [] for m in modalities}
    for sid in seg_ids:
        feat = cache.load(sid)
        for m in modalities:
            val = feat.get(m)
            if val is not None:
                per_mod[m].append(_pool_segment(val))
    out: Dict[str, np.ndarray] = {}
    for m in modalities:
        d = dims[m]
        if per_mod[m]:
            stack = np.stack(per_mod[m], axis=0)
            out[m] = np.concatenate([stack.mean(0), stack.std(0)])
        else:
            out[m] = np.zeros(2 * d)
    return out


def _infer_dims(segments: pd.DataFrame, cache: FeatureCache, modalities: List[str]) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    for sid in segments["seg_id"]:
        feat = cache.load(sid)
        for m in modalities:
            if m not in dims and feat.get(m) is not None:
                dims[m] = int(_pool_segment(feat[m]).shape[-1])
        if len(dims) == len(modalities):
            break
    for m in modalities:
        dims.setdefault(m, 1)
    return dims


def _make_clf(C: float, penalty: str):
    from sklearn.linear_model import LogisticRegression

    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return LogisticRegression(
        C=C, penalty=penalty, solver=solver, class_weight="balanced", max_iter=5000
    )


def _select_C(X: np.ndarray, y: np.ndarray, penalty: str, seed: int) -> float:
    """Pick C by inner stratified CV AUC (falls back to 1.0 if degenerate)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if len(np.unique(y)) < 2 or len(y) < 6:
        return 1.0
    best_C, best = 1.0, -np.inf
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    for C in C_GRID:
        try:
            s = cross_val_score(_make_clf(C, penalty), X, y, cv=inner, scoring="roc_auc").mean()
        except ValueError:
            continue
        if s > best:
            best, best_C = s, C
    return best_C


def _fit_predict(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    C: str,
    penalty: str,
    pca: int,
    calibrate: bool,
    seed: int,
) -> np.ndarray:
    """Standardise -> (PCA) -> LogReg (-> calibrate); return test probabilities."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    if pca and pca > 0 and pca < min(Xtr_s.shape):
        from sklearn.decomposition import PCA

        red = PCA(n_components=pca, random_state=seed).fit(Xtr_s)
        Xtr_s, Xte_s = red.transform(Xtr_s), red.transform(Xte_s)

    C_val = _select_C(Xtr_s, ytr, penalty, seed) if C == "auto" else float(C)

    if calibrate and len(ytr) >= 12 and len(np.unique(ytr)) == 2:
        from sklearn.calibration import CalibratedClassifierCV

        base = _make_clf(C_val, penalty)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(Xtr_s, ytr)
    else:
        clf = _make_clf(C_val, penalty).fit(Xtr_s, ytr)
    return clf.predict_proba(Xte_s)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifests/all.csv")
    ap.add_argument("--segments", default="data/manifests/segments.csv")
    ap.add_argument("--cache_dir", default="data/interim/features")
    ap.add_argument("--modalities", default="audio,text")
    ap.add_argument("--fusion", default="early", choices=["early", "late"])
    ap.add_argument("--C", default="auto", help="'auto' (inner-CV) or a float.")
    ap.add_argument("--penalty", default="l2", choices=["l2", "l1"])
    ap.add_argument("--pca", type=int, default=30, help="PCA components (0=off).")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--exp_name", default="baseline_logreg")
    args = ap.parse_args()

    modalities = [m.strip() for m in args.modalities.split(",")]
    manifest = pd.read_csv(args.manifest)
    if "corpus_id" not in manifest.columns:
        manifest = add_corpus_id(manifest)
    segments = pd.read_csv(args.segments)
    cache = FeatureCache(args.cache_dir)

    dims = _infer_dims(segments, cache, modalities)
    logger.info("Per-segment pooled dims: %s | fusion=%s C=%s pca=%d calib=%s",
                dims, args.fusion, args.C, args.pca, args.calibrate)

    segs_by_pid = {pid: g["seg_id"].tolist() for pid, g in segments.groupby("participant_id")}
    pids = [p for p in manifest["participant_id"] if p in segs_by_pid]
    logger.info("Building pooled vectors for %d participants...", len(pids))

    # Per-modality feature matrices (so late fusion can train separately).
    mod_mats: Dict[str, List[np.ndarray]] = {m: [] for m in modalities}
    for p in pids:
        vecs = _modality_vectors(segs_by_pid[p], cache, dims, modalities)
        for m in modalities:
            mod_mats[m].append(vecs[m])
    Xmod = {m: np.stack(v) for m, v in mod_mats.items()}
    Xfull = np.concatenate([Xmod[m] for m in modalities], axis=1)

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

            if args.fusion == "late" and len(modalities) > 1:
                probs = []
                for m in modalities:
                    probs.append(_fit_predict(
                        Xmod[m][tr_idx], y[tr_idx], Xmod[m][te_idx],
                        args.C, args.penalty, args.pca, args.calibrate, seed))
                prob = np.mean(probs, axis=0)
            else:
                prob = _fit_predict(
                    Xfull[tr_idx], y[tr_idx], Xfull[te_idx],
                    args.C, args.penalty, args.pca, args.calibrate, seed)

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
        "fusion": args.fusion,
        "C": args.C, "penalty": args.penalty, "pca": args.pca, "calibrate": args.calibrate,
        "feature_dim": int(Xfull.shape[1]),
        "aggregate": agg,
        "auc_bootstrap": {"mean": auc_mean, "ci95": [lo, hi]},
    }
    with open(out_dir / f"{args.exp_name}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    pred_df.to_csv(out_dir / f"{args.exp_name}_preds.csv", index=False)

    logger.info("=== Simple baseline (%s) ===", args.exp_name)
    logger.info("AUC per-run: %.3f +/- %.3f (n=%d)",
                agg["auc"]["mean"], agg["auc"]["std"], agg["auc"]["n"])
    logger.info("AUC pooled bootstrap: %.3f  CI95 [%.3f, %.3f]", auc_mean, lo, hi)
    logger.info("F1 %.3f | ECE %.3f | Brier %.3f",
                agg["f1"]["mean"], agg["ece"]["mean"], agg["brier"]["mean"])


if __name__ == "__main__":
    main()
