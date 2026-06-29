#!/usr/bin/env python
"""Run multi-seed cross-validation for one experiment.

Merges ``configs/default.yaml`` with an optional experiment override, builds a
:class:`BagBuilder` over the cached features and runs :class:`CVRunner`.

Usage:
    python scripts/02_run_cv.py --experiment configs/experiments/E2_corpus_adv.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bag_builder import BagBuilder  # noqa: E402
from src.data.features import FeatureCache, MODALITIES  # noqa: E402
from src.models.transval_net import TransValNet  # noqa: E402
from src.train.cv_runner import CVConfig, CVRunner  # noqa: E402
from src.train.trainer import TrainerConfig  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("run_cv")


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_configs(cfg) -> (TrainerConfig, CVConfig):
    trainer_cfg = TrainerConfig(
        epochs=cfg.train.epochs,
        lr=float(cfg.train.lr),
        grad_clip=cfg.train.grad_clip,
        patience=cfg.train.patience,
        alpha=cfg.loss.alpha,
        beta=cfg.loss.beta,
        gamma=cfg.loss.gamma,
        delta=cfg.loss.delta,
        pos_weight=cfg.loss.pos_weight,
        use_group_dro=cfg.model.use_group_dro,
        use_irm=cfg.model.use_irm,
        grl_gamma=cfg.train.grl_gamma,
        weight_decay=float(cfg.train.weight_decay),
        device=cfg.train.device,
        calib_method=cfg.calib.method,
        threshold_strategy=cfg.calib.threshold,
    )
    cv_cfg = CVConfig(
        n_folds=cfg.cv.n_folds,
        seeds=list(cfg.cv.seeds),
        mode=cfg.cv.mode,
        batch_bags=cfg.train.batch_bags,
        inner_val_frac=cfg.cv.inner_val_frac,
        out_dir=cfg.cv.out_dir,
        exp_name=cfg.cv.exp_name,
    )
    return trainer_cfg, cv_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--manifest", default="data/manifests/all.csv")
    ap.add_argument("--segments", default="data/manifests/segments.csv")
    ap.add_argument("--cache_dir", default="data/interim/features")
    ap.add_argument("--modalities", default=None,
                    help="Comma-separated modality subset, e.g. 'acoustic' for E0.")
    ap.add_argument("--device", default=None,
                    help="Torch device for training (cuda, cuda:0, cpu). "
                         "Overrides train.device; defaults to cuda if available.")
    args = ap.parse_args()

    base = load_config(args.config).to_dict()
    if args.experiment:
        override = load_config(args.experiment).to_dict()
        merged = _deep_merge(base, override)
    else:
        merged = base
    from src.utils.config import Config

    cfg = Config(merged)

    # Resolve training device (CLI override > config > auto-detect).
    if args.device:
        cfg.train.device = args.device
    elif cfg.train.device == "cpu":
        import torch

        if torch.cuda.is_available():
            cfg.train.device = "cuda"
    logger.info("Training device: %s", cfg.train.device)

    manifest = pd.read_csv(args.manifest)
    segments = pd.read_csv(args.segments)
    cache = FeatureCache(args.cache_dir)

    modalities = (
        [m.strip() for m in args.modalities.split(",")] if args.modalities else None
    )
    if cfg.cv.exp_name.startswith("E0") and modalities is None:
        modalities = ["acoustic"]  # acoustic-only baseline

    builder = BagBuilder(manifest, segments, cache, modalities=modalities)
    n_groups = builder.n_groups()
    n_corpus = int(builder.manifest["corpus_id"].max()) + 1

    # Acoustic input dim is inferred from the cached features (e.g. 79 for
    # COVAREP, 25 for openSMILE eGeMAPS) so the encoder matches the data.
    in_dims = None
    sample = builder.build(list(builder.manifest.index[:1]))
    if sample:
        for seg in sample[0].segments:
            ac = seg.get("acoustic")
            if ac is not None:
                in_dims = {"acoustic": int(np.asarray(ac).shape[-1])}
                break

    def model_factory() -> TransValNet:
        return TransValNet(
            d=cfg.model.d,
            n_corpus=n_corpus,
            n_gender=cfg.model.n_gender,
            use_adv=cfg.model.use_adv,
            use_visual=cfg.model.use_visual,
            dropout=float(cfg.model.get("dropout", 0.0)),
            in_dims=in_dims,
        )

    trainer_cfg, cv_cfg = build_configs(cfg)
    runner = CVRunner(
        manifest_df=manifest,
        bag_provider=builder.build,
        model_factory=model_factory,
        trainer_cfg=trainer_cfg,
        cv_cfg=cv_cfg,
        n_groups=n_groups,
    )
    result = runner.run()

    out_dir = Path(cfg.cv.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{cfg.cv.exp_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(result["aggregate"], fh, indent=2)
    logger.info("Wrote summary to %s", summary_path)
    for metric, stats in result["aggregate"].items():
        logger.info("  %-14s mean=%.4f std=%.4f (n=%d)",
                    metric, stats["mean"], stats["std"], stats["n"])


if __name__ == "__main__":
    main()
