"""Multi-seed, K-fold (or LOCO) cross-validation orchestrator (NV1 + NV5).

Drives the full leakage-free protocol:

1. For each seed and fold, split participants (test fold held out).
2. Carve an inner-validation slice out of the training fold.
3. Train, then fit the calibrator/threshold on the inner slice only.
4. Predict each test participant exactly once and persist predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ..data.dataset import Bag, BagDataset, collate_bags
from ..data.splitter import LeakageFreeSplitter
from ..eval.metrics import compute_all_metrics
from ..eval.stats import aggregate_seeds
from ..models.transval_net import TransValNet
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .trainer import Trainer, TrainerConfig

logger = get_logger("transval.cv")

BagProvider = Callable[[List[str]], List[Bag]]
ModelFactory = Callable[[], TransValNet]


@dataclass
class CVConfig:
    """Cross-validation orchestration settings."""

    n_folds: int = 5
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    mode: str = "pooled"            # pooled | loco
    batch_bags: int = 8
    inner_val_frac: float = 0.2
    num_workers: int = 0
    out_dir: str = "outputs"
    exp_name: str = "exp"


class CVRunner:
    """Run multi-seed cross-validation for one experiment.

    Args:
        manifest_df: Participant-level manifest (needs ``participant_id``,
            ``label``, ``corpus``).
        bag_provider: Maps a list of participant ids to their :class:`Bag`s.
        model_factory: Returns a fresh :class:`TransValNet` per fold/seed.
        trainer_cfg: Training hyper-parameters.
        cv_cfg: CV orchestration settings.
        n_groups: Number of Group-DRO groups.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        bag_provider: BagProvider,
        model_factory: ModelFactory,
        trainer_cfg: TrainerConfig,
        cv_cfg: CVConfig,
        n_groups: int,
    ) -> None:
        self.manifest = manifest_df.reset_index(drop=True)
        self.bag_provider = bag_provider
        self.model_factory = model_factory
        self.trainer_cfg = trainer_cfg
        self.cv_cfg = cv_cfg
        self.n_groups = n_groups
        self._label_by_pid = dict(
            zip(self.manifest["participant_id"], self.manifest["label"])
        )
        self.out_dir = Path(cv_cfg.out_dir)
        (self.out_dir / "preds" / cv_cfg.exp_name).mkdir(parents=True, exist_ok=True)
        (self.out_dir / "splits").mkdir(parents=True, exist_ok=True)

    # -- helpers ------------------------------------------------------------
    def _loader(self, bags: List[Bag], shuffle: bool) -> DataLoader:
        return DataLoader(
            BagDataset(bags),
            batch_size=self.cv_cfg.batch_bags,
            shuffle=shuffle,
            num_workers=self.cv_cfg.num_workers,
            collate_fn=lambda b: collate_bags(b),
        )

    def _inner_split(self, train_ids: List[str], seed: int) -> tuple:
        """Stratified participant-level inner train/val split."""
        from sklearn.model_selection import train_test_split

        labels = [int(self._label_by_pid[p]) for p in train_ids]
        stratify = labels if len(set(labels)) > 1 else None
        inner_tr, inner_val = train_test_split(
            train_ids,
            test_size=self.cv_cfg.inner_val_frac,
            random_state=seed,
            stratify=stratify,
        )
        return inner_tr, inner_val

    # -- main loop ----------------------------------------------------------
    def run(self) -> Dict[str, object]:
        """Execute all seeds x folds.

        Returns:
            Dict with ``per_run`` metric dicts, ``aggregate`` (mean/std/n) and
            ``pred_files`` written.
        """
        per_run: List[Dict[str, float]] = []
        pred_files: List[str] = []

        for seed in self.cv_cfg.seeds:
            set_seed(seed)
            splitter = LeakageFreeSplitter(
                self.manifest, n_folds=self.cv_cfg.n_folds, seed=seed, mode=self.cv_cfg.mode
            )
            for fold, (train_ids, test_ids) in enumerate(splitter.folds()):
                metrics, preds_df = self._run_fold(train_ids, test_ids, fold, seed)
                per_run.append(metrics)

                pf = self.out_dir / "preds" / self.cv_cfg.exp_name / f"{fold}_{seed}.parquet"
                self._save_parquet(preds_df, pf)
                pred_files.append(str(pf))

                self._save_split(train_ids, test_ids, fold, seed)
                logger.info(
                    "[%s] seed=%d fold=%d AUC=%.4f ECE=%.4f",
                    self.cv_cfg.exp_name, seed, fold, metrics.get("auc", float("nan")),
                    metrics.get("ece", float("nan")),
                )

        return {
            "per_run": per_run,
            "aggregate": aggregate_seeds(per_run),
            "pred_files": pred_files,
        }

    def _run_fold(
        self, train_ids: List[str], test_ids: List[str], fold: int, seed: int
    ) -> tuple:
        inner_tr, inner_val = self._inner_split(train_ids, seed)
        train_loader = self._loader(self.bag_provider(inner_tr), shuffle=True)
        val_loader = self._loader(self.bag_provider(inner_val), shuffle=False)

        test_bags = self.bag_provider(test_ids)
        test_loader = self._loader(test_bags, shuffle=False)

        model = self.model_factory()
        trainer = Trainer(model, self.trainer_cfg, self.n_groups)
        trainer.fit(train_loader, val_loader)
        trainer.calibrate(val_loader)

        out = trainer.predict(test_loader)
        metrics = compute_all_metrics(out["labels"], out["prob"], trainer.threshold)
        metrics["threshold"] = float(trainer.threshold)

        preds_df = pd.DataFrame(
            {
                "participant_id": [b.participant_id for b in test_bags],
                "label": out["labels"],
                "prob": out["prob"],
                "prob_raw": out["prob_raw"],
                "logit": out["logits"],
                "gate": out["gate"],
                "fold": fold,
                "seed": seed,
            }
        )
        return metrics, preds_df

    # -- persistence --------------------------------------------------------
    @staticmethod
    def _save_parquet(df: pd.DataFrame, path: Path) -> None:
        try:
            df.to_parquet(path, index=False)
        except (ImportError, ValueError):  # pragma: no cover - no parquet engine
            df.to_csv(path.with_suffix(".csv"), index=False)

    def _save_split(
        self, train_ids: List[str], test_ids: List[str], fold: int, seed: int
    ) -> None:
        rows = [{"participant_id": p, "split": "train"} for p in train_ids] + \
               [{"participant_id": p, "split": "test"} for p in test_ids]
        pd.DataFrame(rows).to_csv(
            self.out_dir / "splits" / f"{self.cv_cfg.exp_name}_{fold}_{seed}.csv",
            index=False,
        )
