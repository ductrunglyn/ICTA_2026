"""Per-fold training loop with adversarial schedule and post-hoc calibration.

Implements the total objective from blueprint section 7.1::

    L = L_dep_bag + alpha * L_dep_seg + beta * L_adv + gamma * L_irm + delta * L_cons

with an annealed gradient-reversal strength, optional Group-DRO for the bag
loss, early stopping on an inner-validation AUC, and calibration fit *only* on
the inner-validation split of the training fold.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..calibration.calibrators import ProbabilityCalibrator, choose_threshold
from ..eval.metrics import roc_auc
from ..losses.consistency import intermodal_consistency
from ..losses.group_dro import GroupDROLoss
from ..losses.irm import irm_penalty
from ..models.transval_net import TransValNet
from ..utils.logging import get_logger

logger = get_logger("transval.trainer")


@dataclass
class TrainerConfig:
    """Hyper-parameters for :class:`Trainer`."""

    epochs: int = 40
    lr: float = 1e-3
    grad_clip: float = 1.0
    patience: int = 10
    alpha: float = 0.3          # segment auxiliary weight
    beta: float = 0.5           # adversary weight
    gamma: float = 0.1          # IRM weight
    delta: float = 0.0          # intermodal consistency weight
    pos_weight: float = 2.0     # BCE positive class weight
    use_group_dro: bool = True
    use_irm: bool = False
    grl_gamma: float = 10.0     # steepness of the lambda warmup schedule
    weight_decay: float = 1e-5
    device: str = "cpu"
    calib_method: str = "isotonic"
    threshold_strategy: str = "youden_inner"


def gather_seg_labels(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Broadcast bag labels to their segments via ``seg2bag``.

    Used only for the auxiliary segment loss (segments have no hard label of
    their own — see the MIL formulation).

    Args:
        batch: A collated batch.

    Returns:
        ``(N_seg,)`` float tensor of bag labels mapped to segments.
    """
    return batch["bag_labels"].float()[batch["seg2bag"]]


def _move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


class Trainer:
    """Train and calibrate a :class:`TransValNet` on one fold.

    Args:
        model: The network to train.
        cfg: Training hyper-parameters.
        n_groups: Number of Group-DRO groups.
    """

    def __init__(self, model: TransValNet, cfg: TrainerConfig, n_groups: int) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.group_dro = GroupDROLoss(n_groups).to(self.device) if cfg.use_group_dro else None
        self.pos_weight = torch.tensor(cfg.pos_weight, device=self.device)
        self.calibrator: Optional[ProbabilityCalibrator] = None
        self.threshold: float = 0.5
        self._best_state: Optional[dict] = None

    # -- loss ---------------------------------------------------------------
    def _compute_loss(
        self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        labels = batch["bag_labels"].float()
        per_sample = F.binary_cross_entropy_with_logits(
            out["logit_bag"], labels, pos_weight=self.pos_weight, reduction="none"
        )
        if self.group_dro is not None:
            # Group-DRO and pos_weight are not stacked; use plain per-sample here.
            per_sample_dro = F.binary_cross_entropy_with_logits(
                out["logit_bag"], labels, reduction="none"
            )
            l_dep = self.group_dro(per_sample_dro, batch["group_ids"])
        else:
            l_dep = per_sample.mean()

        seg_labels = gather_seg_labels(batch)
        l_seg = F.binary_cross_entropy_with_logits(out["logit_seg"], seg_labels)

        total = l_dep + self.cfg.alpha * l_seg

        if self.model.adv is not None and "corpus_logit" in out:
            l_adv = F.cross_entropy(out["corpus_logit"], batch["corpus_ids"]) + \
                F.cross_entropy(out["gender_logit"], batch["gender_ids"])
            total = total + self.cfg.beta * l_adv

        if self.cfg.use_irm:
            dummy_w = torch.ones(1, device=self.device, requires_grad=True)
            total = total + self.cfg.gamma * irm_penalty(
                out["logit_bag"], batch["bag_labels"], dummy_w
            )

        if self.cfg.delta > 0:
            # Consistency between segment and bag logit streams (broadcast).
            bag_logit_for_seg = out["logit_bag"][batch["seg2bag"]]
            total = total + self.cfg.delta * intermodal_consistency(
                [out["logit_seg"], bag_logit_for_seg]
            )
        return total

    # -- schedule -----------------------------------------------------------
    def _lambda(self, progress: float) -> float:
        """DANN-style gradient-reversal warmup: ``2/(1+e^{-gamma p}) - 1``."""
        return 2.0 / (1.0 + math.exp(-self.cfg.grl_gamma * progress)) - 1.0

    # -- training -----------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> "Trainer":
        """Train with early stopping on inner-validation AUC.

        Args:
            train_loader: Inner-training bag loader.
            val_loader: Inner-validation bag loader (for early stopping); if
                ``None`` the last epoch's weights are kept.

        Returns:
            ``self`` (with best weights restored).
        """
        best_auc = -np.inf
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        for epoch in range(self.cfg.epochs):
            progress = epoch / max(self.cfg.epochs - 1, 1)
            if self.model.adv is not None:
                self.model.adv.grl.lambd = self._lambda(progress)

            self.model.train()
            for batch in train_loader:
                batch = _move_batch(batch, self.device)
                out = self.model(batch)
                loss = self._compute_loss(out, batch)
                self.opt.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()

            if val_loader is not None:
                y, _, p = self._infer(val_loader)
                auc = roc_auc(y, p)
                auc = auc if not math.isnan(auc) else -np.inf
                if auc > best_auc + 1e-5:
                    best_auc = auc
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.cfg.patience:
                        logger.info("Early stopping at epoch %d (AUC=%.4f)", epoch, best_auc)
                        break

        self.model.load_state_dict(best_state)
        self._best_state = best_state
        return self

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def _infer(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(labels, logits, sigmoid-probs)`` over a loader's bags."""
        self.model.eval()
        ys, logits = [], []
        for batch in loader:
            batch = _move_batch(batch, self.device)
            out = self.model(batch)
            logits.append(out["logit_bag"].cpu().numpy())
            ys.append(batch["bag_labels"].cpu().numpy())
        y = np.concatenate(ys) if ys else np.array([])
        lg = np.concatenate(logits) if logits else np.array([])
        p = 1.0 / (1.0 + np.exp(-lg)) if len(lg) else lg
        return y, lg, p

    # -- calibration --------------------------------------------------------
    def calibrate(self, val_loader: DataLoader) -> "Trainer":
        """Fit the calibrator + threshold on inner-validation only.

        Args:
            val_loader: Inner-validation bag loader (subset of the train fold).

        Returns:
            ``self`` with ``calibrator`` and ``threshold`` set.
        """
        y, logits, _ = self._infer(val_loader)
        self.calibrator = ProbabilityCalibrator(self.cfg.calib_method).fit(logits, y)
        p_cal = self.calibrator.transform(logits)
        self.threshold = choose_threshold(y, p_cal, self.cfg.threshold_strategy)
        return self

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Dict[str, np.ndarray]:
        """Predict on a (test) loader, applying the train-fold calibrator.

        Args:
            loader: Test bag loader.

        Returns:
            Dict with ``labels``, ``logits``, ``prob`` (calibrated),
            ``prob_raw`` (sigmoid) and ``gate``.
        """
        self.model.eval()
        ys, logits, gates = [], [], []
        for batch in loader:
            batch = _move_batch(batch, self.device)
            out = self.model(batch)
            logits.append(out["logit_bag"].cpu().numpy())
            gates.append(out["gate_bag"].cpu().numpy())
            ys.append(batch["bag_labels"].cpu().numpy())
        y = np.concatenate(ys) if ys else np.array([])
        lg = np.concatenate(logits) if logits else np.array([])
        gate = np.concatenate(gates) if gates else np.array([])
        p_raw = 1.0 / (1.0 + np.exp(-lg)) if len(lg) else lg
        p_cal = self.calibrator.transform(lg) if self.calibrator is not None else p_raw
        return {
            "labels": y,
            "logits": lg,
            "prob": np.asarray(p_cal),
            "prob_raw": p_raw,
            "gate": gate,
        }

    @torch.no_grad()
    def predict_segments(self, loader: DataLoader) -> Dict[str, np.ndarray]:
        """Collect segment-level logits/metadata for the validity probe."""
        self.model.eval()
        seg_logits, seg_labels, qtypes, corpus_ids = [], [], [], []
        for batch in loader:
            batch = _move_batch(batch, self.device)
            out = self.model(batch)
            seg_logits.append(out["logit_seg"].cpu().numpy())
            seg_labels.append(gather_seg_labels(batch).cpu().numpy())
            qtypes.append(batch["qtypes"].cpu().numpy())
            corpus_ids.append(batch["corpus_ids"].cpu().numpy())
        cat = lambda xs: np.concatenate(xs) if xs else np.array([])
        return {
            "seg_logits": cat(seg_logits),
            "seg_labels": cat(seg_labels),
            "qtypes": cat(qtypes),
            "corpus_ids": cat(corpus_ids),
        }
