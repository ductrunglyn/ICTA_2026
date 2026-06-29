"""Question-type validity probe (revived segmentation, used for analysis).

Groups segment-level logits by ``question_type`` x ``corpus`` to measure which
question types carry transferable, confound-robust depression signal. Produces
a table of ``question_type x {AUC_in, AUC_transfer, AUC_residualized}``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.segmentation import QTYPES
from .confound_eval import ConfoundEvaluator


class QuestionTypeProbe:
    """Aggregate segment predictions per question type and corpus.

    Args:
        qtypes: Ordered question-type vocabulary (defaults to
            :data:`src.data.segmentation.QTYPES`).
    """

    def __init__(self, qtypes: Optional[List[str]] = None) -> None:
        self.qtypes = list(qtypes) if qtypes is not None else list(QTYPES)
        self._conf = ConfoundEvaluator()

    def _auc(self, y: np.ndarray, s: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, s))

    def run(
        self,
        seg_logits: np.ndarray,
        seg_labels: np.ndarray,
        qtype_ids: np.ndarray,
        corpus_ids: np.ndarray,
        confounds: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Build the question-type validity table.

        Args:
            seg_logits: ``(N_seg,)`` segment depression logits.
            seg_labels: ``(N_seg,)`` bag labels broadcast to segments.
            qtype_ids: ``(N_seg,)`` question-type ids.
            corpus_ids: ``(N_seg,)`` corpus ids.
            confounds: Optional ``(N_seg, k)`` confound matrix for residualised
                AUC.

        Returns:
            DataFrame with columns ``question_type, n, auc_in, auc_transfer_std,
            auc_residualized``.
        """
        seg_logits = np.asarray(seg_logits, dtype=np.float64).reshape(-1)
        seg_labels = np.asarray(seg_labels).reshape(-1)
        qtype_ids = np.asarray(qtype_ids).reshape(-1)
        corpus_ids = np.asarray(corpus_ids).reshape(-1)
        p = 1.0 / (1.0 + np.exp(-seg_logits))

        rows: List[Dict[str, object]] = []
        for q in np.unique(qtype_ids):
            qmask = qtype_ids == q
            name = self.qtypes[q] if q < len(self.qtypes) else f"q{q}"
            auc_in = self._auc(seg_labels[qmask], p[qmask])

            # Transfer stability: spread of per-corpus AUC for this qtype.
            per_corpus = []
            for c in np.unique(corpus_ids[qmask]):
                cm = qmask & (corpus_ids == c)
                a = self._auc(seg_labels[cm], p[cm])
                if not np.isnan(a):
                    per_corpus.append(a)
            transfer_std = float(np.std(per_corpus)) if len(per_corpus) > 1 else float("nan")

            if confounds is not None and confounds.shape[1] > 0:
                auc_res = self._conf.residualized_auc(
                    seg_labels[qmask], seg_logits[qmask], np.asarray(confounds)[qmask]
                )
            else:
                auc_res = float("nan")

            rows.append(
                {
                    "question_type": name,
                    "n": int(qmask.sum()),
                    "auc_in": auc_in,
                    "auc_transfer_std": transfer_std,
                    "auc_residualized": auc_res,
                }
            )
        return pd.DataFrame(rows).sort_values("auc_in", ascending=False).reset_index(drop=True)
