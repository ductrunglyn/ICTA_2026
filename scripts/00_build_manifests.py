#!/usr/bin/env python
"""Build the participant-level manifest from per-corpus label files.

Reads ``configs/corpora.yaml`` and each corpus' ``label_csv`` to produce a
unified ``data/manifests/all.csv`` whose one-row-per-participant schema is the
unit of leakage-free splitting.

Usage:
    python scripts/00_build_manifests.py \
        --corpora configs/corpora.yaml --out data/manifests/all.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.splitter import add_corpus_id  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("build_manifests")

# Heuristic column name candidates per corpus label file.
ID_CANDIDATES = ["Participant_ID", "participant_id", "id", "pid"]
LABEL_CANDIDATES = ["PHQ8_Binary", "label", "binary", "Depression"]
SCORE_CANDIDATES = ["PHQ8_Score", "SDS", "severity", "score"]
GENDER_CANDIDATES = ["Gender", "gender", "sex"]


def _pick(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def build_for_corpus(name: str, spec) -> pd.DataFrame:
    """Build manifest rows for a single corpus from its label CSV.

    Args:
        name: Corpus key in corpora.yaml.
        spec: Corpus spec (Config).

    Returns:
        DataFrame with the unified manifest columns for this corpus.
    """
    label_csv = Path(spec["label_csv"])
    if not label_csv.exists():
        # TODO(external-data): corpus label files are not distributed with the
        # code. Provide them under data/raw/ then re-run this script.
        logger.warning("Label file missing for %s: %s (skipping)", name, label_csv)
        return pd.DataFrame()

    raw = pd.read_csv(label_csv)
    id_col = _pick(raw, ID_CANDIDATES)
    label_col = _pick(raw, LABEL_CANDIDATES)
    if not id_col or not label_col:
        logger.warning("Could not find id/label columns for %s; columns=%s", name, list(raw.columns))
        return pd.DataFrame()

    corpus = spec.get("corpus", name)
    score_col = _pick(raw, SCORE_CANDIDATES)
    gender_col = _pick(raw, GENDER_CANDIDATES)

    out = pd.DataFrame()
    out["participant_id"] = corpus + "_" + raw[id_col].astype(str)
    out["corpus"] = corpus
    out["language"] = spec.get("language", "unknown")
    out["label"] = raw[label_col].astype(int)
    out["severity"] = raw[score_col].astype(float) if score_col else float("nan")
    out["gender"] = raw[gender_col].astype(int) if gender_col else 0
    out["age"] = raw["age"].astype(float) if "age" in raw.columns else float("nan")
    out["interview_len_s"] = (
        raw["interview_len_s"].astype(float) if "interview_len_s" in raw.columns else float("nan")
    )
    out["comorbidity_ptsd"] = (
        raw["PTSD"].astype(int) if "PTSD" in raw.columns else 0
    )
    out["n_segments"] = 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", default="configs/corpora.yaml")
    ap.add_argument("--out", default="data/manifests/all.csv")
    args = ap.parse_args()

    corpora = load_config(args.corpora)
    frames = [build_for_corpus(name, spec) for name, spec in corpora.items()]
    frames = [f for f in frames if not f.empty]

    if not frames:
        logger.error("No corpus produced rows. Add label files under data/raw/.")
        # Still emit an empty, correctly-typed manifest so downstream imports work.
        cols = [
            "participant_id", "corpus", "language", "label", "severity",
            "gender", "age", "interview_len_s", "comorbidity_ptsd", "n_segments",
        ]
        manifest = pd.DataFrame(columns=cols)
    else:
        manifest = pd.concat(frames, ignore_index=True)
        manifest = add_corpus_id(manifest)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    logger.info("Wrote %d participants to %s", len(manifest), out)


if __name__ == "__main__":
    main()
