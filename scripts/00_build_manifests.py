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
ID_CANDIDATES = ["Participant_ID", "participant_ID", "participant_id", "id", "pid"]
LABEL_CANDIDATES = ["PHQ8_Binary", "PHQ_Binary", "label", "binary", "Depression"]
SCORE_CANDIDATES = ["PHQ8_Score", "PHQ_Score", "SDS", "severity", "score"]
GENDER_CANDIDATES = ["Gender", "gender", "sex"]


def _pick(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def _label_files(spec) -> list:
    """Return label_csv as a list (supports a single path or a list of paths)."""
    val = spec["label_csv"]
    if isinstance(val, (list, tuple)):
        return list(val)
    # Config wraps lists; fall back to its raw form if needed.
    if hasattr(val, "to_dict"):
        return list(val)  # pragma: no cover
    return [val]


def _read_one(path: Path, corpus: str, language: str) -> pd.DataFrame:
    """Parse a single split/label CSV into unified manifest rows.

    Files lacking a usable binary-label column (e.g. the blind
    ``test_split_Depression_AVEC2017.csv`` that only has gender) are skipped.
    """
    raw = pd.read_csv(path)
    id_col = _pick(raw, ID_CANDIDATES)
    label_col = _pick(raw, LABEL_CANDIDATES)
    if not id_col or not label_col:
        logger.warning("Skipping %s: no id/label columns (have %s)", path.name, list(raw.columns))
        return pd.DataFrame()

    score_col = _pick(raw, SCORE_CANDIDATES)
    gender_col = _pick(raw, GENDER_CANDIDATES)

    out = pd.DataFrame()
    out["participant_id"] = corpus + "_" + raw[id_col].astype(str).str.strip()
    out["corpus"] = corpus
    out["language"] = language
    # Drop rows with a missing/blank binary label before int-casting.
    label = pd.to_numeric(raw[label_col], errors="coerce")
    out["label"] = label
    out["severity"] = pd.to_numeric(raw[score_col], errors="coerce") if score_col else float("nan")
    out["gender"] = pd.to_numeric(raw[gender_col], errors="coerce").fillna(0).astype(int) if gender_col else 0
    out["age"] = pd.to_numeric(raw["age"], errors="coerce") if "age" in raw.columns else float("nan")
    out["interview_len_s"] = (
        pd.to_numeric(raw["interview_len_s"], errors="coerce") if "interview_len_s" in raw.columns else float("nan")
    )
    out["comorbidity_ptsd"] = (
        pd.to_numeric(raw["PTSD"], errors="coerce").fillna(0).astype(int) if "PTSD" in raw.columns else 0
    )
    out = out.dropna(subset=["label"])
    out["label"] = out["label"].astype(int)
    return out


def build_for_corpus(name: str, spec) -> pd.DataFrame:
    """Build manifest rows for a single corpus from one or more label CSVs.

    For DAIC-WOZ/AVEC2017 the official train/dev/test splits are concatenated
    here and re-split later by the leakage-free participant-level CV (the AVEC
    split is *not* used as the evaluation split).

    Args:
        name: Corpus key in corpora.yaml.
        spec: Corpus spec (Config).

    Returns:
        DataFrame with the unified manifest columns for this corpus
        (deduplicated by ``participant_id``).
    """
    corpus = spec.get("corpus", name)
    language = spec.get("language", "unknown")
    frames = []
    for lf in _label_files(spec):
        path = Path(lf)
        if not path.exists():
            # TODO(external-data): provide the label files under data/raw/.
            logger.warning("Label file missing for %s: %s (skipping)", name, path)
            continue
        frames.append(_read_one(path, corpus, language))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    # AVEC splits are disjoint; dedup defensively (keep first labelled row).
    out = out.drop_duplicates(subset=["participant_id"], keep="first").reset_index(drop=True)
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
