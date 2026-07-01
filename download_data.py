#!/usr/bin/env python
"""Download DAIC-WOZ from Kaggle and arrange it into the expected layout.

Downloads via ``kagglehub`` then organises files into::

    data/raw/daic/
    ├── train_split_Depression_AVEC2017.csv
    ├── dev_split_Depression_AVEC2017.csv
    ├── full_test_split.csv
    ├── 300_P/  (300_AUDIO.wav, 300_TRANSCRIPT.csv, [300_COVAREP.csv], ...)
    └── ...

Per-participant files (``<pid>_AUDIO.wav``, ``<pid>_TRANSCRIPT.csv`` and any
``<pid>_COVAREP.csv`` / ``<pid>_CLNF*.txt``) are placed under ``<pid>_P/``.
Split/label CSVs (``*split*.csv``) are placed at the dataset root. The Kaggle
cache is the source of truth; by default we *symlink* into ``data/raw/daic`` so
the (large) audio is not duplicated.

Usage:
    pip install kagglehub
    python scripts/download_data.py                 # symlink into data/raw/daic
    python scripts/download_data.py --mode copy      # copy instead of symlink
    python scripts/download_data.py --inspect-only   # just download + print tree
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("download_data")

# <pid>_AUDIO.wav / <pid>_TRANSCRIPT.csv / <pid>_COVAREP.csv / <pid>_CLNF_AUs.txt
PID_FILE_RE = re.compile(r"^(?P<pid>\d+)_.*\.(wav|csv|txt)$", re.IGNORECASE)
SPLIT_RE = re.compile(r"split.*\.csv$", re.IGNORECASE)


def _print_tree(root: Path, max_entries: int = 40) -> None:
    """Print a shallow listing of the download to help diagnose layout."""
    logger.info("Top-level of %s:", root)
    entries = sorted(root.iterdir())
    for e in entries[:max_entries]:
        kind = "DIR " if e.is_dir() else "file"
        logger.info("  [%s] %s", kind, e.name)
    if len(entries) > max_entries:
        logger.info("  ... (%d more)", len(entries) - max_entries)


def _place(src: Path, dst: Path, mode: str) -> None:
    """Symlink or copy ``src`` to ``dst`` (idempotent)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def organise(download_root: Path, dest: Path, mode: str) -> Dict[str, int]:
    """Walk the download and place files into the expected DAIC layout.

    Returns:
        Counts of participants and each per-participant file type, plus split CSVs.
    """
    dest.mkdir(parents=True, exist_ok=True)
    per_pid: Dict[str, List[Path]] = defaultdict(list)
    splits: List[Path] = []

    for path in download_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        m = PID_FILE_RE.match(name)
        if m:
            per_pid[m.group("pid")].append(path)
        elif SPLIT_RE.search(name):
            splits.append(path)

    counts = {"participants": 0, "audio": 0, "transcript": 0, "acoustic": 0,
              "visual": 0, "splits": 0}

    for sp in splits:
        _place(sp, dest / sp.name, mode)
        counts["splits"] += 1

    for pid, files in per_pid.items():
        folder = dest / f"{pid}_P"
        has_any = False
        for f in files:
            _place(f, folder / f.name, mode)
            lower = f.name.lower()
            if lower.endswith("_audio.wav"):
                counts["audio"] += 1
                has_any = True
            elif lower.endswith("_transcript.csv"):
                counts["transcript"] += 1
                has_any = True
            elif "covarep" in lower:
                counts["acoustic"] += 1
            elif "clnf" in lower or "_aus" in lower or "_features" in lower:
                counts["visual"] += 1
        if has_any:
            counts["participants"] += 1

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="alejandropatinio/daic-woz")
    ap.add_argument("--dest", default="data/raw/daic")
    ap.add_argument("--mode", default="symlink", choices=["symlink", "copy"])
    ap.add_argument("--inspect-only", action="store_true",
                    help="Download and print the tree without organising.")
    args = ap.parse_args()

    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    logger.info("Downloading '%s' via kagglehub (this may take a while)...", args.dataset)
    path = Path(kagglehub.dataset_download(args.dataset))
    logger.info("Downloaded to: %s", path)
    _print_tree(path)

    if args.inspect_only:
        logger.info("--inspect-only set; not organising. Re-run without it to link files.")
        return

    counts = organise(path, Path(args.dest), args.mode)
    logger.info("Organised into %s (mode=%s):", args.dest, args.mode)
    logger.info("  participants: %d", counts["participants"])
    logger.info("  audio: %d | transcript: %d | acoustic(COVAREP): %d | visual(CLNF): %d",
                counts["audio"], counts["transcript"], counts["acoustic"], counts["visual"])
    logger.info("  split CSVs: %d", counts["splits"])

    if counts["participants"] == 0:
        logger.warning(
            "No <pid>_AUDIO.wav / <pid>_TRANSCRIPT.csv found. The Kaggle dataset "
            "layout may differ — re-run with --inspect-only and share the tree, "
            "or adjust PID_FILE_RE / paths in configs/corpora.yaml."
        )
    else:
        logger.info("Next: python scripts/00_build_manifests.py --corpora configs/corpora.yaml")


if __name__ == "__main__":
    main()
