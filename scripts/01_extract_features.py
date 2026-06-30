#!/usr/bin/env python
"""Segment interviews, extract per-modality features and cache them.

For every participant in the manifest this:

1. segments the interview (prompt-based for DAIC, VAD fallback otherwise);
2. extracts audio/text/acoustic/visual features per segment;
3. caches a feature dict per segment under ``data/interim/features/``;
4. writes a segment-level manifest ``data/manifests/segments.csv``.

Usage:
    python scripts/01_extract_features.py \
        --manifest data/manifests/all.csv --corpora configs/corpora.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import (  # noqa: E402
    AudioFeatureExtractor,
    FeatureCache,
    OpenSmileAcousticExtractor,
    TextFeatureExtractor,
    build_segment_feature,
    load_acoustic_csv,
    load_full_audio,
    load_visual_txt,
    slice_waveform,
)
from src.data.segmentation import Segment, segment_participant  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("extract_features")

# Budget so repeated failures surface loudly a few times instead of silently.
_WARN_BUDGET: Dict[str, int] = {}


def _warn_once(key: str, msg: str, *args, budget: int = 5) -> None:
    """Emit a warning up to ``budget`` times per key, then go quiet."""
    n = _WARN_BUDGET.get(key, budget)
    if n > 0:
        logger.warning(msg, *args)
        _WARN_BUDGET[key] = n - 1
        if n == 1:
            logger.warning("(further '%s' warnings suppressed)", key)


def _load_prompt_map(path: str = "configs/prompt2qtype.yaml") -> Dict[str, str]:
    cfg = load_config(path)
    mapping = cfg.get("mapping", {})
    return {k: v for k, v in (mapping.to_dict() if hasattr(mapping, "to_dict") else mapping).items()}


def _resolve(template: str, pid_local: str) -> str:
    return template.replace("{pid}", pid_local)


def process_participant(
    row: pd.Series,
    spec,
    prompt_map: Dict[str, str],
    cache: FeatureCache,
    audio_fx: AudioFeatureExtractor,
    text_fx: TextFeatureExtractor,
    extract_audio: bool,
    overwrite: bool = False,
    acoustic_smile: "OpenSmileAcousticExtractor | None" = None,
) -> List[Dict[str, object]]:
    """Segment + feature-extract a single participant. Returns segment rows.

    Heavy backbone inference is guarded so the pipeline still runs end-to-end on
    machines without GPU/model weights (those modalities are cached as ``None``
    and masked downstream).
    """
    pid = row["participant_id"]
    pid_local = pid.split("_", 1)[1] if "_" in pid else pid
    corpus = row["corpus"]

    audio_path = _resolve(str(spec.get("audio_dir", "")), pid_local) or None
    transcript_path = spec.get("transcript")
    transcript_df = None
    if transcript_path:
        tp = Path(_resolve(str(transcript_path), pid_local))
        if tp.exists():
            # DAIC transcripts are TAB-separated; honour an explicit override.
            sep = spec.get("transcript_sep", "\t" if tp.suffix == ".csv" else ",")
            transcript_df = pd.read_csv(tp, sep=sep)

    try:
        segments: List[Segment] = segment_participant(
            participant_id=pid,
            audio_path=audio_path if (audio_path and Path(audio_path).exists()) else None,
            transcript_df=transcript_df,
            prompt_map=prompt_map,
            prompts_speaker=spec.get("prompts_speaker"),
        )
    except Exception as exc:  # pragma: no cover - depends on external data
        logger.warning("Segmentation failed for %s: %s", pid, exc)
        return []

    acoustic_path = spec.get("acoustic")
    visual_path = spec.get("visual")
    rows: List[Dict[str, object]] = []

    def _row(seg) -> Dict[str, object]:
        return {
            "participant_id": pid,
            "seg_id": seg.seg_id,
            "qtype": seg.qtype_id,
            "start_s": seg.start_s,
            "end_s": seg.end_s,
        }

    # Modalities we are able to produce in this run (used by the smart resume).
    acoustic_available = (
        acoustic_smile is not None
        or (acoustic_path and Path(_resolve(str(acoustic_path), pid_local)).exists())
    )
    visual_available = bool(
        visual_path and Path(_resolve(str(visual_path), pid_local)).exists()
    )

    # Decode the participant's waveform ONCE (not per segment) for efficiency.
    need_wave = (extract_audio or acoustic_smile is not None)
    full_wave = None
    if need_wave:
        if not audio_path or not Path(audio_path).exists():
            _warn_once("audio_path", "Audio file missing for %s: %s", pid, audio_path)
        else:
            try:
                full_wave = load_full_audio(audio_path)
            except Exception as exc:  # pragma: no cover - external audio
                _warn_once("audio_load", "Audio load failed for %s: %s", pid, exc)

    for seg in segments:
        # Smart resume: skip only if every modality we could add is already
        # cached. Otherwise recompute so newly-enabled modalities (e.g. audio)
        # get filled in without needing --overwrite.
        if not overwrite and cache.exists(seg.seg_id):
            cached = cache.load(seg.seg_id)
            want = set()
            if seg.text:
                want.add("text")
            if extract_audio:
                want.add("audio")
            if acoustic_available:
                want.add("acoustic")
            if visual_available:
                want.add("visual")
            if all(cached.get(m) is not None for m in want):
                rows.append(_row(seg))
                continue

        audio_arr = acoustic_arr = visual_arr = text_arr = None

        if acoustic_path:
            ap = Path(_resolve(str(acoustic_path), pid_local))
            if ap.exists():
                acoustic_arr = load_acoustic_csv(ap, seg.start_s, seg.end_s)
        if visual_path:
            vp = Path(_resolve(str(visual_path), pid_local))
            if vp.exists():
                visual_arr = load_visual_txt(vp, seg.start_s, seg.end_s)
        if seg.text:
            try:
                text_arr = text_fx.extract(seg.text)
            except Exception as exc:  # pragma: no cover - external model
                logger.debug("Text features skipped for %s: %s", seg.seg_id, exc)

        # Slice the pre-decoded participant waveform for this segment.
        wav = None
        if full_wave is not None:
            wav = slice_waveform(full_wave, seg.start_s, seg.end_s)
        if extract_audio and wav is not None:
            try:
                audio_arr = audio_fx.extract(wav)
            except Exception as exc:  # pragma: no cover - external model
                _warn_once("audio_fx", "Audio feature extraction failed for %s: %s",
                           seg.seg_id, exc)
        if acoustic_smile is not None and wav is not None:
            try:
                acoustic_arr = acoustic_smile.extract(wav)  # overrides COVAREP
            except Exception as exc:  # pragma: no cover - external model
                logger.debug("openSMILE features skipped for %s: %s", seg.seg_id, exc)

        feat = build_segment_feature(
            seg, corpus=corpus, gender=int(row.get("gender", 0)),
            language=row.get("language", "unknown"),
            audio_arr=audio_arr, acoustic_arr=acoustic_arr,
            visual_arr=visual_arr, text_arr=text_arr,
        )
        cache.save(seg.seg_id, feat)
        rows.append(_row(seg))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifests/all.csv")
    ap.add_argument("--corpora", default="configs/corpora.yaml")
    ap.add_argument("--cache_dir", default="data/interim/features")
    ap.add_argument("--segments_out", default="data/manifests/segments.csv")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--extract_audio", action="store_true",
                    help="Run the (heavy) audio backbone; off by default.")
    ap.add_argument("--device", default=None,
                    help="Torch device for backbones (e.g. cuda, cuda:0, cpu). "
                         "Defaults to cuda if available, else cpu.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute features even if a segment is already cached.")
    ap.add_argument("--acoustic_backend", default="none",
                    choices=["none", "opensmile"],
                    help="Acoustic branch source. 'opensmile' extracts eGeMAPS "
                         "LLDs from raw audio (when COVAREP is unavailable).")
    ap.add_argument("--opensmile_set", default="eGeMAPSv02",
                    choices=["eGeMAPSv02", "ComParE_2016"])
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    corpora = load_config(args.corpora)
    cfg = load_config(args.config)
    prompt_map = _load_prompt_map()
    cache = FeatureCache(args.cache_dir)

    if args.device:
        device = args.device
    else:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Extracting features on device: %s", device)

    audio_fx = AudioFeatureExtractor(model_name=cfg.backbones.audio, device=device)
    text_fx = TextFeatureExtractor(model_name=cfg.backbones.text, device=device)
    acoustic_smile = (
        OpenSmileAcousticExtractor(feature_set=args.opensmile_set)
        if args.acoustic_backend == "opensmile" else None
    )
    if acoustic_smile is not None:
        logger.info("Acoustic backend: openSMILE %s (dim=%d)",
                    args.opensmile_set, acoustic_smile.dim)

    # Map corpus name -> spec.
    spec_by_corpus = {spec.get("corpus", name): spec for name, spec in corpora.items()}

    all_segment_rows: List[Dict[str, object]] = []
    seg_counts: Dict[str, int] = {}
    for _, row in manifest.iterrows():
        spec = spec_by_corpus.get(row["corpus"])
        if spec is None:
            logger.warning("No corpus spec for %s", row["corpus"])
            continue
        rows = process_participant(
            row, spec, prompt_map, cache, audio_fx, text_fx, args.extract_audio,
            overwrite=args.overwrite, acoustic_smile=acoustic_smile,
        )
        all_segment_rows.extend(rows)
        seg_counts[row["participant_id"]] = len(rows)

    seg_df = pd.DataFrame(all_segment_rows)
    Path(args.segments_out).parent.mkdir(parents=True, exist_ok=True)
    seg_df.to_csv(args.segments_out, index=False)

    # Update n_segments in the participant manifest.
    manifest["n_segments"] = manifest["participant_id"].map(seg_counts).fillna(0).astype(int)
    manifest.to_csv(args.manifest, index=False)

    # --- Coverage report (diagnoses the "only N/Total participants used" issue) ---
    with_seg = int((manifest["n_segments"] > 0).sum())
    zero = manifest.loc[manifest["n_segments"] == 0, "participant_id"].tolist()
    logger.info("Cached %d segments | participants WITH segments: %d / %d",
                len(seg_df), with_seg, len(manifest))
    if zero:
        logger.warning(
            "%d participants have ZERO segments (dropped from CV). First 20: %s",
            len(zero), zero[:20],
        )
        logger.warning(
            "Likely causes: missing transcript/audio file, or all answers < 3s "
            "(MIN_SEG_S). Check that each participant has a data folder + transcript."
        )

    # --- Per-modality coverage (diagnoses 'audio is empty in cache') ---
    counts = {m: 0 for m in ("audio", "acoustic", "text", "visual")}
    for sid in seg_df["seg_id"] if not seg_df.empty else []:
        try:
            feat = cache.load(sid)
        except Exception:  # pragma: no cover - corrupt cache entry
            continue
        for m in counts:
            if feat.get(m) is not None:
                counts[m] += 1
    total = len(seg_df)
    logger.info("Modality coverage (segments with non-None feature):")
    for m, c in counts.items():
        pct = 100.0 * c / total if total else 0.0
        logger.info("  %-9s %d / %d (%.1f%%)", m, c, total, pct)
    if counts["audio"] == 0 and total:
        logger.warning(
            "AUDIO is empty in the cache. Re-run with '--extract_audio --overwrite' "
            "(and --device cuda). Check the warnings above for the root cause."
        )


if __name__ == "__main__":
    main()
