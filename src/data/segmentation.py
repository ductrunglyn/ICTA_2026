"""Segmentation of interviews into response segments (NV: validity probe).

Two strategies are supported:

* **Prompt-based** (DAIC-WOZ): cut every answer between two consecutive
  interviewer (``Ellie``) prompts and tag it with a ``question_type`` derived
  from a *versioned, public* ``prompt2qtype.yaml`` mapping.
* **VAD fallback** (EATD / Androids): cut turns with a voice-activity
  detector (silero-vad) and tag ``question_type = "unknown"``.

Segments shorter than ``MIN_SEG_S`` are dropped; longer than ``MAX_SEG_S`` are
capped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

MIN_SEG_S: float = 3.0
MAX_SEG_S: float = 60.0

# Question-type vocabulary (kept in sync with configs/prompt2qtype.yaml).
QTYPES: List[str] = [
    "greeting",
    "background",
    "family",
    "mood",
    "sleep",
    "work",
    "stress",
    "therapy",
    "other",
    "unknown",
]
QTYPE2ID: Dict[str, int] = {q: i for i, q in enumerate(QTYPES)}


@dataclass
class Segment:
    """A single interview response segment.

    Attributes:
        seg_id: Globally unique segment id.
        participant_id: Owning participant.
        start_s: Start time in seconds within the source audio.
        end_s: End time in seconds.
        question_type: One of :data:`QTYPES`.
        text: Optional transcript text for the segment.
        audio_path: Path to the source audio file (offset/dur used to slice).
    """

    seg_id: str
    participant_id: str
    start_s: float
    end_s: float
    question_type: str = "unknown"
    text: str = ""
    audio_path: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.end_s - self.start_s)

    @property
    def qtype_id(self) -> int:
        return QTYPE2ID.get(self.question_type, QTYPE2ID["unknown"])


def _match_qtype(prompt: str, prompt_map: Dict[str, str]) -> str:
    """Map an interviewer prompt to a question type via substring rules.

    ``prompt_map`` maps a (lower-cased) keyword/regex to a question type. The
    first matching pattern wins; falls back to ``"other"``.
    """
    p = prompt.lower().strip()
    for pattern, qtype in prompt_map.items():
        if re.search(pattern.lower(), p):
            return qtype if qtype in QTYPE2ID else "other"
    return "other"


def _filter_and_cap(segments: List[Segment]) -> List[Segment]:
    """Drop too-short segments and cap too-long ones at :data:`MAX_SEG_S`."""
    kept: List[Segment] = []
    for seg in segments:
        if seg.duration < MIN_SEG_S:
            continue
        if seg.duration > MAX_SEG_S:
            seg.end_s = seg.start_s + MAX_SEG_S
        kept.append(seg)
    return kept


def segment_with_prompts(
    participant_id: str,
    transcript_df: pd.DataFrame,
    prompt_map: Dict[str, str],
    audio_path: Optional[str] = None,
    prompts_speaker: str = "Ellie",
    speaker_col: str = "speaker",
    start_col: str = "start_time",
    stop_col: str = "stop_time",
    text_col: str = "value",
) -> List[Segment]:
    """Cut participant answers between consecutive interviewer prompts.

    Args:
        participant_id: Participant identifier.
        transcript_df: DAIC-style transcript with speaker/time/text columns.
        prompt_map: Keyword -> question_type mapping (see prompt2qtype.yaml).
        audio_path: Source audio path attached to each segment.
        prompts_speaker: Speaker name of the interviewer.
        speaker_col, start_col, stop_col, text_col: Column names.

    Returns:
        List of filtered :class:`Segment` objects.
    """
    df = transcript_df.sort_values(start_col).reset_index(drop=True)
    segments: List[Segment] = []
    current_qtype = "greeting"
    pending_start: Optional[float] = None
    pending_text: List[str] = []
    pending_stop: Optional[float] = None
    idx = 0

    for _, row in df.iterrows():
        speaker = str(row[speaker_col])
        if speaker == prompts_speaker:
            # Flush the participant answer accumulated before this prompt.
            if pending_start is not None and pending_stop is not None:
                segments.append(
                    Segment(
                        seg_id=f"{participant_id}_seg{idx:04d}",
                        participant_id=participant_id,
                        start_s=float(pending_start),
                        end_s=float(pending_stop),
                        question_type=current_qtype,
                        text=" ".join(pending_text).strip(),
                        audio_path=audio_path,
                    )
                )
                idx += 1
                pending_start, pending_stop, pending_text = None, None, []
            # The new prompt determines the next answer's question type.
            current_qtype = _match_qtype(str(row[text_col]), prompt_map)
        else:
            if pending_start is None:
                pending_start = float(row[start_col])
            pending_stop = float(row[stop_col])
            pending_text.append(str(row[text_col]))

    # Trailing answer with no closing prompt.
    if pending_start is not None and pending_stop is not None:
        segments.append(
            Segment(
                seg_id=f"{participant_id}_seg{idx:04d}",
                participant_id=participant_id,
                start_s=float(pending_start),
                end_s=float(pending_stop),
                question_type=current_qtype,
                text=" ".join(pending_text).strip(),
                audio_path=audio_path,
            )
        )
    return _filter_and_cap(segments)


def segment_with_vad(
    participant_id: str,
    audio_path: Union[str, Path],
    sample_rate: int = 16000,
) -> List[Segment]:
    """Segment audio into turns using silero-vad (VAD fallback).

    Used for corpora without an interviewer-prompt structure (EATD, Androids);
    every segment receives ``question_type = "unknown"``.

    Args:
        participant_id: Participant identifier.
        audio_path: Path to a 16 kHz wav file.
        sample_rate: Expected sample rate.

    Returns:
        List of filtered :class:`Segment` objects.

    Note:
        Requires ``silero-vad`` and ``torchaudio``. If unavailable the function
        raises a clear error rather than silently degrading.
    """
    try:
        import torch
        import torchaudio  # noqa: F401  (used to load audio)
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    except ImportError as exc:  # pragma: no cover - external dependency
        raise ImportError(
            "VAD segmentation needs 'silero-vad' and 'torchaudio'. "
            "Install via `pip install silero-vad torchaudio`."
        ) from exc

    model = load_silero_vad()
    wav = read_audio(str(audio_path), sampling_rate=sample_rate)
    ts = get_speech_timestamps(
        wav, model, sampling_rate=sample_rate, return_seconds=True
    )
    segments: List[Segment] = []
    for i, span in enumerate(ts):
        segments.append(
            Segment(
                seg_id=f"{participant_id}_seg{i:04d}",
                participant_id=participant_id,
                start_s=float(span["start"]),
                end_s=float(span["end"]),
                question_type="unknown",
                audio_path=str(audio_path),
            )
        )
    return _filter_and_cap(segments)


def segment_participant(
    participant_id: str,
    audio_path: Optional[Union[str, Path]],
    transcript_df: Optional[pd.DataFrame],
    prompt_map: Optional[Dict[str, str]],
    prompts_speaker: Optional[str],
) -> List[Segment]:
    """Dispatch to prompt-based or VAD segmentation based on availability.

    If ``prompts_speaker`` and a transcript are provided, prompt-based
    segmentation is used; otherwise the VAD fallback runs on the audio.
    """
    if prompts_speaker and transcript_df is not None and prompt_map is not None:
        return segment_with_prompts(
            participant_id,
            transcript_df,
            prompt_map,
            audio_path=str(audio_path) if audio_path else None,
            prompts_speaker=prompts_speaker,
        )
    if audio_path is None:
        raise ValueError(
            f"{participant_id}: no transcript/prompts and no audio for VAD."
        )
    return segment_with_vad(participant_id, audio_path)
