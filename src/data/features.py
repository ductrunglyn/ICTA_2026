"""Per-modality feature extraction with on-disk caching.

Each modality is an independent branch (acoustic is the mandatory strong
baseline per prior findings). Backbones are *frozen*; only the light encoders
downstream are trained. Extracted features are cached as ``.pt`` files keyed by
segment id so they are never recomputed.

Output shapes per segment:

==========  =========================  ==================
modality    backbone                   shape
==========  =========================  ==================
audio       wav2vec2 / HuBERT / XLS-R  ``(T_a, 1024)``
acoustic    COVAREP + FORMANT z-score  ``(T_c, 79)``
text        BERT / XLM-R ``[CLS]``     ``(768,)``
visual      CLNF AUs + gaze + pose     ``(T_v, 50)``
==========  =========================  ==================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from .segmentation import Segment

# Canonical feature dimensions (must match the model's encoder ``in_dim``s).
AUDIO_DIM = 1024
ACOUSTIC_DIM = 79
TEXT_DIM = 768
VISUAL_DIM = 50
MODALITIES = ("audio", "acoustic", "text", "visual")


class AudioFeatureExtractor:
    """Frame-level audio embeddings from a frozen wav2vec2-style backbone.

    Args:
        model_name: HuggingFace model id (e.g. ``facebook/wav2vec2-xls-r-300m``).
        sample_rate: Expected audio sample rate.
        device: Torch device string.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        sample_rate: int = 16000,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device
        self._model = None
        self._fe = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:  # pragma: no cover - external dependency
            raise ImportError(
                "Audio features need 'transformers' and 'torch'."
            ) from exc
        self._fe = AutoFeatureExtractor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    def extract(self, waveform: "np.ndarray") -> "np.ndarray":
        """Return ``(T_a, 1024)`` last-hidden-state frames for a waveform."""
        import torch

        self._lazy_init()
        inputs = self._fe(
            waveform, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        with torch.no_grad():
            out = self._model(inputs.input_values.to(self.device))
        return out.last_hidden_state.squeeze(0).cpu().numpy()


class TextFeatureExtractor:
    """``[CLS]`` sentence embedding from a frozen (multilingual) BERT.

    Args:
        model_name: HuggingFace model id (e.g. ``xlm-roberta-base``).
        device: Torch device string.
    """

    def __init__(self, model_name: str = "xlm-roberta-base", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tok = None

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - external dependency
            raise ImportError("Text features need 'transformers' and 'torch'.") from exc
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    def extract(self, text: str) -> "np.ndarray":
        """Return a ``(768,)`` ``[CLS]`` embedding for ``text``."""
        import torch

        self._lazy_init()
        enc = self._tok(
            text or "", return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
        with torch.no_grad():
            out = self._model(**enc)
        return out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()


def load_acoustic_csv(
    path: Union[str, Path],
    start_s: float,
    end_s: float,
    frame_rate: float = 100.0,
    dim: int = ACOUSTIC_DIM,
) -> np.ndarray:
    """Load and z-score a slice of a COVAREP/FORMANT CSV.

    Args:
        path: CSV path with one row per frame.
        start_s, end_s: Segment bounds in seconds.
        frame_rate: Frames per second of the CSV (COVAREP default 100 Hz).
        dim: Expected feature dimension.

    Returns:
        ``(T_c, dim)`` z-scored array (per-segment standardisation).
    """
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr[None, :]
    i0, i1 = int(start_s * frame_rate), int(end_s * frame_rate)
    sl = arr[i0:i1, :dim]
    if sl.shape[0] == 0:
        sl = np.zeros((1, dim), dtype=np.float32)
    mu, sigma = sl.mean(0, keepdims=True), sl.std(0, keepdims=True) + 1e-6
    return ((sl - mu) / sigma).astype(np.float32)


def load_audio_slice(
    path: Union[str, Path],
    start_s: float,
    end_s: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Load a mono waveform slice ``[start_s, end_s)`` resampled to ``sample_rate``.

    Tries ``torchaudio`` first, then ``soundfile``. The returned 1-D float32
    array is what :meth:`AudioFeatureExtractor.extract` expects.

    Args:
        path: Path to a wav file.
        start_s, end_s: Segment bounds in seconds.
        sample_rate: Target sample rate.

    Returns:
        1-D ``float32`` waveform for the requested span.
    """
    try:
        import torch  # noqa: F401
        import torchaudio

        wav, sr = torchaudio.load(str(path))           # (channels, n)
        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)            # to mono
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        wav = wav.squeeze(0).numpy()
    except ImportError:
        import soundfile as sf  # type: ignore

        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != sample_rate:  # pragma: no cover - depends on input rate
            raise ValueError(
                f"{path}: sample rate {sr} != {sample_rate} and torchaudio is "
                "unavailable for resampling."
            )
    i0, i1 = int(start_s * sample_rate), int(end_s * sample_rate)
    sl = np.asarray(wav[i0:i1], dtype=np.float32)
    if sl.shape[0] == 0:
        sl = np.zeros(sample_rate // 10, dtype=np.float32)  # 0.1s of silence
    return sl


class OpenSmileAcousticExtractor:
    """Frame-level acoustic LLDs via openSMILE (eGeMAPS by default).

    Replaces COVAREP when only raw audio is available. eGeMAPSv02 yields 25
    low-level descriptors per frame; ComParE_2016 yields 65. The chosen set's
    dimension must match the model's ``acoustic`` encoder ``in_dim``.

    Args:
        feature_set: ``eGeMAPSv02`` (25-d) or ``ComParE_2016`` (65-d).
        sample_rate: Expected sample rate of the sliced waveform.
    """

    DIMS = {"eGeMAPSv02": 25, "ComParE_2016": 65}

    def __init__(self, feature_set: str = "eGeMAPSv02", sample_rate: int = 16000) -> None:
        self.feature_set = feature_set
        self.sample_rate = sample_rate
        self._smile = None

    @property
    def dim(self) -> int:
        return self.DIMS.get(self.feature_set, 25)

    def _lazy_init(self) -> None:
        if self._smile is not None:
            return
        try:
            import opensmile
        except ImportError as exc:  # pragma: no cover - external dependency
            raise ImportError(
                "openSMILE acoustic features need 'opensmile' "
                "(`pip install opensmile`)."
            ) from exc
        self._smile = opensmile.Smile(
            feature_set=getattr(opensmile.FeatureSet, self.feature_set),
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """Return a ``(T, dim)`` z-scored LLD sequence for a waveform slice."""
        self._lazy_init()
        sig = np.asarray(waveform, dtype=np.float32)
        df = self._smile.process_signal(sig, self.sample_rate)
        arr = df.to_numpy().astype(np.float32)
        if arr.shape[0] == 0:
            arr = np.zeros((1, self.dim), dtype=np.float32)
        mu, sigma = arr.mean(0, keepdims=True), arr.std(0, keepdims=True) + 1e-6
        return ((arr - mu) / sigma).astype(np.float32)


def load_visual_txt(
    path: Union[str, Path],
    start_s: float,
    end_s: float,
    frame_rate: float = 30.0,
    dim: int = VISUAL_DIM,
) -> np.ndarray:
    """Load a slice of CLNF AU/gaze/pose features (DAIC only).

    Args:
        path: CLNF feature file (comma-separated, header skipped).
        start_s, end_s: Segment bounds in seconds.
        frame_rate: Frames per second (OpenFace/CLNF default 30 Hz).
        dim: Expected feature dimension.

    Returns:
        ``(T_v, dim)`` array.
    """
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    i0, i1 = int(start_s * frame_rate), int(end_s * frame_rate)
    sl = arr[i0:i1, -dim:]
    if sl.shape[0] == 0:
        sl = np.zeros((1, dim), dtype=np.float32)
    return sl.astype(np.float32)


class FeatureCache:
    """Read/write ``.pt`` feature dicts keyed by segment id."""

    def __init__(self, cache_dir: Union[str, Path]) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, seg_id: str) -> Path:
        return self.cache_dir / f"{seg_id}.pt"

    def exists(self, seg_id: str) -> bool:
        return self.path_for(seg_id).exists()

    def save(self, seg_id: str, feat: Dict[str, object]) -> None:
        import torch

        torch.save(feat, self.path_for(seg_id))

    def load(self, seg_id: str) -> Dict[str, object]:
        import torch

        # weights_only=False: the cache is produced by this pipeline and stores
        # numpy arrays / metadata dicts (not untrusted checkpoints).
        return torch.load(self.path_for(seg_id), map_location="cpu", weights_only=False)


def build_segment_feature(
    segment: Segment,
    corpus: str,
    gender: int,
    language: str,
    audio_arr: Optional[np.ndarray] = None,
    acoustic_arr: Optional[np.ndarray] = None,
    visual_arr: Optional[np.ndarray] = None,
    text_arr: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Assemble the per-segment feature dict consumed by the dataset.

    Any modality passed as ``None`` is stored as ``None`` and later masked in
    the collate function (never imputed).

    Args:
        segment: The source :class:`Segment`.
        corpus, gender, language: Metadata copied into ``meta``.
        audio_arr, acoustic_arr, visual_arr, text_arr: Pre-extracted arrays.

    Returns:
        Dict with modality tensors (as numpy arrays), ``qtype`` and ``meta``.
    """
    return {
        "audio": audio_arr,
        "acoustic": acoustic_arr,
        "text": text_arr,
        "visual": visual_arr,
        "qtype": segment.qtype_id,
        "meta": {
            "participant_id": segment.participant_id,
            "seg_id": segment.seg_id,
            "corpus": corpus,
            "gender": int(gender),
            "language": language,
        },
    }
