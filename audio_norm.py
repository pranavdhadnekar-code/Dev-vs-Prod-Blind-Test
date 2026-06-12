"""Audio normalization pipeline for blindness integrity.

Every clip served to a rater must be indistinguishable by *format*: identical
container, codec, sample-rate and channel layout, and loudness-normalized to a
common LUFS target. This removes the codec/container fingerprint (e.g. Murf MP3
vs. Omni WAV) that would otherwise let a rater identify the provider.

Pipeline per clip:
    decode (ffmpeg via pydub)  ->  mono + common sample-rate
    ->  loudness-normalize to target LUFS (BS.1770 / EBU R128)
    ->  peak-limit to avoid clipping
    ->  re-encode to a single common container/codec (lossless PCM WAV)
    ->  strip provider-identifying metadata

The output WAV carries only canonical PCM format chunks (no tags), so no
provider metadata leaks. The exact parameters used are returned alongside the
bytes so they can be recorded with the battle for reproducibility/audit.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import numpy as np

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "audio_norm requires 'soundfile'. Install with: pip install soundfile"
    ) from exc

try:
    from pydub import AudioSegment
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "audio_norm requires 'pydub' (and ffmpeg on PATH). "
        "Install with: pip install pydub and a system ffmpeg."
    ) from exc

try:
    import pyloudnorm as pyln
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "audio_norm requires 'pyloudnorm'. Install with: pip install pyloudnorm"
    ) from exc


# --- Common served format (single codec/container for ALL providers) ---------
DEFAULT_TARGET_LUFS: float = -23.0          # EBU R128 integrated loudness target
DEFAULT_TRUE_PEAK_DBFS: float = -1.0        # ceiling to prevent clipping
DEFAULT_SAMPLE_RATE: int = 24000            # Hz, common rate for all clips
DEFAULT_CHANNELS: int = 1                   # mono
OUTPUT_CONTAINER: str = "wav"
OUTPUT_CODEC: str = "pcm_s16le"
OUTPUT_SUBTYPE: str = "PCM_16"

# pyloudnorm/BS.1770 needs >= 400 ms to compute a gated integrated loudness.
_MIN_LOUDNESS_SECONDS: float = 0.4


@dataclass
class NormalizationParams:
    """The exact, disclosed settings applied to a clip (stored per battle)."""
    target_lufs: float = DEFAULT_TARGET_LUFS
    true_peak_dbfs: float = DEFAULT_TRUE_PEAK_DBFS
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    container: str = OUTPUT_CONTAINER
    codec: str = OUTPUT_CODEC

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizationResult:
    """Normalized audio plus the lineage needed for audit/reproducibility."""
    audio: bytes
    params: NormalizationParams
    measured_lufs_in: Optional[float]
    measured_lufs_out: Optional[float]
    duration_seconds: float
    sha256: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> Dict[str, Any]:
        """Flat, JSON-serializable record for the battle row."""
        meta = self.params.to_dict()
        meta.update(
            {
                "measured_lufs_in": self.measured_lufs_in,
                "measured_lufs_out": self.measured_lufs_out,
                "duration_seconds": round(self.duration_seconds, 3),
                "sha256": self.sha256,
            }
        )
        meta.update(self.extra)
        return meta


def _decode_to_samples(
    data: bytes, source_format: Optional[str], params: NormalizationParams
) -> tuple[np.ndarray, AudioSegment]:
    """Decode arbitrary audio bytes to float32 mono samples at the target rate."""
    seg = AudioSegment.from_file(io.BytesIO(data), format=source_format)
    seg = seg.set_channels(params.channels).set_frame_rate(params.sample_rate)

    samples = np.array(seg.get_array_of_samples())
    if seg.channels > 1:  # defensive; we already forced mono above
        samples = samples.reshape((-1, seg.channels)).mean(axis=1)
    max_int = float(1 << (8 * seg.sample_width - 1))
    floats = samples.astype(np.float64) / max_int
    return floats, seg


def _integrated_loudness(samples: np.ndarray, sample_rate: int) -> Optional[float]:
    if samples.size / sample_rate < _MIN_LOUDNESS_SECONDS:
        return None
    try:
        meter = pyln.Meter(sample_rate)
        loud = float(meter.integrated_loudness(samples))
        if not np.isfinite(loud):
            return None
        return loud
    except Exception:
        return None


def _apply_gain_db(samples: np.ndarray, gain_db: float) -> np.ndarray:
    return samples * (10.0 ** (gain_db / 20.0))


def _peak_limit(samples: np.ndarray, ceiling_dbfs: float) -> np.ndarray:
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > ceiling and peak > 0:
        samples = samples * (ceiling / peak)
    return samples


def normalize_audio(
    data: bytes,
    source_format: Optional[str] = None,
    params: Optional[NormalizationParams] = None,
) -> NormalizationResult:
    """Normalize one clip to the common, brand-free served format.

    Args:
        data: raw audio bytes as returned by a provider (mp3, wav, etc.).
        source_format: optional decoder hint (e.g. "mp3", "wav"); ffmpeg can
            usually auto-detect, so this is best-effort only.
        params: normalization settings; defaults to the common target.

    Returns:
        NormalizationResult with the re-encoded WAV bytes and full lineage.
    """
    if not data:
        raise ValueError("normalize_audio received empty audio data")

    params = params or NormalizationParams()

    floats, _seg = _decode_to_samples(data, source_format, params)
    duration = floats.size / float(params.sample_rate) if params.sample_rate else 0.0

    measured_in = _integrated_loudness(floats, params.sample_rate)
    if measured_in is not None:
        floats = _apply_gain_db(floats, params.target_lufs - measured_in)
    else:
        # Too short / silent to gate: peak-normalize toward the ceiling instead.
        peak = float(np.max(np.abs(floats))) if floats.size else 0.0
        if peak > 0:
            ceiling = 10.0 ** (params.true_peak_dbfs / 20.0)
            floats = floats * (ceiling / peak)

    floats = _peak_limit(floats, params.true_peak_dbfs)
    floats = np.clip(floats, -1.0, 1.0)

    measured_out = _integrated_loudness(floats, params.sample_rate)

    pcm16 = (floats * float(1 << 15)).astype(np.int16)

    buf = io.BytesIO()
    # soundfile writes a canonical PCM WAV with no metadata tags.
    sf.write(buf, pcm16, params.sample_rate, subtype=OUTPUT_SUBTYPE, format="WAV")
    out_bytes = buf.getvalue()

    return NormalizationResult(
        audio=out_bytes,
        params=params,
        measured_lufs_in=measured_in,
        measured_lufs_out=measured_out,
        duration_seconds=duration,
        sha256=hashlib.sha256(out_bytes).hexdigest(),
    )


def clip_hash(data: bytes) -> str:
    """Stable content hash for a clip (used in battle lineage)."""
    return hashlib.sha256(data).hexdigest()
