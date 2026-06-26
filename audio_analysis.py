from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np


@dataclass
class TranscriptResult:
    text: str
    language: str
    language_probability: float
    duration_sec: float
    segments: list[dict]
    error: str = ""


@dataclass
class VoiceResult:
    duration_sec: float
    speaking_rate_wpm: float
    mean_rms: float
    median_pitch_hz: float
    silence_ratio: float
    delivery_score: float
    interpretation: str
    error: str = ""


def extract_audio_to_wav(video_path: str | Path) -> Path:
    output = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(output),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Audio extraction failed. Confirm that the video contains audio.")
    return output


@lru_cache(maxsize=3)
def _load_whisper(model_size: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio(audio_path: str, model_size: str = "tiny") -> TranscriptResult:
    try:
        model = _load_whisper(model_size)
        generator, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
        segments, text_parts = [], []
        for segment in generator:
            cleaned = segment.text.strip()
            if cleaned:
                text_parts.append(cleaned)
                segments.append({
                    "start_sec": round(float(segment.start), 2),
                    "end_sec": round(float(segment.end), 2),
                    "text": cleaned,
                })
        duration = max((row["end_sec"] for row in segments), default=0.0)
        return TranscriptResult(
            " ".join(text_parts).strip(),
            str(getattr(info, "language", "unknown")),
            round(float(getattr(info, "language_probability", 0.0)), 3),
            round(duration, 2),
            segments,
        )
    except Exception as exc:
        return TranscriptResult("", "unknown", 0.0, 0.0, [], str(exc))


def analyze_voice(audio_path: str, transcript_word_count: int = 0) -> VoiceResult:
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))
        if duration <= 0 or y.size == 0:
            raise ValueError("No usable audio samples were found.")

        mean_rms = float(np.mean(librosa.feature.rms(y=y)[0]))
        try:
            f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr)
            finite_f0 = f0[np.isfinite(f0)]
            median_pitch = float(np.median(finite_f0)) if finite_f0.size else 0.0
        except Exception:
            median_pitch = 0.0

        intervals = librosa.effects.split(y, top_db=30)
        active = sum(int(end - start) for start, end in intervals)
        silence_ratio = max(0.0, min(1.0, 1 - active / max(len(y), 1)))
        speaking_rate = transcript_word_count / duration * 60 if transcript_word_count else 0.0

        energy_component = max(0.0, min(100.0, mean_rms * 900))
        pause_component = max(0.0, 100 - abs(silence_ratio - 0.20) * 220)
        if speaking_rate:
            rate_component = max(0.0, 100 - abs(speaking_rate - 140) * 0.75)
            score = 0.35 * energy_component + 0.30 * pause_component + 0.35 * rate_component
        else:
            score = 0.55 * energy_component + 0.45 * pause_component
        score = round(max(0.0, min(100.0, score)), 2)

        interpretation = (
            "Clear and reasonably steady delivery" if score >= 75
            else "Moderate delivery; review pacing, pauses, and volume" if score >= 55
            else "Delivery needs practice or the audio quality is weak"
        )
        return VoiceResult(round(duration, 2), round(speaking_rate, 2), round(mean_rms, 5),
                           round(median_pitch, 2), round(silence_ratio * 100, 2), score, interpretation)
    except Exception as exc:
        return VoiceResult(0.0, 0.0, 0.0, 0.0, 0.0, 50.0, "Voice analysis unavailable", str(exc))
