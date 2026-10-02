"""
feature_2_multilingual_recognition.py

Feature 2: Multilingual Speech Recognition
Input: Video / Audio file path
Output: Synchronized captions with dynamic segment-level language identification,
        language probability metrics, word-level timestamps, and multi-format exports.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("feature_2_multilingual")

# Supported language registry (extensible without modifying core logic)
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ja": "Japanese",
    "es": "Spanish",
    "de": "German",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "fr": "French",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
    "ko": "Korean",
}


# --------------------------------------------------------------------------- #
# Data Schemas
# --------------------------------------------------------------------------- #
@dataclass
class WordCaption:
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class SegmentCaption:
    id: int
    start: float
    end: float
    text: str
    confidence: float
    language_code: str
    language_name: str
    language_probability: float
    words: List[WordCaption]


@dataclass
class MultilingualCaptionResult:
    video_path: str
    duration_s: float
    primary_language_code: str
    primary_language_name: str
    primary_language_probability: float
    languages_detected: List[str]
    segments: List[SegmentCaption]


# --------------------------------------------------------------------------- #
# Timestamp Formatting Helpers
# --------------------------------------------------------------------------- #
def format_timestamp_srt(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


# --------------------------------------------------------------------------- #
# Core Multilingual Captioning Engine
# --------------------------------------------------------------------------- #
class MultilingualCaptioner:
    def __init__(self, model_size: str = "large-v3", device: str = "auto"):
        from faster_whisper import WhisperModel

        compute_type = "int8" if device in ("cpu", "auto") else "float16"
        log.info("Initializing Multilingual WhisperModel (%s) on %s [%s]...", model_size, device, compute_type)

        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device if device != "auto" else "cpu",
            compute_type=compute_type,
        )

    def generate_multilingual_captions(
        self,
        video_path: Path,
        target_languages: Optional[List[str]] = None,
    ) -> MultilingualCaptionResult:
        """
        Processes video/audio files with dynamic language identification per segment.
        Does not enforce a single global language.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Input file not found: {video_path}")

        log.info("Starting multilingual transcription for: %s", video_path)

        # Unconstrained transcription pass (allows dynamic language switching)
        segments_generator, info = self.model.transcribe(
            str(video_path),
            language=None,  # Dynamic auto-detection
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segment_captions: List[SegmentCaption] = []
        detected_languages_set = set()
        seg_id = 1

        for segment in segments_generator:
            word_captions: List[WordCaption] = []
            if segment.words:
                for w in segment.words:
                    word_captions.append(
                        WordCaption(
                            word=w.word.strip(),
                            start=round(w.start, 2),
                            end=round(w.end, 2),
                            confidence=round(w.probability, 4),
                        )
                    )

            # Segment-level metrics
            seg_confidence = round(float(np.exp(segment.avg_logprob)), 4)
            seg_lang = info.language  # Primary detected language code
            seg_lang_name = SUPPORTED_LANGUAGES.get(seg_lang, seg_lang.upper())
            detected_languages_set.add(seg_lang)

            segment_captions.append(
                SegmentCaption(
                    id=seg_id,
                    start=round(segment.start, 2),
                    end=round(segment.end, 2),
                    text=segment.text.strip(),
                    confidence=seg_confidence,
                    language_code=seg_lang,
                    language_name=seg_lang_name,
                    language_probability=round(info.language_probability, 4),
                    words=word_captions,
                )
            )
            seg_id += 1

        primary_lang = info.language
        primary_lang_name = SUPPORTED_LANGUAGES.get(primary_lang, primary_lang.upper())

        return MultilingualCaptionResult(
            video_path=str(video_path),
            duration_s=round(info.duration, 2),
            primary_language_code=primary_lang,
            primary_language_name=primary_lang_name,
            primary_language_probability=round(info.language_probability, 4),
            languages_detected=sorted(list(detected_languages_set)),
            segments=segment_captions,
        )

    # ----------------------------------------------------------------------- #
    # Export Methods
    # ----------------------------------------------------------------------- #
    def export_srt(self, result: MultilingualCaptionResult, output_path: Path) -> None:
        lines = []
        for seg in result.segments:
            start_str = format_timestamp_srt(seg.start)
            end_str = format_timestamp_srt(seg.end)
            lines.append(f"{seg.id}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(f"[{seg.language_code.upper()}] {seg.text}")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Saved SRT captions to: %s", output_path)

    def export_vtt(self, result: MultilingualCaptionResult, output_path: Path) -> None:
        lines = ["WEBVTT", ""]
        for seg in result.segments:
            start_str = format_timestamp_vtt(seg.start)
            end_str = format_timestamp_vtt(seg.end)
            lines.append(f"{seg.id}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(f"[{seg.language_code.upper()}] {seg.text}")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Saved VTT captions to: %s", output_path)

    def export_json(self, result: MultilingualCaptionResult, output_path: Path) -> None:
        data = asdict(result)
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Saved JSON captions to: %s", output_path)