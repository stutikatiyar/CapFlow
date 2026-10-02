"""
feature_1_automatic_captioning.py

Feature 1: Automatic Video Caption Generation
Input: Video / Audio file path
Output: Synchronized captions with timestamps, confidence scores, detected language, 
        and ready-to-use export formats (JSON, SRT, VTT).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger("feature_1_captioning")


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
    language: str
    words: List[WordCaption]


@dataclass
class VideoCaptionResult:
    video_path: str
    duration_s: float
    detected_language: str
    language_probability: float
    segments: List[SegmentCaption]


# --------------------------------------------------------------------------- #
# Helper Functions for Formatting (SRT & VTT)
# --------------------------------------------------------------------------- #
def format_timestamp_srt(seconds: float) -> str:
    """Converts seconds (e.g. 12.34) to SRT format: HH:MM:SS,mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Converts seconds to WebVTT format: HH:MM:SS.mmm"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"


# --------------------------------------------------------------------------- #
# Core Pipeline Implementation
# --------------------------------------------------------------------------- #
class AutomaticCaptioner:
    def __init__(self, model_size: str = "large-v3", device: str = "auto"):
        from faster_whisper import WhisperModel

        compute_type = "int8" if device == "cpu" or device == "auto" else "float16"
        log.info("Initializing WhisperModel (%s) on %s [%s]...", model_size, device, compute_type)
        
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device if device != "auto" else "cpu",
            compute_type=compute_type,
        )

    def generate_captions(self, video_path: Path) -> VideoCaptionResult:
        """
        Processes a video or audio file and extracts synchronized captions
        with word-level timestamps and segment confidence.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Input file not found: {video_path}")

        log.info("Starting transcription for: %s", video_path)

        # Execute transcription with word timestamps enabled
        segments_generator, info = self.model.transcribe(
            str(video_path),
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segment_captions: List[SegmentCaption] = []
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

            # Segment-level confidence calculated from token log probabilities
            seg_confidence = round(float(np.exp(segment.avg_logprob)), 4)

            segment_captions.append(
                SegmentCaption(
                    id=seg_id,
                    start=round(segment.start, 2),
                    end=round(segment.end, 2),
                    text=segment.text.strip(),
                    confidence=seg_confidence,
                    language=info.language,
                    words=word_captions,
                )
            )
            seg_id += 1

        return VideoCaptionResult(
            video_path=str(video_path),
            duration_s=round(info.duration, 2),
            detected_language=info.language,
            language_probability=round(info.language_probability, 4),
            segments=segment_captions,
        )

    def export_srt(self, result: VideoCaptionResult, output_path: Path) -> None:
        """Exports caption result as a standard .srt subtitle file."""
        lines = []
        for seg in result.segments:
            start_str = format_timestamp_srt(seg.start)
            end_str = format_timestamp_srt(seg.end)
            lines.append(f"{seg.id}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(seg.text)
            lines.append("")  # Empty newline separator

        output_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Saved SRT captions to: %s", output_path)

    def export_vtt(self, result: VideoCaptionResult, output_path: Path) -> None:
        """Exports caption result as a WebVTT (.vtt) file for HTML5 video players."""
        lines = ["WEBVTT", ""]
        for seg in result.segments:
            start_str = format_timestamp_vtt(seg.start)
            end_str = format_timestamp_vtt(seg.end)
            lines.append(f"{seg.id}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(seg.text)
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Saved VTT captions to: %s", output_path)

    def export_json(self, result: VideoCaptionResult, output_path: Path) -> None:
        """Exports complete payload with word-level confidence and timestamps."""
        data = asdict(result)
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Saved JSON captions to: %s", output_path)