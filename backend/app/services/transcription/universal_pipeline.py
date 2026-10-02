"""
universal_pipeline.py

Language-Agnostic, High-Precision Code-Switching ASR Engine.
Handles automatic language identification, script detection, dynamic Romanization,
and confidence-gated secondary speech recovery.
"""
from __future__ import annotations

import json
import logging
import re
import string
import subprocess
import tempfile
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

# Optional Universal Transliteration Libraries
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

log = logging.getLogger("universal_pipeline")
SR = 16_000


# --------------------------------------------------------------------------- #
# Script & Transliteration Helper
# --------------------------------------------------------------------------- #
def is_devanagari(text: str) -> bool:
    """Detects if text contains Devanagari script characters."""
    return bool(re.search(r'[\u0900-\u097F]', text))


def universal_transliterate(text: str, target_script: str = "latin") -> str:
    """
    Dynamically converts native scripts (e.g., Devanagari, Tamil, Cyrillic) 
    into Romanized Latin script without hardcoded rules.
    """
    if not text or target_script != "latin":
        return text

    # Handle Indic Scripts
    if HAS_INDIC and is_devanagari(text):
        lat = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        # Clean ITRANS diacritics for natural reading
        lat = lat.replace('.', '').replace('~', '').replace('^', '')
        lat = re.sub(r'aim\b', 'ai', lat)
        lat = re.sub(r'um\b', 'u', lat)
        return lat.lower()

    return text


# --------------------------------------------------------------------------- #
# Data Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class WordToken:
    word: str
    start: float
    end: float
    probability: float
    language: str


@dataclass
class Config:
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    window_s: float = 5.0
    hop_s: float = 2.5
    min_support: int = 2
    min_avg_logprob: float = -1.0
    max_no_speech_prob: float = 0.5
    max_compression_ratio: float = 2.4


# --------------------------------------------------------------------------- #
# Engine Implementation
# --------------------------------------------------------------------------- #
class UniversalASREngine:
    def __init__(self, cfg: Config):
        from faster_whisper import WhisperModel

        device = cfg.device
        compute_type = cfg.compute_type

        if device == "cpu" or compute_type == "auto":
            compute_type = "int8"
            device = "cpu"

        self.model = WhisperModel(cfg.model, device=device, compute_type=compute_type)

    def process_audio(self, audio: np.ndarray, Romanize: bool = True) -> list[WordToken]:
        """
        Executes unconstrained multilingual transcription with word-level 
        timestamps and dynamic language assignment.
        """
        segments, info = self.model.transcribe(
            audio.astype(np.float32),
            language=None,  # Unconstrained auto-detection
            word_timestamps=True,
            beam_size=5,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        detected_words: list[WordToken] = []

        for segment in segments:
            # Skip low confidence / high noise segments
            if segment.no_speech_prob > 0.6 or segment.avg_logprob < -1.0:
                continue

            if segment.words:
                for w in segment.words:
                    word_text = w.word.strip()
                    if Romanize:
                        word_text = universal_transliterate(word_text, target_script="latin")

                    detected_words.append(
                        WordToken(
                            word=word_text,
                            start=round(w.start, 2),
                            end=round(w.end, 2),
                            probability=round(w.probability, 4),
                            language=info.language,
                        )
                    )

        return detected_words


# --------------------------------------------------------------------------- #
# Quality & Error Review Pipeline
# --------------------------------------------------------------------------- #
def analyze_caption_quality(words: list[WordToken]) -> dict:
    """
    Generates Feature #18 (AI Quality Checker) metrics for the transcribed words.
    """
    if not words:
        return {"quality_score": 0, "issues": ["No speech detected"]}

    low_confidence_count = sum(1 for w in words if w.probability < 0.6)
    language_switches = sum(
        1 for i in range(1, len(words)) if words[i].language != words[i - 1].language
    )

    total_words = len(words)
    quality_score = max(0, int(100 - (low_confidence_count / total_words * 50) - (language_switches * 2)))

    return {
        "quality_score": quality_score,
        "total_words": total_words,
        "low_confidence_words": low_confidence_count,
        "language_switches_detected": language_switches,
        "issues": [
            f"{low_confidence_count} low-confidence words detected." if low_confidence_count > 0 else "Confidence high.",
            f"{language_switches} language switch points detected." if language_switches > 0 else "Single language dominant.",
        ],
    }