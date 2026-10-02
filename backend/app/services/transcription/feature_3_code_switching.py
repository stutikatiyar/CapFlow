"""
feature_3_code_switching.py

Feature 3: Code-Switching Detection ⭐
Input: Video / Audio file path
Output: Word-level language tagging (Hindi vs. English vs. target language), 
        dynamic Hinglish script normalization, and code-switched caption payloads.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional Transliteration Integration
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

from app.services.transcription.llm_post_processor import LLMPostProcessor

log = logging.getLogger("feature_3_code_switching")


def is_devanagari(text: str) -> bool:
    """Checks if string contains Devanagari Unicode characters."""
    return bool(re.search(r'[\u0900-\u097F]', text))


def detect_word_language_and_normalize(word_raw: str) -> Tuple[str, str]:
    """
    Analyzes a raw word token, identifies its script/language origin, 
    and converts native scripts into Romanized Latin text.
    """
    clean_word = word_raw.strip().strip(",.!?\"'").lower()

    if not clean_word:
        return word_raw, "en"

    # If already Latin/ASCII alphabets
    if clean_word.isascii():
        hindi_markers = {"mai", "hu", "hai", "ka", "ki", "ke", "ko", "se", "me", "pe", "par", "raha", "rahi", "aayi"}
        lang = "hi" if clean_word in hindi_markers else "en"
        return clean_word, lang

    # If Devanagari script is detected
    if is_devanagari(word_raw) and HAS_INDIC:
        # Transliterate to ITRANS Latin
        lat = transliterate(word_raw, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
        lat = lat.replace('.', '').replace('~', '').replace('^', '')
        lat = re.sub(r'aim\b', 'ai', lat)
        lat = re.sub(r'um\b', 'u', lat)

        return lat, "hi"

    return word_raw, "hi"


# --------------------------------------------------------------------------- #
# Data Schemas
# --------------------------------------------------------------------------- #
@dataclass
class CodeSwitchedWord:
    word: str
    original_word: str
    start: float
    end: float
    confidence: float
    detected_language: str  # e.g., 'en' or 'hi'


@dataclass
class CodeSwitchedSegment:
    id: int
    start: float
    end: float
    code_switched_text: str
    confidence: float
    words: List[CodeSwitchedWord]


@dataclass
class CodeSwitchingResult:
    video_path: str
    duration_s: float
    primary_language: str
    hindi_word_count: int
    english_word_count: int
    code_switching_ratio: float  # Percentage of secondary language words
    segments: List[CodeSwitchedSegment]


# --------------------------------------------------------------------------- #
# Formatting Helpers
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
# Code-Switching Engine Implementation
# --------------------------------------------------------------------------- #
class CodeSwitchingCaptioner:
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        use_llm_fix: bool = True,
        llm_provider: str = "groq",
        llm_model: str = "llama-3.1-8b-instant",
    ):
        from faster_whisper import WhisperModel

        compute_type = "int8" if device in ("cpu", "auto") else "float16"
        log.info("Initializing Code-Switching Whisper Engine (%s) on %s...", model_size, device)

        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device if device != "auto" else "cpu",
            compute_type=compute_type,
        )

        self.use_llm_fix = use_llm_fix
        if self.use_llm_fix:
            self.llm_processor = LLMPostProcessor(provider=llm_provider, model_name=llm_model)

    def process_code_switched_video(self, video_path: Path) -> CodeSwitchingResult:
        """
        Executes word-level code-switching analysis across the input video,
        followed by an automated LLM post-processing correction pass.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Input file not found: {video_path}")

        log.info("Analyzing code-switched audio stream for: %s", video_path)

        segments_generator, info = self.model.transcribe(
            str(video_path),
            language="hi",  # Lock primary language to Hindi for better tokenization
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250), # Reduced from 500ms to break long segments
            condition_on_previous_text=False, # Prevents Whisper hallucination loops across long pauses
        )

        segment_results: List[CodeSwitchedSegment] = []
        total_hi_words = 0
        total_en_words = 0
        seg_id = 1

        for segment in segments_generator:
            word_tokens: List[CodeSwitchedWord] = []
            normalized_sentence_words = []

            if segment.words:
                for w in segment.words:
                    raw_w = w.word.strip()
                    norm_w, word_lang = detect_word_language_and_normalize(raw_w)

                    if word_lang == "en":
                        total_en_words += 1
                    else:
                        total_hi_words += 1

                    word_tokens.append(
                        CodeSwitchedWord(
                            word=norm_w,
                            original_word=raw_w,
                            start=round(w.start, 2),
                            end=round(w.end, 2),
                            confidence=round(w.probability, 4),
                            detected_language=word_lang,
                        )
                    )
                    normalized_sentence_words.append(norm_w)

            full_cs_text = " ".join(normalized_sentence_words) if normalized_sentence_words else segment.text.strip()
            seg_confidence = round(float(np.exp(segment.avg_logprob)), 4)

            segment_results.append(
                CodeSwitchedSegment(
                    id=seg_id,
                    start=round(segment.start, 2),
                    end=round(segment.end, 2),
                    code_switched_text=full_cs_text,
                    confidence=seg_confidence,
                    words=word_tokens,
                )
            )
            seg_id += 1

        total_words = total_hi_words + total_en_words
        cs_ratio = round((total_en_words / total_words) * 100, 2) if total_words > 0 else 0.0

        # Dynamic LLM Post-Correction Pass
        # Dynamic LLM Post-Correction Pass
        # Dynamic LLM Post-Correction Pass
        if self.use_llm_fix and segment_results:
            log.info("Executing LLM post-processing correction pass...")
            try:
                raw_texts = [seg.code_switched_text for seg in segment_results]
                corrected_texts = self.llm_processor.process_segments(raw_texts)

                for seg, clean_text in zip(segment_results, corrected_texts):
                    seg.code_switched_text = clean_text
                log.info("LLM post-processing pass completed successfully!")
            except Exception as e:
                # PRINT THE EXACT ERROR TO TERMINAL
                print(f"\n❌ GROQ POST-PROCESSING ERROR: {e}\n")
                log.error("LLM Post-Processing failed: %s. Falling back to raw transcript.", e)
        return CodeSwitchingResult(
            video_path=str(video_path),
            duration_s=round(info.duration, 2),
            primary_language=info.language,
            hindi_word_count=total_hi_words,
            english_word_count=total_en_words,
            code_switching_ratio=cs_ratio,
            segments=segment_results,
        )

    # ----------------------------------------------------------------------- #
    # Export Capabilities
    # ----------------------------------------------------------------------- #
    def export_srt(self, result: CodeSwitchingResult, output_path: Path) -> None:
        lines = []
        for seg in result.segments:
            start_str = format_timestamp_srt(seg.start)
            end_str = format_timestamp_srt(seg.end)
            lines.append(f"{seg.id}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(seg.code_switched_text)
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Saved Code-Switched SRT to: %s", output_path)

    def export_vtt(self, result: CodeSwitchingResult, output_path: Path) -> None:
        lines = ["WEBVTT", ""]
        for seg in result.segments:
            start_str = format_timestamp_vtt(seg.start)
            end_str = format_timestamp_vtt(seg.end)
            lines.append(f"{seg.id}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(seg.code_switched_text)
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        log.info("Saved Code-Switched VTT to: %s", output_path)

    def export_json(self, result: CodeSwitchingResult, output_path: Path) -> None:
        data = asdict(result)
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Saved Code-Switched JSON payload to: %s", output_path)