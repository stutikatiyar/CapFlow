from pathlib import Path
from typing import Any
import json


from faster_whisper import WhisperModel


# ============================================================
# BENCHMARK 2 — VERIFIED GROUND TRUTH VS WHISPER
# ============================================================
#
# PURPOSE:
# Compare Whisper's word-level timestamps/transcription against
# the user's manually verified transcript for the ORIGINAL video.
#
# IMPORTANT:
# - Uses the ORIGINAL test_hinglish.mp4.
# - No Adobe enhancement.
# - No FFmpeg enhancement/filtering.
# - No forced language.
# - Word timestamps are enabled.
# - condition_on_previous_text=False.
#
# IMPORTANT EVALUATION NOTE:
# The ground-truth times below are the user's verified coarse
# ranges. They are NOT millisecond-perfect boundaries.
#
# Therefore this benchmark DOES NOT calculate WER yet.
# It reports the Whisper words/timestamps that fall inside each
# ground-truth range so we can inspect alignment first.
#
# 0:07–0:08 is CONFIRMED SILENCE.
# ============================================================


BACKEND_DIR = Path(__file__).resolve().parents[3]

VIDEO_PATH = (
    BACKEND_DIR
    / "storage"
    / "uploads"
    / "test_hinglish.mp4"
)

OUTPUT_DIR = (
    BACKEND_DIR
    / "storage"
    / "exports"
    / "whisper_benchmark"
    / "benchmark_2"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "benchmark_2_ground_truth_comparison.json"
)


# ============================================================
# VERIFIED GROUND TRUTH
# ============================================================
#
# The transcript below preserves the user's Roman Hinglish and
# English wording.
#
# Parenthetical/context notes are NOT spoken words.
#
# 0:07–0:08 is silence.
# ============================================================

GROUND_TRUTH = [
    {
        "id": 1,
        "start": 0.00,
        "end": 3.00,
        "text": "Mai yaha pe chai peene aayi hu",
        "note": "",
    },
    {
        "id": 2,
        "start": 3.00,
        "end": 6.00,
        "text": (
            "Vo waha pe egg rice kha raha hai baithke, "
            "idhar dekho"
        ),
        "note": "teiling Tanay",
    },
    {
        "id": 3,
        "start": 7.00,
        "end": 8.00,
        "text": "",
        "type": "silence",
        "note": "Confirmed silence — no one is speaking.",
    },
    {
        "id": 4,
        "start": 9.00,
        "end": 12.00,
        "text": (
            "Ki mai chai peene aayi hu or ye mere haath ke bane "
            "egg rice kha raha hai"
        ),
        "note": "",
    },
    {
        "id": 5,
        "start": 13.00,
        "end": 15.00,
        "text": (
            "I am veg. And I said ki ye nhi I am veg. "
            "And I am in background and I say ye dekho guys"
        ),
        "note": (
            "Source notes: background Tanay / camera is towards Tanay. "
            "These notes are not spoken words."
        ),
    },
    {
        "id": 6,
        "start": 16.00,
        "end": 20.00,
        "text": (
            "Ye nahi ki meri tareef kar dein, to bolta hai ki "
            "masala kam hai rice me"
        ),
        "note": "",
    },
    {
        "id": 7,
        "start": 21.00,
        "end": 22.00,
        "text": "Apna banana bolungi to nhi bana payega",
        "note": "",
    },
    {
        "id": 8,
        "start": 22.00,
        "end": 24.00,
        "text": "Lekin sahi bola masala mujhe bhi kam laga",
        "note": "",
    },
    {
        "id": 9,
        "start": 25.00,
        "end": 26.00,
        "text": "Guys mai apni bhot badi critique hu",
        "note": "",
    },
    {
        "id": 10,
        "start": 27.00,
        "end": 30.00,
        "text": (
            "Fir koi aise log mujhe bol dete hai, kami nikal dete hai, "
            "toh mera confidence thoda sa down ho jaata hai"
        ),
        "note": "",
    },
    {
        "id": 11,
        "start": 30.00,
        "end": 31.00,
        "text": "ye dekho guys firse",
        "note": "",
    },
    {
        "id": 12,
        "start": 32.00,
        "end": 37.00,
        "text": (
            "chalo guys baad me milte hain vo mai video record "
            "karri thi kyuki mujhe practice karni h"
        ),
        "note": "",
    },
]


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def run_whisper() -> dict[str, Any]:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Original test video not found:\n"
            f"{VIDEO_PATH}\n\n"
            "Expected file:\n"
            f"{VIDEO_PATH}"
        )

    print("=" * 80)
    print("BENCHMARK 2 — VERIFIED GROUND TRUTH VS WHISPER")
    print("=" * 80)

    print(f"\nInput:")
    print(VIDEO_PATH)

    print("\nLoading Whisper model: large-v3")

    model = WhisperModel(
        "large-v3",
        device="cpu",
        compute_type="int8",
    )

    print("Whisper model loaded successfully.")

    print("\nTranscribing ORIGINAL video...")
    print("-" * 80)

    segments, info = model.transcribe(
        str(VIDEO_PATH),
        language=None,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
    )

    whisper_segments = []

    for segment in segments:
        segment_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": [],
        }

        if segment.words:
            for word in segment.words:
                segment_data["words"].append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                )

        whisper_segments.append(segment_data)

        print(
            f"[{segment.start:.2f}s → {segment.end:.2f}s] "
            f"{segment.text.strip()}"
        )

    print("-" * 80)

    print(f"\nDetected language: {info.language}")
    print(
        f"Language probability: "
        f"{info.language_probability:.4f}"
    )

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "settings": {
            "language": None,
            "word_timestamps": True,
            "vad_filter": True,
            "beam_size": 5,
            "temperature": 0.0,
            "condition_on_previous_text": False,
        },
        "segments": whisper_segments,
    }


# ============================================================
# WORD/TIMESTAMP EXTRACTION
# ============================================================

def whisper_words_in_range(
    whisper_segments: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    """
    Collect Whisper words whose midpoint falls inside the
    ground-truth time range.

    This is only a timing inspection tool.

    It does NOT prove that Whisper is correct or incorrect,
    because the ground-truth boundaries are still coarse.
    """

    words = []

    for segment in whisper_segments:
        for word in segment["words"]:

            midpoint = (
                word["start"] + word["end"]
            ) / 2

            if start <= midpoint <= end:
                words.append(word)

    return words


def build_timing_review(
    whisper_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build a timing review for every ground-truth entry.

    We deliberately do NOT calculate WER here.

    Reason:
    Whisper currently outputs Devanagari while the verified
    transcript is Roman Hinglish. A normal text WER would mix
    script differences with actual speech-recognition errors.

    First we inspect the word timestamps.
    """

    review = []

    for item in GROUND_TRUTH:

        words = whisper_words_in_range(
            whisper_result["segments"],
            item["start"],
            item["end"],
        )

        whisper_text = " ".join(
            word["word"]
            for word in words
        ).strip()

        review_item = {
            "id": item["id"],
            "start": item["start"],
            "end": item["end"],
            "ground_truth": (
                "[SILENCE]"
                if item.get("type") == "silence"
                else item["text"]
            ),
            "note": item.get("note", ""),
            "whisper_text_in_time_range": whisper_text,
            "whisper_words_in_time_range": [],
        }

        for word in words:
            review_item["whisper_words_in_time_range"].append(
                {
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                    "probability": word["probability"],
                }
            )

        review.append(review_item)

    return review


# ============================================================
# PRINT TIMING REVIEW
# ============================================================

def print_timing_review(
    review: list[dict[str, Any]],
) -> None:

    print("\n")
    print("=" * 80)
    print("GROUND TRUTH / WHISPER TIMING REVIEW")
    print("=" * 80)

    for item in review:

        print(
            f"\n[{item['start']:.2f}s → "
            f"{item['end']:.2f}s]"
        )

        print(
            f"GROUND TRUTH:"
            f"\n  {item['ground_truth']}"
        )

        if item["note"]:
            print(
                f"NOTE:"
                f"\n  {item['note']}"
            )

        print(
            "\nWHISPER WORDS IN THIS RANGE:"
        )

        words = item[
            "whisper_words_in_time_range"
        ]

        if not words:
            print("  [none]")

        else:
            for word in words:
                print(
                    f"  {word['word']!r:<20} "
                    f"[{word['start']:.2f}s → "
                    f"{word['end']:.2f}s] "
                    f"confidence="
                    f"{word['probability']:.3f}"
                )

        print(
            "\nWHISPER TEXT IN RANGE:"
        )

        if item["whisper_text_in_time_range"]:
            print(
                f"  {item['whisper_text_in_time_range']}"
            )
        else:
            print("  [none]")


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    whisper_result: dict[str, Any],
    timing_review: list[dict[str, Any]],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "benchmark": (
            "02_verified_ground_truth_vs_whisper"
        ),
        "input": str(VIDEO_PATH),
        "model": "large-v3",
        "audio_preprocessing": (
            "none — original video input"
        ),
        "evaluation_note": (
            "Ground-truth times are the user's "
            "verified coarse ranges. "
            "This benchmark inspects Whisper word "
            "timestamps inside those ranges. "
            "It does not calculate final WER yet."
        ),
        "ground_truth": GROUND_TRUTH,
        "whisper_result": whisper_result,
        "timing_review": timing_review,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    whisper_result = run_whisper()

    timing_review = build_timing_review(
        whisper_result
    )

    print_timing_review(
        timing_review
    )

    save_result(
        whisper_result,
        timing_review,
    )

    print("\n")
    print("=" * 80)
    print("BENCHMARK 2 COMPLETE")
    print("=" * 80)

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
