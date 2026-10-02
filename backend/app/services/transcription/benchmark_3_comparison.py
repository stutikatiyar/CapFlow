from pathlib import Path
import json

from faster_whisper import WhisperModel


# ============================================================
# PATHS
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
)

JSON_OUTPUT = OUTPUT_DIR / "benchmark_3_comparison.json"
TXT_OUTPUT = OUTPUT_DIR / "benchmark_3_comparison.txt"


# ============================================================
# MANUAL TRANSCRIPTION
# Verified transcription of what was actually spoken.
# ============================================================

MANUAL_TRANSCRIPTION = [
    {
        "id": 1,
        "text": "Mai yaha pe chai peene aayi hu",
    },
    {
        "id": 2,
        "text": "Vo waha pe egg rice kha raha hai baithke, idhar dekho",
    },
    {
        "id": 3,
        "text": "[SILENCE]",
    },
    {
        "id": 4,
        "text": "Ki mai chai peene aayi hu or ye mere haath ke bane egg rice kha raha hai",
    },
    {
        "id": 5,
        "text": "I am veg. And I said ki ye nhi I am veg. And I am in background and I say ye dekho guys",
    },
    {
        "id": 6,
        "text": "Ye nahi ki meri tareef kar dein, to bolta hai ki masala kam hai rice me",
    },
    {
        "id": 7,
        "text": "Apna banana bolungi to nhi bana payega",
    },
    {
        "id": 8,
        "text": "Lekin sahi bola masala mujhe bhi kam laga",
    },
    {
        "id": 9,
        "text": "Guys mai apni bhot badi critique hu",
    },
    {
        "id": 10,
        "text": "Fir koi aise log mujhe bol dete hai, kami nikal dete hai, toh mera confidence thoda sa down ho jaata hai",
    },
    {
        "id": 11,
        "text": "ye dekho guys firse",
    },
    {
        "id": 12,
        "text": "chalo guys baad me milte hain vo mai video record karri thi kyuki mujhe practice karni h",
    },
]


# ============================================================
# MANUAL TIME RANGES
#
# These are ONLY used to find the corresponding Whisper
# words for the side-by-side comparison.
#
# We are NOT calculating WER here.
# ============================================================

MANUAL_RANGES = [
    {"id": 1, "start": 0.00, "end": 3.00},
    {"id": 2, "start": 3.00, "end": 6.00},
    {"id": 3, "start": 7.00, "end": 8.00},
    {"id": 4, "start": 9.00, "end": 12.00},
    {"id": 5, "start": 13.00, "end": 15.00},
    {"id": 6, "start": 16.00, "end": 20.00},
    {"id": 7, "start": 21.00, "end": 22.00},
    {"id": 8, "start": 22.00, "end": 24.00},
    {"id": 9, "start": 25.00, "end": 26.00},
    {"id": 10, "start": 27.00, "end": 30.00},
    {"id": 11, "start": 30.00, "end": 31.00},
    {"id": 12, "start": 32.00, "end": 37.00},
]


# ============================================================
# RUN WHISPER
# ============================================================

def transcribe_video():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"\nVideo not found:\n{VIDEO_PATH}\n"
        )

    print("=" * 80)
    print("BENCHMARK 3 — MANUAL VS MODEL TRANSCRIPTION")
    print("=" * 80)

    print("\nInput video:")
    print(VIDEO_PATH)

    print("\nLoading Whisper large-v3...")

    model = WhisperModel(
        "large-v3",
        device="cpu",
        compute_type="int8",
    )

    print("Model loaded.")

    print("\nTranscribing original video...")
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

    model_segments = []

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

        model_segments.append(segment_data)

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

    return model_segments, info


# ============================================================
# GET WHISPER WORDS FOR A MANUAL TIME RANGE
#
# We use word midpoint to decide whether a Whisper word
# belongs inside the manual section.
# ============================================================

def get_words_in_range(model_segments, start, end):

    words = []

    for segment in model_segments:

        for word in segment["words"]:

            midpoint = (
                word["start"] + word["end"]
            ) / 2

            if start <= midpoint < end:

                words.append(word)

    return words


# ============================================================
# BUILD SIDE-BY-SIDE COMPARISON
# ============================================================

def build_comparison(model_segments):

    comparison = []

    for manual_item, time_range in zip(
        MANUAL_TRANSCRIPTION,
        MANUAL_RANGES,
    ):

        words = get_words_in_range(
            model_segments,
            time_range["start"],
            time_range["end"],
        )

        model_text = " ".join(
            word["word"].strip()
            for word in words
        ).strip()

        if not model_text:
            model_text = "[NO MODEL OUTPUT]"

        comparison.append(
            {
                "id": manual_item["id"],
                "start": time_range["start"],
                "end": time_range["end"],
                "manual_transcription": manual_item["text"],
                "model_transcription": model_text,
            }
        )

    return comparison


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    comparison,
    model_segments,
    info,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_model_transcription = " ".join(
        segment["text"]
        for segment in model_segments
    ).strip()

    result = {
        "benchmark": "03_manual_vs_model_side_by_side",

        "input": str(VIDEO_PATH),

        "model": "faster-whisper large-v3",

        "audio_preprocessing": (
            "none — original video input"
        ),

        "settings": {
            "language": None,
            "word_timestamps": True,
            "vad_filter": True,
            "beam_size": 5,
            "temperature": 0.0,
            "condition_on_previous_text": False,
        },

        "detected_language": info.language,

        "language_probability": (
            info.language_probability
        ),

        "comparison_note": (
            "Manual transcription is the verified "
            "ground truth. Model transcription is the "
            "raw Whisper output found inside the "
            "corresponding manual time range. "
            "No correction, translation, transliteration, "
            "or WER calculation is performed."
        ),

        "side_by_side": comparison,

        "raw_model_segments": model_segments,

        "full_raw_model_transcription": (
            full_model_transcription
        ),
    }

    with open(
        JSON_OUTPUT,
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
# SAVE HUMAN-READABLE SIDE-BY-SIDE TXT
# ============================================================

def save_text(
    comparison,
    model_segments,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_model_transcription = " ".join(
        segment["text"]
        for segment in model_segments
    ).strip()

    with open(
        TXT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "=" * 100
            + "\n"
        )

        file.write(
            "MANUAL TRANSCRIPTION  VS  MODEL TRANSCRIPTION\n"
        )

        file.write(
            "=" * 100
            + "\n\n"
        )

        for item in comparison:

            file.write(
                f"{item['id']}. "
                f"[{item['start']:.2f}s → {item['end']:.2f}s]\n"
            )

            file.write(
                "-" * 100
                + "\n"
            )

            file.write(
                "MANUAL TRANSCRIPTION:\n"
            )

            file.write(
                item["manual_transcription"]
                + "\n\n"
            )

            file.write(
                "MODEL TRANSCRIPTION:\n"
            )

            file.write(
                item["model_transcription"]
                + "\n\n"
            )

            file.write(
                "=" * 100
                + "\n\n"
            )

        file.write(
            "\n"
        )

        file.write(
            "=" * 100
            + "\n"
        )

        file.write(
            "COMPLETE RAW MODEL TRANSCRIPTION\n"
        )

        file.write(
            "=" * 100
            + "\n\n"
        )

        file.write(
            full_model_transcription
            + "\n"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    model_segments, info = transcribe_video()

    comparison = build_comparison(
        model_segments
    )

    save_json(
        comparison,
        model_segments,
        info,
    )

    save_text(
        comparison,
        model_segments,
    )

    print("\n")
    print("=" * 80)
    print("BENCHMARK 3 COMPLETE")
    print("=" * 80)

    print("\nJSON saved to:")
    print(JSON_OUTPUT)

    print("\nTXT saved to:")
    print(TXT_OUTPUT)

    print("\n")
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)

    for item in comparison:

        print(
            f"\n{item['id']}."
            f" [{item['start']:.2f}s → "
            f"{item['end']:.2f}s]"
        )

        print(
            f"MANUAL : "
            f"{item['manual_transcription']}"
        )

        print(
            f"MODEL  : "
            f"{item['model_transcription']}"
        )

    print("\n")
    print("=" * 80)


if __name__ == "__main__":
    main()