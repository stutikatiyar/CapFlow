from pathlib import Path
import json

from faster_whisper import WhisperModel


BASE_DIR = Path(__file__).resolve().parents[3]

AUDIO_PATH = BASE_DIR / "storage" / "audio" / "test_hinglish_enhanced.wav"
OUTPUT_PATH = (
    BASE_DIR
    / "storage"
    / "exports"
    / "whisper_benchmark"
    / "07_enhanced_audio.json"
)


def main():
    print(f"Loading enhanced audio: {AUDIO_PATH}")

    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(
        "large-v3",
        device="cpu",
        compute_type="int8",
    )

    print("Transcribing enhanced WAV...")
    print()

    segments, info = model.transcribe(
        str(AUDIO_PATH),
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
    )

    results = {
        "experiment": "07_enhanced_audio",
        "audio_file": str(AUDIO_PATH),
        "model": "large-v3",
        "device": "cpu",
        "compute_type": "int8",
        "settings": {
            "word_timestamps": True,
            "vad_filter": True,
            "beam_size": 5,
            "temperature": 0.0,
            "condition_on_previous_text": False,
        },
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": [],
    }

    print("Transcription:")
    print("-" * 80)

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
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                )

        results["segments"].append(segment_data)

        print(
            f"[{segment.start:.2f}s → {segment.end:.2f}s] "
            f"{segment.text.strip()}"
        )

    print("-" * 80)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()