from pathlib import Path
import json

from faster_whisper import WhisperModel


def main() -> None:
    project_root = Path(__file__).resolve().parents[4]

    audio_path = (
        project_root
        / "backend"
        / "storage"
        / "audio"
        / "test_hinglish.wav"
    )

    output_path = (
        project_root
        / "backend"
        / "storage"
        / "exports"
        / "whisper_benchmark"
        / "06_clean_audio.json"
    )

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print(f"Loading audio: {audio_path}")

    model = WhisperModel(
        "large-v3",
        device="cpu",
        compute_type="int8",
    )

    print("Transcribing clean WAV...")

    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
    )

    result_segments = []

    for segment in segments:
        words = []

        if segment.words:
            for word in segment.words:
                words.append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                )

        result_segments.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": words,
            }
        )

    result = {
        "experiment": "06_clean_audio",
        "audio": str(audio_path),
        "language": info.language,
        "language_probability": info.language_probability,
        "settings": {
            "vad_filter": True,
            "beam_size": 5,
            "temperature": 0.0,
            "condition_on_previous_text": False,
        },
        "segments": result_segments,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nTranscription:")
    print("-" * 80)

    for segment in result_segments:
        print(
            f"[{segment['start']:.2f}s → "
            f"{segment['end']:.2f}s] "
            f"{segment['text']}"
        )

    print("-" * 80)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()