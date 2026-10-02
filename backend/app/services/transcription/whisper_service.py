from pathlib import Path
from typing import Any
import json


from faster_whisper import WhisperModel


class WhisperTranscriber:
    """
    Handles speech-to-text transcription using faster-whisper.

    This version is designed for controlled baseline experiments.

    We can vary:
    - language
    - VAD
    - beam size
    - temperature
    - previous-text conditioning

    while keeping the rest of the pipeline consistent.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        print(f"Loading Whisper model: {model_size}")

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        print("Whisper model loaded successfully.")

    def transcribe(
        self,
        media_path: str,
        *,
        language: str | None = None,
        vad_filter: bool = True,
        beam_size: int = 5,
        temperature: float = 0.0,
        condition_on_previous_text: bool = True,
    ) -> dict[str, Any]:
        """
        Transcribe a video/audio file.

        Args:
            media_path:
                Path to the video/audio file.

            language:
                Language hint for Whisper.
                Examples:
                    "hi"
                    "en"
                    None = automatic detection

            vad_filter:
                Whether to use Voice Activity Detection.

            beam_size:
                Beam search size used during decoding.

            temperature:
                Decoding temperature.

            condition_on_previous_text:
                Whether Whisper should use previous text
                when decoding the next segment.

        Returns:
            {
                "language": "...",
                "language_probability": 0.0,
                "settings": {...},
                "segments": [...]
            }
        """

        path = Path(media_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Media file not found: {path}"
            )

        print(f"Transcribing: {path}")

        print(
            "Settings:"
            f"\n  language={language}"
            f"\n  vad_filter={vad_filter}"
            f"\n  beam_size={beam_size}"
            f"\n  temperature={temperature}"
            f"\n  condition_on_previous_text={condition_on_previous_text}"
        )

        segments, info = self.model.transcribe(
            str(path),
            language=language,
            word_timestamps=True,
            vad_filter=vad_filter,
            beam_size=beam_size,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
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

        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "settings": {
                "language": language,
                "vad_filter": vad_filter,
                "beam_size": beam_size,
                "temperature": temperature,
                "condition_on_previous_text": (
                    condition_on_previous_text
                ),
            },
            "segments": result_segments,
        }


def save_result(
    result: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save a transcription result as JSON.
    """

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


def print_result(
    experiment_name: str,
    result: dict[str, Any],
) -> None:
    """
    Print a readable transcription result.
    """

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)

    print(
        f"\nDetected language: "
        f"{result['language']}"
    )

    print(
        "Language probability: "
        f"{result['language_probability']:.4f}"
    )

    print("\nSettings:")

    for key, value in result["settings"].items():
        print(f"  {key}: {value}")

    print("\nTranscript:\n")

    for segment in result["segments"]:
        print(
            f"[{segment['start']:.2f}s → "
            f"{segment['end']:.2f}s] "
            f"{segment['text']}"
        )

        for word in segment["words"]:
            print(
                f"    {word['word']!r:<25} "
                f"{word['start']:.2f}s → "
                f"{word['end']:.2f}s | "
                f"confidence="
                f"{word['probability']:.3f}"
            )


def run_experiment(
    transcriber: WhisperTranscriber,
    video_path: Path,
    output_directory: Path,
    experiment_name: str,
    *,
    language: str | None,
    vad_filter: bool,
    beam_size: int,
    temperature: float,
    condition_on_previous_text: bool,
) -> None:
    """
    Run one controlled transcription experiment.
    """

    print("\n")
    print("#" * 80)
    print(f"# STARTING: {experiment_name}")
    print("#" * 80)

    result = transcriber.transcribe(
        str(video_path),
        language=language,
        vad_filter=vad_filter,
        beam_size=beam_size,
        temperature=temperature,
        condition_on_previous_text=condition_on_previous_text,
    )

    output_path = (
        output_directory
        / f"{experiment_name}.json"
    )

    save_result(
        result,
        output_path,
    )

    print_result(
        experiment_name,
        result,
    )

    print(
        f"\nSaved to:\n{output_path}"
    )


def main() -> None:
    """
    Run controlled Whisper experiments on the same test video.

    IMPORTANT:
    Every experiment uses the exact same video.

    Only selected transcription parameters change.
    """

    project_root = (
        Path(__file__).resolve().parents[4]
    )

    video_path = (
        project_root
        / "backend"
        / "storage"
        / "uploads"
        / "test_hinglish.mp4"
    )

    output_directory = (
        project_root
        / "backend"
        / "storage"
        / "exports"
        / "whisper_benchmark"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not video_path.exists():
        raise FileNotFoundError(
            "\nTest video was not found.\n"
            f"Expected location:\n{video_path}\n\n"
            "Put your test video in:\n"
            "backend/storage/uploads/\n"
            "and rename it to test_hinglish.mp4"
        )

    print("\n" + "=" * 80)
    print("WHISPER BENCHMARK")
    print("=" * 80)

    print(
        f"\nVideo:\n{video_path}"
    )

    print(
        "\nOutput directory:\n"
        f"{output_directory}"
    )

    # ---------------------------------------------------------
    # Load the model ONCE.
    # ---------------------------------------------------------

    transcriber = WhisperTranscriber(
        model_size="large-v3",
        device="cpu",
        compute_type="int8",
    )

    # ---------------------------------------------------------
    # EXPERIMENT 1
    # Current baseline
    # ---------------------------------------------------------

    run_experiment(
        transcriber,
        video_path,
        output_directory,
        "01_baseline",
        language=None,
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=True,
    )

    # ---------------------------------------------------------
    # EXPERIMENT 2
    # Force Hindi
    # ---------------------------------------------------------

    run_experiment(
        transcriber,
        video_path,
        output_directory,
        "02_forced_hindi",
        language="hi",
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=True,
    )

    # ---------------------------------------------------------
    # EXPERIMENT 3
    # Force English
    # ---------------------------------------------------------

    run_experiment(
        transcriber,
        video_path,
        output_directory,
        "03_forced_english",
        language="en",
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=True,
    )

    # ---------------------------------------------------------
    # EXPERIMENT 4
    # Disable VAD
    # ---------------------------------------------------------

    run_experiment(
        transcriber,
        video_path,
        output_directory,
        "04_no_vad",
        language=None,
        vad_filter=False,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=True,
    )

    # ---------------------------------------------------------
    # EXPERIMENT 5
    # Reduce previous-text influence
    #
    # This is particularly important because our baseline
    # repeatedly hallucinated:
    #
    # "सब्सक्राइब करना पाएगा"
    #
    # We want to see whether removing previous-text
    # conditioning reduces this repetition.
    # ---------------------------------------------------------

    run_experiment(
        transcriber,
        video_path,
        output_directory,
        "05_no_previous_text",
        language=None,
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
    )

    # ---------------------------------------------------------
    # DONE
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    print(
        "\nResults saved in:"
        f"\n{output_directory}"
    )

    print("\nFiles:")

    for file in sorted(
        output_directory.glob("*.json")
    ):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()