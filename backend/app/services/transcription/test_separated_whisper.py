from pathlib import Path

from faster_whisper import WhisperModel


MODEL_SIZE = "large-v3"

FILES = [
    Path(r"C:\Projects\intelligent-captioning\backend\storage\audio\sepformer\speaker_1.wav"),
    Path(r"C:\Projects\intelligent-captioning\backend\storage\audio\sepformer\speaker_2.wav"),
]


print("Loading Whisper large-v3...")

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8",
)

print("Whisper loaded.\n")


for audio_file in FILES:

    print("=" * 70)
    print(f"FILE: {audio_file.name}")
    print("=" * 70)

    segments, info = model.transcribe(
        str(audio_file),
        language=None,
        beam_size=5,
        temperature=0,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    segments = list(segments)

    print(f"LANGUAGE: {info.language}")
    print(f"PROBABILITY: {info.language_probability}")
    print()

    if not segments:
        print("[NO SPEECH DETECTED]")
        continue

    for segment in segments:
        print(
            f"{segment.start:.2f} -> {segment.end:.2f}: "
            f"{segment.text.strip()}"
        )

        if segment.words:
            for word in segment.words:
                print(
                    f"    {word.start:.2f} -> "
                    f"{word.end:.2f}: "
                    f"{word.word.strip()} "
                    f"(p={word.probability:.3f})"
                )

    print()