"""
run_feature_2.py

CLI runner to execute Feature 2 (Multilingual Speech Recognition) on any input video.
"""
import argparse
import logging
from pathlib import Path

from app.services.transcription.feature_2_multilingual_recognition import MultilingualCaptioner

BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "storage" / "exports" / "feature_2"


def main():
    parser = argparse.ArgumentParser(description="Feature 2: Multilingual Speech Recognition")
    parser.add_argument("--input", required=True, type=Path, help="Path to input video/audio file")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Export directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    captioner = MultilingualCaptioner(model_size="large-v3", device="cpu")
    result = captioner.generate_multilingual_captions(args.input)

    # Save exports
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    captioner.export_json(result, args.out_dir / f"{stem}_multilingual.json")
    captioner.export_srt(result, args.out_dir / f"{stem}_multilingual.srt")
    captioner.export_vtt(result, args.out_dir / f"{stem}_multilingual.vtt")

    print("\n" + "=" * 85)
    print("FEATURE 2: MULTILINGUAL SPEECH RECOGNITION SUMMARY")
    print("=" * 85)
    print(f"Video File                   : {result.video_path}")
    print(f"Duration                     : {result.duration_s}s")
    print(f"Primary Language Detected    : {result.primary_language_name} ({result.primary_language_code})")
    print(f"Primary Language Confidence  : {result.primary_language_probability}")
    print(f"All Detected Languages       : {', '.join(result.languages_detected)}")
    print(f"Total Segments               : {len(result.segments)}")
    print("-" * 85)

    for seg in result.segments[:5]:  # Print preview of first 5 segments
        print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] [{seg.language_code.upper()}] (Conf: {seg.confidence}) {seg.text}")

    print("-" * 85)
    print(f"Exports saved to: {args.out_dir}")


if __name__ == "__main__":
    main()