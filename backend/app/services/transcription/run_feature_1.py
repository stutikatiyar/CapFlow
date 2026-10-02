"""
run_feature_1.py

CLI runner to execute Feature 1 (Automatic Caption Generation) on any input video.
"""
import argparse
import logging
from pathlib import Path

from app.services.transcription.feature_1_automatic_captioning import AutomaticCaptioner

BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "storage" / "exports" / "feature_1"


def main():
    parser = argparse.ArgumentParser(description="Feature 1: Automatic Video Caption Generation")
    parser.add_argument("--input", required=True, type=Path, help="Path to input video file")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Export directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    captioner = AutomaticCaptioner(model_size="large-v3", device="cpu")
    result = captioner.generate_captions(args.input)

    # Save outputs in JSON, SRT, and VTT formats
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    captioner.export_json(result, args.out_dir / f"{stem}_captions.json")
    captioner.export_srt(result, args.out_dir / f"{stem}_captions.srt")
    captioner.export_vtt(result, args.out_dir / f"{stem}_captions.vtt")

    print("\n" + "=" * 80)
    print("FEATURE 1: AUTOMATIC CAPTION GENERATION SUMMARY")
    print("=" * 80)
    print(f"Video File           : {result.video_path}")
    print(f"Duration             : {result.duration_s}s")
    print(f"Detected Language    : {result.detected_language} (confidence: {result.language_probability})")
    print(f"Total Segments       : {len(result.segments)}")
    print("-" * 80)

    for seg in result.segments[:5]:  # Print first 5 segments
        print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] (Conf: {seg.confidence}) {seg.text}")

    print("-" * 80)
    print(f"Exports saved to: {args.out_dir}")


if __name__ == "__main__":
    main()