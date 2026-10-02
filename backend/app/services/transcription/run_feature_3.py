"""
run_feature_3.py

CLI runner to execute Feature 3 (Code-Switching Detection) on any input video.
"""
import argparse
import logging
from pathlib import Path

from app.services.transcription.feature_3_code_switching import CodeSwitchingCaptioner

BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "storage" / "exports" / "feature_3"


def main():
    parser = argparse.ArgumentParser(description="Feature 3: Code-Switching Detection ⭐")
    parser.add_argument("--input", required=True, type=Path, help="Path to input video/audio file")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Export directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    captioner = CodeSwitchingCaptioner(model_size="large-v3", device="cpu")
    result = captioner.process_code_switched_video(args.input)

    # Save exports
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    captioner.export_json(result, args.out_dir / f"{stem}_code_switched.json")
    captioner.export_srt(result, args.out_dir / f"{stem}_code_switched.srt")
    captioner.export_vtt(result, args.out_dir / f"{stem}_code_switched.vtt")

    print("\n" + "=" * 85)
    print("FEATURE 3: CODE-SWITCHING DETECTION SUMMARY ⭐")
    print("=" * 85)
    print(f"Video File             : {result.video_path}")
    print(f"Duration               : {result.duration_s}s")
    print(f"Primary Stream Language : {result.primary_language}")
    print(f"Hindi Word Count       : {result.hindi_word_count}")
    print(f"English Word Count     : {result.english_word_count}")
    print(f"Code-Switching Ratio   : {result.code_switching_ratio}% English tokens")
    print(f"Total Segments         : {len(result.segments)}")
    print("-" * 85)

    for seg in result.segments:
        print(f"\n[Segment {seg.id}] ({seg.start:.2f}s -> {seg.end:.2f}s)")
        print(f"Rendered Text: \"{seg.code_switched_text}\"")
        print("Word-Level Tagging:")
        for w in seg.words[:8]:  # Print first 8 words as sample
            print(f"  └─ [{w.start:.2f}s-{w.end:.2f}s] word: '{w.word}' (orig: '{w.original_word}') -> lang: {w.detected_language.upper()} ({w.confidence})")

    print("-" * 85)
    print(f"Exports saved to: {args.out_dir}")


if __name__ == "__main__":
    main()