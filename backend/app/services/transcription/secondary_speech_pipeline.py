"""
secondary_speech_pipeline.py

Recover quiet / overlapping background speech automatically without pre-fed prompts,
normalizing outputs to conversational Hinglish spellings.
"""
from __future__ import annotations

import argparse
import json
import logging
import string
import subprocess
import tempfile
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from app.services.transcription.hinglish_normalizer import normalize_to_manual_hinglish

log = logging.getLogger("secondary_speech")

SR = 16_000
BACKEND_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = BACKEND_ROOT / "storage" / "exports" / "secondary_speech"

BOOST_FILTER = (
    "highpass=f=100,"
    "acompressor=threshold=0.03:ratio=10:attack=5:release=80:makeup=6,"
    "dynaudnorm=f=100:g=11:p=0.9:m=20:s=5"
)

KNOWN_HALLUCINATIONS = (
    "thanks for watching",
    "thank you for watching",
    "subscribe",
    "amara.org",
    "subtitles by",
    "please like",
    "thank you guys",
    "thank you very much",
)


@dataclass
class Config:
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    window_s: float = 5.0
    hop_s: float = 2.5
    use_channels: bool = True
    use_boost: bool = True
    min_support: int = 2
    agree_jaccard: float = 0.4
    novelty_min: float = 0.4
    max_compression_ratio: float = 2.4
    min_avg_logprob: float = -1.0
    max_no_speech_prob: float = 0.6
    max_words_per_sec: float = 6.0
    boost_filter: str = BOOST_FILTER
    out_dir: Path = DEFAULT_OUT_DIR


@dataclass
class Candidate:
    start: float
    end: float
    variant: str
    lang: str
    engine: str
    boosted: bool
    text: str
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None
    gate_reasons: list[str] = field(default_factory=list)

    @property
    def passes_gate(self) -> bool:
        return not self.gate_reasons


def load_audio(path: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "in.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-vn",
             "-ar", str(SR), "-c:a", "pcm_s16le", str(wav)],
            check=True,
        )
        data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    assert sr == SR
    return data


def channel_variants(data: np.ndarray, use_channels: bool) -> dict[str, np.ndarray]:
    mid = data.mean(axis=1)
    variants = {"mid": mid}
    if use_channels and data.shape[1] >= 2:
        left, right = data[:, 0], data[:, 1]
        variants.update({"left": left, "right": right, "side": left - right})
    return variants


def boost_chunk(chunk: np.ndarray, filt: str) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
         "-i", "pipe:0", "-af", filt, "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"],
        input=chunk.astype(np.float32).tobytes(),
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return chunk
    out = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    return out / peak * 0.9 if peak > 1.0 else out


_STRIP = string.punctuation + "।“”‘’…"


def tokens(text: str) -> list[str]:
    return [t for t in (w.strip(_STRIP).lower() for w in text.split()) if t]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def compression_ratio(text: str) -> float:
    b = text.encode("utf-8")
    return len(b) / max(len(zlib.compress(b)), 1)


def max_consecutive_repeat(toks: list[str]) -> int:
    best = 1
    for n in (1, 2, 3):
        i = 0
        while i + n <= len(toks):
            gram, reps, j = toks[i:i + n], 1, i + n
            while toks[j:j + n] == gram:
                reps += 1
                j += n
            best = max(best, reps)
            i += 1
    return best


def run_gate(c: Candidate, cfg: Config) -> list[str]:
    reasons: list[str] = []
    toks = tokens(c.text)
    dur = max(c.end - c.start, 0.1)
    if not toks:
        return ["empty"]
    if compression_ratio(c.text) > cfg.max_compression_ratio:
        reasons.append("high_compression_ratio")
    if max_consecutive_repeat(toks) >= 3:
        reasons.append("repeated_ngram")
    if c.avg_logprob is not None and c.avg_logprob < cfg.min_avg_logprob:
        reasons.append("low_logprob")
    if c.no_speech_prob is not None and c.no_speech_prob > cfg.max_no_speech_prob:
        reasons.append("high_no_speech_prob")
    if len(toks) / dur > cfg.max_words_per_sec:
        reasons.append("too_many_words_for_duration")
    low = c.text.lower()
    if any(p in low for p in KNOWN_HALLUCINATIONS):
        reasons.append("known_hallucination_phrase")
    return reasons


class WhisperEngine:
    name = "faster-whisper"

    def __init__(self, cfg: Config):
        from faster_whisper import WhisperModel

        device = cfg.device
        compute_type = cfg.compute_type

        if device == "cpu" or compute_type == "auto":
            compute_type = "int8"
            device = "cpu"

        self.model = WhisperModel(cfg.model, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray):
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            language=None,
            beam_size=5,
            temperature=[0.0, 0.2, 0.4],
            condition_on_previous_text=False,
            vad_filter=False,
            no_speech_threshold=None,
            log_prob_threshold=None,
            compression_ratio_threshold=2.4,
        )
        return list(segments)


def sliding_windows(total_s: float, win: float, hop: float) -> list[tuple[float, float]]:
    out, t = [], 0.0
    while t < total_s:
        out.append((t, min(t + win, total_s)))
        if t + win >= total_s:
            break
        t += hop
    return out


def collect_candidates(variants: dict[str, np.ndarray], windows, engine: WhisperEngine, cfg: Config) -> list[Candidate]:
    cands: list[Candidate] = []
    for (w0, w1) in windows:
        i0, i1 = int(w0 * SR), int(w1 * SR)
        for vname, sig in variants.items():
            raw = sig[i0:i1]
            if raw.size < SR // 2:
                continue
            passes: list[tuple[str, np.ndarray, bool]] = [(vname, raw, False)]
            if cfg.use_boost:
                boosted = boost_chunk(raw, cfg.boost_filter)
                passes.append((vname, boosted, True))

            for pname, audio, is_boosted in passes:
                for seg in engine.transcribe(audio):
                    raw_text = seg.text.strip()
                    if not raw_text:
                        continue
                    
                    # Apply automated Hinglish normalization
                    final_text = normalize_to_manual_hinglish(raw_text)

                    cands.append(Candidate(
                        start=w0 + seg.start, end=w0 + seg.end, variant=pname, lang="auto",
                        engine=engine.name, boosted=is_boosted, text=final_text,
                        avg_logprob=seg.avg_logprob,
                        compression_ratio=seg.compression_ratio,
                        no_speech_prob=seg.no_speech_prob,
                    ))
    for c in cands:
        c.gate_reasons = run_gate(c, cfg)
    return cands


def group_and_decide(cands: list[Candidate], baseline_tokens: set[str], cfg: Config):
    good = sorted((c for c in cands if c.passes_gate), key=lambda c: c.start)

    clusters: list[list[Candidate]] = []
    for c in good:
        ct = tokens(c.text)
        for cl in clusters:
            rep = cl[0]
            overlaps = min(rep.end, c.end) - max(rep.start, c.start) > 0
            if overlaps and jaccard(tokens(rep.text), ct) >= cfg.agree_jaccard:
                cl.append(c)
                break
        else:
            clusters.append([c])

    raw_records = []
    for cl in clusters:
        support_set = {(c.variant, c.lang, c.boosted, c.engine) for c in cl}
        engines = {c.engine for c in cl}
        best = max(cl, key=lambda c: (c.avg_logprob if c.avg_logprob is not None else -0.5))
        toks = tokens(best.text)
        novelty = (sum(t not in baseline_tokens for t in toks) / len(toks)) if toks else 0.0

        raw_records.append({
            "start": round(best.start, 2),
            "end": round(best.end, 2),
            "text": best.text,
            "support": len(support_set),
            "engines": sorted(engines),
            "novelty_vs_baseline": round(novelty, 2),
        })

    merged_records = []
    for rec in sorted(raw_records, key=lambda x: x["start"]):
        if not merged_records:
            merged_records.append(rec)
            continue

        prev = merged_records[-1]
        same_text = jaccard(tokens(prev["text"]), tokens(rec["text"])) >= 0.4
        adjacent_or_overlapping = rec["start"] <= prev["end"] + 1.0

        if same_text and adjacent_or_overlapping:
            prev["end"] = max(prev["end"], rec["end"])
            prev["start"] = min(prev["start"], rec["start"])
            prev["support"] += rec["support"]
            prev["engines"] = sorted(list(set(prev["engines"] + rec["engines"])))
        else:
            merged_records.append(rec)

    accepted, flagged = [], []
    for rec in merged_records:
        agreed = rec["support"] >= cfg.min_support or len(rec["engines"]) >= 2
        if agreed and rec["novelty_vs_baseline"] >= cfg.novelty_min:
            accepted.append(rec)
        elif rec["novelty_vs_baseline"] >= cfg.novelty_min:
            rec["status"] = "[unclear]"
            flagged.append(rec)

    return accepted, flagged


def run(input_path: Path, cfg: Config) -> dict:
    data = load_audio(input_path)
    total_s = data.shape[0] / SR
    variants = channel_variants(data, cfg.use_channels)

    engine = WhisperEngine(cfg)
    baseline_segs = engine.transcribe(variants["mid"])
    baseline_text = " ".join(normalize_to_manual_hinglish(s.text.strip()) for s in baseline_segs)
    baseline_tokens = set(tokens(baseline_text))

    windows = sliding_windows(total_s, cfg.window_s, cfg.hop_s)
    cands = collect_candidates(variants, windows, engine, cfg)
    accepted, flagged = group_and_decide(cands, baseline_tokens, cfg)

    result = {
        "input": str(input_path),
        "duration_s": round(total_s, 2),
        "baseline_text": baseline_text,
        "recovered_secondary": accepted,
        "unclear": flagged,
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.out_dir / f"{input_path.stem}_secondary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--window", type=float, default=5.0)
    ap.add_argument("--hop", type=float, default=2.5)
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO)
    cfg = Config(model=a.model, window_s=a.window, hop_s=a.hop, min_support=a.min_support)
    res = run(a.input, cfg)

    print("\n=== BASELINE ===\n", res["baseline_text"])
    print("\n=== RECOVERED SECONDARY SPEECH ===")
    for r in res["recovered_secondary"]:
        print(f"[{r['start']:.1f}-{r['end']:.1f}] {r['text']}  (support={r['support']})")


if __name__ == "__main__":
    main()