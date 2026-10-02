from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from speechbrain.inference.separation import SepformerSeparation
from speechbrain.utils.fetching import LocalStrategy


# ============================================================
# PATHS
# ============================================================

INPUT = Path(
    r"C:\Projects\intelligent-captioning\backend\storage\audio\veg_context_demucs.wav"
)

OUTPUT_DIR = Path(
    r"C:\Projects\intelligent-captioning\backend\storage\audio\sepformer"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cpu"

MODEL_SOURCE = "speechbrain/sepformer-wsj02mix"

MODEL_DIR = "pretrained_models/sepformer-wsj02mix"


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT}")


# ============================================================
# LOAD SEPFORER
# ============================================================

print("Loading SepFormer...")

separator = SepformerSeparation.from_hparams(
    source=MODEL_SOURCE,
    savedir=MODEL_DIR,
    run_opts={"device": DEVICE},
    local_strategy=LocalStrategy.COPY,
)

print("SepFormer loaded.")
print(f"Input: {INPUT}")


# ============================================================
# LOAD WAV WITHOUT TORCHAUDIO / TORCHCODEC
# ============================================================

print("Loading audio with SoundFile...")

audio, sample_rate = sf.read(
    str(INPUT),
    dtype="float32",
    always_2d=True,
)

print(f"Original sample rate: {sample_rate}")
print(f"Original shape: {audio.shape}")


# Convert stereo/multi-channel -> mono
audio = audio.mean(axis=1)

print(f"Mono samples: {len(audio)}")
print(f"Duration: {len(audio) / sample_rate:.2f}s")


# ============================================================
# SEPFORER WSJ02MIX EXPECTS 8 kHz AUDIO
# ============================================================

TARGET_SAMPLE_RATE = 8000

if sample_rate != TARGET_SAMPLE_RATE:
    print(
        f"Resampling audio: "
        f"{sample_rate} Hz -> {TARGET_SAMPLE_RATE} Hz"
    )

    audio = resample_poly(
        audio,
        TARGET_SAMPLE_RATE,
        sample_rate,
    ).astype(np.float32)

    sample_rate = TARGET_SAMPLE_RATE


print(f"Final sample rate: {sample_rate}")
print(f"Final duration: {len(audio) / sample_rate:.2f}s")


# ============================================================
# NUMPY -> PYTORCH
# ============================================================

waveform = torch.from_numpy(audio)

# [samples] -> [batch, samples]
waveform = waveform.unsqueeze(0)

print(f"Waveform shape: {waveform.shape}")


# ============================================================
# SEPARATE SPEAKERS
# ============================================================

print("Separating speech...")

with torch.no_grad():
    estimated_sources = separator.separate_batch(waveform)


print(
    f"Separated tensor shape: "
    f"{estimated_sources.shape}"
)


# ============================================================
# SAVE SEPARATED SOURCES
# ============================================================

sources = estimated_sources[0]

print(f"Number of separated sources: {sources.shape[-1]}")


for i in range(sources.shape[-1]):

    source = sources[:, i].detach().cpu().numpy()

    output_path = OUTPUT_DIR / f"speaker_{i + 1}.wav"

    sf.write(
        str(output_path),
        source,
        TARGET_SAMPLE_RATE,
        subtype="PCM_16",
    )

    print(f"Saved: {output_path}")


print()
print("=" * 60)
print("DONE")
print("=" * 60)