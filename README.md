
---

# CapFlow

**CapFlow** is an automated, multi-stage caption generation engine built to process code-switched **Hinglish (Hindi + English)** videos.

When speakers mix Hindi and English mid-sentence, standard Automatic Speech Recognition (ASR) engines struggle—often outputting English loanwords as phonetic Hindi transliterations or mangled spellings (e.g., turning *"egg rice"* into `"egaraysa"` or *"practice"* into `"praiktisa"`).

CapFlow resolves these phonetic artifacts by executing a 3-step pipeline: transcribing audio with word-level timestamps, transliterating native script to Roman letters, and applying an LLM post-processing pass to produce human-readable Hinglish captions.

---

## Architecture Overview

```
                          ┌───────────────────────────┐
                          │   Input Video (.mp4)      │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Acoustic Pass (faster-whisper large-v3)                              │
│ • Extracts audio & applies VAD filtering (silence removal)                    │
│ • Emits word-level timestamps & language tags (Hindi vs. English)             │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Transliteration Layer (indic_transliteration)                        │
│ • Converts Hindi Devanagari script into Romanized Latin text (ITRANS format)  │
│ • Preserves exact token timing while unifying character encodings             │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Contextual LLM Cleanup (Groq SDK / Llama 3.1 8B)                     │
│ • Evaluates full sentence context to normalize phonetic artifacts             │
│ • Maps raw ITRANS tokens to standard English/Hinglish ("egaraysa" -> "egg rice")│
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       │
                                       ▼
                          ┌───────────────────────────┐
                          │ Output Subtitles          │
                          │ (.json, .srt, .vtt)       │
                          └───────────────────────────┘

```

---

## Tech Stack & Version Matrix

| Dependency | Minimum Version | Role in Pipeline |
| --- | --- | --- |
| **Python** | `3.10+` | Core Runtime Environment |
| **faster-whisper** | `0.10.0+` | CTranslate2-optimized implementation of OpenAI Whisper (`large-v3`) |
| **indic-transliteration** | `2.3.0+` | Devanagari to ITRANS Latin script conversion |
| **groq** | `0.4.0+` | Low-latency inference client for Llama 3 API execution |
| **numpy** | `1.24.0+` | Probability distribution analysis & confidence scoring |
| **pydantic / dataclasses** | Native | Data schema validation & JSON payload serialization |

---

## Key Pipeline Features

* **Word-Level Code-Switching Detection:** Assigns per-word language origin tags (`hi` vs. `en`) along with probability confidence scores.
* **Contextual Orthographic Normalization:** Automatically fixes spoken Indian English loanwords without relying on brittle, hardcoded word dictionaries.
* **Automatic LLM Failover:** Cycles dynamically through Groq candidate endpoints (`llama-3.1-8b-instant` $\rightarrow$ `llama3-8b-8192` $\rightarrow$ `llama-3.3-70b-versatile`) to protect against single-model API downtime.
* **Payload Length Guarantee:** Enforces structured JSON output matching input segment array sizes ($N \rightarrow N$), preventing subtitle timing drift.

---

## How It Works

### Step 1: Acoustic Processing & VAD

`faster-whisper` processes the input video's audio stream. Voice Activity Detection (VAD) filters out silent intervals to prevent acoustic hallucination loops. The model assigns a confidence score and language tag to every spoken word token.

### Step 2: Indic Transliteration

Whisper output generated in native Devanagari script (e.g., `"और"`, `"चाय"`) is mapped into Latin characters (`"aura"`, `"chaya"`) while maintaining token-level start and end timestamps.

### Step 3: Contextual Cleanup Pass

The transliterated lines are sent to Groq's API in a single JSON payload. The LLM evaluates surrounding sentence context to normalize phonetic artifacts (e.g., `"gaisa"` $\rightarrow$ `"guys"`, `"kritika"` $\rightarrow$ `"critique"`, `"vidiyo"` $\rightarrow$ `"video"`).

---

## Quickstart Guide

### 1. Installation

```bash
git clone https://github.com/stutikatiyar/CapFlow.git
cd CapFlow/backend

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate      # On Windows
# source venv/bin/activate   # On Linux/macOS

pip install -r requirements.txt

```

### 2. Set API Key

Get an API key from [console.groq.com](https://console.groq.com) and declare it in your environment:

```powershell
$env:GROQ_API_KEY="gsk_YOUR_GROQ_API_KEY"

```

### 3. Run Pipeline

```bash
python -m app.services.transcription.run_feature_3 --input storage/uploads/test_hinglish.mp4

```

---

## Output Artifacts

Exported caption payloads are saved automatically under `storage/exports/feature_3/`:

* **`.json`**: Complete structured transcript containing word-level language tags, timing, and confidence metrics.
* **`.srt`**: Standard SubRip subtitle format for video players and video editing software.
* **`.vtt`**: WebVTT format for web-based HTML5 media players.

---

## System Design Q&A (Interview Reference)

### Q1: Why use an LLM post-processing pass instead of retraining Whisper?

Fine-tuning Whisper requires massive datasets of annotated code-switched audio and substantial GPU compute. Retraining also risks degrading Whisper's core acoustic speech recognition. Decoupling **acoustic decoding** from **text cleanup** via a fast LLM (`llama-3.1-8b-instant`) provides zero-shot normalization without altering the underlying ASR model.

### Q2: How do subtitle timings remain synchronized after the LLM modifies text?

Array structure preservation is enforced using Groq's structured JSON mode (`response_format={"type": "json_object"}`). The prompt mandates that the output array length matches the input array length ($N \rightarrow N$). Word timings computed in Step 1 remain mapped to the exact corresponding segment in Step 3.

### Q3: Why choose `faster-whisper` over standard PyTorch Whisper?

`faster-whisper` utilizes CTranslate2, an optimized C++ inference engine for Transformer models. It achieves up to **4x faster execution speeds** and significantly reduces RAM consumption through `int8` quantization on CPU and CUDA runtimes.

---

## Future Roadmap & Vision

### Current Version: `v1.0.0` (MVP Pipeline)

* [x] Word-level language tagging (`hi` vs. `en`).
* [x] Devanagari to ITRANS script conversion.
* [x] Zero-shot LLM contextual normalization pass.
* [x] Automated `.json`, `.srt`, and `.vtt` export generation.

### Phase 2: Pipeline Optimization (`v1.1.0` - Planned)

* **Dedicated LoRA Fine-Tuning:** Replace general LLM prompts with a fine-tuned 1B/3B model trained specifically on Phonetic-to-Hinglish transcript mappings.
* **Dynamic VAD Tuning:** Adjust VAD silence thresholds dynamically based on audio speech rates to prevent long monologues from merging into single oversized segments.
* **Speaker Diarization:** Integrate `pyannote.audio` to identify speaker switches (`Speaker 1:`, `Speaker 2:`) directly within caption payloads.

### Phase 3: Platform Integration (`v2.0.0` - Vision)

* **Real-Time Streaming Captions:** Transition from file-based batch processing to WebSocket streaming for live video sources.
* **Interactive Web Subtitle Editor:** A React/Next.js interface allowing creators to adjust generated caption timing, style fonts, and burn captions into video clips.