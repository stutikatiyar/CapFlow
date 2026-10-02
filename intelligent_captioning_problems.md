# Intelligent Captioning — Problems & Worklist

## Project Goal

Build a captioning system for **natural multilingual / Hinglish / code-switched speech** that produces accurate, creator-ready captions instead of raw Whisper output.

---

# Problems Identified So Far

## 1. Transcription Accuracy / Missing Speech

**Problem:** Whisper sometimes distorts or fails to recover spoken content.

Examples:
- `I am veg` section was not properly recovered.
- `confidence thoda sa down` became something like `मेरा फॉर डाउन...`.
- Some sentences are only partially reconstructed.

### Work on
- Improve speech recognition accuracy.
- Test alternative decoding/settings.
- Benchmark additional STT models if needed.
- Measure missing words and semantic errors against ground truth.

**Priority: 🔴 High**

---

## 2. Gender / Grammatical Agreement

**Problem:** Whisper can change gender-dependent Hindi forms.

Example:

Actual:
`Mai yaha pe chai peene aayi hu`

Whisper:
`मैं यहां पर चाय पीना आया हूं`

Another example:
`mai video record karri thi`

was reconstructed closer to:
`मैं वीडियो रिकॉर्ड करेंगे...`

### Work on
- Speaker-aware processing.
- Preserve actual spoken morphology.
- Avoid blindly applying grammar correction.
- Investigate speaker-specific linguistic context.

**Priority: 🔴 High**

---

## 3. Code-Switching

**Problem:** Natural speech switches between Hindi and English, but Whisper does not always preserve the language choice.

Examples:
- `egg rice`
- `I am veg`
- `critique`
- `confidence`
- `video record`
- `practice`

### Work on
- Detect language at word/phrase level.
- Identify Hindi ↔ English switches.
- Preserve English words instead of translating/paraphrasing them.
- Build a code-switch-aware representation.

**Priority: 🔴 High**

---

## 4. Roman Hinglish Output

**Problem:** Whisper commonly outputs Hindi in Devanagari, while our target caption style is Roman Hinglish.

Example:

Desired:
`Mai yaha pe chai peene aayi hu`

Whisper:
`और मैं यहां पर चाय पीना आया हूं`

### Work on
- Hindi-script → Roman representation.
- Natural conversational Romanization, not just mechanical transliteration.
- Preserve English words in English.
- Support informal spellings such as:
  - `mai`
  - `hu`
  - `bhot`
  - `pe`
  - `karri`

**Priority: 🔴 High**

---

## 5. Semantic Errors

**Problem:** Some output words are plausible but change the meaning.

Example:

Actual:
`mera confidence thoda sa down ho jaata hai`

Output:
`मेरा फॉर डाउन हो जाता है`

### Work on
- Context-aware correction.
- Detect semantically suspicious words.
- Compare surrounding words and sentence meaning.
- Use an AI quality-checking layer.

**Priority: 🔴 High**

---

## 6. Speaker Awareness

**Problem:** The current pipeline does not identify who is speaking.

This video contains at least two speakers.

### Why it matters

Speaker identity can help with:
- Gender-dependent morphology
- Context
- Dialogue separation
- Multi-speaker captions
- Speaker-specific vocabulary

### Work on
- Speaker diarization.
- Assign words/segments to speaker IDs.
- Connect speaker context to later caption processing.

**Priority: 🟠 Medium-High**

> Note: Speaker diarization has not yet been implemented; this is a planned capability, not a demonstrated Whisper failure in Benchmark #1.

---

## 7. Word-Level Timestamp Reliability

**Problem:** Word timestamps exist, but some can be suspicious.

Potential issues:
- Zero-duration words
- Very long word durations
- Poor alignment
- Low-confidence words with questionable timing

### Work on
- Validate timestamp duration.
- Detect impossible/suspicious timings.
- Re-align words when necessary.
- Build a reliable word-timing layer.

**Priority: 🟠 Medium-High**

---

## 8. Caption Segmentation

**Problem:** Whisper's segments are not necessarily good final captions.

Example:
A long speech section can become one large segment.

The `condition_on_previous_text=False` experiment produced smaller chunks in some areas, suggesting configuration affects segmentation.

### Work on
- Sentence/phrase boundary detection.
- Natural pause detection.
- Maximum caption length.
- Reading-speed constraints.
- Word-level timing based segmentation.
- Keep phrases meaningful rather than splitting randomly.

**Priority: 🟠 Medium-High**

---

## 9. Confidence Is Not Enough

**Problem:** Whisper confidence scores cannot be treated as proof that a word is correct.

A word can have high confidence while the overall sentence is still semantically wrong.

### Work on

Build a **Confidence + Context Quality Checker** using:
- Word confidence
- Sentence context
- Language consistency
- Repetition
- Vocabulary
- Timing
- Semantic plausibility

**Priority: 🟠 Medium-High**

---

## 10. Silence / Hallucination Handling

**Important finding:**

The `0:07–0:08` section of our ground truth is **silence**, not missing speech.

Whisper did not generate a caption for that silent section.

Therefore:

### This is NOT currently a failure.

But our benchmark must distinguish:

- Real speech that was missed
- Intentional silence
- Background noise
- Hallucinated speech

### Work on
- Silence detection.
- Hallucination detection.
- Do not create captions during genuine silence.
- Measure false captions during silence.

**Priority: 🟠 Medium**

---

## 11. Language Forcing Does Not Solve Hinglish

We tested:

- Automatic language detection
- Forced Hindi
- Forced English
- VAD on/off
- Previous-text conditioning on/off

Findings:

### Forced Hindi
Essentially the same output as automatic detection.

### Forced English
Can produce English translations/paraphrases instead of preserving the actual Hinglish speech.

Example:
`I have come here to have tea and he is eating egg rice.`

This is NOT our desired behavior.

### Work on
Do not rely on a single forced language for multilingual/code-switched speech.

**Priority: 🟡 Architectural finding**

---

## 12. VAD Configuration

Disabling VAD changed segmentation but also introduced questionable output in places.

### Current conclusion

Keep VAD enabled for the current baseline while we investigate other improvements.

### Work on later
- Tune VAD.
- Compare silence handling across models/settings.
- Evaluate false speech around pauses/noise.

**Priority: 🟡 Later**

---

# Benchmark #1 Summary

### Test

**Natural Hinglish / Roman-Hinglish video**

Approximate duration:
**37 seconds**

Model:
**faster-whisper large-v3**

Current baseline:
- CPU
- int8
- VAD enabled
- word timestamps enabled
- beam size 5
- temperature 0
- automatic language detection

---

# Current Architecture Direction

Do NOT treat Whisper as the complete captioning system.

Think of it as the **speech-recognition engine**:

```text
VIDEO
  ↓
Audio
  ↓
Whisper / STT
  ↓
Raw word transcription
  ↓
Speaker detection
  ↓
Code-switch detection
  ↓
Confidence analysis
  ↓
Roman Hinglish reconstruction
  ↓
Semantic / grammar-aware correction
  ↓
Timestamp validation
  ↓
Intelligent caption segmentation
  ↓
AI Caption Quality Checker
  ↓
Final captions
```

---

# Priority Roadmap

## Phase 1 — Recognition

1. Improve transcription accuracy
2. Reduce missing speech
3. Investigate code-switch recognition
4. Benchmark alternative STT models if necessary

## Phase 2 — Language Understanding

5. Code-switch detection
6. Speaker awareness
7. Gender/morphology preservation
8. Semantic error detection

## Phase 3 — Caption Representation

9. Roman Hinglish reconstruction
10. English-word preservation
11. Natural conversational normalization

## Phase 4 — Timing

12. Word timestamp validation
13. Re-alignment if required
14. Intelligent caption segmentation

## Phase 5 — Quality Intelligence

15. Confidence-based review
16. Hallucination detection
17. AI caption quality checker
18. AI caption repair

---

# Immediate Next Task

## Benchmark #2 — Roman Hinglish Reconstruction

Start with the existing `01_baseline.json`.

Do NOT build the frontend yet.

Test whether we can transform:

`और मैं यहां पर चाय पीना आया हूं`

into a natural Roman-Hinglish representation closer to:

`Mai yaha pe chai peene aayi hu`

while preserving:

- speaker
- gender
- Hindi/English code-switching
- actual meaning
- conversational style
- word timing

Only after this experiment should we decide how much of the reconstruction needs rules, models, or an LLM.
