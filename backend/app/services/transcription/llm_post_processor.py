import json
import logging
import os
from typing import List, Dict, Any

log = logging.getLogger(__name__)

# Generalized production prompt with zero hardcoded word pairs
SYSTEM_PROMPT = """You are an expert Hinglish subtitle post-processor for short-form social media videos.

Your task:
Convert phonetic ITRANS / Devanagari-transliterated transcripts into clean, modern Romanized Hinglish.

Transformation Principles:
1. Phonetic Loanwords: Identify spoken English loanwords that were phonetically transliterated into Devanagari/ITRANS (e.g., phonetic representations of "video", "practice", "guys", "critique", "egg rice", "down") and output them in standard English spelling.
2. Colloquial Hindi: Normalize phonetic Hindi tokens into standard conversational Romanized Hindi (e.g., "taripha" -> "tareef", "nahim" -> "nahi").
3. Contextual Disambiguation: Use surrounding sentence context to resolve misheard or ambiguous phonetic tokens.
4. Payload Integrity: Preserve segment array length and ordering exactly. Return strictly valid JSON with key "raw_segments".
"""

class LLMPostProcessor:
    def __init__(self, provider: str = "groq", model_name: str = "llama-3.1-8b-instant"):
        self.provider = provider.lower()
        self.model_name = model_name

    def process_segments(self, raw_segments: List[str]) -> List[str]:
        if not raw_segments:
            return []

        payload = {"raw_segments": raw_segments}
        payload_str = json.dumps(payload, ensure_ascii=False)

        if self.provider == "groq":
            return self._call_groq(payload_str, len(raw_segments))
        else:
            log.warning("Unsupported LLM provider '%s'. Returning raw segments.", self.provider)
            return raw_segments

    def _call_groq(self, payload_str: str, expected_count: int) -> List[str]:
        """Executes ultra-fast cloud inference using Groq API with automatic model failover."""
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        client = Groq(api_key=api_key)

        # Active Groq model IDs to attempt sequentially
        candidate_models = [
            self.model_name,
            "llama-3.1-8b-instant",
            "llama3-8b-8192",
            "llama-3.3-70b-versatile"
        ]
        # De-duplicate while preserving priority order
        candidate_models = list(dict.fromkeys([m for m in candidate_models if m]))

        last_error = None
        for model in candidate_models:
            try:
                log.info("Attempting Groq LLM completion with model: %s", model)
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": payload_str},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )

                content = completion.choices[0].message.content
                parsed = json.loads(content)
                corrected_list = parsed.get("raw_segments", parsed.get("corrected_segments", []))

                if len(corrected_list) == expected_count:
                    return corrected_list
                else:
                    log.warning("LLM returned %d segments, expected %d. Truncating/padding.", len(corrected_list), expected_count)
                    return corrected_list[:expected_count]

            except Exception as e:
                log.warning("Groq model '%s' failed: %s. Trying next candidate...", model, e)
                last_error = e
                continue

        log.error("All Groq model attempts failed. Last error: %s", last_error)
        raise last_error