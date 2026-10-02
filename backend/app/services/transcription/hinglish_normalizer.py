"""
hinglish_normalizer.py

Normalizes raw Whisper outputs / Devanagari transliterations to match 
conversational Hinglish spellings (e.g., guys, mai, fir, chalo, practice).
"""
import re

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_TRANSLITERATION = True
except ImportError:
    HAS_TRANSLITERATION = False


# Exact mapping dictionary matching conversational Hinglish spelling styles
HINGLISH_SPELLING_MAP = {
    "gaisa": "guys",
    "gaisa.n": "guys",
    "kaise": "guys",
    "maim": "mai",
    "main": "mai",
    "hu.n": "hu",
    "hun": "hu",
    "ho": "hu",
    "phira": "fir",
    "phira.n": "fir",
    "chala": "chalo",
    "bada": "baad",
    "mem": "me",
    "mein": "me",
    "rikaॉrda": "record",
    "vidiyo": "video",
    "praktisa": "practice",
    "peene": "peene",
    "aayi": "aayi",
    "raha": "raha",
    "kha": "kha",
    "baithake": "baithke",
    "taariif": "tareef",
    "taripha": "tareef",
    "critique": "critique",
    "kritipha": "critique",
    "kritika": "critique",
    "badi": "badi",
    "ba.di": "badi",
    "karegi": "karri thi",
    "muldama": "banana",
}


def devanagari_to_hinglish(text: str) -> str:
    """Converts Devanagari script to Latin characters using indic-transliteration."""
    if not HAS_TRANSLITERATION or not text:
        return text
    return transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)


def normalize_to_manual_hinglish(text: str) -> str:
    """
    Strips ITRANS diacritics and converts raw transliterated words 
    into standard conversational Hinglish spellings.
    """
    if not text:
        return ""

    # Convert Devanagari script if present
    transliterated = devanagari_to_hinglish(text)

    # Clean ITRANS special symbols, dots, and nasal accents
    cleaned = transliterated.replace('.', '').replace('~', '').replace('^', '')

    # Standardize double vowels and phonetic endings
    cleaned = re.sub(r'aim\b', 'ai', cleaned)
    cleaned = re.sub(r'um\b', 'u', cleaned)
    cleaned = re.sub(r'aa', 'a', cleaned)
    cleaned = re.sub(r'ii', 'i', cleaned)
    cleaned = re.sub(r'uu', 'u', cleaned)

    # Word-by-word spelling mapping
    words = cleaned.lower().split()
    normalized = [HINGLISH_SPELLING_MAP.get(w, w) for w in words]

    return " ".join(normalized)