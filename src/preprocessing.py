"""
src/preprocessing.py

Text cleaning, validation, normalization, and label encoding utilities
for the 4-class Dark Pattern Detection research task:
  1. Social Proof
  2. Scarcity
  3. Urgency
  4. Misdirection
"""

import html
import re
from typing import Dict, Any, Tuple, List, Optional

# Primary 4-class taxonomy
TARGET_CLASSES: List[str] = [
    "Social Proof",
    "Scarcity",
    "Urgency",
    "Misdirection",
]

# Excluded categories that must NOT be relabeled or included in the 4-class task
EXCLUDED_CLASSES: List[str] = [
    "Obstruction",
    "Sneaking",
    "Forced Action",
]

LABEL_MAP: Dict[int, str] = {i: name for i, name in enumerate(TARGET_CLASSES)}
REVERSE_LABEL_MAP: Dict[str, int] = {name: i for i, name in enumerate(TARGET_CLASSES)}


def sanitize_text(text: Any) -> str:
    """Sanitizes text, decoding HTML entities, handling non-breaking spaces, and normalizing whitespace."""
    if text is None:
        return ""
    text_str = str(text)
    # Unescape HTML entities (&amp;, &nbsp;, etc.)
    text_str = html.unescape(text_str)
    # Replace non-breaking and zero-width spaces
    text_str = text_str.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    # Normalize control characters except standard whitespace
    text_str = re.sub(r"[\r\n\t]+", " ", text_str)
    # Normalize multiple spaces
    text_str = re.sub(r"\s+", " ", text_str).strip()
    return text_str


def validate_input_text(text: str, max_chars: int = 2000) -> Tuple[bool, str]:
    """Validates input text for inference and training.
    
    Returns (is_valid, sanitized_text_or_error_msg).
    """
    if not isinstance(text, str):
        return False, "Input must be a string."
    cleaned = sanitize_text(text)
    if len(cleaned) == 0:
        return False, "Input text cannot be empty or only whitespace."
    if len(cleaned) > max_chars:
        return False, f"Input text exceeds maximum length of {max_chars} characters (received {len(cleaned)})."
    return True, cleaned


def extract_surface_stats(text: str) -> Dict[str, Any]:
    """Computes basic surface statistics for reporting and inspection."""
    cleaned = sanitize_text(text)
    words = cleaned.split()
    word_count = len(words)
    char_count = len(cleaned)

    uppercase_words = [w for w in words if w.isupper() and len(w) > 1 and w.isalpha()]
    uppercase_ratio = len(uppercase_words) / max(1, word_count)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "uppercase_word_count": len(uppercase_words),
        "uppercase_word_ratio": round(uppercase_ratio, 4),
        "has_exclamation": "!" in cleaned,
        "has_question": "?" in cleaned,
        "has_digits": bool(re.search(r"\d", cleaned)),
    }
