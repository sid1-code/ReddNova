import logging
import re
from typing import Optional


logger = logging.getLogger("preprocess")


try:
    import spacy  # type: ignore
    _NLP = spacy.load("en_core_web_sm")
except Exception as exc:  # pragma: no cover - environment dependent
    # Delay hard failure until masking is actually called
    logger.warning(
        "spaCy model not loaded: %s. Run: python -m spacy download en_core_web_sm",
        exc,
    )
    _NLP = None  # type: ignore


def normalize_disguised(text: str) -> str:
    """Normalize simple disguised characters often used to evade filters.

    Replacements (case-insensitive):
    - 0 -> o
    - 1 -> i
    - @ -> a
    """
    if not text:
        return ""

    # Normalize common obfuscations using regex for broader coverage
    normalized = text

    # Replace zero used as letter 'o'
    normalized = re.sub(r"(?i)0", "o", normalized)

    # Replace one used as letter 'i'
    normalized = re.sub(r"(?i)1", "i", normalized)

    # Replace at-sign used as letter 'a'
    normalized = normalized.replace("@", "a")

    # Collapse multiple spaces produced by replacements
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _ensure_nlp_loaded() -> None:
    """Ensure spaCy model is loaded, with a clear error if missing."""
    global _NLP
    if _NLP is None:  # type: ignore
        try:
            import spacy  # type: ignore

            _NLP = spacy.load("en_core_web_sm")  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "spaCy English model is not available. Install it with: "
                "python -m spacy download en_core_web_sm"
            ) from exc


def mask_entities(text: str) -> str:
    """Mask PERSON and location-like entities with placeholder tags.

    Tags used:
    - [MASK_PERSON] for PERSON
    - [MASK_GPE] for geopolitical entities (countries, cities)
    - [MASK_LOC] for LOC (non-GPE locations)
    """
    if not text:
        return ""

    _ensure_nlp_loaded()
    doc = _NLP(text)  # type: ignore

    # Build replacements from end to start to preserve character indices
    replacements = []
    for ent in doc.ents:
        label = ent.label_
        if label == "PERSON":
            token = "[MASK_PERSON]"
        elif label == "GPE":
            token = "[MASK_GPE]"
        elif label == "LOC":
            token = "[MASK_LOC]"
        else:
            continue
        replacements.append((ent.start_char, ent.end_char, token))

    masked = text
    for start, end, token in sorted(replacements, key=lambda x: x[0], reverse=True):
        masked = masked[:start] + token + masked[end:]

    # Normalize whitespace after masking
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked


def preprocess(text: str) -> str:
    """Run normalization and entity masking in sequence.

    1) normalize_disguised
    2) mask_entities
    """
    normalized = normalize_disguised(text)
    masked = mask_entities(normalized)
    return masked


