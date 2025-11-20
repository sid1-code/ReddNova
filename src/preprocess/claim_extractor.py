#!/usr/bin/env python3
import re
import spacy
from typing import List, Tuple

# Load spaCy model once (fast)
nlp = spacy.load("en_core_web_sm")

# Keywords indicating a factual claim, news, rumor, leak, etc.
KEYWORDS = {
    "confirm", "announc", "leak", "rumor", "rumour", "fake", "true",
    "patch", "update", "release", "banned", "ban",
    "exploit", "cheat", "developer", "studio", "revealed",
    "breaking", "report", "claimed", "statement"
}

def clean_text(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def is_assertive(sent) -> bool:
    """Determine if a sentence expresses a claim."""
    s = sent.text.lower()

    # Reject extremely short or long sentences
    if len(s) < 8 or len(s) > 400:
        return False

    # Keywords appear in the sentence
    if any(k in s for k in KEYWORDS):
        return True

    # Has a verb (basic heuristic for statements)
    for tok in sent:
        if tok.pos_ == "VERB":
            return True

    return False

def extract_claims(title: str, body: str, max_sentences: int = 3) -> List[Tuple[int, str]]:
    """Extract 1–3 meaningful claim sentences from a Reddit post."""
    text = (title or "") + ". " + (body or "")
    text = clean_text(text)
    
    doc = nlp(text)
    sents = list(doc.sents)

    candidates = []

    for i, sent in enumerate(sents):
        if is_assertive(sent):
            candidates.append((i, sent.text.strip()))
        if len(candidates) >= max_sentences:
            break

    # Fallback: if no assertive sentence found, take first sentence
    if not candidates and sents:
        candidates.append((0, sents[0].text.strip()))

    return candidates

# Quick local test
if __name__ == "__main__":
    title = "Valve confirms Steam sale starts tomorrow"
    body = "A leak earlier today showed discounted prices in the Steam database."
    print(extract_claims(title, body))
