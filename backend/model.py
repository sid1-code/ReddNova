import logging
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


logger = logging.getLogger("model")


# Model name for pretrained fake news classification
_MODEL_NAME = "mrm8488/bert-tiny-finetuned-fake-news-detection"


class _ModelBundle:
    """Container for globally loaded tokenizer and model."""

    def __init__(self) -> None:
        logger.info("Loading HuggingFace model: %s", _MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)

        # Prefer GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Normalize labels mapping to ensure FAKE/REAL outputs
        self.id2label = {int(k): str(v).upper() for k, v in self.model.config.id2label.items()} if hasattr(self.model.config, "id2label") else {}


_BUNDLE: _ModelBundle = _ModelBundle()


def _id_to_label(idx: int) -> str:
    """Map class index to a human label.

    If model provides id2label, use it; otherwise fallback to index mapping.
    """
    if _BUNDLE.id2label:
        return _BUNDLE.id2label.get(idx, "FAKE" if idx == 0 else "REAL")

    # Sensible default if config lacks labels
    return "FAKE" if idx == 0 else "REAL"


def predict(text: str) -> Tuple[str, float]:
    """Predict whether the text is FAKE or REAL with confidence.

    Returns:
    (label, confidence) where label in {"FAKE", "REAL"} and confidence in [0, 1].
    """
    if not text:
        return "FAKE", 0.0

    encoded = _BUNDLE.tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    encoded = {k: v.to(_BUNDLE.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = _BUNDLE.model(**encoded)
        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        conf, idx = torch.max(probs, dim=-1)

    label = _id_to_label(int(idx.item()))
    confidence = float(conf.item())
    # Normalize label text strictly to FAKE/REAL if variants exist
    label_upper = label.upper()
    if "FAKE" in label_upper and "REAL" not in label_upper:
        label = "FAKE"
    elif "REAL" in label_upper and "FAKE" not in label_upper:
        label = "REAL"
    else:
        # Fallback by index if ambiguous
        label = "FAKE" if int(idx.item()) == 0 else "REAL"

    return label, confidence


