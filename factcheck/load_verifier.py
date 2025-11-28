import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------
# Correct project root (only go up ONE level)
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "data" / "models" / "domain_classifier"

print("FINAL MODEL PATH:", MODEL_PATH)

# ---------------------------------------------------------
# Load tokenizer + model from local folder only
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)

model.eval()

# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------
LABELS = ["real", "fake", "unverified"]  # Adjust if needed


def classify_claim(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

    idx = torch.argmax(probs).item()
    confidence = float(probs[idx])

    return {
        "label": LABELS[idx],
        "confidence": round(confidence, 4)
    }
