import logging
from typing import Any, Dict

from fastapi import FastAPI, Body, HTTPException

from preprocess import preprocess as preprocess_text
from model import predict as predict_label


# Start command (run from the backend directory):
# uvicorn api:app --reload --port 8000


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(title="Fake News Detection API", version="0.1.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(text: str = Body(..., embed=False)) -> Dict[str, Any]:
    """Predict whether input text is FAKE or REAL.

    Request body should be a raw string (not an object). Example:
    "This is a sample news text..."
    """
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Input text is empty.")

        logger.info("Received /predict request with %d characters", len(text))

        processed = preprocess_text(text)
        label, confidence = predict_label(processed)

        response: Dict[str, Any] = {
            "processed_text": processed,
            "label": label,
            "confidence": round(float(confidence), 4),
        }
        logger.debug("Response: %s", response)
        return response

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime dependent
        logger.exception("/predict failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error")


