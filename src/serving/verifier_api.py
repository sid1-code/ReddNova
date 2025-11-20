from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI(title="FakeNews Verifier API")

# ----- Request Model -----
class VerifyRequest(BaseModel):
    claim: str

# ----- Response Model -----
class VerifyResponse(BaseModel):
    label: str
    confidence: float
    model_version: str

# ----- Dummy model -----
def predict_fake_news(claim: str):
    # Replace this with your real model later
    labels = ["true", "false", "unverified"]
    label = random.choice(labels)
    confidence = round(random.uniform(0.55, 0.98), 3)
    return label, confidence

# ----- API Route -----
@app.post("/verify", response_model=VerifyResponse)
def verify_claim(req: VerifyRequest):
    label, confidence = predict_fake_news(req.claim)
    return VerifyResponse(
        label=label,
        confidence=confidence,
        model_version="v0-dummy"   # Replace later
    )
