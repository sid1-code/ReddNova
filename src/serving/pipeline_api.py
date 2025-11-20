from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI(title="Fake News Pipeline API")

RETRIEVER_URL = "http://127.0.0.1:9000/retrieve"
VERIFIER_URL = "http://127.0.0.1:9001/verify"

class PipelineRequest(BaseModel):
    claim: str

@app.post("/pipeline")
def run_pipeline(req: PipelineRequest):
    # Call retriever
    r = requests.post(RETRIEVER_URL, json={"query": req.claim})
    evidence = r.json().get("results", [])

    # Call verifier
    v = requests.post(VERIFIER_URL, json={"claim": req.claim})
    verdict = v.json()

    return {
        "claim": req.claim,
        "verdict": verdict,
        "evidence": evidence
    }
