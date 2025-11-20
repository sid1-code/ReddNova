#!/usr/bin/env python3
import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize

INDEX_PATH = "data/index/faiss.index"
MAPPING_PATH = "data/index/doc_mapping.pkl"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

app = FastAPI(title="Evidence Retriever API")

# Load model and FAISS index at startup
print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading document mapping...")
with open(MAPPING_PATH, "rb") as f:
    doc_mapping = pickle.load(f)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    query = req.query

    # Encode query
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    # Search FAISS
    D, I = index.search(q_emb, req.top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        url = doc_mapping[idx]
        snippet = generate_snippet(url)

        results.append({
            "url": url,
            "score": float(score),
            "snippet": snippet
        })

    return {"results": results}


def generate_snippet(url: str):
    # Snippet generation using first few sentences
    # For now, we only use URL as identifier — snippet lookup coming soon from DB
    # Temporary solution: return the URL itself as snippet placeholder
    return f"Evidence from: {url}"


@app.get("/")
def home():
    return {"message": "Retriever API is running."}
