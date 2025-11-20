#!/usr/bin/env python3

import os
import csv
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime

INPUT_CSV = "data/evidence/evidence_raw.csv"
INDEX_DIR = "data/index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
MAPPING_FILE = os.path.join(INDEX_DIR, "doc_mapping.pkl")

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def load_evidence(csv_path):
    texts = []
    mapping = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            url = row.get("url")
            text = row.get("text", "").strip()

            if not text or len(text) < 50:
                continue

            texts.append(text)
            mapping.append(url)

    return texts, mapping


def build_faiss_index(texts, mapping):
    print("Loading embedding model:", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Encoding", len(texts), "documents...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Saving FAISS index to:", INDEX_FILE)
    faiss.write_index(index, INDEX_FILE)

    print("Saving document mapping to:", MAPPING_FILE)
    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(mapping, f)

    print("Index built successfully.")


if __name__ == "__main__":
    print("Loading evidence CSV:", INPUT_CSV)
    texts, mapping = load_evidence(INPUT_CSV)

    print("Loaded", len(texts), "valid evidence documents.")
    build_faiss_index(texts, mapping)
