import os
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "data", "documents.json")
INDEX_PATH = os.path.join(BASE_DIR, "data", "nn_index.npy")


# --------------------------------------
# LOAD DOCUMENTS
# --------------------------------------
def load_documents():
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"documents.json not found at {DOCS_PATH}")

    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    return docs


# --------------------------------------
# BUILD INDEX
# --------------------------------------
def build_index(load_existing: bool = True):

    documents = load_documents()
    texts = [d["text"] for d in documents]

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if load_existing and os.path.exists(INDEX_PATH):
        print("Loading existing NN index...")
        embeddings = np.load(INDEX_PATH)

    else:
        print("Building new NN index (sklearn)...")

        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        # Save embeddings to speed up future loads
        np.save(INDEX_PATH, embeddings)

    # Initialize nearest neighbor search
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(embeddings)

    return nn, documents


# --------------------------------------
# RETRIEVE TOP-K
# --------------------------------------
def retrieve(text: str, embedder, index, documents, top_k: int = 3):

    query_vec = embedder.encode([text], convert_to_numpy=True)

    distances, indices = index.kneighbors(query_vec, n_neighbors=top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        doc = documents[idx]
        results.append({
            "text": doc["text"],
            "source": doc.get("source", "unknown"),
            "score": float(1 - dist)  # cosine similarity
        })

    return results
