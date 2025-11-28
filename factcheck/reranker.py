from sentence_transformers import CrossEncoder

cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(claim, docs):
    pairs = [[claim, d["text"][:512]] for d in docs]
    scores = cross.predict(pairs)

    ranked = sorted(
        [
            {
                "text": d["text"],
                "score": float(s),
                "url": d["url"],
                "title": d["title"]
            }
            for d, s in zip(docs, scores)
        ],
        key=lambda x: x["score"],
        reverse=True
    )
    return ranked
