# factcheck/pipeline.py
"""
Optimized NLI-based fact-checking pipeline (BART-large-CNN summarizer, top-3 summaries).

Features:
- Loads sentence-transformers embeddings and FAISS/Sklearn/Numpy retrieval.
- Uses MNLI model (facebook/bart-large-mnli) for NLI entailment/contradiction decisions.
- Uses facebook/bart-large-cnn for abstractive summaries (fast-ish and reliable).
- Summarizes ONLY the top 3 retrieved evidence documents to avoid heavy compute / timeouts.
- Returns classifier-like output with NLI-derived pred/probs, per-evidence NLI scores, summaries
  (for top-3), and lightweight excerpts for others.

Added: Option A (STRICT) evidence filtering:
 - require >=2 claim keywords present in evidence
 - if claim contains an entity (TitleCase token or known brand/acronym), require at least one entity match
 - fallback to >=1 keyword if strict filter returns no documents
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import re
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

# Optional libs
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from sklearn.neighbors import NearestNeighbors
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# NLTK for simple extractive fallback
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("factcheck.pipeline")

# -------------------------
# Configuration
# -------------------------
HERE = Path(__file__).resolve().parent.parent

MODEL_PATH = os.environ.get("MODEL_PATH", "data/models/domain_classifier_final")
EVIDENCE_CLEAN = Path(os.environ.get("EVIDENCE_CLEAN", "data/evidence/evidence_clean.csv"))
EVIDENCE_RAW = Path(os.environ.get("EVIDENCE_RAW", "data/evidence/evidence_raw.csv"))
EMBED_CACHE = Path(os.environ.get("EMBED_CACHE", "data/evidence/embeddings.npy"))

TOP_K = int(os.environ.get("TOP_K", "5"))

# NLI and summarizer choices
NLI_MODEL = os.environ.get("NLI_MODEL", "facebook/bart-large-mnli")
SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", "facebook/bart-large-cnn")  # C2
ENTAILMENT_THRESHOLD = float(os.environ.get("ENTAILMENT_THRESHOLD", "0.75"))
CONTRADICTION_THRESHOLD = float(os.environ.get("CONTRADICTION_THRESHOLD", "0.75"))

# Evidence excerpt/truncation
EVIDENCE_MAX_CHARS = int(os.environ.get("EVIDENCE_MAX_CHARS", "300"))
TOP_SUMMARIES = int(os.environ.get("TOP_SUMMARIES", "3"))  # summarize only top 3

logger.info(f"PIPELINE MODEL PATH: {MODEL_PATH}")
logger.info(f"EVIDENCE CSV PATH: {EVIDENCE_CLEAN if EVIDENCE_CLEAN.exists() else EVIDENCE_RAW}")
logger.info(f"NLI model: {NLI_MODEL}; summarizer: {SUMMARIZER_MODEL}")

# -------------------------
# Load evidence dataframe
# -------------------------
def _load_evidence_dataframe() -> pd.DataFrame:
    if EVIDENCE_CLEAN.exists():
        path = EVIDENCE_CLEAN
    elif EVIDENCE_RAW.exists():
        path = EVIDENCE_RAW
    else:
        raise FileNotFoundError("No evidence CSV found. Place data/evidence/evidence_clean.csv or evidence_raw.csv")

    df = pd.read_csv(path)
    # normalize columns
    df["title"] = df.get("title", "").fillna("")
    df["text"] = df.get("text", "").fillna("")
    df = df[~((df["title"] == "") & (df["text"] == ""))].reset_index(drop=True)
    df["content"] = (df["title"].astype(str) + "\n\n" + df["text"].astype(str)).str.strip()
    return df

df_docs = _load_evidence_dataframe()
documents: List[str] = df_docs["content"].tolist()
logger.info(f"Loaded {len(documents)} evidence documents.")

# -------------------------
# Embeddings + index
# -------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
logger.info("Loaded sentence-transformers/all-MiniLM-L6-v2")

def compute_and_cache_embeddings(docs: List[str], cache_path: Path) -> np.ndarray:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            arr = np.load(cache_path)
            if arr.shape[0] == len(docs):
                logger.info("Loaded embeddings from cache.")
                return arr
            else:
                logger.info("Embedding cache length mismatch; recomputing.")
        except Exception:
            logger.info("Embedding cache unreadable; recomputing.")
    logger.info("Computing embeddings for %d docs...", len(docs))
    embs = embedder.encode(docs, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    np.save(cache_path, embs)
    logger.info("Saved embeddings to cache.")
    return embs

embeddings = compute_and_cache_embeddings(documents, EMBED_CACHE)

_index = None
_index_type = None

def _build_faiss_index(embs: np.ndarray):
    d = embs.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(embs.astype(np.float32))
    return idx

def _build_sklearn_index(embs: np.ndarray):
    nn = NearestNeighbors(n_neighbors=min(32, len(embs)), metric="cosine", algorithm="brute")
    nn.fit(embs)
    return nn

if _HAS_FAISS:
    try:
        _index = _build_faiss_index(embeddings)
        _index_type = "faiss"
        logger.info("Built FAISS index.")
    except Exception as e:
        logger.warning("FAISS build failed: %s", e)
        _index = None

if _index is None and _HAS_SKLEARN:
    try:
        _index = _build_sklearn_index(embeddings)
        _index_type = "sklearn"
        logger.info("Built sklearn NearestNeighbors index.")
    except Exception as e:
        logger.warning("sklearn index failed: %s", e)

if _index is None:
    _index_type = "numpy"
    logger.info("Using numpy-based search (no faiss/sklearn).")

# -------------------------
# NLI model (MNLI)
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device} for NLI and summarizer")

nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(device)
nli_model.eval()

def _get_nli_label_map(model) -> Dict[int, str]:
    cfg = getattr(model, "config", None)
    if cfg and hasattr(cfg, "id2label"):
        try:
            return {int(k): v.lower() for k, v in cfg.id2label.items()}
        except Exception:
            pass
    # fallback
    return {0: "contradiction", 1: "neutral", 2: "entailment"}

nli_id2label = _get_nli_label_map(nli_model)
logger.info(f"NLI id2label: {nli_id2label}")

def nli_predict_batch(premises: List[str], hypothesis: str) -> List[Dict[str, float]]:
    if not premises:
        return []
    enc = nli_tokenizer(premises, [hypothesis]*len(premises), truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = nli_model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    results = []
    for row in probs:
        row_map = {}
        for idx, score in enumerate(row):
            label_name = nli_id2label.get(idx, str(idx))
            row_map[label_name] = float(score)
        # ensure keys
        for k in ("entailment", "neutral", "contradiction"):
            row_map.setdefault(k, 0.0)
        # normalize key names to consistent lowercase
        normalized = {k.lower(): v for k, v in row_map.items()}
        results.append(normalized)
    return results

# -------------------------
# Summarizer (BART-large-CNN)
# -------------------------
summ_tokenizer = None
summ_model = None
try:
    logger.info(f"Loading summarizer: {SUMMARIZER_MODEL}")
    summ_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    summ_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL).to(device)
    summ_model.eval()
except Exception as e:
    logger.warning("Could not load summarizer (%s): %s", SUMMARIZER_MODEL, e)
    summ_tokenizer = None
    summ_model = None

def abstractive_summary(text: str, max_sentences: int = 2) -> str:
    if not text:
        return ""
    # If summarizer isn't available, fallback to extractive first N sentences
    if summ_model is None or summ_tokenizer is None:
        sents = sent_tokenize(text)
        return " ".join(sents[:max_sentences]).strip()
    try:
        inputs = summ_tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
        summary_ids = summ_model.generate(**inputs, max_length=120, num_beams=4, early_stopping=True)
        out = summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # limit to max_sentences
        sents = sent_tokenize(out)
        return " ".join(sents[:max_sentences]).strip()
    except Exception as e:
        logger.warning("Summarization failed: %s", e)
        sents = sent_tokenize(text)
        return " ".join(sents[:max_sentences]).strip()

# -------------------------
# Utilities: keyword/entity extraction + strict filter (Option A)
# -------------------------
_SIMPLE_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "will",
    "have", "has", "not", "you", "your", "they", "their", "its", "but", "what",
    "when", "where", "how", "which", "all", "any", "new", "one", "two", "use",
    "using", "into", "more", "than", "also", "on", "in", "at", "by", "of", "to",
    "a", "an", "is", "it", "as", "be", "or"
}

# Additional common brands / game words to spot as entities (not exhaustive)
_COMMON_BRANDS = {
    "gta", "rockstar", "take-two", "take two", "ea", "electronic arts", "fifa",
    "psn", "playstation", "sony", "xbox", "microsoft", "nintendo", "ubisoft",
    "activision", "treyarch", "codemasters", "steam", "epic", "battlefield",
    "call of duty", "black ops", "gtav", "gta6", "gta-6"
}

def extract_keywords_and_entities(claim: str, max_keywords: int = 8) -> Dict[str, Any]:
    """
    Simple keyword & 'entity' extractor for Option A strict filtering.
    - keywords: tokens (lowercase) length>=3 minus stopwords; returns top unique tokens.
    - entities: TitleCase tokens, all-caps acronyms, and tokens matching the common brand list.
    """
    if not claim:
        return {"keywords": [], "entities": []}
    # normalize and tokenize
    # keep words with letters/digits and apostrophes
    tokens = re.findall(r"[A-Za-z0-9'\-]+", claim)
    tokens_clean = [t.strip(" '\"").lower() for t in tokens if t.strip(" '\"")]
    # keywords: remove stopwords and very short tokens
    keywords = [t for t in tokens_clean if len(t) >= 3 and t not in _SIMPLE_STOPWORDS]
    # deduplicate preserving order
    seen = set()
    keywords_uniq = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            keywords_uniq.append(k)
    keywords_uniq = keywords_uniq[:max_keywords]

    # Entities: TitleCase, ACRONYMS, or known-brand tokens (case-insensitive check)
    entities = set()
    # TitleCase and acronyms from original tokens
    for t in tokens:
        if re.match(r"^[A-Z0-9]{2,}$", t):  # ACRONYMS e.g. PSN, NBA
            entities.add(t.lower())
        # TitleCase heuristic: contains uppercase letter followed by lowercase (e.g. Rockstar, GTA is all-caps handled above)
        if re.match(r"^[A-Z][a-z0-9]+", t):
            entities.add(t.lower())
    # check for brand names inside claim (multiword brand tokens)
    lc = claim.lower()
    for b in _COMMON_BRANDS:
        if b in lc:
            entities.add(b.lower())

    entities_list = list(entities)
    logger.debug("Extracted keywords=%s entities=%s from claim=%s", keywords_uniq, entities_list, claim)
    return {"keywords": keywords_uniq, "entities": entities_list}

def _evidence_matches_strict(ev_text: str, claim_keywords: List[str], claim_entities: List[str]) -> bool:
    """
    Option A strict rule:
      - evidence must contain >= 2 keywords from claim_keywords (case-insensitive)
      - AND if claim_entities is non-empty, evidence must contain at least one entity
    """
    if not ev_text:
        return False
    text_l = ev_text.lower()
    # keyword matches
    kw_matches = 0
    for kw in claim_keywords:
        # match whole word or hyphen variants
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, text_l):
            kw_matches += 1
    # entity matches
    ent_matches = 0
    for ent in claim_entities:
        # entity may be multiword like "take two"
        if ent in text_l:
            ent_matches += 1

    # require at least 2 keyword matches
    if len(claim_keywords) >= 2:
        if kw_matches < 2:
            return False
    else:
        # if claim produced <2 keywords, require at least 1 keyword present
        if kw_matches < 1:
            return False

    # if any entity extracted from claim, evidence must contain at least one entity
    if claim_entities:
        if ent_matches < 1:
            return False

    return True

# -------------------------
# Retrieval
# -------------------------
def retrieve(claim: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Retrieve top_k by embedding similarity, then apply Option A strict filter.
    If strict filter yields zero results, a safe fallback relaxes to >=1 keyword.
    """
    if not claim:
        return []

    q_emb = embedder.encode([claim], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)

    if _index_type == "faiss":
        D, I = _index.search(np.expand_dims(q_emb, axis=0), top_k * 3)  # fetch extra for filtering
        scores = D[0].tolist()
        idxs = I[0].tolist()
    elif _index_type == "sklearn":
        distances, idxs = _index.kneighbors([q_emb], n_neighbors=min(top_k * 3, len(documents)))
        idxs = idxs[0].tolist()
        distances = distances[0].tolist()
        scores = [1.0 - d for d in distances]
    else:
        sims = (embeddings @ q_emb).astype(np.float32)
        # select top_k*3 candidates for filter, but cap at len(documents)
        cand_count = min(len(documents), top_k * 3)
        idxs = np.argsort(-sims)[:cand_count].tolist()
        scores = sims[idxs].tolist()

    # Build preliminary list
    prelim = []
    for idx, score in zip(idxs, scores):
        if idx < 0 or idx >= len(documents):
            continue
        doc_text = documents[idx]
        title = str(df_docs.loc[idx, "title"]) if "title" in df_docs.columns else ""
        text_field = str(df_docs.loc[idx, "text"]) if "text" in df_docs.columns else doc_text
        excerpt = (title + "\n\n" + " ".join(sent_tokenize(text_field)[:2])).strip()
        if not excerpt:
            excerpt = doc_text[:512]
        prelim.append({
            "idx": int(idx),
            "url": df_docs.loc[idx, "url"] if "url" in df_docs.columns else None,
            "text_full": doc_text,
            "text_short": (doc_text[:EVIDENCE_MAX_CHARS] + "…") if len(doc_text) > EVIDENCE_MAX_CHARS else doc_text,
            "excerpt": excerpt,
            "score": float(score)
        })

    # Extract claim keywords/entities
    parsed = extract_keywords_and_entities(claim, max_keywords=10)
    claim_keywords = parsed["keywords"]
    claim_entities = parsed["entities"]

    logger.info("Retrieved %d candidates; applying Option A strict filtering (keywords=%s entities=%s)",
                len(prelim), claim_keywords, claim_entities)

    # Strict filter pass: require >=2 keywords AND entity match (if entities exist)
    strict_filtered = []
    for ev in prelim:
        if _evidence_matches_strict(ev.get("text_full", "") + " " + (ev.get("excerpt", "") or ""), claim_keywords, claim_entities):
            strict_filtered.append(ev)

    # If strict filter produced >= top_k results, take top_k by original score
    if len(strict_filtered) >= top_k:
        strict_filtered = sorted(strict_filtered, key=lambda x: -x["score"])[:top_k]
        logger.info("Strict filter success: returning %d docs", len(strict_filtered))
        return strict_filtered

    # If strict filter produced some but fewer than top_k, keep them and attempt to supplement with relaxed candidates
    if strict_filtered:
        # supplement with prelim candidates that match >=1 keyword (but keep strict ones first)
        supplement = []
        for ev in prelim:
            if ev in strict_filtered:
                continue
            # relaxed: match >=1 keyword OR (if entities present) entity match
            text_all = ev.get("text_full", "") + " " + (ev.get("excerpt", "") or "")
            kw_count = sum(1 for kw in claim_keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_all.lower()))
            ent_ok = any(ent in text_all.lower() for ent in claim_entities) if claim_entities else False
            if kw_count >= 1 or ent_ok:
                supplement.append(ev)
            if len(strict_filtered) + len(supplement) >= top_k:
                break
        combined = strict_filtered + supplement[:max(0, top_k - len(strict_filtered))]
        logger.info("Strict filter partial: returning %d docs (strict=%d supplement=%d)",
                    len(combined), len(strict_filtered), len(supplement[:max(0, top_k - len(strict_filtered))]))
        return combined

    # If strict filtered nothing, fallback to relaxed matching (>=1 keyword), but only return top_k
    relaxed = []
    for ev in prelim:
        text_all = ev.get("text_full", "") + " " + (ev.get("excerpt", "") or "")
        kw_count = sum(1 for kw in claim_keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_all.lower()))
        ent_ok = any(ent in text_all.lower() for ent in claim_entities) if claim_entities else False
        if kw_count >= 1 or ent_ok:
            relaxed.append(ev)
        if len(relaxed) >= top_k:
            break

    if relaxed:
        logger.warning("Strict filter returned 0 results; relaxing to >=1 keyword/entity match and returning %d docs", len(relaxed))
        return relaxed[:top_k]

    # Final fallback: return top K preliminary retrievals (no filtering) but log a warning
    logger.warning("No filtered evidence found for claim; returning top-%d raw retrievals for manual inspection", top_k)
    return prelim[:top_k]

# -------------------------
# NLI aggregation & final labeling
# -------------------------
def aggregate_nli_and_label(claim: str, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not evidence_list:
        return {
            "pred": "unverified",
            "pred_idx": 1,
            "confidence": 0.0,
            "probs": {"false": 0.0, "unverified": 1.0, "true": 0.0},
            "reasoning": "No evidence available to evaluate the claim.",
            "nli_details": []
        }

    premises = [e["excerpt"] for e in evidence_list]
    nli_results = nli_predict_batch(premises, claim)

    supporting = []
    contradicting = []
    neutral = []
    nli_details = []

    for ev, nli in zip(evidence_list, nli_results):
        ent = float(nli.get("entailment", 0.0))
        neu = float(nli.get("neutral", 0.0))
        con = float(nli.get("contradiction", 0.0))

        doc_detail = {
            "idx": ev.get("idx"),
            "url": ev.get("url"),
            "score": ev.get("score"),
            "excerpt": ev.get("excerpt"),
            "nli_entailment": ent,
            "nli_neutral": neu,
            "nli_contradiction": con
        }
        nli_details.append(doc_detail)

        if ent >= ENTAILMENT_THRESHOLD:
            supporting.append(doc_detail)
        elif con >= CONTRADICTION_THRESHOLD:
            contradicting.append(doc_detail)
        else:
            neutral.append(doc_detail)

    final_label = "unverified"
    final_conf = 0.0

    if supporting:
        best_ent = max(d["nli_entailment"] for d in supporting)
        final_label = "true"
        final_conf = best_ent
    elif contradicting:
        best_con = max(d["nli_contradiction"] for d in contradicting)
        final_label = "false"
        final_conf = best_con
    else:
        max_scores = [max(d["nli_entailment"], d["nli_contradiction"]) for d in nli_details] if nli_details else [0.0]
        final_conf = float(np.mean(max_scores)) if max_scores else 0.0
        final_label = "unverified"

    reasoning_parts = []
    if final_label == "true":
        reasoning_parts.append(f"At least one evidence strongly entails the claim (top entailment={final_conf:.2f}).")
        reasoning_parts.append("Supporting evidence URLs: " + ", ".join([str(d["url"]) for d in supporting[:5] if d.get("url")]))
    elif final_label == "false":
        reasoning_parts.append(f"At least one evidence strongly contradicts the claim (top contradiction={final_conf:.2f}).")
        reasoning_parts.append("Contradicting evidence URLs: " + ", ".join([str(d["url"]) for d in contradicting[:5] if d.get("url")]))
    else:
        reasoning_parts.append("No evidence in the database conclusively entails or contradicts the claim.")
        reasoning_parts.append("Returned the most relevant documents for manual inspection.")

    reasoning = " ".join(reasoning_parts)

    probs = {
        "false": float(max([d["nli_contradiction"] for d in nli_details]) if nli_details else 0.0),
        "unverified": float(1.0 - max([d["nli_entailment"] for d in nli_details]) if nli_details else 1.0),
        "true": float(max([d["nli_entailment"] for d in nli_details]) if nli_details else 0.0)
    }

    return {
        "pred": final_label,
        "pred_idx": {"false": 0, "unverified": 1, "true": 2}.get(final_label, 1),
        "confidence": float(final_conf),
        "probs": probs,
        "reasoning": reasoning,
        "nli_details": nli_details,
        "supporting": supporting,
        "contradicting": contradicting,
        "neutral": neutral
    }

# -------------------------
# Helper: find most contradictory sentence (unchanged)
# -------------------------
def find_most_contradictory_sentence(claim: str, evidence: str) -> str:
    try:
        sents = sent_tokenize(evidence)
        if not sents:
            return ""
        claim_emb = embedder.encode([claim], normalize_embeddings=True)[0]
        sent_embs = embedder.encode(sents, normalize_embeddings=True)
        sims = np.dot(sent_embs, claim_emb)
        idx = int(np.argmin(sims))
        return sents[idx].strip()
    except Exception:
        return ""

# -------------------------
# Main API function
# -------------------------
def fact_check_post(text: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Returns:
      {
        "classifier": { pred, pred_idx, confidence, probs, reasoning },
        "evidence": [ { idx, url, text_short, excerpt, score, summary (top3), nli_entailment, nli_contradiction, contradiction_sentence }, ... ]
      }
    """
    if text is None:
        raise ValueError("text is required")

    claim = text.strip()
    evidence = retrieve(claim, top_k=top_k)

    # Only run heavy ops (summary, contradiction sentence) on top N docs
    for i, ev in enumerate(evidence):
        full_text = ev.get("text_full", "")
        if i < TOP_SUMMARIES:
            # Abstractive summary (or extractive fallback)
            ev["summary"] = abstractive_summary(full_text, max_sentences=2)
            # contradiction_sentence - best-effort extract
            ev["contradiction_sentence"] = find_most_contradictory_sentence(claim, full_text)
            # keep excerpt already present
        else:
            # Light-weight: no heavy summarization
            ev["summary"] = None
            ev["contradiction_sentence"] = None
            # shorten excerpt for quick response
            ev["text_short"] = ev.get("text_short", "")[:EVIDENCE_MAX_CHARS] + ("…" if len(ev.get("text_short","")) > EVIDENCE_MAX_CHARS else "")

    # Run NLI aggregation over the retrieved excerpts
    classifier_res = aggregate_nli_and_label(claim, evidence)

    # Merge NLI per-evidence back into evidence list
    idx_to_details = {d["idx"]: d for d in classifier_res.get("nli_details", [])}
    for ev in evidence:
        details = idx_to_details.get(ev.get("idx"))
        if details:
            ev["nli_entailment"] = details.get("nli_entailment", 0.0)
            ev["nli_neutral"] = details.get("nli_neutral", 0.0)
            ev["nli_contradiction"] = details.get("nli_contradiction", 0.0)
        else:
            ev["nli_entailment"] = 0.0
            ev["nli_neutral"] = 0.0
            ev["nli_contradiction"] = 0.0

    response = {
        "classifier": {
            "pred": classifier_res["pred"],
            "pred_idx": classifier_res["pred_idx"],
            "confidence": classifier_res["confidence"],
            "probs": classifier_res["probs"],
            "reasoning": classifier_res["reasoning"]
        },
        "evidence": evidence
    }
    return response

# Quick smoke test when running the module directly
if __name__ == "__main__":
    sample = "GTA 6 will be released tomorrow"
    out = fact_check_post(sample, top_k=3)
    print(json.dumps(out, indent=2, ensure_ascii=False))
