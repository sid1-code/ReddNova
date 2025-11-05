import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st


# ==============================
# Configuration
# ==============================
BACKEND_PREDICT_URL = os.getenv("PREDICT_URL", "http://backend_service:8000/predict")
DEFAULT_BACKEND_CSV = os.path.join("backend", "reddit_posts.csv")


# ==============================
# Utilities: Backend interaction
# ==============================
@st.cache_data(show_spinner=False)
def backend_available(timeout: float = 1.0) -> bool:
    try:
        # Quick ping by small request; using predict with dummy short string
        r = requests.post(BACKEND_PREDICT_URL, json="ping", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def predict_api(texts: List[str], timeout: float = 8.0) -> List[Dict[str, Any]]:
    """Call the backend predict endpoint for a list of texts.

    Returns a list of dicts with keys: processed_text, label, confidence.
    On error, returns a placeholder with UNKNOWN label.
    """
    results: List[Dict[str, Any]] = []
    for t in texts:
        try:
            resp = requests.post(BACKEND_PREDICT_URL, json=t, timeout=timeout)
            if resp.ok:
                results.append(resp.json())
            else:
                results.append({
                    "processed_text": t,
                    "label": "UNKNOWN",
                    "confidence": 0.0,
                })
        except Exception:
            results.append({
                "processed_text": t,
                "label": "UNKNOWN",
                "confidence": 0.0,
            })
    return results


def to_fake_real_probs(label: str, conf: float) -> Tuple[float, float]:
    """Approximate [p_fake, p_real] from single-label confidence returned by backend.

    If backend label is FAKE with confidence c => [c, 1-c], else [1-c, c].
    """
    c = float(conf or 0.0)
    lab = (label or "").upper()
    if "FAKE" in lab and "REAL" not in lab:
        return c, 1 - c
    if "REAL" in lab and "FAKE" not in lab:
        return 1 - c, c
    # Ambiguous; default split
    return 0.5, 0.5


# ==============================
# Explanation: LIME or fallback
# ==============================
def try_import_lime():
    try:
        from lime.lime_text import LimeTextExplainer  # type: ignore
        return LimeTextExplainer
    except Exception:
        return None


def predict_proba_wrapper(texts: List[str]) -> List[List[float]]:
    """Predict proba for LIME: returns [[p_fake, p_real], ...]."""
    out: List[List[float]] = []
    results = predict_api(texts)
    for r in results:
        p_fake, p_real = to_fake_real_probs(r.get("label", ""), float(r.get("confidence", 0.0)))
        out.append([p_fake, p_real])
    return out


def explain_with_lime(text: str, num_features: int = 10) -> List[Tuple[str, float]]:
    LimeTextExplainer = try_import_lime()
    if not LimeTextExplainer:
        return []
    explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])
    exp = explainer.explain_instance(text, predict_proba_wrapper, num_features=num_features)
    # Return importance for both classes combined by taking FAKE perspective (index 0)
    return exp.as_list(label=0)


def explain_with_deletion(text: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """Fallback: token importance via deletion and observing FAKE confidence delta.

    Positive weight => increases FAKE score; negative => increases REAL score.
    """
    base = predict_api([text])[0]
    p_fake_base, _ = to_fake_real_probs(base.get("label", ""), float(base.get("confidence", 0.0)))

    tokens = text.split()
    importances: List[Tuple[str, float]] = []
    for i, tok in enumerate(tokens):
        pruned = " ".join(tokens[:i] + tokens[i + 1 :]) or ""
        alt = predict_api([pruned])[0]
        p_fake_alt, _ = to_fake_real_probs(alt.get("label", ""), float(alt.get("confidence", 0.0)))
        delta = p_fake_base - p_fake_alt  # if removing decreases FAKE prob, token supported FAKE
        importances.append((tok, float(delta)))

    # Sort by absolute importance and keep top_k unique tokens (preserving max abs weight per token)
    agg: Dict[str, float] = {}
    for tok, w in importances:
        if tok not in agg or abs(w) > abs(agg[tok]):
            agg[tok] = w
    ranked = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return ranked


def render_explanation_html(text: str, weights: List[Tuple[str, float]]) -> str:
    """Color highlight tokens by importance: red => FAKE, green => REAL."""
    if not text:
        return "<div>No text provided.</div>"

    token_to_weight = {t: w for t, w in weights}
    parts = []
    for tok in text.split():
        w = token_to_weight.get(tok, 0.0)
        if w == 0:
            parts.append(tok)
            continue
        # Map weight to alpha 0.1..0.9
        alpha = min(0.9, max(0.1, abs(w)))
        if w > 0:
            color = f"rgba(255,0,0,{alpha})"  # red for FAKE
        else:
            color = f"rgba(0,150,0,{alpha})"  # green for REAL
        parts.append(f"<span style='background-color:{color}; padding:2px; border-radius:3px'>{tok}</span>")
    return "<div style='line-height:1.9'>{}</div>".format(" ".join(parts))


# ==============================
# Data loading
# ==============================
@st.cache_data(show_spinner=False)
def load_default_csv(path: str = DEFAULT_BACKEND_CSV) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df
        except Exception:
            return None
    return None


# ==============================
# UI
# ==============================
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("Fake News Detection MVP – Streamlit Frontend")
st.caption("Team: Person A (Backend) & Person B (Frontend)")

with st.expander("Instructions", expanded=True):
    st.markdown(
        "- Enter text to analyze or load a CSV of posts.\n"
        "- Use the sidebar to pick data source (CSV or Live fetch), set confidence threshold, analyze all rows, and clear cache.\n"
        "- Explanations highlight tokens: red increases FAKE score, green increases REAL score.\n"
        "- If the backend is offline, you can still upload a CSV and view data; predictions/explanations will be disabled."
    )


# Sidebar controls
st.sidebar.header("Controls")
data_source = st.sidebar.selectbox("Data source", ["CSV", "Live fetch via API"])  # live fetch placeholder
threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    analyze_all_clicked = st.button("Analyze All")
with col_sb2:
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")


# Load data section
df_loaded: Optional[pd.DataFrame] = None
backend_ok = backend_available()

if data_source == "CSV":
    default_df = load_default_csv()
    st.subheader("Data Source: CSV")
    if default_df is not None:
        st.success(f"Loaded default CSV from {DEFAULT_BACKEND_CSV}")
        df_loaded = default_df
    uploaded = st.file_uploader("Upload CSV with columns: id, title, selftext, permalink", type=["csv"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            df_loaded = pd.read_csv(uploaded)
            st.success("Uploaded CSV loaded.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
else:
    st.subheader("Data Source: Live fetch via API")
    if not backend_ok:
        st.warning("Backend not reachable; live fetch disabled. Switch to CSV or upload a file.")
    else:
        st.info("Live fetch endpoint not implemented in backend. Please use CSV for now.")


# Manual input analysis
st.subheader("Manual Text Analysis")
manual_text = st.text_area("Enter text to analyze", height=140, placeholder="Type or paste news text here...")
if st.button("Analyze Text"):
    if not manual_text.strip():
        st.warning("Please enter some text.")
    elif not backend_ok:
        st.error("Backend is not reachable. Start FastAPI or use CSV upload.")
    else:
        with st.spinner("Analyzing..."):
            res = predict_api([manual_text])[0]
        st.json(res)
        # Explanation
        weights = explain_with_lime(res.get("processed_text", manual_text)) or explain_with_deletion(res.get("processed_text", manual_text))
        html = render_explanation_html(res.get("processed_text", manual_text), weights)
        st.markdown("Token-level explanation:", help="Red increases FAKE; Green increases REAL")
        st.markdown(html, unsafe_allow_html=True)


# Batch analysis for CSV-loaded data
results_df: Optional[pd.DataFrame] = None
if df_loaded is not None:
    # Ensure expected columns exist
    required_cols = {"title", "selftext", "permalink"}
    missing = required_cols - set(c.lower() for c in df_loaded.columns)
    # Try case-insensitive mapping
    col_map = {c.lower(): c for c in df_loaded.columns}
    if missing:
        st.warning("CSV missing some expected columns. Attempting best-effort mapping.")

    title_col = col_map.get("title") or next((c for c in df_loaded.columns if "title" in c.lower()), None)
    text_col = col_map.get("selftext") or next((c for c in df_loaded.columns if "text" in c.lower()), None)
    link_col = col_map.get("permalink") or next((c for c in df_loaded.columns if "link" in c.lower() or "url" in c.lower()), None)

    if not text_col:
        st.error("Could not find a text/selftext column in the CSV.")
    else:
        view_df = pd.DataFrame({
            "Title": df_loaded[title_col] if title_col else "",
            "Excerpt": df_loaded[text_col].fillna("").astype(str).str.slice(0, 250),
            "FullText": df_loaded[text_col].fillna("").astype(str),
            "Link": df_loaded[link_col] if link_col else "",
        })

        # Run predictions when Analyze All clicked
        if analyze_all_clicked:
            if not backend_ok:
                st.error("Backend is not reachable. Start FastAPI to analyze.")
            else:
                with st.spinner("Analyzing all rows..."):
                    preds = predict_api(view_df["FullText"].tolist())
                labels = [p.get("label", "UNKNOWN") for p in preds]
                confs = [float(p.get("confidence", 0.0)) for p in preds]
                results_df = view_df.copy()
                results_df["Label"] = labels
                results_df["Confidence"] = confs

        # If we already computed results this session, keep them visible
        if results_df is None and "results_df" in st.session_state:
            results_df = st.session_state["results_df"]
        elif results_df is not None:
            st.session_state["results_df"] = results_df

        # Display results (filtered by confidence threshold)
        if results_df is not None:
            filtered = results_df[results_df["Confidence"].fillna(0.0) >= threshold]
            st.subheader("Results")
            st.dataframe(filtered[["Title", "Excerpt", "Label", "Confidence", "Link"]], use_container_width=True)

            # Bar chart FAKE vs REAL
            if not filtered.empty and "Label" in filtered.columns:
                counts = filtered["Label"].fillna("UNKNOWN").value_counts().rename_axis("Label").reset_index(name="Count")
                st.bar_chart(data=counts, x="Label", y="Count")

            # Row selection for explanation
            st.markdown("\n")
            idx = st.number_input("Select row index for explanation", min_value=0, max_value=max(len(results_df) - 1, 0), value=0, step=1)
            if len(results_df) > 0:
                row = results_df.iloc[int(idx)]
                st.markdown(f"**Selected Title:** {str(row['Title'])}")
                if backend_ok:
                    with st.spinner("Generating explanation..."):
                        text_for_exp = str(row["FullText"]) or ""
                        res = predict_api([text_for_exp])[0]
                        weights = explain_with_lime(res.get("processed_text", text_for_exp)) or explain_with_deletion(res.get("processed_text", text_for_exp))
                        html = render_explanation_html(res.get("processed_text", text_for_exp), weights)
                        st.markdown(html, unsafe_allow_html=True)
                else:
                    st.info("Backend offline: explanations are unavailable.")

            # Download button
            csv_bytes = filtered[["Title", "Excerpt", "Label", "Confidence", "Link"]].to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", csv_bytes, file_name="results.csv", mime="text/csv")
        else:
            st.info("Load or upload a CSV, then click 'Analyze All' in the sidebar.")


# Footer / attribution
st.markdown("---")
st.caption("MVP Frontend by Person B. Backend by Person A.")


