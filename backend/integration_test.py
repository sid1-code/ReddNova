# integration_test.py
import os
import csv
import json
import time
from typing import List, Dict
import requests

BACKEND_CSV = os.path.join(os.path.dirname(__file__), "reddit_posts.csv")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "predictions_output.csv")
BATCH_DELAY = 0.15  # seconds between requests to avoid rate issues

def load_posts(csv_path: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run reddit_fetch.py first.")
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows

def call_predict(text: str) -> Dict:
    # API expects raw string in body; some servers want JSON string
    headers = {"Content-Type": "application/json"}
    payload = json.dumps(text)
    resp = requests.post(API_URL, data=payload, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()

def run_batch(posts: List[Dict]):
    rows_out = []
    for i, p in enumerate(posts):
        try:
            excerpt = (p.get("selftext") or p.get("title") or "")[:200]
            result = call_predict(excerpt)
            row = {
                "id": p.get("id"),
                "title": p.get("title"),
                "permalink": p.get("permalink"),
                "label": result.get("label"),
                "confidence": result.get("confidence"),
                "processed_text": result.get("processed_text"),
            }
            print(f"[{i+1}/{len(posts)}] {row['label']} ({row['confidence']:.3f}) - {row['title'][:60]}")
            rows_out.append(row)
        except Exception as exc:
            print(f"[{i+1}/{len(posts)}] ERROR: {exc}")
        time.sleep(BATCH_DELAY)
    return rows_out

def save_results(rows: List[Dict], out_path: str):
    if not rows:
        print("No rows to save.")
        return
    keys = rows[0].keys()
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved predictions to", out_path)

if __name__ == "__main__":
    posts = load_posts(BACKEND_CSV)
    preds = run_batch(posts)
    save_results(preds, OUTPUT_CSV)
    # Print basic metrics
    total = len(preds)
    fake = sum(1 for r in preds if r.get("label") == "FAKE")
    real = sum(1 for r in preds if r.get("label") == "REAL")
    print(f"Total: {total}, FAKE: {fake}, REAL: {real}")
