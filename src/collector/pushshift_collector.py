#!/usr/bin/env python3
import os
import csv
import time
import requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read base URL (from .env)
PUSHSHIFT_BASE = os.getenv("PUSHSHIFT_BASE", "https://api.pushshift.io/reddit/search/submission")

# Output CSV file
OUT_CSV = "data/raw/pushshift_posts.csv"


def fetch_subreddit(subreddit, size=100, before=None):
    params = {"subreddit": subreddit, "size": size, "sort": "desc", "sort_type": "created_utc"}
    if before:
        params["before"] = before

    print(f"Fetching posts before={before}...")
    r = requests.get(PUSHSHIFT_BASE, params=params, timeout=20)
    r.raise_for_status()

    data = r.json().get("data", [])
    return data


def save_posts_csv(posts, out_csv=OUT_CSV):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    header = [
        "id",
        "subreddit",
        "title",
        "selftext",
        "created_utc",
        "url",
        "author",
        "score",
        "num_comments",
        "raw_json",
    ]

    write_header = not os.path.exists(out_csv)

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for p in posts:
            writer.writerow([
                p.get("id"),
                p.get("subreddit"),
                p.get("title", ""),
                p.get("selftext", ""),
                p.get("created_utc"),
                p.get("url", ""),
                p.get("author", ""),
                p.get("score", 0),
                p.get("num_comments", 0),
                p
            ])


if __name__ == "__main__":
    subreddit = "gaming"   # You can change this to any subreddit
    before = None

    for i in range(20):  # 20 pages × 100 posts = ~2000 posts
        posts = fetch_subreddit(subreddit, size=100, before=before)

        if not posts:
            print("No more posts found. Stopping.")
            break

        save_posts_csv(posts)

        before = posts[-1].get("created_utc")

        print(f"Saved {len(posts)} posts, next before={before}")
        time.sleep(1.0)  # Prevent rate limits
