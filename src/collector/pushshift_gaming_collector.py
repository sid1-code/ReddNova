import requests
import pandas as pd
import time
import os
from datetime import datetime

OUTPUT_POSTS = "data/raw/reddit_posts_pushshift.csv"
OUTPUT_COMMENTS = "data/raw/reddit_comments_pushshift.csv"

SUBREDDITS = [
    "gaming",
    "pcgaming",
    "games",
    "playstation",
    "xbox",
    "nintendo",
    "Steam"
]

def pushshift_request(endpoint, params):
    url = f"https://api.pushshift.io/reddit/{endpoint}/search/"
    while True:
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json().get("data", [])
                return data
            else:
                print("Rate limited — waiting 5s")
                time.sleep(5)
        except Exception as e:
            print(f"Error: {e}. Retrying in 5 seconds...")
            time.sleep(5)


def collect_posts(subreddit, limit=5000):
    print(f"\n🔍 Collecting posts from r/{subreddit}...")

    all_posts = []
    last_timestamp = int(time.time())

    while len(all_posts) < limit:
        params = {
            "subreddit": subreddit,
            "size": 500,
            "before": last_timestamp
        }

        data = pushshift_request("submission", params)

        if not data:
            print("No more posts found.")
            break

        for post in data:
            all_posts.append({
                "id": post.get("id"),
                "subreddit": subreddit,
                "title": post.get("title", ""),
                "selftext": post.get("selftext", ""),
                "score": post.get("score", 0),
                "created_utc": post.get("created_utc"),
                "url": post.get("url", "")
            })

        last_timestamp = data[-1]["created_utc"]

        print(f"→ Collected {len(all_posts)} posts so far...")

        time.sleep(1)

    return all_posts


def collect_comments(post_ids):
    print("\n💬 Collecting comments for posts...")

    all_comments = []

    for pid in post_ids:
        params = {"link_id": f"t3_{pid}", "size": 500}

        data = pushshift_request("comment", params)

        for c in data:
            all_comments.append({
                "post_id": pid,
                "comment_id": c.get("id"),
                "body": c.get("body", ""),
                "score": c.get("score", 0),
                "created_utc": c.get("created_utc")
            })

        time.sleep(0.5)

    return all_comments


def main():
    os.makedirs("data/raw", exist_ok=True)

    all_posts = []

    # Collect from all subreddits
    for sub in SUBREDDITS:
        posts = collect_posts(sub, limit=3000)
        all_posts.extend(posts)

    print(f"\n✅ TOTAL POSTS COLLECTED: {len(all_posts)}")

    # Save posts
    posts_df = pd.DataFrame(all_posts)
    posts_df.to_csv(OUTPUT_POSTS, index=False, encoding="utf-8")
    print(f"📁 Saved posts to {OUTPUT_POSTS}")

    # Collect comments
    post_ids = posts_df["id"].tolist()
    comments = collect_comments(post_ids)

    comments_df = pd.DataFrame(comments)
    comments_df.to_csv(OUTPUT_COMMENTS, index=False, encoding="utf-8")
    print(f"📁 Saved comments to {OUTPUT_COMMENTS}")

    print("\n🎉 DONE — Ready for preprocessing + ML training!")

if __name__ == "__main__":
    main()
