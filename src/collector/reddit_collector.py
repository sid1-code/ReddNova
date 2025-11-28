import requests
import time
import pandas as pd
import traceback
from datetime import datetime
import praw
import os

#############################
# CONFIG
#############################

SUBREDDITS = [
    "gaming",
    "pcgaming",
    "truegaming",
    "Games",
    "GameDeals",
    "Steam"
]

LIMIT_PER_SUB = 1500        # You can increase later
PUSHSHIFT_URL = "https://api.pushshift.io/reddit/search/submission/"

OUTPUT_POSTS = "data/raw/reddit_posts.csv"
OUTPUT_COMMENTS = "data/raw/reddit_comments.csv"

#############################
# REDDIT API FALLBACK (Optional)
#############################
# Create a Reddit App at https://www.reddit.com/prefs/apps

USE_REDDIT_API = False   # Set to True if you want fallback

if USE_REDDIT_API:
    reddit = praw.Reddit(
        client_id="YOUR_ID",
        client_secret="YOUR_SECRET",
        user_agent="GamingFactCheckerBot/0.1"
    )

#############################
# HELPERS
#############################

def clean(text):
    if text is None:
        return ""
    return (
        text.replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )

def fetch_pushshift(subreddit, limit):
    """Fetch posts from Pushshift"""
    print(f"\n🔵 Fetching posts from r/{subreddit} using Pushshift...")

    posts = []
    params = {
        "subreddit": subreddit,
        "size": 100,
        "sort": "desc",
        "sort_type": "created_utc"
    }

    fetched = 0
    last_created = None

    while fetched < limit:
        if last_created:
            params["before"] = last_created

        try:
            r = requests.get(PUSHSHIFT_URL, params=params, timeout=15)
            data = r.json().get("data", [])

            if not data:
                break

            for item in data:
                posts.append({
                    "id": item.get("id"),
                    "title": clean(item.get("title")),
                    "body": clean(item.get("selftext")),
                    "subreddit": item.get("subreddit"),
                    "created_utc": item.get("created_utc"),
                    "url": item.get("url"),
                    "author": item.get("author"),
                    "score": item.get("score"),
                    "num_comments": item.get("num_comments")
                })
                fetched += 1

            last_created = data[-1]["created_utc"]
            time.sleep(1)

        except Exception as e:
            print("⚠️ Error:", e)
            traceback.print_exc()
            time.sleep(5)

    print(f"✔ Done: {fetched} posts fetched")
    return posts


def fetch_comments(post_ids):
    """Fetch comments using Pushshift"""
    print("\n🟣 Fetching comments for posts...")

    comments_url = "https://api.pushshift.io/reddit/comment/search/"
    all_comments = []

    for post_id in post_ids:
        params = {
            "link_id": post_id,
            "size": 200
        }

        try:
            r = requests.get(comments_url, params=params, timeout=15)
            data = r.json().get("data", [])

            for c in data:
                all_comments.append({
                    "comment_id": c.get("id"),
                    "post_id": c.get("link_id"),
                    "body": clean(c.get("body")),
                    "author": c.get("author"),
                    "score": c.get("score"),
                    "created_utc": c.get("created_utc")
                })
        except:
            pass

        time.sleep(0.3)

    print(f"✔ Done: {len(all_comments)} comments fetched")
    return all_comments


#############################
# MAIN PIPELINE
#############################

def main():
    print("\n🔥 STARTING REDDIT GAMING COLLECTOR\n")

    all_posts = []
    all_comments = []

    # Ensure directories exist
    os.makedirs("data/raw", exist_ok=True)

    for sub in SUBREDDITS:
        posts = fetch_pushshift(sub, LIMIT_PER_SUB)
        all_posts.extend(posts)

        post_ids = [p["id"] for p in posts]
        comments = fetch_comments(post_ids)
        all_comments.extend(comments)

    # Save posts
    print("\n💾 Saving posts CSV...")
    pd.DataFrame(all_posts).to_csv(OUTPUT_POSTS, index=False)
    print(f"✔ Saved: {OUTPUT_POSTS}")

    # Save comments
    print("\n💾 Saving comments CSV...")
    pd.DataFrame(all_comments).to_csv(OUTPUT_COMMENTS, index=False)
    print(f"✔ Saved: {OUTPUT_COMMENTS}")

    print("\n✨ COMPLETED ALL REDDIT COLLECTION")
    print(f"Total Posts: {len(all_posts)}")
    print(f"Total Comments: {len(all_comments)}")


if __name__ == "__main__":
    main()
