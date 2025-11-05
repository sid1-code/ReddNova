import os
import logging
from typing import List, Dict

import pandas as pd
import praw


# Configure basic logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reddit_fetch")


def get_reddit_client() -> praw.Reddit:
    """Create an authenticated PRAW Reddit client using environment variables.

    Required environment variables:
    - REDDIT_CLIENT_ID
    - REDDIT_CLIENT_SECRET
    Optional:
    - REDDIT_USER_AGENT (defaults to a simple identifier)
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "fake-news-mvp/0.1 (by u/your_username)")

    if not client_id or not client_secret:
        raise EnvironmentError(
            "Missing Reddit API credentials. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
        )

    logger.debug("Creating Reddit client with provided credentials")
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def fetch_text_posts(subreddit_name: str, limit: int = 50) -> List[Dict[str, str]]:
    """Fetch text-only posts from the given subreddit using PRAW.

    - Filters posts to those containing non-empty `selftext`.
    - Returns a list of dictionaries ready to be saved as a DataFrame.
    """
    reddit = get_reddit_client()
    subreddit = reddit.subreddit(subreddit_name)

    logger.info("Fetching up to %d posts from r/%s", limit, subreddit_name)
    posts: List[Dict[str, str]] = []

    try:
        for submission in subreddit.hot(limit=limit):
            # Only include posts that have text content (self posts)
            if getattr(submission, "selftext", None):
                text = (submission.selftext or "").strip()
                if text:
                    posts.append(
                        {
                            "id": submission.id,
                            "title": submission.title,
                            "selftext": text,
                            "created_utc": submission.created_utc,
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                            "permalink": f"https://www.reddit.com{submission.permalink}",
                        }
                    )
    except Exception as exc:
        logger.exception("Error while fetching posts: %s", exc)
        raise

    logger.info("Fetched %d text posts", len(posts))
    return posts


def save_posts_to_csv(posts: List[Dict[str, str]], output_path: str) -> None:
    """Save a list of post dictionaries to a CSV file using pandas."""
    if not posts:
        logger.warning("No posts to save. The CSV will not be created.")
        return

    df = pd.DataFrame(posts)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d posts to %s", len(df), output_path)


def main() -> None:
    """Entrypoint for fetching text posts from r/news and writing them to CSV."""
    try:
        posts = fetch_text_posts("news", limit=50)
        # Save CSV in the backend folder to keep assets together with the code
        output_csv = os.path.join(os.path.dirname(__file__), "reddit_posts.csv")
        save_posts_to_csv(posts, output_csv)
    except Exception as exc:
        logger.error("reddit_fetch failed: %s", exc)
        raise


if __name__ == "__main__":
    main()


