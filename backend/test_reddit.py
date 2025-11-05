import os
import praw

reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent=os.environ["REDDIT_USER_AGENT"],
)

sub = reddit.subreddit("news")
for i, post in enumerate(sub.hot(limit=5), 1):
    print(f"{i}. {post.title} ({post.id})")
