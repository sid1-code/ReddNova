import csv
import psycopg2
import os
import json
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

INPUT_CSV = "data/raw/pushshift_posts.csv"

def insert_posts():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    with open(INPUT_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            cur.execute("""
                INSERT INTO reddit_posts (
                    post_id, subreddit, title, body, created_utc, url, author, score, num_comments, raw_json
                )
                VALUES (%s,%s,%s,%s, to_timestamp(%s), %s,%s,%s,%s,%s)
                ON CONFLICT (post_id) DO NOTHING;
            """, (
                row["id"],
                row["subreddit"],
                row["title"],
                row["selftext"],
                row["created_utc"],
                row["url"],
                row["author"],
                row["score"],
                row["num_comments"],
                json.dumps(row)  # <-- FIXED: convert dict to JSON string
            ))

    conn.commit()
    cur.close()
    conn.close()
    print("Finished inserting posts.")

if __name__ == "__main__":
    insert_posts()
