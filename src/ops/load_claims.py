import csv
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

INPUT_CSV = "data/processed/claims.csv"

def insert_claims():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    with open(INPUT_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            cur.execute("""
                INSERT INTO claims (post_id, sentence_idx, claim_text, created_utc)
                VALUES (%s, %s, %s, to_timestamp(%s))
                ;
            """, (
                row["post_id"],
                row["sentence_idx"],
                row["claim_text"],
                row["created_utc"]
            ))

    conn.commit()
    cur.close()
    conn.close()

    print("Finished inserting claims.")

if __name__ == "__main__":
    insert_claims()
