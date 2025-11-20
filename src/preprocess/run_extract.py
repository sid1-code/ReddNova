import csv, os
from claim_extractor import extract_claims

IN = "data/raw/pushshift_posts.csv"
OUT = "data/processed/claims.csv"

print("Starting extraction...")
print("Checking if input file exists:", os.path.exists(IN))

os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(IN, newline='', encoding='utf-8') as fin, open(OUT, 'w', newline='', encoding='utf-8') as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(["post_id", "sentence_idx", "claim_text", "created_utc"])

    count = 0
    for r in reader:
        title = r.get("title", "")
        body = r.get("selftext", "")
        post_id = r.get("id")
        created = r.get("created_utc")

        claims = extract_claims(title, body, max_sentences=2)
        for idx, text in claims:
            writer.writerow([post_id, idx, text, created])
            count += 1

print("Extraction complete.")
print("Total claims written:", count)
print("Output file:", OUT)
