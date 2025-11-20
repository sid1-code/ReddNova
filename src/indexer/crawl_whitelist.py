#!/usr/bin/env python3
"""
Crawl whitelist for evidence collection (gaming domain).

Creates/updates CSV: data/evidence/evidence_raw.csv
Optionally upserts into Postgres if DATABASE_URL is present in .env

Requirements:
- pip install requests pyyaml feedparser beautifulsoup4 psycopg2-binary python-dotenv readability-lxml
  (readability-lxml is optional; falls back to simple parsing)

Usage:
  python src/indexer/crawl_whitelist.py --sources src/indexer/evidence_sources.yaml --out data/evidence/evidence_raw.csv --insert-db

This script:
- Reads YAML of sources (rss and/or url)
- For RSS feeds: iterates entries and fetches article pages
- For plain URLs: fetches homepage and tries to find article links (basic)
- Extracts main text (readability or <p> join fallback)
- Writes CSV with columns: url, domain, title, text, published_at, fetched_at, metadata
- (Optional) Upserts into Postgres evidence_docs table

Be polite: respects robots.txt (via requests + delay). Add your REDDIT_USER_AGENT or custom UA in .env as CRAWLER_USER_AGENT.
"""

import os
import time
import csv
import yaml
import json
import argparse
import logging
import socket
from urllib.parse import urlparse, urljoin
from datetime import datetime

import requests
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Optional: readability for better main content extraction
try:
    from readability import Document
    _HAS_READABILITY = True
except Exception:
    _HAS_READABILITY = False

# Optional DB
try:
    import psycopg2
    from psycopg2.extras import Json
    _HAS_PG = True
except Exception:
    _HAS_PG = False

# Basic configuration
load_dotenv()
DEFAULT_USER_AGENT = os.getenv("CRAWLER_USER_AGENT") or "live-fake-news-crawler/0.1 (+https://example.com)"
REQUESTS_TIMEOUT = 15
RATE_LIMIT_SECONDS = float(os.getenv("CRAWLER_RATE_LIMIT", "1.0"))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_sources(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # flatten into list of dicts with name,url,rss,domain
    sources = []
    for section in ("gaming_sources", "fact_check_sources", "general_news_sources"):
        for item in data.get(section, []) if data else []:
            url = item.get("url")
            rss = item.get("rss") or ""
            domain = urlparse(url).netloc if url else item.get("domain")
            sources.append({"name": item.get("name"), "url": url, "rss": rss, "domain": domain, "section": section})
    return sources


def fetch_url(url: str, headers: dict = None):
    headers = headers or {"User-Agent": DEFAULT_USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        logger.warning("Request failed for %s: %s", url, e)
        return None


def extract_text_from_html(html: str, url: str = None) -> str:
    if not html:
        return ""
    if _HAS_READABILITY:
        try:
            doc = Document(html)
            cleaned = doc.summary()
            soup = BeautifulSoup(cleaned, "html.parser")
            text = "\n\n".join([p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)])
            return text
        except Exception:
            pass
    # fallback simple extraction
    soup = BeautifulSoup(html, "html.parser")
    # remove scripts and styles
    for s in soup(["script", "style", "header", "footer", "nav", "aside"]):
        s.decompose()
    # try to get title first
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    text = "\n\n".join(paragraphs)
    # heuristic: if text is too short and there is article tag
    if len(text) < 200:
        article = soup.find("article")
        if article:
            paragraphs = [p.get_text(strip=True) for p in article.find_all("p") if p.get_text(strip=True)]
            text = "\n\n".join(paragraphs)
    return text or title


def parse_rss_and_yield_entries(rss_url: str):
    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            # Some entries have 'link', 'published' or 'updated'
            yield {
                "link": entry.get("link"),
                "title": entry.get("title"),
                "published": entry.get("published") or entry.get("updated") or None,
                "summary": entry.get("summary") if entry.get("summary") else None,
            }
    except Exception as e:
        logger.warning("Failed to parse RSS %s: %s", rss_url, e)


def absolute_links_from_page(base_url: str, html: str):
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a", href=True)
    links = set()
    for a in anchors:
        href = a.get("href")
        if href.startswith("#"):
            continue
        # make absolute
        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)
        if parsed.scheme in ("http", "https"):
            links.add(abs_url.split("#")[0])
    return links


def upsert_evidence_row(conn, row: dict):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO evidence_docs (url, domain, title, text, published_at, metadata)
        VALUES (%s,%s,%s,%s,to_timestamp(%s),%s)
        ON CONFLICT (url) DO UPDATE SET text = EXCLUDED.text, title = EXCLUDED.title, published_at = EXCLUDED.published_at;
        """,
        (
            row.get("url"),
            row.get("domain"),
            row.get("title"),
            row.get("text"),
            row.get("published_at"),
            Json(row.get("metadata") or {}),
        ),
    )
    conn.commit()


def save_rows_to_csv(out_path: str, rows: list):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["url", "domain", "title", "text", "published_at", "fetched_at", "metadata"])
        for r in rows:
            writer.writerow([
                r.get("url"),
                r.get("domain"),
                r.get("title"),
                r.get("text"),
                r.get("published_at"),
                r.get("fetched_at"),
                json.dumps(r.get("metadata") or {}),
            ])


def crawl_sources(sources, out_csv: str, insert_db: bool = False, max_per_source: int = 100):
    rows_written = 0
    pg_conn = None
    if insert_db:
        if not _HAS_PG:
            raise RuntimeError("psycopg2 not installed but insert_db=True")
        pg_url = os.getenv("DATABASE_URL")
        pg_conn = psycopg2.connect(pg_url)

    for src in sources:
        name = src.get("name")
        domain = src.get("domain")
        logger.info("Processing source: %s (%s)", name, domain)

        source_rows = []
        # First try RSS
        if src.get("rss"):
            for entry in parse_rss_and_yield_entries(src.get("rss")):
                if not entry.get("link"):
                    continue
                link = entry.get("link")
                logger.info("Fetching article: %s", link)
                resp = fetch_url(link)
                time.sleep(RATE_LIMIT_SECONDS)
                if not resp:
                    continue
                text = extract_text_from_html(resp.text, link)
                fetched_at = datetime.utcnow().isoformat()
                row = {
                    "url": link,
                    "domain": domain,
                    "title": entry.get("title") or "",
                    "text": text,
                    "published_at": entry.get("published"),
                    "fetched_at": fetched_at,
                    "metadata": {"source_name": name, "via": "rss"},
                }
                source_rows.append(row)
                if insert_db:
                    upsert_evidence_row(pg_conn, row)
                rows_written += 1
                if len(source_rows) >= max_per_source:
                    break
        else:
            # No RSS: fetch homepage and find links
            home = src.get("url")
            resp = fetch_url(home)
            time.sleep(RATE_LIMIT_SECONDS)
            if resp:
                links = absolute_links_from_page(home, resp.text)
                count = 0
                for link in links:
                    if domain not in link:
                        continue
                    logger.info("Fetching linked article: %s", link)
                    article_resp = fetch_url(link)
                    time.sleep(RATE_LIMIT_SECONDS)
                    if not article_resp:
                        continue
                    text = extract_text_from_html(article_resp.text, link)
                    fetched_at = datetime.utcnow().isoformat()
                    row = {
                        "url": link,
                        "domain": domain,
                        "title": None,
                        "text": text,
                        "published_at": None,
                        "fetched_at": fetched_at,
                        "metadata": {"source_name": name, "via": "homepage"},
                    }
                    source_rows.append(row)
                    if insert_db:
                        upsert_evidence_row(pg_conn, row)
                    count += 1
                    rows_written += 1
                    if count >= max_per_source:
                        break

        # save per-source batch to CSV
        if source_rows:
            save_rows_to_csv(out_csv, source_rows)

    if pg_conn:
        pg_conn.close()
    logger.info("Crawl complete. Total rows written: %d", rows_written)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="src/indexer/evidence_sources.yaml")
    parser.add_argument("--out", default="data/evidence/evidence_raw.csv")
    parser.add_argument("--insert-db", action="store_true")
    parser.add_argument("--max-per-source", type=int, default=50)
    args = parser.parse_args()

    sources = load_sources(args.sources)
    logger.info("Loaded %d sources", len(sources))

    crawl_sources(sources, args.out, insert_db=args.insert_db, max_per_source=args.max_per_source)


if __name__ == "__main__":
    main()
