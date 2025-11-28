#!/usr/bin/env python3
"""
Upgraded Mass Reddit Collector (NO API KEYS)
-------------------------------------------
Features:
 - Reddit JSON + RSS hybrid collection
 - Adaptive rate limiting
 - Auto slowdown & cooldown
 - Per-subreddit health tracking
 - Resume from progress files
 - Chunked CSV writing
 - Automatic dedup merge
"""

import requests
import time
import json
import os
import csv
from typing import Optional, List

# ======================================================
# CONFIG
# ======================================================
GAMING_SUBREDDITS = [
    "gaming","pcgaming","games","PS5","PlayStation","xbox","xboxone",
    "NintendoSwitch","nintendo","Steam","SteamDeck","GameDeals","buildapc","pcmasterrace",
    "leagueoflegends","Valorant","Minecraft","FortniteCompetitive","GrandTheftAutoV","GTA6",
    "PUBATTLEGROUNDS","pokemon","halo","dota2","CounterStrike","R6ProLeague","eldenring",
    "starfield","cyberpunkgame","witcher","skyrim","fallout","BaldursGate3","callofduty",
    "mw3","ApexLegends","Overwatch","Overwatch2","rocketleague","HuntShowdown","arma",
    "dayz","spacestation13","HonkaiStarRail","Genshin_Impact","wow","classicwow","FFXIV",
    "FFXV","KingdomHearts","zelda","smashbros","StreetFighter","MortalKombat","Tekken",
    "assassinscreed","battlefield","rainbow6","EscapeFromTarkov","vrgaming","oculus",
    "virtualreality","metaquest","gamedev","unity3d","unrealengine","indiegames"
]

OUTPUT_CSV = "data/raw/gaming_posts_full.csv"
PROGRESS_DIR = "data/raw/progress"
TEMP_DIR = "data/raw/chunks"

PER_SUB_LIMIT = 25000
REQUEST_LIMIT = 100

USER_AGENT = "live-fake-news-collector/1.0 (by u/temp_user_llm)"

# Adaptive delays
BASE_SLEEP = 2
MAX_SLEEP = 60
SUBREDDIT_DELAY = {s: BASE_SLEEP for s in GAMING_SUBREDDITS}


# ======================================================
# FILE HELPERS
# ======================================================

def mkdir(path):
    os.makedirs(path, exist_ok=True)

def progress_file(sub):
    mkdir(PROGRESS_DIR)
    return f"{PROGRESS_DIR}/progress_{sub}.json"

def load_progress(sub):
    f = progress_file(sub)
    if os.path.exists(f):
        try:
            return json.load(open(f, "r", encoding="utf8"))
        except:
            return {}
    return {}

def save_progress(sub, data):
    json.dump(data, open(progress_file(sub), "w", encoding="utf8"))


def save_chunk(rows: List[dict], idx: int):
    mkdir(TEMP_DIR)
    path = f"{TEMP_DIR}/chunk_{idx:04d}.csv"
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["id","subreddit","title","selftext","created_utc","author",
                        "score","num_comments","url"])

        for r in rows:
            w.writerow([
                r.get("id"),
                r.get("subreddit"),
                r.get("title"),
                r.get("selftext"),
                r.get("created_utc"),
                r.get("author"),
                r.get("score"),
                r.get("num_comments"),
                r.get("url")
            ])

    print(f"💾 Saved chunk {path} ({len(rows)} rows)")


def merge_chunks(final_path=OUTPUT_CSV):
    mkdir(os.path.dirname(final_path) or ".")
    seen = set()
    total = 0

    with open(final_path + ".tmp", "w", newline="", encoding="utf8") as out:
        w = csv.writer(out)
        w.writerow(["id","subreddit","title","selftext","created_utc","author",
                    "score","num_comments","url"])

        chunks = sorted(os.listdir(TEMP_DIR))
        for c in chunks:
            if not c.endswith(".csv"):
                continue

            with open(f"{TEMP_DIR}/{c}", "r", encoding="utf8") as f:
                r = csv.DictReader(f)
                for row in r:
                    pid = row["id"]
                    if pid and pid in seen:
                        continue
                    seen.add(pid)
                    w.writerow([row["id"], row["subreddit"], row["title"], row["selftext"],
                                row["created_utc"], row["author"], row["score"],
                                row["num_comments"], row["url"]])
                    total += 1

    os.replace(final_path + ".tmp", final_path)
    print(f"🎉 MERGED {total} unique posts → {final_path}")


# ======================================================
# FETCHERS
# ======================================================

def fetch_json(sub, after=None):
    url = f"https://www.reddit.com/r/{sub}/new.json"
    params = {"limit": REQUEST_LIMIT}
    if after:
        params["after"] = after

    try:
        r = requests.get(url, params=params,
                         headers={"User-Agent": USER_AGENT}, timeout=15)
    except:
        return [], None, "ERR"

    if r.status_code != 200:
        return [], None, r.status_code

    j = r.json()
    children = j.get("data", {}).get("children", [])
    next_after = j.get("data", {}).get("after")

    rows = []
    for c in children:
        d = c.get("data", {})
        rows.append({
            "id": d.get("id"),
            "subreddit": sub,
            "title": d.get("title"),
            "selftext": d.get("selftext"),
            "created_utc": d.get("created_utc"),
            "author": d.get("author"),
            "score": d.get("score"),
            "num_comments": d.get("num_comments"),
            "url": d.get("url"),
        })

    return rows, next_after, 200


def fetch_rss(sub):
    """ Lightweight RSS fallback """
    url = f"https://www.reddit.com/r/{sub}/.rss"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code != 200:
            return []
        txt = r.text

        entries = txt.split("<entry>")[1:]
        rows = []
        for e in entries[:REQUEST_LIMIT]:
            try:
                title = e.split("<title>")[1].split("</title>")[0]
                link = e.split('href="')[1].split('"')[0]
                updated = e.split("<updated>")[1].split("</updated>")[0]
            except:
                continue

            rows.append({
                "id": None,
                "subreddit": sub,
                "title": title,
                "selftext": None,
                "created_utc": updated,
                "author": None,
                "score": None,
                "num_comments": None,
                "url": link
            })

        return rows

    except:
        return []


# ======================================================
# ADAPTIVE RATE LIMITER
# ======================================================

def adjust_delay(sub, success=True, rate_limited=False):
    global SUBREDDIT_DELAY

    if rate_limited:
        SUBREDDIT_DELAY[sub] = min(SUBREDDIT_DELAY[sub] + 10, MAX_SLEEP)
    elif success:
        SUBREDDIT_DELAY[sub] = max(SUBREDDIT_DELAY[sub] - 1, BASE_SLEEP)

    return SUBREDDIT_DELAY[sub]


# ======================================================
# MAIN COLLECTION LOGIC
# ======================================================

def collect_subreddit(sub, target):
    print(f"\n🚀 START r/{sub} target={target}")

    prog = load_progress(sub)
    after = prog.get("after")
    collected = prog.get("collected", 0)
    chunk_idx = prog.get("chunk_idx", 0)

    buffer = []

    while collected < target:
        sleep_time = SUBREDDIT_DELAY[sub]
        time.sleep(sleep_time)

        rows, next_after, status = fetch_json(sub, after)

        # 429 RATE LIMIT
        if status == 429:
            print(f"⛔ rate limited r/{sub} → sleeping & slowing down")
            adjust_delay(sub, rate_limited=True)
            time.sleep(30)
            continue

        # 403 FORBIDDEN → Switch to RSS permanently
        if status == 403:
            print(f"❌ 403 Forbidden → using RSS only for r/{sub}")
            rss = fetch_rss(sub)
            for r in rss:
                buffer.append(r)
                collected += 1
            adjust_delay(sub, success=True)
            continue

        # ERRORS → fallback to RSS
        if status != 200:
            print(f"⚠️ JSON error {status} → RSS fallback")
            rss = fetch_rss(sub)
            for r in rss:
                buffer.append(r)
                collected += 1
            continue

        # JSON SUCCESS → reduce delay
        adjust_delay(sub, success=True)

        # if empty → RSS fallback
        if not rows:
            print("⚠️ Empty JSON feed, switching to RSS")
            rss = fetch_rss(sub)
            for r in rss:
                buffer.append(r)
                collected += 1
            continue

        # Append fetched rows
        for r in rows:
            buffer.append(r)
            collected += 1

            if len(buffer) >= 5000:
                save_chunk(buffer, chunk_idx)
                chunk_idx += 1
                buffer = []

        # Progress save
        after = next_after
        save_progress(sub, {
            "after": after,
            "collected": collected,
            "chunk_idx": chunk_idx
        })

        print(f"📌 r/{sub}: collected={collected}, next_after={after}, delay={SUBREDDIT_DELAY[sub]}s")

        if after is None:
            print("⏳ End of history for this subreddit.")
            break

    # Write leftover buffer
    if buffer:
        save_chunk(buffer, chunk_idx)
        chunk_idx += 1

    save_progress(sub, {
        "after": after,
        "collected": collected,
        "chunk_idx": chunk_idx
    })

    print(f"🎯 DONE r/{sub}: total={collected}")
    return collected


def collect_all():
    mkdir("data/raw")
    mkdir(PROGRESS_DIR)
    mkdir(TEMP_DIR)

    total = 0

    for sub in GAMING_SUBREDDITS:
        total += collect_subreddit(sub, PER_SUB_LIMIT)

    print(f"\n🎉 COLLECTION FINISHED — TOTAL POSTS: {total}")
    merge_chunks(OUTPUT_CSV)


if __name__ == "__main__":
    collect_all()
