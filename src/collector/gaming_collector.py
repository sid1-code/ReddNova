import requests
import pandas as pd
import time
import os

# ============================================
#  FULL LIST OF GAMING SUBREDDITS
# ============================================
GAMING_SUBREDDITS = [
    "gaming", "pcgaming", "games", "PS5", "PlayStation", "xbox", "xboxone",
    "NintendoSwitch", "nintendo", "Steam", "SteamDeck", "GameDeals",
    "buildapc", "pcmasterrace", "leagueoflegends", "Valorant",
    "Minecraft", "FortniteCompetitive", "GrandTheftAutoV",
    "GTA6", "PUBATTLEGROUNDS", "pokemon", "halo", "dota2",
    "CounterStrike", "R6ProLeague", "eldenring", "starfield",
    "cyberpunkgame", "witcher", "skyrim", "fallout", "BaldursGate3",
    "callofduty", "mw3", "ApexLegends", "Overwatch", "Overwatch2",
    "rocketleague", "HuntShowdown", "arma", "dayz", "spacestation13",
    "HonkaiStarRail", "Genshin_Impact", "wow", "classicwow",
    "FFXIV", "FFXV", "KingdomHearts", "zelda", "smashbros",
    "StreetFighter", "MortalKombat", "Tekken", "assassinscreed",
    "battlefield", "rainbow6", "EscapeFromTarkov", "vrgaming",
    "oculus", "virtualreality", "metaquest", "gamedev",
    "unity3d", "unrealengine", "indiegames"
]


# ============================================
#  Pushshift single-subreddit collector
# ============================================
def fetch_pushshift(subreddit, limit=2000):
    base_url = (
        "https://api.pushshift.io/reddit/search/submission/"
        f"?subreddit={subreddit}&size=500&sort=desc"
    )

    print(f"\n🔍 Collecting posts from r/{subreddit} ...")
    posts = []
    last_created_utc = None

    while len(posts) < limit:
        url = base_url
        if last_created_utc:
            url += f"&before={last_created_utc}"

        try:
            r = requests.get(url, timeout=20)
            data = r.json().get("data", [])
        except Exception as e:
            print(f"⚠ Error fetching from Pushshift: {e}")
            break

        if not data:
            print("No more posts available.")
            break

        for post in data:
            posts.append({
                "id": post.get("id"),
                "subreddit": subreddit,
                "title": post.get("title"),
                "selftext": post.get("selftext"),
                "created_utc": post.get("created_utc"),
                "author": post.get("author"),
                "score": post.get("score"),
                "num_comments": post.get("num_comments"),
                "url": post.get("url"),
            })

        last_created_utc = data[-1].get("created_utc")
        print(f"  → {len(posts)} collected so far...")

        time.sleep(1)

    print(f"✔ Done r/{subreddit}: {len(posts)} posts.\n")
    return posts


# ============================================
#  MASTER COLLECTOR — RUN ALL SUBREDDITS
# ============================================
def collect_all_gaming_posts(output_path="data/raw/gaming_posts.csv", limit_per_sub=2000):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_data = []
    for sub in GAMING_SUBREDDITS:
        try:
            posts = fetch_pushshift(sub, limit=limit_per_sub)
            all_data.extend(posts)
        except Exception as e:
            print(f"❌ Failed for r/{sub}: {e}")

    print(f"\n🔥 TOTAL posts collected: {len(all_data)}")

    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset="id", inplace=True)
    df.to_csv(output_path, index=False)

    print(f"📁 Saved dataset to: {output_path}")


# ============================================
#  MAIN
# ============================================
if __name__ == "__main__":
    collect_all_gaming_posts(
        output_path="data/raw/gaming_posts.csv",
        limit_per_sub=2000   # You can increase to 5000 or 10000 later
    )
