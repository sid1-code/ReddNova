import requests
import json
from textwrap import shorten

API_URL = "http://127.0.0.1:8000/factcheck"

TEST_CLAIMS = [
    # Clear false
    "GTA 6 launched today on all platforms for free.",
    "PlayStation has announced they are shutting down PSN permanently.",
    "EA confirmed they are releasing FIFA 26 next week.",

    # Clear true
    "Next year's F1 World Championships won't get a new game, but a paid expansion instead.",
    "The Evercade EXP-R handheld received a Black Friday discount.",
    "Meta Quest 3S bundle includes Amazon credit.",

    # Unverified
    "Elden Ring 2 is releasing in January 2026.",
    "Rockstar is working on a VR version of Red Dead Redemption.",
    "A new console called PlayStation Zero was secretly trademarked.",

    # Complex / ambiguous
    "Xbox Partner Preview leaked a new Halo game during the November event.",
    "PS Plus is removing ten games next month, including Sonic Frontiers.",
    "EA is skipping the F1 game release next year.",

    # Sarcasm / rumor
    "Apparently GTA 6 is coming out tomorrow because someone's uncle at Rockstar said so.",
    "Meta Quest 3S now costs ₹500 in India.",

    # Short claims
    "GTA 6 delayed?",
    "Fortnite banned?",
    "PS5 Pro release?",
]

def run_test(claim: str):
    payload = {"title": claim}
    try:
        res = requests.post(API_URL, json=payload, timeout=20)
        data = res.json()
    except Exception as e:
        print(f"❌ ERROR calling API: {e}")
        return

    clf = data["classifier"]
    evidence = data["evidence"]

    print("\n" + "="*90)
    print(f"📝 CLAIM: {claim}")
    print("="*90)

    print(f"Prediction: {clf['pred'].upper()}   (confidence: {clf['confidence']:.3f})")
    print(f"Reasoning: {clf['reasoning']}")

    # Extract supporting or contradicting evidence URLs
    supporting = []
    contradicting = []

    for ev in evidence:
        ent = ev.get("nli_entailment", 0)
        con = ev.get("nli_contradiction", 0)

        if ent >= 0.75:
            supporting.append(ev["url"])
        if con >= 0.75:
            contradicting.append(ev["url"])

    print("\nSupporting evidence URLs:")
    if supporting:
        for url in supporting:
            print("   ✔", url)
    else:
        print("   (none)")

    print("\nContradicting evidence URLs:")
    if contradicting:
        for url in contradicting:
            print("   ✘", url)
    else:
        print("   (none)")

    print("\nTop evidence excerpts:")
    for ev in evidence[:3]:
        excerpt = shorten(ev.get("excerpt", ""), width=120)
        print(f" - {ev['url']}  |  entail={ev.get('nli_entailment',0):.2f}, contradict={ev.get('nli_contradiction',0):.2f}")
        print(f"   {excerpt}")
    print("-"*90)


if __name__ == "__main__":
    print("\n🚀 Running fact-check tests...\n")
    for claim in TEST_CLAIMS:
        run_test(claim)

    print("\n🎉 Tests completed!\n")
