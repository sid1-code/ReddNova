import requests

SERPAPI_KEY = "f4dca3936519122d272cd62efa182f88d06027099755ad84559beedae03b0878"   # replace with your real key later

def search_web(query, num=5):
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num
    }
    data = requests.get("https://serpapi.com/search.json", params=params).json()

    urls = []
    for r in data.get("organic_results", [])[:num]:
        urls.append({
            "title": r.get("title"),
            "url": r.get("link")
        })
    return urls
