// src/api/fakeApi.js
// Dummy implementations of /pipeline and /retrieve.
// Replace these with real fetch(...) calls when your backend is ready.

export async function pipeline(claim) {
  // simulate latency
  await new Promise((r) => setTimeout(r, 900 + Math.random() * 700));

  const t = (claim || "").toLowerCase();
  let verdict = "UNCERTAIN";
  let confidence = 55;

  if (!t.trim()) {
    verdict = "INSUFFICIENT";
    confidence = 30;
  } else if (t.includes("official") || t.includes("study") || t.includes("reported")) {
    verdict = "LIKELY TRUE";
    confidence = 88;
  } else if (t.includes("hoax") || t.includes("fake") || t.includes("false")) {
    verdict = "LIKELY FAKE";
    confidence = 94;
  } else {
    verdict = Math.random() > 0.5 ? "LIKELY TRUE" : "LIKELY FAKE";
    confidence = Math.round(60 + Math.random() * 30);
  }

  return {
    verdict,
    confidence, // integer 0-100
    evidence: [
      "This sentence partially matches a forum post with unverified sources.",
      "No major outlet has published a matching headline in the last 72 hours.",
      "An older press release contains some similar wording but differs in context."
    ],
    sources: [
      "https://example-news.org/article/123",
      "https://factcheck.example/report/456",
      "https://archive.example/related"
    ]
  };
}

export async function retrieve(query) {
  await new Promise((r) => setTimeout(r, 500 + Math.random() * 600));
  if (!query || !query.trim()) throw new Error("No query provided");
  return {
    snippets: [
      `Manual snippet 1 about "${query}" — low confidence.`,
      `Manual snippet 2: small excerpt from a comment thread discussing the claim.`
    ],
  };
}

