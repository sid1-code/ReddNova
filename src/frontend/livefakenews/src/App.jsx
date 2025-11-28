import React, { useState } from "react";
import "./App.css";

// Use the uploaded image file path as hero/post image
const HERO_IMAGE = "/mnt/data/8d6c0c64-e1da-4406-b362-9ccaabe5d5fb.png";

function Skeleton({ width = "100%", height = 12, style = {} }) {
  return <div className="skeleton" style={{ width, height, ...style }} />;
}

export default function App() {
  const [claim, setClaim] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  // -------------------------
  // Dummy backend (replace later)
  // -------------------------
  async function pipeline(claimText) {
    // simulate network latency
    await new Promise((r) => setTimeout(r, 900 + Math.random() * 700));

    const t = (claimText || "").toLowerCase().trim();
    let verdict = "UNCERTAIN";
    let confidence = 58;

    if (!t) {
      verdict = "INSUFFICIENT";
      confidence = 30;
    } else if (t.includes("official") || t.includes("study") || t.includes("reported")) {
      verdict = "LIKELY TRUE";
      confidence = 88;
    } else if (t.includes("hoax") || t.includes("fake") || t.includes("false")) {
      verdict = "LIKELY FAKE";
      confidence = 95;
    } else {
      verdict = Math.random() > 0.5 ? "LIKELY TRUE" : "LIKELY FAKE";
      confidence = Math.round(60 + Math.random() * 30);
    }

    return {
      verdict,
      confidence,
      evidence: [
        "Matches phrasing from a low-credibility forum thread.",
        "No matching headlines from major outlets in the last 72 hours.",
        "A similar claim appeared in a press release — context differs."
      ],
      sources: [
        "https://example-news.org/article/123",
        "https://factcheck.example/report/456"
      ],
    };
  }

  async function retrieve(query) {
    await new Promise((r) => setTimeout(r, 500 + Math.random() * 500));
    if (!query || !query.trim()) throw new Error("No query provided");
    return {
      snippets: [
        `Manual snippet: discussion mentioning "${query}".`,
        "Manual snippet: short excerpt from a blog with partial match."
      ]
    };
  }

  // -------------------------
  // Handlers
  // -------------------------
  async function handleAnalyze(e) {
  e?.preventDefault();
  setError("");
  setResult(null);

  if (!claim.trim()) {
    setError("Please enter a claim to analyze.");
    return;
  }

  try {
    setLoading(true);

    const resp = await fetch("http://127.0.0.1:8000/factcheck", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        title: claim,
        selftext: ""
      })
    });

    if (!resp.ok) {
      throw new Error("Server returned " + resp.status);
    }

    const data = await resp.json();

    // -----------------------------
    // Map backend → frontend format
    // -----------------------------
    const verdict =
      data.classifier?.pred === "true"
        ? "LIKELY TRUE"
        : data.classifier?.pred === "false"
        ? "LIKELY FAKE"
        : "UNVERIFIED";

    const confidence = Math.round((data.classifier?.confidence || 0) * 100);

    const evidence = (data.evidence || []).map((ev) => {
      // Use summary > excerpt > text_short
      const text =
        ev.summary ||
        ev.excerpt ||
        ev.text_short ||
        "No summary available.";

      return text;
    });

    const sources = (data.evidence || [])
      .map((ev) => ev.url)
      .filter(Boolean);

    setResult({
      verdict,
      confidence,
      evidence,
      sources
    });

  } catch (err) {
    console.error(err);
    setError("Pipeline error — please try again later.");
  } finally {
    setLoading(false);
  }
}


  async function handleRetrieve() {
    setError("");
    try {
      setLoading(true);
      const r = await retrieve(claim);
      setResult((prev) => ({ ...(prev || {}), manualSnippets: r.snippets }));
    } catch (err) {
      console.error(err);
      setError(err.message || "Retrieve failed");
    } finally {
      setLoading(false);
    }
  }

  // verdict styling helper
  function verdictClass(v) {
  if (!v) return "verdict neutral";
  if (v.includes("TRUE")) return "verdict true";
  if (v.includes("FAKE")) return "verdict fake";
  if (v.includes("UNVERIFIED")) return "verdict neutral";
  return "verdict uncertain";
}


  return (
    <div className="app-root">
      <header className="topbar">
        <div className="brand">
          <div className="logo">r/</div>
          <div className="brand-texts">
            <div className="brand-title">Verification Ledger</div>
            <div className="brand-sub">Reddit claim verification — demo UI</div>
          </div>
        </div>
      </header>

      <main className="main-grid">
        {/* Left: post card */}
        <section className="post-column">
          <article className="post-card">
            <div className="vote-col">
              <button className="vote">▲</button>
              <div className="score">1.2k</div>
              <button className="vote">▼</button>
            </div>

            <div className="post-content">
              <div className="post-meta">
                <span className="subreddit">r/news</span>
                <span className="dot">•</span>
                <span className="meta">Posted by u/example • 2h</span>
              </div>

              <h2 className="post-title">Example Reddit post / claim (demo)</h2>

              <div className="post-hero">
                <img src={HERO_IMAGE} alt="hero" />
              </div>

              <p className="post-text">
                Paste a Reddit claim in the panel on the right and click <strong>Analyze Claim</strong> to see evidence, sources, verdict and confidence.
              </p>

              <div className="analysis-block">
                {!result && !loading && (
                  <div className="muted">No analysis yet — use the right panel to analyze a claim.</div>
                )}

                {loading && (
                  <>
                    <div className="analysis-header">
                      <Skeleton width="140px" height={28} style={{ borderRadius: 20 }} />
                      <Skeleton width="120px" height={18} style={{ borderRadius: 6 }} />
                    </div>

                    <div className="evidence-clean">
                      <Skeleton width="80%" height={12} />
                      <Skeleton width="90%" height={12} />
                      <Skeleton width="70%" height={12} />
                    </div>

                    <div className="sources-clean">
                      <Skeleton width="60%" height={12} />
                      <Skeleton width="95%" height={12} />
                    </div>
                  </>
                )}

                {result && !loading && (
                  <>
                    <div className="analysis-header">
                      <div className={verdictClass(result.verdict)}>{result.verdict}</div>
                      <div className="confidence">Confidence: <strong>{result.confidence}%</strong></div>
                    </div>

                    <div className="evidence-clean">
                      {result.evidence.map((e, i) => (
                        <p key={i} className="evidence-item">• {e}</p>
                      ))}

                      {result.manualSnippets?.map((s, i) => (
                        <p key={`m-${i}`} className="evidence-item manual">• {s}</p>
                      ))}
                    </div>

                    <div className="sources-clean">
                      <strong>Sources:</strong>
                      <ul>
                        {result.sources.map((s, i) => (
                          <li key={i}><a href={s} target="_blank" rel="noreferrer">{s}</a></li>
                        ))}
                      </ul>
                    </div>
                  </>
                )}
              </div>
            </div>
          </article>
        </section>

        {/* Right: control panel */}
        <aside className="control-column">
          <div className="control-card">
            <label className="label">Enter claim</label>
            <textarea
              className="claim-input"
              placeholder="Paste a Reddit claim or headline..."
              value={claim}
              onChange={(e) => setClaim(e.target.value)}
            />

            <div className="controls">
              <button className="btn primary" onClick={handleAnalyze} disabled={loading}>
                {loading ? "Analyzing…" : "Analyze Claim"}
              </button>
              <button className="btn" onClick={handleRetrieve} disabled={loading}>
                Manual retrieve
              </button>
            </div>

            {error && <div className="error">{error}</div>}

            <div className="help">
              Tip: concise claims/headlines work best. This demo uses a dummy backend. To hook your real API, replace the <code>pipeline</code> call in <code>handleAnalyze</code>.
            </div>
          </div>

          <div className="footer-card">
            <div className="footer-brand">Verification Ledger</div>
            <div className="footer-sub">Demo UI • no backend required</div>
          </div>
        </aside>
      </main>
    </div>
  );
}
