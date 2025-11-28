// src/components/RedditPost.jsx
import React, { useState } from "react";
import { FACTCHECK_ENDPOINT } from "../config";
import "../styles/reddit.css";

function VerdictBadge({ label }) {
  const cls =
    label === "true" ? "badge-true" : label === "false" ? "badge-false" : "badge-unverified";
  const text = label ? label.toUpperCase() : "UNKNOWN";
  return <span className={`verdict-badge ${cls}`}>{text}</span>;
}

export default function RedditPost() {
  const [title, setTitle] = useState("Black Ops 7 includes the Endgame mode that unlocks after completing the campaign.");
  const [body, setBody] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function handleFactCheck() {
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const payload = { title, selftext: body || null };
      const resp = await fetch(FACTCHECK_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${txt}`);
      }
      const data = await resp.json();
      // attach original claim for display
      data.claim = title + (body ? `\n\n${body}` : "");
      setResult(data);
    } catch (e) {
      console.error("Fact-check error:", e);
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  function renderEvidence(ev, i) {
    return (
      <div className="evidence-card" key={i}>
        <div className="evidence-meta">
          {ev.url ? <a href={ev.url} target="_blank" rel="noreferrer">{ev.url}</a> : <span>(no url)</span>}
          <span className="evidence-score">score: {ev.score ? ev.score.toFixed(2) : "—"}</span>
        </div>
        {ev.summary ? <div className="evidence-summary">{ev.summary}</div> : <div className="evidence-short">{ev.text_short}</div>}
        {ev.contradiction_sentence ? <div className="evidence-contradict">✖ {ev.contradiction_sentence}</div> : null}
        <div className="evidence-nli">
          entailment: {(ev.nli_entailment ?? 0).toFixed(2)} · contradiction: {(ev.nli_contradiction ?? 0).toFixed(2)}
        </div>
      </div>
    );
  }

  return (
    <div className="reddit-mock-root">
      <div className="post-card">
        <div className="post-vote">▲<br/>1.2k<br/>▼</div>
        <div className="post-main">
          <div className="post-header">
            <span className="subreddit">r/gaming</span>
            <span className="poster"> • u/you</span>
          </div>
          <h3 className="post-title">{title}</h3>
          {body && <div className="post-body">{body}</div>}

          <div className="post-actions">
            <button className="btn-fact" onClick={handleFactCheck} disabled={loading}>
              {loading ? "Checking…" : "Fact Check"}
            </button>
            <button className="btn-draft" onClick={() => { setTitle(title); setBody(body); }}>Save (local)</button>
          </div>

          {error && <div className="error">{error}</div>}

          {/* Render the fact-check comment */}
          {result && (
            <div className="comment-thread">
              <div className="comment-card">
                <div className="comment-side">▲</div>
                <div className="comment-body">
                  <div className="comment-header">
                    <strong>LiveFakeNewsBot</strong> <span className="comment-meta">•  just now</span>
                    <div className="verdict-inline">
                      <VerdictBadge label={result.classifier.pred} />
                      <span className="confidence"> confidence: {(result.classifier.confidence ?? 0).toFixed(2)}</span>
                    </div>
                  </div>

                  <div className="comment-content">
                    <p><em>Reasoning:</em> {result.classifier.reasoning}</p>

                    <details open>
                      <summary>Top evidence (click to expand)</summary>
                      <div className="evidence-list">
                        {result.evidence && result.evidence.length ? result.evidence.slice(0,5).map(renderEvidence) : <div>No evidence returned.</div>}
                      </div>
                    </details>

                    <div className="comment-footer">
                      <small>I'm an automated fact-checker. Source: local evidence DB.</small>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

        </div>
      </div>

      {/* Controls to edit claim */}
      <div className="composer">
        <h4>Edit mock Reddit post</h4>
        <input className="input-title" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Post title" />
        <textarea className="input-body" value={body} onChange={(e) => setBody(e.target.value)} placeholder="(Optional) post body" rows={4} />
      </div>
    </div>
  );
}
