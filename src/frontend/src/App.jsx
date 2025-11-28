// src/App.jsx
import React from "react";
import RedditPost from "./components/RedditPost";
import "./styles/reddit.css";

function App() {
  return (
    <div className="app-root">
      <header className="app-header">
        <h1>Live Fake News — Mock Reddit</h1>
      </header>
      <main>
        <RedditPost />
      </main>
    </div>
  );
}

export default App;

