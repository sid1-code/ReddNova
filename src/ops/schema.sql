CREATE TABLE IF NOT EXISTS reddit_posts (
  post_id TEXT PRIMARY KEY,
  subreddit TEXT,
  title TEXT,
  body TEXT,
  created_utc TIMESTAMP,
  url TEXT,
  author TEXT,
  score INTEGER,
  num_comments INTEGER,
  raw_json JSONB
);

CREATE TABLE IF NOT EXISTS claims (
  claim_id SERIAL PRIMARY KEY,
  post_id TEXT REFERENCES reddit_posts(post_id),
  claim_text TEXT,
  sentence_idx INTEGER,
  created_utc TIMESTAMP,
  processed BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS evidence_docs (
  doc_id SERIAL PRIMARY KEY,
  url TEXT UNIQUE,
  domain TEXT,
  title TEXT,
  text TEXT,
  published_at TIMESTAMP,
  metadata JSONB
);

CREATE TABLE IF NOT EXISTS feedback (
  feedback_id SERIAL PRIMARY KEY,
  claim_id INTEGER REFERENCES claims(claim_id),
  predicted_label TEXT,
  predicted_confidence FLOAT,
  user_vote TEXT,
  corrected_label TEXT,
  user_id TEXT,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_versions (
  model_name TEXT,
  version TEXT,
  trained_on TEXT,
  metrics JSONB,
  path TEXT,
  created_at TIMESTAMP DEFAULT now(),
  PRIMARY KEY (model_name, version)
);
