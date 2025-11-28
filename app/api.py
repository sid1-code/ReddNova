# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from factcheck.pipeline import fact_check_post

# Create FastAPI app
app = FastAPI()

# -------------------------
# CORS FIX (required for frontend)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow all origins (OK for local dev)
    allow_credentials=True,
    allow_methods=["*"],          # allow POST, OPTIONS, GET, etc
    allow_headers=["*"],          # allow all headers
)

# Request model
class Post(BaseModel):
    title: str
    selftext: str | None = None   # optional field

# Fact-check endpoint
@app.post("/factcheck")
def run(post: Post):
    # Combine title + body into one text
    full_text = post.title
    if post.selftext:
        full_text += " " + post.selftext

    # Pass a single string into the pipeline
    return fact_check_post(full_text)
