import pickle
import gzip
from difflib import get_close_matches
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------- Load artifacts once at startup ----------
with open("movies.pkl", "rb") as f:
    MOVIES: pd.DataFrame = pickle.load(f)

# Replace this with your actual similarity matrix
SIM = ...  # your numpy array

with gzip.open("similarity2.pkl.gz", "wb") as f:
    pickle.dump(SIM, f)

if not {"movie_id", "title"}.issubset(MOVIES.columns):
    raise RuntimeError("movies.pkl must have columns: ['movie_id','title']")

MOVIES["_norm_title"] = MOVIES["title"].str.strip().str.lower()
TITLE_TO_IDX = {t: i for i, t in enumerate(MOVIES["_norm_title"])}

class RecommendRequest(BaseModel):
    title: str = Field(..., description="Movie title to find similar movies for")
    k: int = Field(5, ge=1, le=50, description="Number of recommendations")

class MovieOut(BaseModel):
    title: str
    movie_id: int
    score: float
    poster_url: Optional[str] = None

class RecommendResponse(BaseModel):
    query: str
    used_title: str
    results: List[MovieOut]

def _resolve_title(raw: str):
    norm = raw.strip().lower()
    if norm in TITLE_TO_IDX:
        idx = TITLE_TO_IDX[norm]
        return MOVIES.loc[idx, "title"], idx
    match = get_close_matches(norm, list(TITLE_TO_IDX.keys()), n=1, cutoff=0.6)
    if match:
        idx = TITLE_TO_IDX[match[0]]
        return MOVIES.loc[idx, "title"], idx
    return None

def _recommend(title: str, k: int = 5):
    resolved = _resolve_title(title)
    if not resolved:
        return None, []
    used_title, idx = resolved
    distances = SIM[idx]
    ranked = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)

    out = []
    for j, score in ranked:
        if j == idx:
            continue
        row = MOVIES.iloc[j]
        out.append(
            MovieOut(
                title=str(row["title"]),
                movie_id=int(row["movie_id"]),
                score=float(score),
                poster_url=None,
            )
        )
        if len(out) >= k:
            break
    return used_title, out

app = FastAPI(title="Movie Recommender", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok", "movies": int(len(MOVIES))}

@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: RecommendRequest):
    used_title, items = _recommend(req.title, req.k)
    if not items:
        raise HTTPException(status_code=404, detail="Title not found or no recommendations.")
    return RecommendResponse(query=req.title, used_title=used_title, results=items)
