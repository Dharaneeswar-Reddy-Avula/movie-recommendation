import pickle
import gzip
from difflib import get_close_matches
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------- Load artifacts once at startup ----------
with open("movies.pkl", "rb") as f:
    MOVIES: pd.DataFrame = pickle.load(f)

# Initialize SIM with a default value
SIM: Optional[np.ndarray] = None

# Try to load the similarity matrix from the original file first
try:
    with gzip.open("similarity2.pkl.gz", "rb") as f:
        loaded_sim = pickle.load(f)
        # Check if the loaded object is actually a numpy array
        if isinstance(loaded_sim, np.ndarray):
            SIM = loaded_sim
        else:
            print("Warning: similarity2.pkl.gz did not contain a valid numpy array")
except Exception as e:
    print(f"Warning: Could not load similarity2.pkl.gz: {e}")

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

class ErrorMessage(BaseModel):
    detail: str

def _resolve_title(raw: str) -> Optional[Tuple[str, int]]:
    norm = raw.strip().lower()
    if norm in TITLE_TO_IDX:
        idx = TITLE_TO_IDX[norm]
        return MOVIES.loc[idx, "title"], idx
    match = get_close_matches(norm, list(TITLE_TO_IDX.keys()), n=1, cutoff=0.6)
    if match:
        idx = TITLE_TO_IDX[match[0]]
        return MOVIES.loc[idx, "title"], idx
    return None

def _recommend(title: str, k: int = 5) -> Tuple[str, List[MovieOut]]:
    # Check if similarity matrix is available
    if SIM is None:
        raise HTTPException(status_code=500, detail="Recommendation system not properly initialized: similarity matrix missing or invalid")
    
    resolved = _resolve_title(title)
    if not resolved:
        return title, []  # Return the original title if not found
    used_title, idx = resolved
    
    # Check if idx is valid for the SIM array
    if idx >= len(SIM):
        raise HTTPException(status_code=500, detail="Internal error: index out of bounds for similarity matrix")
    
    distances = SIM[idx]
    ranked = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)

    out: List[MovieOut] = []
    for j, score in ranked:
        if j == idx:
            continue
        row = MOVIES.iloc[j]
        # Generate poster URL using TMDB image URL pattern
        # Using w500 size for poster images (500px wide)
        poster_url = f"https://image.tmdb.org/t/p/w500/{row['movie_id']}" if 'movie_id' in row else None
        out.append(
            MovieOut(
                title=str(row["title"]),
                movie_id=int(row["movie_id"]),
                score=float(score),
                poster_url=poster_url,
            )
        )
        if len(out) >= k:
            break
    return used_title, out

app = FastAPI(title="Movie Recommender", version="1.0")

@app.get("/health")
def health():
    sim_status = "loaded" if SIM is not None else "missing/invalid"
    return {"status": "ok", "movies": int(len(MOVIES)), "similarity_matrix": sim_status}

@app.get("/recommend", response_model=RecommendResponse, responses={500: {"model": ErrorMessage}})
def recommend_endpoint(movie: str, k: int = 5):
    if k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="k must be between 1 and 50")
    try:
        used_title, items = _recommend(movie, k)
        if not items:
            raise HTTPException(status_code=404, detail="Title not found or no recommendations.")
        return RecommendResponse(query=movie, used_title=used_title, results=items)
    except Exception as e:
        # Re-raise HTTP exceptions
        if isinstance(e, HTTPException):
            raise e
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")