#!/usr/bin/env python3
"""
VentureGraph API — FastAPI backend.

This is the deployable backend service: it wraps the same pipeline
`main.py` drives from the CLI (`src/parser`, `src/schemas`, `src/agents`,
`src/comparison`, `src/report`) behind HTTP endpoints, so a separate
frontend (see `app_gui.py`, the Streamlit dashboard) can call it over the
network instead of importing it directly. That split matters for
deployment: this backend needs the heavy dependencies (Docling, its model
weights, PyTorch) and holds all the API keys (`ANTHROPIC_API_KEY`,
`TAVILY_API_KEY`/`BRAVE_API_KEY`); the frontend only needs `streamlit` and
`requests` and never sees those keys. See `deployment_guide.md` for how the
two are deployed separately (this backend on Railway, the frontend on
Hugging Face Spaces).

Run locally:
    uvicorn app_api:app --reload --port 8000

Endpoints (see `src/api/routes.py` for the implementation of each):
    GET  /health   - liveness check
    POST /parse    - PDF -> Markdown + ScientificEssence (no LLM)
    POST /orkg     - PDF -> Markdown + ORKG contribution triples
    POST /compare  - 2+ PDFs -> benchmark comparison table
    POST /thesis   - PDF -> full Investment Thesis
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router

app = FastAPI(
    title="VentureGraph API",
    description="Turns scientific PDFs into ORKG contributions and startup Investment Theses.",
    version="1.0.0",
)

_allowed_origins = os.environ.get("CORS_ALLOW_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    # A public demo's frontend (e.g. a Hugging Face Space) is typically on a
    # different origin than this API, so CORS must be opened for it explicitly.
    # Defaults to "*" for ease of getting a demo running; set CORS_ALLOW_ORIGINS
    # to your frontend's exact URL in production instead of leaving this open.
    allow_origins=[origin.strip() for origin in _allowed_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
