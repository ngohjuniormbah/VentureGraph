# VentureGraph Deployment Guide

VentureGraph deploys as two separate services:

1. **Backend API** (`app_api.py`, FastAPI + Docling + the LLM/search agents)
   → deployed on **Railway**, using the provided `Dockerfile`. This is where
   every secret (`ANTHROPIC_API_KEY`, `TAVILY_API_KEY`/`BRAVE_API_KEY`) lives.
2. **Frontend dashboard** (`app_gui.py`, Streamlit) → deployed on
   **Hugging Face Spaces**. It never holds an LLM/search API key - it only
   needs to know the backend's URL, and calls it over HTTPS.

This split exists for a real reason, not just to use two platforms: the
backend needs Docling's model weights and PyTorch (a large, slow-to-build
image, and the only place your API keys should live), while the frontend is
a thin HTTP client that should stay small and fast to redeploy. Keeping keys
out of the frontend also matters because Spaces are often public - anyone
can view a public Space's files.

```
┌────────────────────┐        HTTPS        ┌──────────────────────────┐
│ Hugging Face Space  │ ──────────────────▶ │  Railway service          │
│ app_gui.py (Streamlit)│                    │  app_api.py (FastAPI)     │
│ requirements-gui.txt │ ◀────────────────── │  Dockerfile (Docling,     │
│ (no API keys)        │      JSON           │  ANTHROPIC_API_KEY,       │
└────────────────────┘                        │  TAVILY_API_KEY, ...)    │
                                               └──────────────────────────┘
```

## 1. Deploying the backend to Railway

Railway builds and runs the `Dockerfile` at the repo root, which installs
the system libraries Docling needs (see the comments in `Dockerfile`) plus
`requirements-api.txt`.

### 1.1 Create the service

1. In the [Railway dashboard](https://railway.app), click **New Project →
   Deploy from GitHub repo**, and select this repository.
2. Railway auto-detects the `Dockerfile` and builds from it. No build
   command configuration is needed.
3. Railway injects a `PORT` environment variable at runtime; the
   `Dockerfile`'s `CMD` already binds to `0.0.0.0:${PORT:-8000}`, so no
   changes are needed there either.

### 1.2 Set the API keys (Railway → Variables tab)

Open the service, go to the **Variables** tab, and add:

| Variable | Value |
|---|---|
| `LLM_PROVIDER` | `anthropic` (default, paid) or `gemini` (free tier, good for testing before paying for Claude - see below). |
| `ANTHROPIC_API_KEY` | Your Anthropic API key (required for `/orkg`, `/compare`, `/thesis` if `LLM_PROVIDER=anthropic`). |
| `GEMINI_API_KEY` | Your Gemini API key from [aistudio.google.com](https://aistudio.google.com) (required instead of `ANTHROPIC_API_KEY` if `LLM_PROVIDER=gemini`). |
| `TAVILY_API_KEY` | Your Tavily API key (required for `/thesis`, if using the default search provider). |
| `BRAVE_API_KEY` | Your Brave Search API key (only if you set `SEARCH_PROVIDER=brave` instead). |
| `SEARCH_PROVIDER` | `tavily` (default) or `brave`. Omit to use the default. |
| `CORS_ALLOW_ORIGINS` | Your Hugging Face Space's exact URL (e.g. `https://your-username-venturegraph.hf.space`). **Set this** instead of leaving the app's `"*"` default in production. |
| `VENTUREGRAPH_MODEL` | Optional: pins a specific model id for whichever provider is configured. Leave unset to use the default. |

**Testing without paying for Claude**: set `LLM_PROVIDER=gemini` and
`GEMINI_API_KEY` instead of `ANTHROPIC_API_KEY` - Google AI Studio has an
ongoing free tier, unlike Anthropic's API. Once you're satisfied with the
results and ready to go live for real users, switch `LLM_PROVIDER` back to
`anthropic` (or just remove the variable, since that's the default) and add
`ANTHROPIC_API_KEY` - no code or redeploy-process changes needed, only
these two variables. See the "Switching LLM providers" section of
`README.md` for how this works under the hood.

Railway encrypts and injects these as environment variables at container
start - they're never baked into the image or visible in the repo. Do not
put real keys in `.env` and commit it; `.env` is gitignored specifically to
prevent that (see §4).

After saving variables, Railway automatically redeploys. Once it's live,
note the public URL Railway assigns the service (**Settings → Networking →
Public Networking**, or generate a domain if one isn't assigned yet) - the
frontend needs this URL.

### 1.3 Verify

```bash
curl https://<your-railway-app>.up.railway.app/health
# {"status":"ok"}
```

Interactive API docs are available at `/docs` (Swagger UI) once deployed.

### 1.4 A note on request timeouts

`/thesis` makes several sequential-looking-but-actually-concurrent Claude
calls plus live web searches, and can take on the order of a minute or two
even with the pipeline's internal `asyncio.gather` concurrency. Railway's
own proxy timeout is generous (several minutes) by default, but if you sit
behind an additional reverse proxy or CDN, make sure its timeout is set
well above that (see §2.3 for the Streamlit-side timeout).

## 2. Deploying the frontend to Hugging Face Spaces

The frontend (`app_gui.py`) only needs `requirements-gui.txt`
(`streamlit` + `requests`) - it does not need the `Dockerfile` and does not
need Docling, so use a plain **Streamlit** Space rather than a Docker Space.

### 2.1 Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose **Streamlit** as the Space SDK.
3. Either push this repo's contents to the Space's own git remote, or
   connect the Space to this GitHub repo (Settings → "Repository" on the
   Space, if you want it to track this repo directly).
4. In the Space's root, `app_gui.py` is auto-detected as the entry point by
   the Streamlit SDK. If your Space requires a specific dependency file
   name, either rename `requirements-gui.txt` to `requirements.txt` **for
   the Space's copy only** (don't rename it in this repo - the two files
   mean different things here), or add a small `requirements.txt` at the
   Space root that does `-r requirements-gui.txt`.

### 2.2 Point it at the Railway backend

Hugging Face Spaces has its own secret store, separate from your repo:

1. Open the Space → **Settings → Variables and secrets**.
2. Add a **secret** (not a public variable, to keep it out of the Space's
   visible config) named `VENTUREGRAPH_API_URL`, set to your Railway
   backend's public URL from step 1.3, e.g.
   `https://your-railway-app.up.railway.app`.

`app_gui.py` reads this via `os.environ.get("VENTUREGRAPH_API_URL", ...)` on
load, and also exposes it as an editable field in the sidebar so you can
point it at a different backend (e.g. `http://localhost:8000` while
developing) without changing any secrets.

The frontend needs **no** `ANTHROPIC_API_KEY` or search API key at all -
only the backend URL. If you ever see a prompt asking for an Anthropic key
on the frontend side, something is wrong with the deployment split.

### 2.3 Verify

Open the Space's URL, use the sidebar's **Check connection** button to
confirm it can reach `/health` on the Railway backend, then try the
**Parse** tab with a sample PDF. For the **Investment Thesis** tab
specifically, expect it to take a minute or two - the request timeout for
that call is already set higher (`THESIS_TIMEOUT_SECONDS` in `app_gui.py`)
than the other endpoints to accommodate this.

## 3. Local development (both services on one machine)

```bash
# Terminal 1: backend
pip install -r requirements-api.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY / TAVILY_API_KEY
export $(grep -v '^#' .env | xargs)   # or use python-dotenv / honcho
uvicorn app_api:app --reload --port 8000

# Terminal 2: frontend
pip install -r requirements-gui.txt
streamlit run app_gui.py
```

The Streamlit sidebar defaults to `http://localhost:8000`, so no
configuration is needed for a fully local run.

## 4. API keys and secrets hygiene

- `.env` (and any `.env.*` variant) is gitignored - see `.gitignore` - so
  copying `.env.example` to `.env` and filling in real keys locally will
  never accidentally get committed. `.env.example` itself is tracked and
  contains no real values, only variable names, so it's safe as a template.
- Never put real API keys in `requirements*.txt`, the `Dockerfile`, or
  anywhere under `src/` - every key is read from the environment at
  runtime (`os.environ`), never hardcoded.
- Railway's **Variables** tab and Hugging Face's **Settings → Variables and
  secrets** are the only places real keys should be entered for a deployed
  instance - both are encrypted at rest and injected as environment
  variables at runtime, not committed to any repository.
- Set `CORS_ALLOW_ORIGINS` on the Railway backend to your actual Space URL
  once you know it, instead of leaving the default `"*"` - otherwise any
  website can call your backend (and burn your Anthropic/search API
  budget) from a user's browser.

## 5. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Docker build fails with `libGL.so.1: cannot open shared object file` | A system library the `Dockerfile` installs is missing on your build platform's base image; confirm you're building from the provided `Dockerfile` unmodified (it installs `libgl1`/`libglib2.0-0` specifically for this). |
| `/thesis` or `/orkg` returns a 500 with an Anthropic auth error | `ANTHROPIC_API_KEY` isn't set (or is invalid), or `LLM_PROVIDER` is set to `anthropic` but you meant to set `gemini`. |
| `/thesis` or `/orkg` returns a 500 with a Gemini/`API_KEY_INVALID` error | `LLM_PROVIDER=gemini` but `GEMINI_API_KEY` isn't set or is invalid - check it was copied from [aistudio.google.com](https://aistudio.google.com). |
| `/thesis` returns a 500 mentioning Tavily/Brave | The matching search API key (`TAVILY_API_KEY` or `BRAVE_API_KEY`) isn't set, or `SEARCH_PROVIDER` doesn't match which key you set. |
| Streamlit shows "Could not reach the VentureGraph API" | `VENTUREGRAPH_API_URL` (Space secret) is wrong, the Railway service isn't running, or `CORS_ALLOW_ORIGINS` on the backend doesn't include the Space's origin. |
| First Docling call is very slow | Expected on cold start - Docling downloads its layout/table-structure model weights from Hugging Face on first use and caches them; subsequent calls are fast. Make sure the Railway service has outbound internet access and enough disk for the cache. |

> Note: the `Dockerfile` in this repo has not been build-verified against a
> live Docker daemon in the environment this guide was written in (no
> Docker daemon was available there); the system packages listed are the
> well-documented requirements for Docling/OpenCV on a Debian slim image,
> but if you hit a missing shared library on your specific platform, add it
> to the `apt-get install` list in `Dockerfile` and rebuild.
