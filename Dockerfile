# VentureGraph API backend image.
#
# This builds the FastAPI backend (app_api.py) only - the one that needs
# Docling (PDF parsing), PyTorch, and the LLM/search API keys. The
# Streamlit frontend (app_gui.py) is a thin HTTP client with much lighter
# dependencies (see requirements-gui.txt) and is meant to be deployed
# separately (e.g. Hugging Face Spaces) - see deployment_guide.md.
FROM python:3.11-slim

# System libraries needed for PDF processing / Docling's document pipeline:
#   - libgl1, libglib2.0-0: runtime shared libraries OpenCV (used by
#     Docling's layout/table-structure models) links against; without
#     these, importing docling fails with "libGL.so.1: cannot open shared
#     object file" on a minimal Debian image.
#   - libgomp1: OpenMP runtime required by PyTorch/onnxruntime at inference
#     time.
#   - poppler-utils: provides pdftoppm/pdftotext, the standard system-level
#     PDF rendering/extraction utilities several PDF-processing Python
#     packages shell out to or bundle as a fallback.
#   - curl: used by the container HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (before copying the rest of the source)
# so Docker can cache this layer across builds that only change source code.
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Railway (and most PaaS hosts) inject the port to listen on via $PORT at
# runtime; 8000 is only the local-development default.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8000}/health" || exit 1

# Shell form (not exec form) so ${PORT:-8000} is expanded from the
# container's environment at startup.
CMD uvicorn app_api:app --host 0.0.0.0 --port ${PORT:-8000}
