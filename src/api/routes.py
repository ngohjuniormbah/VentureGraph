"""
FastAPI routes exposing the VentureGraph pipeline over HTTP.

This is a thin HTTP layer over the same functions `main.py` calls for the
CLI - it does not reimplement any pipeline logic. Each route: saves the
uploaded PDF(s) to a temporary file, calls the existing pipeline functions
(running blocking calls like Docling conversion and the synchronous LLM
agents in a worker thread via `asyncio.to_thread`, so the event loop stays
free for other requests), and returns the result as JSON.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from src.agents.benchmark_agent import extract_paper_benchmarks
from src.agents.orkg_agent import extract_orkg_contribution
from src.agents.venture_catalyst import run_venture_catalyst
from src.api.schemas import (
    CompareResponse,
    ComparePaperResult,
    HealthResponse,
    OrkgResponse,
    ParseResponse,
    ThesisResponse,
)
from src.comparison.comparison_engine import generate_comparison_report
from src.extractor.essence_extractor import extract_essence_stub
from src.parser.pdf_parser import convert_pdf_to_markdown
from src.report.investment_thesis import render_investment_thesis_markdown
from src.schemas.venture import InvestmentThesis

router = APIRouter()


async def _save_upload_to_temp_pdf(upload: UploadFile) -> Path:
    """
    Persist an uploaded file to a temporary `.pdf` path on disk.

    Data flow:
        Reads the full contents of `upload` into memory (uploads are
        expected to be single scientific papers, not bulk data, so this is
        an acceptable size), and writes them to a fresh temp file.
        `convert_pdf_to_markdown` (and everything downstream of it) takes a
        filesystem path rather than raw bytes, so this bridges the two.

    Args:
        upload: The incoming FastAPI `UploadFile`.

    Returns:
        Path to the written temporary PDF file. Callers are responsible for
        deleting it (see `_cleanup`).

    Raises:
        HTTPException: 400 if the filename doesn't end in `.pdf`.
    """
    if not upload.filename or not upload.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"Not a PDF file: {upload.filename!r}")

    contents = await upload.read()
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(contents)
    return Path(temp_path)


def _cleanup(*paths: Path) -> None:
    """Best-effort deletion of temporary files created by `_save_upload_to_temp_pdf`."""
    for path in paths:
        path.unlink(missing_ok=True)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check used by hosting platforms (Railway, etc.) and the Streamlit frontend."""
    return HealthResponse(status="ok")


@router.post("/parse", response_model=ParseResponse)
async def parse_pdf(file: UploadFile) -> ParseResponse:
    """
    `POST /parse`: PDF -> Markdown + ScientificEssence (no LLM call).

    Data flow:
        Mirrors `main.py`'s `run_parse`: saves the upload to a temp file,
        runs Docling conversion in a worker thread (it's CPU/IO-bound and
        synchronous), then the heuristic essence extractor, then returns
        both.
    """
    pdf_path = await _save_upload_to_temp_pdf(file)
    try:
        markdown = await asyncio.to_thread(convert_pdf_to_markdown, pdf_path)
        essence = extract_essence_stub(markdown, source_file=file.filename)
        return ParseResponse(markdown=markdown, essence=essence)
    finally:
        _cleanup(pdf_path)


@router.post("/orkg", response_model=OrkgResponse)
async def orkg_pdf(file: UploadFile) -> OrkgResponse:
    """
    `POST /orkg`: PDF -> Markdown + ORKG contribution triples (uses Claude).

    Data flow:
        Mirrors `main.py`'s `run_orkg`: converts the PDF, then runs the
        (synchronous, blocking) ORKG specialist agent in a worker thread so
        the event loop isn't blocked while Claude responds.
    """
    pdf_path = await _save_upload_to_temp_pdf(file)
    try:
        markdown = await asyncio.to_thread(convert_pdf_to_markdown, pdf_path)
        contribution = await asyncio.to_thread(
            extract_orkg_contribution, markdown, None, file.filename
        )
        return OrkgResponse(markdown=markdown, contribution=contribution)
    finally:
        _cleanup(pdf_path)


@router.post("/compare", response_model=CompareResponse)
async def compare_pdfs(files: list[UploadFile]) -> CompareResponse:
    """
    `POST /compare`: two or more PDFs -> a Markdown benchmark comparison table.

    Data flow:
        Mirrors `main.py`'s `run_compare`: converts and extracts benchmarks
        for each uploaded PDF (sequentially per file, since each file's own
        Docling + LLM work already runs off the event loop via
        `asyncio.to_thread`), then builds the deterministic comparison
        table via `generate_comparison_report`.
    """
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="'compare' requires at least two PDF files.")

    pdf_paths = [await _save_upload_to_temp_pdf(f) for f in files]
    try:
        results = []
        for upload, pdf_path in zip(files, pdf_paths):
            markdown = await asyncio.to_thread(convert_pdf_to_markdown, pdf_path)
            benchmarks = await asyncio.to_thread(extract_paper_benchmarks, markdown, None, upload.filename)
            results.append(ComparePaperResult(filename=upload.filename, benchmarks=benchmarks))

        comparison_markdown = generate_comparison_report([r.benchmarks for r in results])
        return CompareResponse(papers=results, comparison_markdown=comparison_markdown)
    finally:
        _cleanup(*pdf_paths)


@router.post("/thesis", response_model=ThesisResponse)
async def thesis_pdf(file: UploadFile) -> ThesisResponse:
    """
    `POST /thesis`: PDF -> full Investment Thesis (ORKG + Venture Catalyst).

    Data flow:
        Mirrors `main.py`'s `run_thesis`: converts the PDF, extracts the
        essence, then runs ORKG extraction and the full Venture Catalyst
        pipeline concurrently via `asyncio.gather` (the same concurrency
        this endpoint inherits from `run_venture_catalyst` internally),
        before rendering the final Markdown report.

        This is the slowest endpoint by far - it makes multiple Claude
        calls and several live web searches - so callers (including the
        Streamlit frontend) should expect this to take on the order of a
        minute and set client-side timeouts accordingly.
    """
    pdf_path = await _save_upload_to_temp_pdf(file)
    try:
        markdown = await asyncio.to_thread(convert_pdf_to_markdown, pdf_path)
        essence = extract_essence_stub(markdown, source_file=file.filename)

        orkg_contribution, idea_dossiers = await asyncio.gather(
            asyncio.to_thread(extract_orkg_contribution, markdown, essence.paper_title, file.filename),
            run_venture_catalyst(essence),
        )

        thesis = InvestmentThesis(
            paper_title=essence.paper_title,
            source_file=file.filename,
            scientific_essence=essence,
            orkg_contribution=orkg_contribution,
            idea_dossiers=idea_dossiers,
        )
        report_markdown = render_investment_thesis_markdown(essence, orkg_contribution, idea_dossiers)
        return ThesisResponse(thesis=thesis, report_markdown=report_markdown)
    finally:
        _cleanup(pdf_path)
