"""
Tests for the FastAPI backend (`app_api.py` / `src/api/routes.py`).

These monkeypatch the pipeline functions the routes call (Docling
conversion and the LLM agents) rather than running them for real, since
this test suite is meant to run with no API key, no network access, and no
Docling model download - it's testing the HTTP wiring (request validation,
response shapes, status codes), not pipeline correctness (which is covered
by the other test modules).
"""

from fastapi.testclient import TestClient

import src.api.routes as routes
from src.schemas.benchmarks import PaperBenchmarks
from src.schemas.orkg import ORKGContribution
from src.schemas.scientific_essence import ScientificEssence
from src.schemas.venture import IdeaDossier


def _fake_essence(source_file=None) -> ScientificEssence:
    return ScientificEssence(
        paper_title="Fake Paper",
        problem_statement="p",
        methodology="m",
        key_results="k",
        source_file=source_file,
    )


def _fake_orkg_contribution() -> ORKGContribution:
    return ORKGContribution(
        reasoning="r",
        paper_title="Fake Paper",
        research_problem="p",
        contribution_label="Contribution 1",
        triples=[],
    )


def _fake_benchmarks(source_file=None) -> PaperBenchmarks:
    return PaperBenchmarks(reasoning="r", paper_title="Fake Paper", benchmarks=[], source_file=source_file)


def _pdf_upload(name="paper.pdf"):
    return {"file": (name, b"%PDF-1.4 fake pdf bytes", "application/pdf")}


def test_health():
    client = TestClient(routes.router)
    # Mount router directly via a minimal app to avoid pulling in app_api's CORS setup.
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_parse_rejects_non_pdf_upload():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    response = client.post("/parse", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "Not a PDF file" in response.json()["detail"]


def test_compare_rejects_fewer_than_two_files():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    response = client.post("/compare", files=[("files", ("a.pdf", b"%PDF-1.4", "application/pdf"))])
    assert response.status_code == 400
    assert "at least two" in response.json()["detail"]


def test_parse_returns_markdown_and_essence(monkeypatch):
    from fastapi import FastAPI

    monkeypatch.setattr(routes, "convert_pdf_to_markdown", lambda pdf_path: "# Fake Paper\n\nSome text.")
    monkeypatch.setattr(routes, "extract_essence_stub", lambda markdown, source_file=None: _fake_essence(source_file))

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    response = client.post("/parse", files=_pdf_upload())
    assert response.status_code == 200
    body = response.json()
    assert body["markdown"] == "# Fake Paper\n\nSome text."
    assert body["essence"]["paper_title"] == "Fake Paper"
    assert body["essence"]["source_file"] == "paper.pdf"


def test_orkg_returns_contribution(monkeypatch):
    from fastapi import FastAPI

    monkeypatch.setattr(routes, "convert_pdf_to_markdown", lambda pdf_path: "# Fake Paper")
    monkeypatch.setattr(
        routes, "extract_orkg_contribution", lambda markdown, paper_title, source_file: _fake_orkg_contribution()
    )

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    response = client.post("/orkg", files=_pdf_upload())
    assert response.status_code == 200
    assert response.json()["contribution"]["contribution_label"] == "Contribution 1"


def test_compare_returns_comparison_markdown(monkeypatch):
    from fastapi import FastAPI

    monkeypatch.setattr(routes, "convert_pdf_to_markdown", lambda pdf_path: "# Fake Paper")
    monkeypatch.setattr(
        routes,
        "extract_paper_benchmarks",
        lambda markdown, paper_title, source_file: _fake_benchmarks(source_file),
    )
    monkeypatch.setattr(routes, "generate_comparison_report", lambda papers: "| Dataset |\n|---|")

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    response = client.post(
        "/compare",
        files=[
            ("files", ("a.pdf", b"%PDF-1.4", "application/pdf")),
            ("files", ("b.pdf", b"%PDF-1.4", "application/pdf")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["papers"]) == 2
    assert body["papers"][0]["filename"] == "a.pdf"
    assert body["comparison_markdown"] == "| Dataset |\n|---|"


async def test_thesis_returns_full_report(monkeypatch):
    from fastapi import FastAPI

    from src.schemas.venture import CompetitorLandscape, StartupIdea, TAMEstimate

    async def fake_run_venture_catalyst(essence, model=None):
        idea = StartupIdea(
            reasoning="r",
            name="Idea",
            description="d",
            target_customer="c",
            commercial_angle="a",
            methodology_basis="m",
        )
        return [
            IdeaDossier(
                idea=idea,
                competitors=CompetitorLandscape(idea_name="Idea", summary="s"),
                tam=TAMEstimate(
                    idea_name="Idea",
                    reasoning="r",
                    calculation_code="print(1)",
                    calculation_output="1\n",
                    tam_usd=1.0,
                ),
            )
        ]

    monkeypatch.setattr(routes, "convert_pdf_to_markdown", lambda pdf_path: "# Fake Paper")
    monkeypatch.setattr(routes, "extract_orkg_contribution", lambda markdown, paper_title, source_file: _fake_orkg_contribution())
    monkeypatch.setattr(routes, "run_venture_catalyst", fake_run_venture_catalyst)

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    response = client.post("/thesis", files=_pdf_upload())
    assert response.status_code == 200
    body = response.json()
    assert body["thesis"]["paper_title"] == "Fake Paper"
    assert len(body["thesis"]["idea_dossiers"]) == 1
    assert "Idea" in body["report_markdown"]
