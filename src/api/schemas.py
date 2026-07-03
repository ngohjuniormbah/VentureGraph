"""
Response schemas for the VentureGraph API.

These wrap the existing pipeline schemas (`ScientificEssence`,
`ORKGContribution`, `PaperBenchmarks`, `InvestmentThesis`) with the extra
bits an HTTP client needs alongside them - the raw Markdown that was
extracted, and/or a filename - rather than inventing a parallel set of API-
only data models.
"""

from pydantic import BaseModel

from src.schemas.benchmarks import PaperBenchmarks
from src.schemas.orkg import ORKGContribution
from src.schemas.scientific_essence import ScientificEssence
from src.schemas.venture import InvestmentThesis


class ParseResponse(BaseModel):
    """Response for `POST /parse`."""

    markdown: str
    essence: ScientificEssence


class OrkgResponse(BaseModel):
    """Response for `POST /orkg`."""

    markdown: str
    contribution: ORKGContribution


class ComparePaperResult(BaseModel):
    """One paper's contribution to a `POST /compare` response."""

    filename: str
    benchmarks: PaperBenchmarks


class CompareResponse(BaseModel):
    """Response for `POST /compare`."""

    papers: list[ComparePaperResult]
    comparison_markdown: str


class ThesisResponse(BaseModel):
    """Response for `POST /thesis`."""

    thesis: InvestmentThesis
    report_markdown: str


class HealthResponse(BaseModel):
    """Response for `GET /health`."""

    status: str
