"""
PDF -> Markdown conversion for scientific papers.

Why Docling?
------------
VentureGraph's whole pipeline depends on getting a *faithful* text
representation of a scientific PDF, because everything downstream (essence
extraction, ORKG contribution generation, startup-opportunity reasoning)
reads that text, not the original PDF. Three things make Docling the right
choice for that job over a plain-text extractor (PyPDF2, pdfminer) or a
generic OCR pipeline:

1. Layout-aware parsing: Docling uses a trained layout model (based on IBM
   Research's DocLayNet/TableFormer work) to understand the *structure* of
   a page - headings, paragraphs, columns, captions - instead of just
   dumping characters in reading order. Scientific papers are almost always
   multi-column, so naive extractors frequently interleave text from the
   left and right columns into garbage.

2. Table structure recognition (TableFormer): Docling reconstructs tables
   as actual rows/columns and can export them as Markdown tables (or
   HTML/CSV), rather than as a flat blob of whitespace-separated numbers.
   Since "Key Results" in a paper are very often reported in tables, this
   directly protects the data our Pydantic schema needs to capture.

3. Formula and code-block handling: Docling has explicit classifiers for
   formulas, code blocks, and captions, and can preserve formulas as LaTeX
   in the exported Markdown. This matters for methodology-heavy papers
   (e.g. ML, physics, statistics) where the formula *is* the method.

4. It's local and open-source (no per-page API cost, no sending
   unpublished research to a third-party OCR service), which suits a tool
   meant to run over a researcher's own paper stash.

The alternative most people reach for, `PyMuPDF`/`pdfplumber`, is faster
but gives you raw text/coordinates with no semantic structure - you'd have
to reimplement table and reading-order detection yourself. `Unstructured`
is a reasonable second choice but its table/formula fidelity is generally
a step behind Docling's specialized models. Docling was built specifically
for the "PDF -> structured Markdown for LLM ingestion" use case that
VentureGraph needs, so we standardize on it here.
"""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def _build_converter(enable_ocr: bool) -> DocumentConverter:
    """
    Build a Docling `DocumentConverter` configured for scientific papers.

    Data flow:
        Constructs `PdfPipelineOptions` with table-structure recognition
        always on (papers are full of results tables) and OCR toggled by
        `enable_ocr`, then wraps them in a `PdfFormatOption` so
        `DocumentConverter` applies them specifically to PDF inputs.

    Why OCR defaults to off: the vast majority of scientific papers
    (arXiv, journal PDFs, conference proceedings) are "born-digital" -
    their text is embedded, not a scanned image - so OCR is unnecessary
    work that also requires downloading OCR model weights from the
    network on first use. Callers processing scanned/photographed papers
    can pass `enable_ocr=True` to turn OCR back on for those documents.

    Args:
        enable_ocr: Whether to run OCR (needed for scanned/image-only PDFs).

    Returns:
        A `DocumentConverter` ready to convert PDF files.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = enable_ocr
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def convert_pdf_to_markdown(pdf_path: str | Path, enable_ocr: bool = False) -> str:
    """
    Convert a scientific PDF into a clean Markdown string.

    Data flow:
        1. `pdf_path` (a path to a .pdf file on disk) is handed to Docling's
           `DocumentConverter`, which runs layout analysis, optional OCR,
           table-structure recognition, and formula detection on every page.
        2. Docling returns a `ConversionResult` whose `.document` attribute
           is a structured `DoclingDocument` - an in-memory tree of the
           paper's headings, paragraphs, tables, and formulas.
        3. We call `.export_to_markdown()` on that document, which walks
           the tree and serializes it to Markdown: `#`/`##` headings for
           section titles, GitHub-style pipe tables for tables, and LaTeX
           (`$...$` / `$$...$$`) for formulas.
        4. The resulting Markdown string is the single artifact that every
           later stage of VentureGraph (essence extraction, ORKG mapping,
           startup-idea generation) consumes as its source of truth.

    Args:
        pdf_path: Path to the input PDF file (str or pathlib.Path).
        enable_ocr: Set True for scanned/image-only PDFs that have no
            embedded text layer. Defaults to False since most scientific
            papers are born-digital and OCR would just add latency (and a
            one-time OCR-model download) for no benefit.

    Returns:
        The full document rendered as a Markdown string, with headings,
        tables, and formulas preserved.

    Raises:
        FileNotFoundError: If `pdf_path` does not point to an existing file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    converter = _build_converter(enable_ocr=enable_ocr)
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown()
    return markdown


def convert_pdf_to_markdown_file(pdf_path: str | Path, output_path: str | Path) -> Path:
    """
    Convert a PDF to Markdown and write the result to disk.

    Data flow:
        1. Delegates the actual PDF -> Markdown conversion to
           `convert_pdf_to_markdown`.
        2. Writes the returned Markdown string to `output_path`, creating
           any missing parent directories along the way, so callers (e.g.
           `main.py`) don't need their own file-handling boilerplate.

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Path where the resulting `.md` file should be written.

    Returns:
        The `Path` the Markdown was written to (same as `output_path`,
        resolved to a `Path` object).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    markdown = convert_pdf_to_markdown(pdf_path)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path
