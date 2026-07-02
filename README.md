# VentureGraph

VentureGraph is a project built to analyse scientific papers and transform them
into structured ORKG (Open Research Knowledge Graph) contributions and startup
opportunities. It extracts the core "scientific essence" of a paper (problem,
methodology, results), then uses that structured data to generate comparison
tables and suggest startup ideas grounded in the research.

## Project structure

```
VentureGraph/
├── main.py                       # CLI entry point
├── requirements.txt
└── src/
    ├── parser/
    │   └── pdf_parser.py         # PDF -> Markdown conversion (Docling)
    ├── schemas/
    │   └── scientific_essence.py # Pydantic schema for a paper's essence
    └── extractor/
        └── essence_extractor.py  # Markdown -> ScientificEssence (heuristic stub)
```

## Why Docling for PDF parsing?

Scientific papers are multi-column, table-heavy, and often formula-heavy.
Docling was chosen over simpler extractors (PyPDF2, pdfminer) because it uses
a trained layout model to understand page structure (so it doesn't interleave
text from side-by-side columns), a dedicated table-structure model
(TableFormer) to reconstruct real Markdown tables instead of whitespace blobs,
and explicit formula/code-block detection so equations survive as LaTeX in the
output. It also runs fully locally, so unpublished research never leaves your
machine. See the docstring in `src/parser/pdf_parser.py` for the full
rationale.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: on its first run, Docling downloads its layout/table-structure
> model weights from Hugging Face (cached locally afterwards), so the
> machine running it needs outbound internet access at least once.

## Usage

```bash
python main.py path/to/paper.pdf --output-dir output
```

This writes two files to `output/`:
- `<paper>.md` — the cleaned Markdown extracted from the PDF.
- `<paper>.essence.json` — a `ScientificEssence` JSON document (title,
  authors, problem statement, methodology, key results).

> Note: `essence_extractor.py` currently ships as a heuristic **stub** — it
> reliably extracts the paper title from the document structure, but leaves
> problem statement / methodology / key results as clearly-labeled
> placeholders, since real extraction of those fields requires an LLM reading
> the paper's content. That LLM-based extraction step is the natural next
> task on top of this scaffolding.
