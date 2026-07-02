#!/usr/bin/env python3
"""
VentureGraph CLI.

Usage:
    python main.py path/to/paper.pdf
    python main.py path/to/paper.pdf --output-dir output

Data flow for a single run:
    1. Argument parsing (this file) resolves which PDF to process and
       where outputs should go.
    2. `src.parser.pdf_parser.convert_pdf_to_markdown` runs Docling on the
       PDF and returns clean Markdown (tables/formulas preserved).
    3. That Markdown is written to `<output-dir>/<pdf-stem>.md`.
    4. `src.extractor.essence_extractor.extract_essence_stub` reads the
       Markdown and produces a `ScientificEssence` Pydantic instance.
    5. That instance is serialized with `.model_dump_json()` and written to
       `<output-dir>/<pdf-stem>.essence.json`.
    6. Both file paths are printed so the user can find the results.
"""

import argparse
import sys
from pathlib import Path

from src.extractor.essence_extractor import extract_essence_stub
from src.parser.pdf_parser import convert_pdf_to_markdown
from src.schemas.scientific_essence import ScientificEssence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the VentureGraph CLI.

    Data flow:
        Takes the raw CLI argument list (or `sys.argv` if `argv` is None)
        and returns a populated `argparse.Namespace` with `pdf_path` and
        `output_dir`, which `main()` uses to drive the rest of the pipeline.

    Args:
        argv: Optional explicit argument list, mainly for testing. Defaults
            to reading from `sys.argv` when None.

    Returns:
        Parsed arguments as an `argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert a scientific PDF into clean Markdown and extract its "
            "Scientific Essence as structured JSON."
        )
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the input scientific PDF.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write the Markdown and JSON outputs to (default: ./output).",
    )
    return parser.parse_args(argv)


def run(pdf_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """
    Run the full PDF -> Markdown -> ScientificEssence pipeline for one file.

    Data flow:
        1. Converts `pdf_path` to Markdown via Docling
           (`convert_pdf_to_markdown`).
        2. Writes that Markdown to `<output_dir>/<pdf stem>.md`.
        3. Extracts a `ScientificEssence` from the Markdown
           (`extract_essence_stub`).
        4. Writes the essence as pretty-printed JSON to
           `<output_dir>/<pdf stem>.essence.json`.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory where the two output files will be written
            (created if it doesn't already exist).

    Returns:
        A tuple `(markdown_path, json_path)` pointing to the two files that
        were written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown = convert_pdf_to_markdown(pdf_path)
    markdown_path = output_dir / f"{pdf_path.stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    essence: ScientificEssence = extract_essence_stub(markdown, source_file=str(pdf_path))
    json_path = output_dir / f"{pdf_path.stem}.essence.json"
    json_path.write_text(essence.model_dump_json(indent=2), encoding="utf-8")

    return markdown_path, json_path


def main() -> None:
    """
    CLI entry point: parse arguments, run the pipeline, report results.

    Data flow:
        Calls `parse_args()` to get `pdf_path`/`output_dir`, validates the
        PDF exists, calls `run()` to do the actual conversion + extraction,
        and prints the resulting file paths for the user. Exits with a
        non-zero status and an error message if the PDF is missing.
    """
    args = parse_args()

    if not args.pdf_path.is_file():
        print(f"Error: PDF not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    markdown_path, json_path = run(args.pdf_path, args.output_dir)

    print(f"Markdown written to:  {markdown_path}")
    print(f"Essence JSON written to: {json_path}")


if __name__ == "__main__":
    main()
