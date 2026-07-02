# VentureGraph

VentureGraph is a project built to analyse scientific papers and transform them
into structured ORKG (Open Research Knowledge Graph) contributions and startup
opportunities. It extracts the core "scientific essence" of a paper (problem,
methodology, results), then uses that structured data to generate comparison
tables and suggest startup ideas grounded in the research.

## Project structure

```
VentureGraph/
├── main.py                        # CLI entry point (parse / orkg / compare)
├── requirements.txt
├── requirements-dev.txt           # + pytest
├── tests/
└── src/
    ├── parser/
    │   └── pdf_parser.py          # PDF -> Markdown conversion (Docling)
    ├── schemas/
    │   ├── scientific_essence.py  # Pydantic schema for a paper's essence
    │   ├── orkg.py                # Triple / ORKGContribution schema
    │   └── benchmarks.py          # MetricType / BenchmarkResult / PaperBenchmarks schema
    ├── extractor/
    │   └── essence_extractor.py   # Markdown -> ScientificEssence (heuristic stub)
    ├── agents/
    │   ├── llm_client.py          # shared Instructor + Anthropic client
    │   ├── orkg_agent.py          # Markdown -> ORKG contribution triples
    │   └── benchmark_agent.py     # Markdown -> structured benchmark results
    └── comparison/
        └── comparison_engine.py   # deterministic cross-paper comparison table
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
# PDF -> Markdown + ScientificEssence JSON (no LLM, no API key needed)
python main.py parse path/to/paper.pdf --output-dir output

# PDF -> Markdown + ORKG contribution triples (needs ANTHROPIC_API_KEY)
python main.py orkg path/to/paper.pdf --output-dir output

# Two or more PDFs -> a Markdown benchmark comparison table (needs ANTHROPIC_API_KEY)
python main.py compare path/to/paper_a.pdf path/to/paper_b.pdf --output-dir output
```

`orkg` and `compare` call Claude and require `ANTHROPIC_API_KEY` to be set
in the environment.

`parse` writes two files to `output/`:
- `<paper>.md` — the cleaned Markdown extracted from the PDF.
- `<paper>.essence.json` — a `ScientificEssence` JSON document (title,
  authors, problem statement, methodology, key results).

> Note: `essence_extractor.py` currently ships as a heuristic **stub** — it
> reliably extracts the paper title from the document structure, but leaves
> problem statement / methodology / key results as clearly-labeled
> placeholders, since real extraction of those fields requires an LLM reading
> the paper's content.

`orkg` writes `<paper>.md` and `<paper>.orkg.json` (an `ORKGContribution`:
research problem, contribution label, and a list of Subject-Predicate-Object
triples, each with a verbatim `evidence_quote`).

`compare` writes each paper's `<paper>.md` plus a single `comparison.md`
containing a Markdown table of benchmark results grouped by dataset, with
one column per paper.

## The Scientific Intelligence layer (ORKG agent + Comparison Engine)

### ORKG specialist agent (`src/agents/orkg_agent.py`)

Takes a paper's Markdown and returns an `ORKGContribution`: a research
problem plus a list of `Triple(subject, predicate, object)` statements
shaped to match ORKG's contribution model (e.g. `"Our Method" —
"evaluates on" → "ImageNet"`).

**Chain-of-Thought strategy, and why it stops hallucination:**

1. **Ordered "reason-then-answer" schema fields.** The response model
   (`ORKGContribution`) declares a free-text `reasoning` field *before*
   `triples`. Claude's tool-calling fills a JSON object's fields in
   declaration order, and generation is autoregressive — so the model is
   forced to write out, for each fact it's about to extract, which
   sentence(s) in the paper justify it, *before* it's allowed to commit to
   the structured triple list. This is the "think before you answer" CoT
   pattern, implemented through schema field order (which the model can't
   skip) rather than a "please think step by step" instruction (which it
   can).
2. **Mandatory verbatim evidence + programmatic verification.** Every
   `Triple` must include an `evidence_quote` — an exact quote from the
   source Markdown. This isn't just a prompt request: after the LLM
   response comes back, `_filter_ungrounded_triples()` checks each quote
   against the real source text (whitespace/case-normalized substring
   match) and drops any triple whose quote can't be found. A fabricated
   fact can still *sound* plausible; it cannot survive this check, because
   its "supporting quote" simply isn't in the paper. See
   `tests/test_orkg_agent.py` for a test that proves this: a fabricated
   triple is dropped while a genuinely-grounded one is kept.
3. Temperature is set to `0` for both agents to keep extraction as
   deterministic as possible, and the system prompt explicitly tells the
   model to prefer omitting a fact over guessing at one.

### Comparison Engine (`src/comparison/comparison_engine.py`)

Two or more papers each get reduced to a `PaperBenchmarks` object (via
`src/agents/benchmark_agent.py`, using the same reasoning-field +
evidence-quote-verification strategy as the ORKG agent) — a list of
`(dataset, metric_type, value)` results. `comparison_engine.py` then builds
the cross-paper Markdown table with **plain, deterministic Python and no
LLM call**, since grouping and formatting already-structured data doesn't
need another model in the loop — it's mechanical and needs to be
reproducible.

**Handling different units of measurement (e.g. Accuracy vs. F1-score):**
the engine does **not** attempt to convert one into the other. Accuracy and
F1 measure different things (F1 is the harmonic mean of precision/recall on
a chosen positive class; accuracy is overall correctness) and are only
mutually derivable if you also have the full confusion matrix, which a
benchmark table essentially never gives you — silently converting or
averaging them would be inventing a number neither paper reported. Instead,
for every dataset row the engine:

1. Groups all reported results for that dataset across every input paper.
2. Shows each paper's column with *every* metric that paper reported for
   that dataset, labeled by type (e.g. `accuracy: 94.3%`, `f1: 0.87`) — so
   nothing is dropped or merged.
3. Computes the set of metric types reported for that dataset across all
   papers. If more than one metric type appears, the row is flagged with
   `⚠` and a footnote is appended underneath the table naming the
   mismatched metrics and explaining why they aren't numerically
   comparable.

Example output (see `tests/test_comparison_engine.py` for the underlying
test cases):

```markdown
| Dataset | FastNet (2024) | DeepQA (2023) |
|---|---|---|
| CoQA | — | f1: 0.81 |
| ImageNet | accuracy: 94.3% | — |
| SQuAD 2.0 ⚠ | accuracy: 88.1% | f1: 0.87 |

**⚠ Metric mismatches:**
- **SQuAD 2.0**: papers report different metric types (accuracy, f1). These
  are not numerically comparable without additional information (e.g.
  Accuracy and F1-score measure different things and can't be converted
  between each other without the underlying confusion matrix) - values are
  shown side-by-side above, not merged or converted.
```

### On Instructor vs. LangChain, and the model version

`src/agents/llm_client.py` uses **Instructor** wrapping the official
`anthropic` client, rather than LangChain: Instructor's entire job is
turning a Pydantic model into a validated, retried structured output from a
single API call, which is exactly what both agents need and nothing more —
LangChain would add a much larger dependency and abstraction surface (chains,
retrievers, memory) for a use case that doesn't need any of that.

The task brief mentions Claude 3.5 Sonnet; `DEFAULT_MODEL` instead defaults
to the current Sonnet model (3.5 Sonnet has since been superseded by newer,
more capable snapshots), overridable via the `VENTUREGRAPH_MODEL`
environment variable if you need to pin a specific snapshot for
reproducibility.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The comparison-engine and ORKG-agent-grounding tests run with no API key
and no network access (they use hand-built fixtures / a fake LLM client),
since they test deterministic logic, not live model behavior.
