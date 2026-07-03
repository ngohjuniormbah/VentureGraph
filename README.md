# VentureGraph

VentureGraph is a project built to analyse scientific papers and transform them
into structured ORKG (Open Research Knowledge Graph) contributions and startup
opportunities. It extracts the core "scientific essence" of a paper (problem,
methodology, results), then uses that structured data to generate comparison
tables and suggest startup ideas grounded in the research.

## Project structure

```
VentureGraph/
├── main.py                        # CLI entry point (parse / orkg / compare / thesis)
├── app_api.py                     # FastAPI backend (deploy: Railway, via Dockerfile)
├── app_gui.py                     # Streamlit frontend (deploy: Hugging Face Spaces)
├── Dockerfile                     # backend image: system libs + requirements-api.txt
├── deployment_guide.md            # Railway (backend) + Hugging Face Spaces (frontend) setup
├── .env.example                   # template for local env vars (.env is gitignored)
├── requirements.txt                # core pipeline deps (Docling, Instructor, Anthropic, Tavily)
├── requirements-api.txt           # + fastapi, uvicorn, python-multipart
├── requirements-gui.txt           # streamlit + requests only (no Docling - thin client)
├── requirements-dev.txt           # + pytest, pytest-asyncio
├── pytest.ini
├── tests/
└── src/
    ├── parser/
    │   └── pdf_parser.py          # PDF -> Markdown conversion (Docling)
    ├── schemas/
    │   ├── scientific_essence.py  # Pydantic schema for a paper's essence
    │   ├── orkg.py                # Triple / ORKGContribution schema
    │   ├── benchmarks.py          # MetricType / BenchmarkResult / PaperBenchmarks schema
    │   └── venture.py             # StartupIdea(s) / CompetitorLandscape / TAMEstimate / InvestmentThesis
    ├── extractor/
    │   └── essence_extractor.py   # Markdown -> ScientificEssence (heuristic stub)
    ├── agents/
    │   ├── llm_client.py          # shared sync/async Instructor + Anthropic clients
    │   ├── orkg_agent.py          # Markdown -> ORKG contribution triples
    │   ├── benchmark_agent.py     # Markdown -> structured benchmark results
    │   ├── tool_loop.py           # generic async ReAct-style tool-use loop
    │   └── venture_catalyst.py    # Startup Ideator + Market Intelligence + TAM Calculator
    ├── tools/
    │   ├── web_search.py          # Tavily / Brave search providers (the `web_search` tool)
    │   └── code_executor.py       # sandboxed subprocess Python execution (the `execute_python` tool)
    ├── comparison/
    │   └── comparison_engine.py   # deterministic cross-paper comparison table
    ├── report/
    │   └── investment_thesis.py   # deterministic final Investment Thesis Markdown renderer
    └── api/
        ├── routes.py              # FastAPI routes wrapping the pipeline (used by app_api.py)
        └── schemas.py             # API response models
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

# PDF -> full Investment Thesis: ORKG + 3 startup ideas + competitors + TAM
# (needs ANTHROPIC_API_KEY, and TAVILY_API_KEY or BRAVE_API_KEY for live search)
python main.py thesis path/to/paper.pdf --output-dir output
```

`orkg`, `compare`, and `thesis` call Claude and require `ANTHROPIC_API_KEY`
to be set in the environment. `thesis` additionally needs a search API key
(`TAVILY_API_KEY` by default, or `BRAVE_API_KEY` with `SEARCH_PROVIDER=brave`)
for its market-research and TAM-calculation steps.

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

`thesis` writes `<paper>.md`, `<paper>.thesis.json` (the full structured
`InvestmentThesis`), and `<paper>.thesis.md` (the human-readable report -
see the Venture Catalyst section below).

## Running the demo (API + dashboard)

Besides the CLI, the same pipeline is exposed as a web demo: a FastAPI
backend (`app_api.py`) and a Streamlit dashboard (`app_gui.py`) that calls
it over HTTP. Locally:

```bash
# Terminal 1: backend
pip install -r requirements-api.txt
export ANTHROPIC_API_KEY=...  # + TAVILY_API_KEY, etc. - see .env.example
uvicorn app_api:app --reload --port 8000

# Terminal 2: frontend
pip install -r requirements-gui.txt
streamlit run app_gui.py
```

The dashboard has one tab per CLI subcommand (Parse, ORKG, Compare,
Investment Thesis) and never needs an LLM/search API key itself - it only
needs to know the backend's URL (`VENTUREGRAPH_API_URL`, defaulting to
`http://localhost:8000`). This split is deliberate: the backend is where
Docling, the LLM calls, and every API key live; the frontend is a thin
client, which is what makes it possible to deploy the two on different
platforms - see **`deployment_guide.md`** for step-by-step instructions on
deploying the backend to **Railway** (via the included `Dockerfile`) and
the frontend to **Hugging Face Spaces**, including exactly where each API
key gets set on each platform.

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

## The Venture Catalyst (Startup Ideator + Market Intelligence + TAM Calculator)

`src/agents/venture_catalyst.py` turns a paper's methodology into a full
Investment Thesis, in three steps:

1. **Startup Ideator** (`generate_startup_ideas`) — a single-shot Instructor
   call that reads `ScientificEssence.methodology` and returns exactly three
   `StartupIdea`s, each with a `methodology_basis` quote grounding it in the
   actual paper text (same reasoning-field-first + verbatim-quote pattern as
   the ORKG agent, described above).
2. **Market Intelligence** (`research_competitors`) — for each idea, runs an
   agentic tool-use loop (`src/agents/tool_loop.py`) that gives Claude a
   `web_search` tool and lets it decide how many searches to run and with
   what queries, then structures the findings into a `CompetitorLandscape`.
3. **TAM Calculator** (`calculate_tam`) — for each idea, runs the same kind
   of tool loop, but with *two* tools: `web_search` (to find market-size
   figures/industry reports) and `execute_python` (to compute the TAM from
   whatever numbers it finds), then structures the result into a
   `TAMEstimate` that keeps the exact code executed and its output.

### Why the "Search" integration, not just asking Claude

Claude's parametric knowledge has a training cutoff, but competitor
landscapes and market-size figures are exactly the kind of fact that goes
stale continuously - a list of competitors from a training-data snapshot
could be a year (or several product pivots) out of date, and the model has
no way to know that on its own. Simply asking "who are the competitors" or
"what's the market size" and trusting the answer would silently launder
possibly-stale or invented numbers as if they were current.

The fix is **tool use**, not prompting: `src/tools/web_search.py` gives
Claude an actual `web_search` tool backed by a live search API (Tavily by
default, since it's built for LLM-agent search and returns clean
pre-summarized snippets; Brave Search as an alternative via
`SEARCH_PROVIDER=brave`). Every call the model makes to this tool is a real
HTTPS request made *at run time* - not anything baked into the model - and
the actual retrieved titles/URLs/snippets are what get fed back into the
conversation. Claude's final synthesis is grounded in that fresh text, and
every `CompetitorLandscape`/`TAMEstimate` keeps the source URLs that were
actually used (`sources`/`data_sources`), so a human can click through and
verify the numbers instead of trusting the model's say-so. The provider
construction is also lazy (see `_web_search_tool_spec` in
`venture_catalyst.py`) - it only runs, and only requires an API key, the
moment the model actually decides to search, not just because the tool was
made available to it.

The agentic loop that makes this possible lives in
`src/agents/tool_loop.py`: Claude is given the tool definitions and decides
for itself, turn by turn, whether to call one, read the result, call
another (e.g. refine a search query), or give a final answer - this is why
that loop is implemented against the raw `anthropic` tool-use API rather
than through Instructor (which is built for single-shot structured
extraction, not multi-turn tool decisions). Once the loop produces a final
free-text answer, a short *second* Instructor call reformats it into the
strict `CompetitorLandscape`/`TAMEstimate` schema - explicitly instructed to
only restructure what's already there, not add anything, so this final
formatting step can't reintroduce hallucination after the grounded loop.

### The TAM Calculator's code-execution tool

`src/tools/code_executor.py` gives Claude an `execute_python` tool: instead
of doing TAM arithmetic "in its head" (where unit mistakes are common and
invisible), it writes a short script and we actually run it, so the
reported figure is the real, verifiable output of that script rather than
a number Claude asserts. It runs in a separate OS subprocess
(`asyncio.create_subprocess_exec`, no shell involved) under `python3 -I`
with a wall-clock timeout - process-level isolation plus a timeout, not a
full security sandbox (the subprocess still shares the host's filesystem/
network access). That's an acceptable trust boundary for a single-user
local tool computing arithmetic; a shared/multi-tenant deployment should run
this inside a locked-down container or VM instead. `TAMEstimate` keeps both
`calculation_code` and `calculation_output`, so the arithmetic behind every
TAM figure is fully auditable in the final report.

### Why this module is fully asynchronous

A single idea's dossier needs two independent multi-turn agentic
conversations (competitor research, TAM calculation), and the Ideator
produces three ideas - so a fully sequential run would be up to six full
tool-use conversations, each several Claude round trips plus live search/
code-execution calls, stacked one after another. None of those six
branches read each other's output, so there's no correctness reason to
serialize them:

- `build_idea_dossier` runs `research_competitors` and `calculate_tam`
  concurrently via `asyncio.gather`.
- `run_venture_catalyst` runs all three ideas' `build_idea_dossier` calls
  concurrently via `asyncio.gather`.
- `main.py`'s `thesis` command additionally runs the ORKG extraction
  (Scientific Intelligence layer) concurrently with the entire Venture
  Catalyst pipeline, since neither depends on the other's output (the
  synchronous `extract_orkg_contribution` call is dispatched via
  `asyncio.to_thread` so it doesn't block the event loop the rest of the
  pipeline runs on).

The net effect: wall-clock time for the whole `thesis` command is roughly
the cost of the *slowest single branch*, not the sum of ~7 sequential LLM
conversations. `tests/test_venture_catalyst_concurrency.py` proves this is
real concurrency (not just `async def` syntax) by using fake clients with
artificial delays and asserting the total elapsed time stays close to one
branch's cost rather than scaling with the number of calls made.

### The final Investment Thesis report

`src/report/investment_thesis.py` deterministically combines everything the
three VentureGraph layers produced - the paper's `ScientificEssence`, its
`ORKGContribution` (Subject/Predicate/Object table), and the Venture
Catalyst's `IdeaDossier`s (idea, competitors table, TAM calculation with
code and output) - into one Markdown report, written to
`<paper>.thesis.md` by `main.py`'s `thesis` command (and as structured JSON
to `<paper>.thesis.json`). See `tests/test_investment_thesis_report.py` for
example output.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

All tests run with no API key and no network access - they use hand-built
fixtures and fake LLM/tool clients, since they test deterministic logic and
control flow (grounding filters, table/report formatting, the tool loop's
turn-taking, and the Venture Catalyst's concurrency), not live model
behavior.

## Environment variables

| Variable | Required for | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | `orkg`, `compare`, `thesis` | Standard Anthropic API key. |
| `TAVILY_API_KEY` | `thesis` (default search provider) | From https://tavily.com |
| `BRAVE_API_KEY` | `thesis`, only if `SEARCH_PROVIDER=brave` | From https://brave.com/search/api |
| `SEARCH_PROVIDER` | optional | `tavily` (default) or `brave`. |
| `VENTUREGRAPH_MODEL` | optional | Overrides the default Claude model id for all agents. |
| `CORS_ALLOW_ORIGINS` | `app_api.py`, optional | Comma-separated allowed origins for the API's CORS policy; set to your deployed frontend's exact URL in production instead of the `"*"` default. |
| `VENTUREGRAPH_API_URL` | `app_gui.py` | Base URL of the backend API the Streamlit dashboard calls; defaults to `http://localhost:8000`. |

See `.env.example` for a copyable template of all of the above.
