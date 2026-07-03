"""
Pydantic schema for a paper's reported benchmark results.

This is the structured intermediate representation the Comparison Engine
runs on: instead of comparing raw Markdown text, we first pull each paper's
(dataset, metric, value) results into this schema, then compare the
structured results across papers. See `src/comparison/comparison_engine.py`
for how mismatched metric types (e.g. one paper reports Accuracy, another
reports F1-score) are handled once we get here.
"""

from enum import Enum

from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """
    A controlled vocabulary of common evaluation metrics.

    Papers describe the same metric in many ways ("Acc.", "Top-1 Accuracy",
    "accuracy (%)"). Normalizing to this enum (while keeping the original
    wording in `BenchmarkResult.metric_name_raw`) is what lets the
    Comparison Engine detect "these two papers used different metrics for
    the same dataset" with a simple set comparison, instead of fuzzy string
    matching.
    """

    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    AUC = "auc"
    BLEU = "bleu"
    ROUGE = "rouge"
    RMSE = "rmse"
    MAE = "mae"
    PERPLEXITY = "perplexity"
    OTHER = "other"


class BenchmarkResult(BaseModel):
    """
    A single reported (dataset, metric, value) result from a paper.

    Data flow:
        Produced by `src.agents.benchmark_agent.extract_paper_benchmarks`,
        one per result table/sentence the LLM finds in the paper. Collected
        into the `benchmarks` list on `PaperBenchmarks`, which
        `comparison_engine.build_comparison_table` groups by dataset across
        papers.
    """

    dataset: str = Field(
        ..., description="Name of the dataset/benchmark the result was measured on, e.g. 'ImageNet', 'SQuAD 2.0'."
    )
    metric_type: MetricType = Field(
        ...,
        description="The metric family this result belongs to, normalized to the controlled vocabulary.",
    )
    metric_name_raw: str = Field(
        ..., description="The metric name exactly as written in the paper, e.g. 'Top-1 Accuracy', 'F1-score'."
    )
    value: float = Field(..., description="The numeric value of the result, as reported (e.g. 94.3 or 0.943).")
    value_raw: str = Field(
        ..., description="The value exactly as printed in the paper, including any unit, e.g. '94.3%', '0.87'."
    )
    evidence_quote: str = Field(
        ...,
        description=(
            "A verbatim quote from the source Markdown (e.g. the table row or "
            "sentence) that this result was read from. Must be an exact "
            "substring of the source text, used to verify grounding."
        ),
    )


class PaperBenchmarks(BaseModel):
    """
    All benchmark results extracted from a single paper.

    Data flow:
        One `PaperBenchmarks` instance is built per input paper by
        `extract_paper_benchmarks`. `main.py`'s `compare` command collects
        one of these per PDF into a list and passes that list to
        `comparison_engine.build_comparison_table` to produce the final
        cross-paper Markdown comparison.
    """

    reasoning: str = Field(
        ...,
        description=(
            "Step-by-step reasoning written BEFORE the benchmarks list: "
            "identify each results table/sentence in the source text and "
            "quote it, before turning it into a structured BenchmarkResult. "
            "Omit anything you can't find explicit numeric support for."
        ),
    )
    paper_title: str = Field(..., description="The title of the paper these benchmarks were extracted from.")
    benchmarks: list[BenchmarkResult] = Field(default_factory=list)
    source_file: str | None = Field(default=None, description="Filename/path of the source PDF.")
