"""
Tests for the deterministic Comparison Engine.

These tests build `PaperBenchmarks` objects by hand (no LLM call involved)
to verify the table-building and metric-mismatch logic in isolation, since
`comparison_engine.py` is pure Python with no network dependency.
"""

import pytest

from src.comparison.comparison_engine import build_comparison_table
from src.schemas.benchmarks import BenchmarkResult, MetricType, PaperBenchmarks


def _result(dataset: str, metric_type: MetricType, value: float, value_raw: str) -> BenchmarkResult:
    return BenchmarkResult(
        dataset=dataset,
        metric_type=metric_type,
        metric_name_raw=metric_type.value,
        value=value,
        value_raw=value_raw,
        evidence_quote="irrelevant for this test",
    )


def test_requires_at_least_two_papers():
    """A single paper can't be "compared" - the function should reject it."""
    lone_paper = PaperBenchmarks(reasoning="r", paper_title="Solo Paper", benchmarks=[])
    with pytest.raises(ValueError):
        build_comparison_table([lone_paper])


def test_same_metric_type_is_shown_without_mismatch_marker():
    """When both papers report Accuracy on the same dataset, no ⚠ marker or footnote should appear."""
    paper_a = PaperBenchmarks(
        reasoning="r",
        paper_title="Paper A",
        benchmarks=[_result("ImageNet", MetricType.ACCURACY, 94.3, "94.3%")],
    )
    paper_b = PaperBenchmarks(
        reasoning="r",
        paper_title="Paper B",
        benchmarks=[_result("ImageNet", MetricType.ACCURACY, 91.0, "91.0%")],
    )

    table = build_comparison_table([paper_a, paper_b])

    assert "| ImageNet |" in table
    assert "⚠" not in table
    assert "accuracy: 94.3%" in table
    assert "accuracy: 91.0%" in table


def test_different_metric_types_are_flagged_not_converted():
    """
    The key requirement: Paper A reports Accuracy, Paper B reports F1-score
    on the same dataset. The engine must show both raw values side-by-side,
    mark the row as mismatched, and explain why in a footnote - it must
    NOT attempt to convert one metric into the other.
    """
    paper_a = PaperBenchmarks(
        reasoning="r",
        paper_title="Paper A",
        benchmarks=[_result("SQuAD 2.0", MetricType.ACCURACY, 88.1, "88.1%")],
    )
    paper_b = PaperBenchmarks(
        reasoning="r",
        paper_title="Paper B",
        benchmarks=[_result("SQuAD 2.0", MetricType.F1, 0.87, "0.87")],
    )

    table = build_comparison_table([paper_a, paper_b])

    # Row is flagged.
    assert "| SQuAD 2.0 ⚠ |" in table
    # Both raw values are preserved verbatim, not merged/converted.
    assert "accuracy: 88.1%" in table
    assert "f1: 0.87" in table
    # A footnote explains the mismatch by name.
    assert "Metric mismatches" in table
    assert "SQuAD 2.0" in table.split("Metric mismatches")[1]
    assert "accuracy" in table.split("Metric mismatches")[1]
    assert "f1" in table.split("Metric mismatches")[1]


def test_missing_dataset_shown_as_em_dash():
    """If only one paper reports a dataset, the other paper's cell should be '—', not blank or an error."""
    paper_a = PaperBenchmarks(
        reasoning="r",
        paper_title="Paper A",
        benchmarks=[_result("CIFAR-10", MetricType.ACCURACY, 99.0, "99.0%")],
    )
    paper_b = PaperBenchmarks(reasoning="r", paper_title="Paper B", benchmarks=[])

    table = build_comparison_table([paper_a, paper_b])

    assert "| CIFAR-10 | accuracy: 99.0% | — |" in table


def test_multiple_metrics_from_same_paper_on_same_dataset_are_both_shown():
    """A single paper reporting both Accuracy and F1 on one dataset should list both in its cell."""
    paper_a = PaperBenchmarks(
        reasoning="r",
        paper_title="Paper A",
        benchmarks=[
            _result("GLUE", MetricType.ACCURACY, 90.0, "90.0%"),
            _result("GLUE", MetricType.F1, 0.89, "0.89"),
        ],
    )
    paper_b = PaperBenchmarks(
        reasoning="r",
        paper_title="Paper B",
        benchmarks=[_result("GLUE", MetricType.F1, 0.85, "0.85")],
    )

    table = build_comparison_table([paper_a, paper_b])

    assert "accuracy: 90.0%; f1: 0.89" in table
