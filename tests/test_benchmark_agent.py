"""
Tests for the benchmark agent's grounding filter (see test_orkg_agent.py for
the equivalent test on the ORKG agent - same strategy, same reasoning).
"""

from src.agents.benchmark_agent import extract_paper_benchmarks
from src.schemas.benchmarks import BenchmarkResult, MetricType, PaperBenchmarks

SOURCE_MARKDOWN = """\
# FastNet: A Widget Classifier

| Dataset | Accuracy |
|---|---|
| WidgetBench | 94.3% |
"""


class _FakeMessages:
    def __init__(self, response: PaperBenchmarks):
        self._response = response

    def create(self, **kwargs):
        return self._response


class _FakeInstructorClient:
    def __init__(self, response: PaperBenchmarks):
        self.messages = _FakeMessages(response)


def test_grounded_result_kept_fabricated_result_dropped():
    canned_response = PaperBenchmarks(
        reasoning="Found one results table row.",
        paper_title="FastNet: A Widget Classifier",
        benchmarks=[
            BenchmarkResult(
                dataset="WidgetBench",
                metric_type=MetricType.ACCURACY,
                metric_name_raw="Accuracy",
                value=94.3,
                value_raw="94.3%",
                evidence_quote="| WidgetBench | 94.3% |",
            ),
            BenchmarkResult(
                dataset="FakeBench",
                metric_type=MetricType.F1,
                metric_name_raw="F1",
                value=0.99,
                value_raw="0.99",
                # Fabricated - not present anywhere in SOURCE_MARKDOWN.
                evidence_quote="| FakeBench | 0.99 F1 |",
            ),
        ],
    )
    fake_client = _FakeInstructorClient(canned_response)

    result = extract_paper_benchmarks(SOURCE_MARKDOWN, source_file="fastnet.pdf", client=fake_client)

    assert len(result.benchmarks) == 1
    assert result.benchmarks[0].dataset == "WidgetBench"
    assert result.source_file == "fastnet.pdf"
