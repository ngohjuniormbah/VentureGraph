"""
Tests that the Venture Catalyst actually runs its independent branches
concurrently, not just that it's written with `async def`. Each fake LLM
call sleeps for a fixed delay; if `build_idea_dossier` / `run_venture_catalyst`
were secretly sequential, the wall-clock time would scale with the number of
calls. With real `asyncio.gather` concurrency, it should stay close to the
slowest single branch instead.
"""

import time
import types

import pytest

import src.agents.venture_catalyst as venture_catalyst
from src.schemas.venture import CompetitorLandscape, StartupIdea, StartupIdeas, TAMEstimate

DELAY = 0.2


class _FakeToolMessages:
    """Simulates AsyncAnthropic.messages: always gives a final text answer immediately (no tool calls)."""

    def __init__(self, delay: float):
        self._delay = delay

    async def create(self, **kwargs):
        import asyncio

        await asyncio.sleep(self._delay)
        return types.SimpleNamespace(
            stop_reason="end_turn",
            content=[types.SimpleNamespace(type="text", text="synthesized findings")],
        )


class _FakeToolClient:
    def __init__(self, delay: float):
        self.messages = _FakeToolMessages(delay)


class _FakeStructuringMessages:
    """Simulates an async Instructor client: returns a canned instance of whatever response_model was asked for."""

    def __init__(self, delay: float):
        self._delay = delay

    async def create(self, **kwargs):
        import asyncio

        await asyncio.sleep(self._delay)
        response_model = kwargs["response_model"]

        if response_model is StartupIdeas:
            return StartupIdeas(
                reasoning="r",
                paper_title="Paper",
                ideas=[
                    StartupIdea(
                        reasoning="r",
                        name=f"Idea {i}",
                        description="d",
                        target_customer="c",
                        commercial_angle="a",
                        methodology_basis="m",
                    )
                    for i in range(3)
                ],
            )
        if response_model is CompetitorLandscape:
            return CompetitorLandscape(idea_name="Idea", summary="s")
        if response_model is TAMEstimate:
            return TAMEstimate(
                idea_name="Idea",
                reasoning="r",
                calculation_code="print(1)",
                calculation_output="1\n",
                tam_usd=1.0,
            )
        raise AssertionError(f"unexpected response_model: {response_model}")


class _FakeStructuringClient:
    def __init__(self, delay: float):
        self.messages = _FakeStructuringMessages(delay)


@pytest.fixture
def fake_clients(monkeypatch):
    """Patch the Venture Catalyst's client constructors with delayed fakes."""
    monkeypatch.setattr(venture_catalyst, "get_async_anthropic_client", lambda: _FakeToolClient(DELAY))
    monkeypatch.setattr(venture_catalyst, "get_async_instructor_client", lambda: _FakeStructuringClient(DELAY))


async def test_build_idea_dossier_runs_competitors_and_tam_concurrently(fake_clients):
    """
    research_competitors and calculate_tam each make 2 fake calls (tool loop +
    structuring), so sequentially this would take ~4*DELAY. Concurrently, it
    should take ~2*DELAY (the cost of one branch, since both branches run at
    the same time).
    """
    idea = StartupIdea(
        reasoning="r", name="X", description="d", target_customer="c", commercial_angle="a", methodology_basis="m"
    )

    start = time.monotonic()
    dossier = await venture_catalyst.build_idea_dossier(idea, model="fake-model")
    elapsed = time.monotonic() - start

    assert dossier.idea.name == "X"
    assert elapsed < DELAY * 3  # well under the ~4*DELAY a sequential run would take


async def test_run_venture_catalyst_runs_all_three_ideas_concurrently(fake_clients):
    """
    Full pipeline: 1 ideation call (~1*DELAY) + 3 idea dossiers, each costing
    ~2*DELAY if run concurrently with each other. Sequentially, 3 dossiers at
    ~4*DELAY each would add ~12*DELAY on top of ideation. Concurrently, total
    should stay close to 1*DELAY (ideation) + 2*DELAY (one dossier's cost).
    """
    from src.schemas.scientific_essence import ScientificEssence

    essence = ScientificEssence(
        paper_title="P",
        problem_statement="p",
        methodology="m",
        key_results="k",
    )

    start = time.monotonic()
    dossiers = await venture_catalyst.run_venture_catalyst(essence, model="fake-model")
    elapsed = time.monotonic() - start

    assert len(dossiers) == 3
    assert elapsed < DELAY * 5  # well under the ~13*DELAY a sequential run would take
