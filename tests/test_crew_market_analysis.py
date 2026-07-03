from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from crewai.types.usage_metrics import UsageMetrics

from crewai_groq_demo.crew import NO_RESEARCH_TEXT, MarketAnalysisResult, run_market_analysis
from crewai_groq_demo.models import MarketAnalysis, MarketNiche
from crewai_groq_demo.settings import get_settings


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _market_analysis_kickoff_returning(analysis: MarketAnalysis, usage: UsageMetrics) -> Any:
    def _side_effect(crew_self: Any, inputs: dict[str, Any] | None = None) -> str:
        crew_self.usage_metrics = usage
        return analysis.model_dump_json()

    return _side_effect


def _sample_analysis(niche_count: int = 1) -> MarketAnalysis:
    return MarketAnalysis(
        niches=[
            MarketNiche(
                niche=f"Niche {i}",
                audience=f"Audience {i}",
                evidence=[f"Some finding {i}: https://example.com/{i}"],
            )
            for i in range(niche_count)
        ]
    )


def test_run_market_analysis_returns_analysis_and_usage() -> None:
    usage = UsageMetrics(prompt_tokens=100, completion_tokens=40, total_tokens=140)
    analysis = _sample_analysis()

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_market_analysis_kickoff_returning(analysis, usage),
    ):
        result = run_market_analysis("user prompt", "some research findings")

    assert result == MarketAnalysisResult(analysis=analysis, usage=usage)


def test_run_market_analysis_passes_research_result_to_task_inputs() -> None:
    usage = UsageMetrics()
    analysis = _sample_analysis()

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_market_analysis_kickoff_returning(analysis, usage),
    ) as mocked_kickoff:
        run_market_analysis("user prompt", "specific research findings")

    inputs = mocked_kickoff.call_args.kwargs["inputs"]
    assert inputs == {
        "user_prompt": "user prompt",
        "research_result": "specific research findings",
    }


def test_run_market_analysis_forces_empty_niches_when_no_research_was_run() -> None:
    usage = UsageMetrics()
    analysis = _sample_analysis(niche_count=2)  # model claims niches anyway

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_market_analysis_kickoff_returning(analysis, usage),
    ):
        result = run_market_analysis("user prompt", NO_RESEARCH_TEXT)

    assert result.analysis.niches == []


def test_run_market_analysis_leaves_niches_alone_when_research_was_run() -> None:
    usage = UsageMetrics()
    analysis = _sample_analysis(niche_count=2)

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_market_analysis_kickoff_returning(analysis, usage),
    ):
        result = run_market_analysis("user prompt", "real research findings")

    assert len(result.analysis.niches) == 2
