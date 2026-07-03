from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from crewai.types.usage_metrics import UsageMetrics

from crewai_groq_demo.crew import NO_MARKET_ANALYSIS_TEXT, BuilderResult, run_builder
from crewai_groq_demo.models import BuildAssessment, BuildPlan
from crewai_groq_demo.settings import get_settings


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _builder_kickoff_returning(plan: BuildPlan, usage: UsageMetrics) -> Any:
    def _side_effect(crew_self: Any, inputs: dict[str, Any] | None = None) -> str:
        crew_self.usage_metrics = usage
        return plan.model_dump_json()

    return _side_effect


def _sample_plan(assessment_count: int = 1) -> BuildPlan:
    return BuildPlan(
        assessments=[
            BuildAssessment(
                niche=f"Niche {i}",
                differentiation="medium",
                differentiation_rationale="Because reasons",
                recommended_package="CrewAI",
                key_requirements=["A tool"],
                risks=["A risk"],
            )
            for i in range(assessment_count)
        ]
    )


def test_run_builder_returns_plan_and_usage() -> None:
    usage = UsageMetrics(prompt_tokens=80, completion_tokens=30, total_tokens=110)
    plan = _sample_plan()

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_builder_kickoff_returning(plan, usage),
    ):
        result = run_builder("user prompt", "1. Niche: Something\n   Audience: Someone")

    assert result == BuilderResult(plan=plan, usage=usage)


def test_run_builder_passes_niches_text_to_task_inputs() -> None:
    usage = UsageMetrics()
    plan = _sample_plan()

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_builder_kickoff_returning(plan, usage),
    ) as mocked_kickoff:
        run_builder("user prompt", "specific niches text")

    inputs = mocked_kickoff.call_args.kwargs["inputs"]
    assert inputs == {"user_prompt": "user prompt", "niches_text": "specific niches text"}


def test_run_builder_accepts_no_market_analysis_sentinel() -> None:
    usage = UsageMetrics()
    plan = _sample_plan()

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_builder_kickoff_returning(plan, usage),
    ) as mocked_kickoff:
        run_builder("user prompt", NO_MARKET_ANALYSIS_TEXT)

    inputs = mocked_kickoff.call_args.kwargs["inputs"]
    assert inputs["niches_text"] == NO_MARKET_ANALYSIS_TEXT
