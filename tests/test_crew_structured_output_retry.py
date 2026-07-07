from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from crewai.types.usage_metrics import UsageMetrics
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError

from crewai_groq_demo.crew import run_builder
from crewai_groq_demo.exceptions import RateLimitError, StructuredOutputParseError
from crewai_groq_demo.models import BuildAssessment, BuildPlan
from crewai_groq_demo.settings import get_settings


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_real_sleep() -> Any:
    with patch("crewai_groq_demo.crew.time.sleep") as mocked_sleep:
        yield mocked_sleep


def _sample_plan() -> BuildPlan:
    return BuildPlan(
        assessments=[
            BuildAssessment(
                niche="Niche",
                differentiation="medium",
                differentiation_rationale="Because reasons",
                recommended_package="CrewAI",
                key_requirements=["A tool"],
                risks=["A risk"],
            )
        ]
    )


def _kickoff_side_effect(outcomes: list[Any]) -> Any:
    calls = iter(outcomes)

    def _side_effect(crew_self: Any, inputs: dict[str, Any] | None = None) -> Any:
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome
        crew_self.usage_metrics = UsageMetrics()
        return outcome

    return _side_effect


def test_retries_once_after_malformed_json_then_succeeds(_no_real_sleep: MagicMock) -> None:
    plan = _sample_plan()
    side_effect = _kickoff_side_effect(["not valid json at all", plan.model_dump_json()])

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect
    ) as mocked_kickoff:
        result = run_builder("user prompt", "some niches text")

    assert result.plan == plan
    assert mocked_kickoff.call_count == 2
    _no_real_sleep.assert_called_once_with(2)


def test_exhausts_retries_on_repeated_malformed_json() -> None:
    side_effect = _kickoff_side_effect(["not json", "still not json", "nope"])

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        with pytest.raises(StructuredOutputParseError) as exc_info:
            run_builder("user prompt", "some niches text")

    assert exc_info.value.attempts == 3
    assert exc_info.value.task_name == "builder_task"


def test_rate_limit_retries_and_succeeds_using_groq_reported_retry_after(
    _no_real_sleep: MagicMock,
) -> None:
    plan = _sample_plan()
    error = LiteLLMRateLimitError(
        message=(
            "Rate limit reached for model `openai/gpt-oss-120b` on tokens per "
            "minute (TPM): Limit 8000, Used 7000, Requested 2000. Please try "
            "again in 15.5s."
        ),
        llm_provider="groq",
        model="groq/openai/gpt-oss-120b",
    )
    side_effect = _kickoff_side_effect([error, plan.model_dump_json()])

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect
    ) as mocked_kickoff:
        result = run_builder("user prompt", "some niches text")

    assert result.plan == plan
    assert mocked_kickoff.call_count == 2
    _no_real_sleep.assert_called_once_with(15.5)


def test_rate_limit_exhausts_retries_when_unparseable(_no_real_sleep: MagicMock) -> None:
    error = LiteLLMRateLimitError(
        message="rate limited", llm_provider="groq", model="groq/openai/gpt-oss-120b"
    )
    side_effect = _kickoff_side_effect([error, error, error])

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect
    ) as mocked_kickoff:
        with pytest.raises(RateLimitError) as exc_info:
            run_builder("user prompt", "some niches text")

    assert mocked_kickoff.call_count == 3
    assert exc_info.value.retry_after is None
    assert exc_info.value.limit_type is None


def test_daily_quota_rate_limit_gives_up_without_retrying() -> None:
    error = LiteLLMRateLimitError(
        message=(
            "Rate limit reached for model `openai/gpt-oss-120b` on tokens per "
            "day (TPD): Limit 200000, Used 200000, Requested 500. Please try "
            "again in 43200s."
        ),
        llm_provider="groq",
        model="groq/openai/gpt-oss-120b",
    )
    side_effect = _kickoff_side_effect([error])

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect
    ) as mocked_kickoff:
        with pytest.raises(RateLimitError) as exc_info:
            run_builder("user prompt", "some niches text")

    mocked_kickoff.assert_called_once()
    assert exc_info.value.limit_type == "TPD"
