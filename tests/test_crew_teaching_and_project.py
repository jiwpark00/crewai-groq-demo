from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from crewai.types.usage_metrics import UsageMetrics
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError

from crewai_groq_demo.crew import (
    NO_BUILDER_TEXT,
    NO_RESEARCH_TEXT,
    ProjectResult,
    TeachingResult,
    run_project,
    run_teaching,
)
from crewai_groq_demo.exceptions import RateLimitError
from crewai_groq_demo.models import ProjectIdea, ProjectIdeaList
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


def _kickoff_returning(text: str, usage: UsageMetrics) -> Any:
    def _side_effect(crew_self: Any, inputs: dict[str, Any] | None = None) -> str:
        crew_self.usage_metrics = usage
        return text

    return _side_effect


def _kickoff_side_effect(outcomes: list[Any]) -> Any:
    calls = iter(outcomes)

    def _side_effect(crew_self: Any, inputs: dict[str, Any] | None = None) -> Any:
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome
        crew_self.usage_metrics = UsageMetrics()
        return outcome

    return _side_effect


def test_run_teaching_returns_text_and_usage() -> None:
    usage = UsageMetrics(prompt_tokens=200, completion_tokens=80, total_tokens=280)

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_kickoff_returning("an explanation", usage),
    ):
        result = run_teaching("user prompt", "research findings")

    assert result == TeachingResult(text="an explanation", usage=usage)


def test_run_teaching_retries_once_and_succeeds_using_groq_reported_retry_after(
    _no_real_sleep: MagicMock,
) -> None:
    error = LiteLLMRateLimitError(
        message=(
            "Rate limit reached for model `openai/gpt-oss-120b` on tokens per "
            "minute (TPM): Limit 8000, Used 7000, Requested 2000. Please try "
            "again in 15.5s."
        ),
        llm_provider="groq",
        model="groq/openai/gpt-oss-120b",
    )
    side_effect = _kickoff_side_effect([error, "an explanation"])

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect
    ) as mocked_kickoff:
        result = run_teaching("user prompt", "research findings")

    assert result.text == "an explanation"
    assert mocked_kickoff.call_count == 2
    _no_real_sleep.assert_called_once_with(15.5)


def test_run_teaching_gives_up_after_one_retry_on_persistent_rate_limit(
    _no_real_sleep: MagicMock,
) -> None:
    error = LiteLLMRateLimitError(
        message="rate limited", llm_provider="groq", model="groq/openai/gpt-oss-120b"
    )
    side_effect = _kickoff_side_effect([error, error])

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect
    ) as mocked_kickoff:
        with pytest.raises(RateLimitError):
            run_teaching("user prompt", "research findings")

    assert mocked_kickoff.call_count == 2
    _no_real_sleep.assert_called_once_with(5.0)


def test_run_teaching_daily_quota_rate_limit_gives_up_without_retrying(
    _no_real_sleep: MagicMock,
) -> None:
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
            run_teaching("user prompt", "research findings")

    mocked_kickoff.assert_called_once()
    _no_real_sleep.assert_not_called()
    assert exc_info.value.limit_type == "TPD"


def _project_kickoff_returning(ideas: ProjectIdeaList, usage: UsageMetrics) -> Any:
    def _side_effect(crew_self: Any, inputs: dict[str, Any] | None = None) -> str:
        crew_self.usage_metrics = usage
        return ideas.model_dump_json()

    return _side_effect


def _sample_ideas(confidence: str = "high") -> ProjectIdeaList:
    return ProjectIdeaList(
        ideas=[
            ProjectIdea(
                name="Idea",
                goal="Goal",
                kpi="KPI",
                package="CrewAI",
                rationale="Because",
                niche_rationale="Because of specifics in the research",
                evidence=["Some finding: https://example.com"],
                confidence=confidence,  # type: ignore[arg-type]
                open_questions=["Will this work?"],
            )
        ]
    )


def test_run_project_does_not_write_file_by_default(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    usage = UsageMetrics(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    ideas = _sample_ideas()

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_project_kickoff_returning(ideas, usage),
    ):
        result = run_project("user prompt", "teaching result", "some research findings")

    assert result == ProjectResult(ideas=ideas, usage=usage)
    assert not (tmp_path / "output.md").exists()


def test_run_project_writes_file_when_output_path_given(tmp_path: Path) -> None:
    usage = UsageMetrics()
    ideas = _sample_ideas()
    output_path = tmp_path / "ideas.md"

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_project_kickoff_returning(ideas, usage),
    ):
        run_project(
            "user prompt",
            "teaching result",
            "some research findings",
            output_path=str(output_path),
        )

    assert output_path.read_text(encoding="utf-8") == ideas.to_markdown()


def test_run_project_passes_research_result_to_task_inputs() -> None:
    usage = UsageMetrics()
    ideas = _sample_ideas()

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_project_kickoff_returning(ideas, usage),
    ) as mocked_kickoff:
        run_project("user prompt", "teaching result", "specific research findings")

    inputs = mocked_kickoff.call_args.kwargs["inputs"]
    assert inputs == {
        "user_prompt": "user prompt",
        "teaching_result": "teaching result",
        "research_result": "specific research findings",
        "builder_result": NO_BUILDER_TEXT,
    }


def test_run_project_passes_builder_result_to_task_inputs_when_given() -> None:
    usage = UsageMetrics()
    ideas = _sample_ideas()

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_project_kickoff_returning(ideas, usage),
    ) as mocked_kickoff:
        run_project(
            "user prompt",
            "teaching result",
            "specific research findings",
            builder_result="specific builder assessment",
        )

    inputs = mocked_kickoff.call_args.kwargs["inputs"]
    assert inputs["builder_result"] == "specific builder assessment"


def test_run_project_forces_low_confidence_when_no_research_was_run() -> None:
    usage = UsageMetrics()
    ideas = _sample_ideas(confidence="high")  # model claims high confidence anyway

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_project_kickoff_returning(ideas, usage),
    ):
        result = run_project("user prompt", "teaching result", NO_RESEARCH_TEXT)

    assert all(idea.confidence == "low" for idea in result.ideas.ideas)


def test_run_project_leaves_confidence_alone_when_research_was_run() -> None:
    usage = UsageMetrics()
    ideas = _sample_ideas(confidence="high")

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff",
        autospec=True,
        side_effect=_project_kickoff_returning(ideas, usage),
    ):
        result = run_project("user prompt", "teaching result", "real research findings")

    assert all(idea.confidence == "high" for idea in result.ideas.ideas)
