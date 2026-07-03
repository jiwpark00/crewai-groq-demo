from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from crewai.types.usage_metrics import UsageMetrics

from crewai_groq_demo.crew import (
    NO_BUILDER_TEXT,
    NO_RESEARCH_TEXT,
    ProjectResult,
    TeachingResult,
    run_project,
    run_teaching,
)
from crewai_groq_demo.models import ProjectIdea, ProjectIdeaList
from crewai_groq_demo.settings import get_settings


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _kickoff_returning(text: str, usage: UsageMetrics) -> Any:
    def _side_effect(crew_self: Any, inputs: dict[str, Any] | None = None) -> str:
        crew_self.usage_metrics = usage
        return text

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
