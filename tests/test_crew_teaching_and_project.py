from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from crewai.types.usage_metrics import UsageMetrics

from crewai_groq_demo.crew import ProjectResult, TeachingResult, run_project, run_teaching
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
    def _side_effect(crew_self: Any, inputs: dict[str, Any] | None = None) -> Any:
        crew_self.usage_metrics = usage
        result = type("Result", (), {"pydantic": ideas})()
        return result

    return _side_effect


def _sample_ideas() -> ProjectIdeaList:
    return ProjectIdeaList(
        ideas=[
            ProjectIdea(
                name="Idea",
                goal="Goal",
                kpi="KPI",
                package="CrewAI",
                rationale="Because",
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
        result = run_project("user prompt", "teaching result")

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
        run_project("user prompt", "teaching result", output_path=str(output_path))

    assert output_path.read_text(encoding="utf-8") == ideas.to_markdown()
