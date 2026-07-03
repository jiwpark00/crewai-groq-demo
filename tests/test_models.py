from crewai_groq_demo.models import ProjectIdea, ProjectIdeaList


def _make_idea(name: str = "Idea One") -> ProjectIdea:
    return ProjectIdea(
        name=name,
        goal="Do the thing",
        kpi="Number of things done",
        package="CrewAI",
        rationale="It fits because reasons",
    )


def test_to_markdown_single_idea() -> None:
    ideas = ProjectIdeaList(ideas=[_make_idea()])

    markdown = ideas.to_markdown()

    assert markdown == (
        "## Idea One\n\n"
        "- **Goal:** Do the thing\n"
        "- **KPI:** Number of things done\n"
        "- **Package:** CrewAI\n"
        "- **Why this package:** It fits because reasons\n"
    )


def test_to_markdown_multiple_ideas() -> None:
    ideas = ProjectIdeaList(ideas=[_make_idea("Idea One"), _make_idea("Idea Two")])

    markdown = ideas.to_markdown()

    assert "## Idea One" in markdown
    assert "## Idea Two" in markdown
    assert markdown.count("- **Goal:**") == 2
    assert markdown.count("- **KPI:**") == 2
    assert markdown.count("- **Package:**") == 2
    assert markdown.count("- **Why this package:**") == 2
    # Blocks are joined with "\n" and each block already ends with "\n",
    # so consecutive blocks are separated by exactly one blank line.
    assert "reasons\n\n## Idea Two" in markdown


def test_to_markdown_empty_list() -> None:
    ideas = ProjectIdeaList(ideas=[])

    assert ideas.to_markdown() == ""
