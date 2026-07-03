from crewai_groq_demo.models import ProjectIdea, ProjectIdeaList


def _make_idea(name: str = "Idea One", **overrides: object) -> ProjectIdea:
    fields: dict[str, object] = {
        "name": name,
        "goal": "Do the thing",
        "kpi": "Number of things done",
        "package": "CrewAI",
        "rationale": "It fits because reasons",
        "niche_rationale": "This niche fits because of specifics",
        "evidence": ["Finding one: https://example.com/a"],
        "confidence": "medium",
        "open_questions": ["Will users actually pay for this?"],
    }
    fields.update(overrides)
    return ProjectIdea(**fields)  # type: ignore[arg-type]


def test_to_markdown_single_idea() -> None:
    ideas = ProjectIdeaList(ideas=[_make_idea()])

    markdown = ideas.to_markdown()

    assert markdown == (
        "## Idea One\n\n"
        "- **Goal:** Do the thing\n"
        "- **KPI:** Number of things done\n"
        "- **Package:** CrewAI\n"
        "- **Why this package:** It fits because reasons\n"
        "- **Why this niche:** This niche fits because of specifics\n"
        "- **Confidence:** medium\n"
        "- **Evidence:**\n"
        "  - Finding one: https://example.com/a\n"
        "- **Open questions:**\n"
        "  - Will users actually pay for this?\n"
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
    assert "?\n\n## Idea Two" in markdown


def test_to_markdown_empty_list() -> None:
    ideas = ProjectIdeaList(ideas=[])

    assert ideas.to_markdown() == ""


def test_to_markdown_no_evidence_or_open_questions() -> None:
    idea = _make_idea(evidence=[], open_questions=[], confidence="low")

    markdown = idea.to_markdown_block()

    assert "- **Evidence:** none — no research was run for this idea\n" in markdown
    assert "Open questions" not in markdown
