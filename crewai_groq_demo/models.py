from typing import Literal

from pydantic import BaseModel, Field


class ProjectIdea(BaseModel):
    name: str = Field(description="Short, descriptive name for the project idea")
    goal: str = Field(description="What the project accomplishes")
    kpi: str = Field(description="The key metric that shows the project is working")
    package: str = Field(
        description="Recommended agentic AI package (CrewAI, LangGraph, Google ADK, AutoGen, etc.)"
    )
    rationale: str = Field(description="Why that package fits this idea")
    niche_rationale: str = Field(
        description="Why this specific idea/niche fits the request, grounded in the "
        "explanation and research findings rather than generic reasoning"
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Specific research findings (each citing its source URL) that back "
        "this idea; empty if no research was run",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in this idea given the available evidence"
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Open questions to validate before committing to build this",
    )

    def to_markdown_block(self) -> str:
        evidence_block = (
            "- **Evidence:** none — no research was run for this idea\n"
            if not self.evidence
            else "- **Evidence:**\n" + "\n".join(f"  - {item}" for item in self.evidence) + "\n"
        )
        open_questions_block = (
            ""
            if not self.open_questions
            else "- **Open questions:**\n"
            + "\n".join(f"  - {question}" for question in self.open_questions)
            + "\n"
        )
        return (
            f"## {self.name}\n\n"
            f"- **Goal:** {self.goal}\n"
            f"- **KPI:** {self.kpi}\n"
            f"- **Package:** {self.package}\n"
            f"- **Why this package:** {self.rationale}\n"
            f"- **Why this niche:** {self.niche_rationale}\n"
            f"- **Confidence:** {self.confidence}\n"
            f"{evidence_block}"
            f"{open_questions_block}"
        )


class ProjectIdeaList(BaseModel):
    ideas: list[ProjectIdea]

    def to_markdown(self) -> str:
        return "\n".join(idea.to_markdown_block() for idea in self.ideas)
