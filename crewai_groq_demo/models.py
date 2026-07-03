from pydantic import BaseModel, Field


class ProjectIdea(BaseModel):
    name: str = Field(description="Short, descriptive name for the project idea")
    goal: str = Field(description="What the project accomplishes")
    kpi: str = Field(description="The key metric that shows the project is working")
    package: str = Field(
        description="Recommended agentic AI package (CrewAI, LangGraph, Google ADK, AutoGen, etc.)"
    )
    rationale: str = Field(description="Why that package fits this idea")


class ProjectIdeaList(BaseModel):
    ideas: list[ProjectIdea]

    def to_markdown(self) -> str:
        return "\n".join(
            f"## {idea.name}\n\n"
            f"- **Goal:** {idea.goal}\n"
            f"- **KPI:** {idea.kpi}\n"
            f"- **Package:** {idea.package}\n"
            f"- **Why this package:** {idea.rationale}\n"
            for idea in self.ideas
        )
