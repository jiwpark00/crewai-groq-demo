from crewai_tools import TavilySearchTool
from pydantic import Field


class CountingTavilySearchTool(TavilySearchTool):
    """TavilySearchTool that tracks how many searches it has actually run and what was searched."""

    call_count: int = 0
    queries: list[str] = Field(default_factory=list)

    def _run(self, query: str) -> str:
        self.call_count += 1
        self.queries.append(query)
        return super()._run(query)
