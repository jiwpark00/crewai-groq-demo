from crewai_tools import TavilySearchTool


class CountingTavilySearchTool(TavilySearchTool):
    """TavilySearchTool that tracks how many searches it has actually run."""

    call_count: int = 0

    def _run(self, query: str) -> str:
        self.call_count += 1
        return super()._run(query)
