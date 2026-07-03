from unittest.mock import patch

from crewai_tools import TavilySearchTool

from crewai_groq_demo.tools.counting_tavily_search_tool import CountingTavilySearchTool


def _make_tool() -> CountingTavilySearchTool:
    # Passing api_key explicitly avoids depending on TAVILY_API_KEY being set in
    # the environment (or on the real project .env, which this suite must not
    # read). Construction only builds a TavilyClient locally — it does not
    # make a network call or validate the key.
    return CountingTavilySearchTool(api_key="test-key", max_results=5, search_depth="basic")


def test_construction_does_not_require_env_var() -> None:
    tool = _make_tool()

    assert tool.call_count == 0
    assert tool.queries == []


def test_run_increments_call_count_and_records_query() -> None:
    tool = _make_tool()

    with patch.object(TavilySearchTool, "_run", return_value="canned result") as mocked_run:
        result = tool._run("agentic AI frameworks")

    assert result == "canned result"
    assert tool.call_count == 1
    assert tool.queries == ["agentic AI frameworks"]
    # patch.object without autospec replaces _run with a plain MagicMock,
    # which isn't a descriptor, so super()._run(query) doesn't get `self`
    # auto-bound — the mock only ever sees the query argument.
    mocked_run.assert_called_once_with("agentic AI frameworks")


def test_run_tracks_multiple_calls_in_order() -> None:
    tool = _make_tool()

    with patch.object(TavilySearchTool, "_run", return_value="canned result"):
        tool._run("first query")
        tool._run("second query")

    assert tool.call_count == 2
    assert tool.queries == ["first query", "second query"]
