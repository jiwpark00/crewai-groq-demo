from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError

from crewai_groq_demo.crew import ResearchResult, run_research
from crewai_groq_demo.exceptions import RateLimitError, ResearchRetryExhaustedError
from crewai_groq_demo.settings import get_settings


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """GroqDemoCrew() and .researcher() validate these are set before doing
    anything else, and get_settings() is lru_cache-d at module scope so the
    cache must be cleared before and after each test.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_real_sleep() -> Any:
    """Retries back off with time.sleep(2**attempt); patch it out so the
    suite doesn't actually wait several seconds per retry scenario.
    """
    with patch("crewai_groq_demo.crew.time.sleep") as mocked_sleep:
        yield mocked_sleep


def _kickoff_side_effect(
    outcomes: list[Callable[[Any], Any]],
) -> Callable[..., Any]:
    """Build a Crew.kickoff side_effect that pops one outcome per call.

    Each outcome is a callable taking the live Crew instance (so it can poke
    at crew.agents[0].tools[0] the way the real tool would after running)
    and either returns a result or raises.
    """
    calls = iter(outcomes)

    def _side_effect(crew_self: Any, inputs: dict[str, Any] | None = None) -> Any:
        outcome = next(calls)
        return outcome(crew_self)

    return _side_effect


def _record_one_search(query: str) -> Callable[[Any], None]:
    def _apply(crew_self: Any) -> None:
        tool = crew_self.agents[0].tools[0]
        tool.call_count += 1
        tool.queries.append(query)

    return _apply


def _succeed_with(text: str, query: str) -> Callable[[Any], str]:
    def _outcome(crew_self: Any) -> str:
        _record_one_search(query)(crew_self)
        return text

    return _outcome


def _fail_with(error: Exception, query: str) -> Callable[[Any], Any]:
    def _outcome(crew_self: Any) -> Any:
        _record_one_search(query)(crew_self)
        raise error

    return _outcome


def test_succeeds_on_first_attempt() -> None:
    side_effect = _kickoff_side_effect([_succeed_with("great findings", "first query")])

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        result = run_research("teach me about agents")

    assert result == ResearchResult(
        text="great findings",
        successful_search_count=1,
        total_search_count=1,
        retries=0,
        queries=("first query",),
    )


def test_retries_once_after_tool_use_failed_then_succeeds() -> None:
    side_effect = _kickoff_side_effect(
        [
            _fail_with(RuntimeError("tool_use_failed: malformed call"), "bad query"),
            _succeed_with("second attempt findings", "good query"),
        ]
    )

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        result = run_research("teach me about agents")

    assert result.text == "second attempt findings"
    assert result.retries == 1
    assert result.successful_search_count == 1
    assert result.total_search_count == 2
    assert result.queries == ("good query",)


def test_reraises_non_tool_use_failed_exceptions_immediately() -> None:
    boom = ValueError("something unrelated broke")
    side_effect = _kickoff_side_effect([_fail_with(boom, "irrelevant")])

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        with pytest.raises(ValueError, match="something unrelated broke"):
            run_research("teach me about agents")


def test_exhausts_retries_on_repeated_tool_use_failed() -> None:
    side_effect = _kickoff_side_effect(
        [
            _fail_with(RuntimeError("tool_use_failed: 1"), "q1"),
            _fail_with(RuntimeError("tool_use_failed: 2"), "q2"),
            _fail_with(RuntimeError("tool_use_failed: 3"), "q3"),
        ]
    )

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        with pytest.raises(ResearchRetryExhaustedError) as exc_info:
            run_research("teach me about agents")

    assert exc_info.value.attempts == 3


def test_raises_rate_limit_error_when_all_attempts_rate_limited() -> None:
    def _rate_limited(_query: str) -> Callable[[Any], Any]:
        error = LiteLLMRateLimitError(
            message="rate limited", llm_provider="groq", model="groq/llama-3.3-70b-versatile"
        )
        return _fail_with(error, _query)

    side_effect = _kickoff_side_effect(
        [_rate_limited("q1"), _rate_limited("q2"), _rate_limited("q3")]
    )

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        with pytest.raises(RateLimitError) as exc_info:
            run_research("teach me about agents")

    assert exc_info.value.provider == "groq"


def test_backs_off_with_exponential_sleep_between_retries(_no_real_sleep: MagicMock) -> None:
    side_effect = _kickoff_side_effect(
        [
            _fail_with(RuntimeError("tool_use_failed: 1"), "q1"),
            _succeed_with("ok", "q2"),
        ]
    )

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        run_research("teach me about agents")

    _no_real_sleep.assert_called_once_with(2)
