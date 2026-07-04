from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from crewai.types.usage_metrics import UsageMetrics
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError

from crewai_groq_demo.crew import GroqDemoCrew, ResearchResult, _research_cache, run_research
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
def _clear_research_cache() -> Iterator[None]:
    """run_research() memoizes results in the module-level `_research_cache`
    dict keyed by prompt text; several tests below reuse the same prompt
    string, so the cache must be cleared before and after each test or a
    later test would silently get an earlier test's mocked result.
    """
    _research_cache.clear()
    yield
    _research_cache.clear()


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


def _succeed_with_usage(text: str, query: str, usage: UsageMetrics) -> Callable[[Any], str]:
    """Like _succeed_with, but also sets crew.usage_metrics the way a real
    kickoff() would, so token-usage accumulation can be tested.
    """

    def _outcome(crew_self: Any) -> str:
        _record_one_search(query)(crew_self)
        crew_self.usage_metrics = usage
        return text

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
        usage=UsageMetrics(),
        search_depth="basic",
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
            message="rate limited", llm_provider="groq", model="groq/openai/gpt-oss-120b"
        )
        return _fail_with(error, _query)

    side_effect = _kickoff_side_effect(
        [_rate_limited("q1"), _rate_limited("q2"), _rate_limited("q3")]
    )

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        with pytest.raises(RateLimitError) as exc_info:
            run_research("teach me about agents")

    assert exc_info.value.provider == "groq"
    # This test's error message doesn't match Groq's real wording, so parsing
    # comes up empty — confirmed separately in test_crew_rate_limit_parsing.py.
    assert exc_info.value.retry_after is None
    assert exc_info.value.limit_type is None


def test_rate_limit_error_carries_retry_after_and_limit_type_when_parseable() -> None:
    def _rate_limited(_query: str) -> Callable[[Any], Any]:
        error = LiteLLMRateLimitError(
            message=(
                "Rate limit reached for model `openai/gpt-oss-120b` on tokens "
                "per minute (TPM): Limit 8000, Used 7492, Requested 4458. "
                "Please try again in 29.625s."
            ),
            llm_provider="groq",
            model="groq/openai/gpt-oss-120b",
        )
        return _fail_with(error, _query)

    side_effect = _kickoff_side_effect(
        [_rate_limited("q1"), _rate_limited("q2"), _rate_limited("q3")]
    )

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        with pytest.raises(RateLimitError) as exc_info:
            run_research("teach me about agents")

    assert exc_info.value.retry_after == 29.625
    assert exc_info.value.limit_type == "TPM"


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


def test_rate_limit_retry_waits_for_groq_reported_retry_after(_no_real_sleep: MagicMock) -> None:
    """Confirmed live: Groq's real TPM cooldowns (20-30s) are much longer than
    the fixed 2s/4s exponential schedule, so retrying on that schedule was
    guaranteed to hit the same wall again. Retries must wait as long as
    Groq's own error says is actually needed instead.
    """
    rate_limit_error = LiteLLMRateLimitError(
        message=(
            "Rate limit reached for model `openai/gpt-oss-120b` on tokens per "
            "minute (TPM): Limit 8000, Used 7000, Requested 2000. Please try "
            "again in 15.5s."
        ),
        llm_provider="groq",
        model="groq/openai/gpt-oss-120b",
    )
    side_effect = _kickoff_side_effect(
        [_fail_with(rate_limit_error, "q1"), _succeed_with("ok", "q2")]
    )

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        run_research("teach me about agents")

    _no_real_sleep.assert_called_once_with(15.5)


def test_rate_limit_retry_caps_wait_at_30_seconds(_no_real_sleep: MagicMock) -> None:
    rate_limit_error = LiteLLMRateLimitError(
        message=(
            "Rate limit reached for model `openai/gpt-oss-120b` on tokens per "
            "minute (TPM): Limit 8000, Used 7999, Requested 2000. Please try "
            "again in 90s."
        ),
        llm_provider="groq",
        model="groq/openai/gpt-oss-120b",
    )
    side_effect = _kickoff_side_effect(
        [_fail_with(rate_limit_error, "q1"), _succeed_with("ok", "q2")]
    )

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        run_research("teach me about agents")

    _no_real_sleep.assert_called_once_with(30.0)


def test_rate_limit_retry_falls_back_to_exponential_backoff_when_unparseable(
    _no_real_sleep: MagicMock,
) -> None:
    rate_limit_error = LiteLLMRateLimitError(
        message="rate limited", llm_provider="groq", model="groq/openai/gpt-oss-120b"
    )
    side_effect = _kickoff_side_effect(
        [_fail_with(rate_limit_error, "q1"), _succeed_with("ok", "q2")]
    )

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        run_research("teach me about agents")

    _no_real_sleep.assert_called_once_with(2)


def test_daily_quota_rate_limit_gives_up_without_retrying(_no_real_sleep: MagicMock) -> None:
    """TPD/RPD won't refill within this call's lifetime, so burning the
    remaining attempts on a guaranteed repeat failure just wastes tokens —
    give up after the first hit instead of retrying.
    """
    rate_limit_error = LiteLLMRateLimitError(
        message=(
            "Rate limit reached for model `openai/gpt-oss-120b` on tokens per "
            "day (TPD): Limit 200000, Used 200000, Requested 500. Please try "
            "again in 43200s."
        ),
        llm_provider="groq",
        model="groq/openai/gpt-oss-120b",
    )
    side_effect = _kickoff_side_effect([_fail_with(rate_limit_error, "q1")])

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect
    ) as mocked_kickoff:
        with pytest.raises(RateLimitError) as exc_info:
            run_research("teach me about agents")

    mocked_kickoff.assert_called_once()
    _no_real_sleep.assert_not_called()
    assert exc_info.value.limit_type == "TPD"


def test_researcher_agent_uses_tavily_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "9")
    monkeypatch.setenv("TAVILY_SEARCH_DEPTH", "advanced")
    get_settings.cache_clear()

    researcher_agent = GroqDemoCrew().researcher()
    tool = researcher_agent.tools[0]

    assert tool.max_results == 9
    assert tool.search_depth == "advanced"


def test_researcher_agent_search_depth_override_wins_over_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_SEARCH_DEPTH", "advanced")
    get_settings.cache_clear()

    researcher_agent = GroqDemoCrew(search_depth="basic").researcher()

    assert researcher_agent.tools[0].search_depth == "basic"


def test_researcher_instance_matches_task_resolved_agent() -> None:
    """Regression test for a real bug caught via live testing: CrewBase's
    metaclass eagerly resolves tasks.yaml's `agent: researcher` string by
    calling `self.researcher()` with *no arguments* during `GroqDemoCrew.__init__`
    (see `_map_task_variables` in crewai/project/crew_base.py), and @agent's
    memoize decorator keys its cache by exact call args. `researcher()` used to
    take a `search_depth` kwarg — calling it that way built a second, different
    Agent/tool instance than the one CrewAI's sequential executor actually runs
    (which uses `task.agent`, not whatever's passed to `Crew(agents=[...])` —
    see `_get_agent_to_use` in crewai/crew.py). Confirmed live: Tavily's own
    usage counter went up by 2 credits on a run our code reported as 0 searches,
    because we were inspecting the untracked, never-executed instance.

    `search_depth` now lives on `GroqDemoCrew.__init__` instead, so every call
    to `researcher()` is args-identical and always resolves to the one real,
    memoized instance. This test doesn't call kickoff() (no live API calls) —
    it only checks object identity through CrewAI's real, unmocked
    Task-config resolution.
    """
    crew_definition = GroqDemoCrew(search_depth="advanced")

    research_task = crew_definition.research_task()
    researcher_agent = crew_definition.researcher()

    assert research_task.agent is researcher_agent
    assert research_task.agent.tools[0] is researcher_agent.tools[0]
    assert researcher_agent.tools[0].search_depth == "advanced"


def test_captures_token_usage_from_successful_attempt() -> None:
    usage = UsageMetrics(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    side_effect = _kickoff_side_effect([_succeed_with_usage("findings", "q1", usage)])

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        result = run_research("teach me about agents")

    assert result.usage == usage


def test_serves_repeat_prompt_from_cache_without_calling_kickoff_again() -> None:
    side_effect = _kickoff_side_effect([_succeed_with("first run findings", "q1")])

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect
    ) as mocked_kickoff:
        first = run_research("teach me about agents")
        second = run_research("teach me about agents")

    mocked_kickoff.assert_called_once()
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.text == "first run findings"
    assert second.successful_search_count == first.successful_search_count


def test_run_research_uses_search_depth_override() -> None:
    side_effect = _kickoff_side_effect([_succeed_with("findings", "q1")])

    with patch("crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect):
        result = run_research("teach me about agents", search_depth="advanced")

    assert result.search_depth == "advanced"


def test_run_research_cache_key_distinguishes_by_search_depth() -> None:
    """A repeat prompt with a *different* search_depth is a different request —
    it must not be served from a cached result for the other depth.
    """
    side_effect = _kickoff_side_effect(
        [_succeed_with("basic findings", "q1"), _succeed_with("advanced findings", "q2")]
    )

    with patch(
        "crewai_groq_demo.crew.Crew.kickoff", autospec=True, side_effect=side_effect
    ) as mocked_kickoff:
        basic_result = run_research("teach me about agents", search_depth="basic")
        advanced_result = run_research("teach me about agents", search_depth="advanced")

    assert mocked_kickoff.call_count == 2
    assert basic_result.from_cache is False
    assert advanced_result.from_cache is False
    assert basic_result.text == "basic findings"
    assert advanced_result.text == "advanced findings"
