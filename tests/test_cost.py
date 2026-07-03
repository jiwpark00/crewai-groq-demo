from crewai.types.usage_metrics import UsageMetrics

from crewai_groq_demo.cost import estimate_groq_cost, estimate_tavily_cost, format_groq_cost
from crewai_groq_demo.settings import (
    GROQ_INPUT_PRICE_PER_MILLION_TOKENS,
    GROQ_OUTPUT_PRICE_PER_MILLION_TOKENS,
    TAVILY_PRICE_PER_CREDIT,
)


def test_estimate_groq_cost_zero_usage() -> None:
    assert estimate_groq_cost(UsageMetrics()) == 0.0


def test_estimate_groq_cost_uses_published_pricing() -> None:
    usage = UsageMetrics(prompt_tokens=1_000_000, completion_tokens=1_000_000)

    cost = estimate_groq_cost(usage)

    assert cost == GROQ_INPUT_PRICE_PER_MILLION_TOKENS + GROQ_OUTPUT_PRICE_PER_MILLION_TOKENS


def test_estimate_tavily_cost_zero_searches() -> None:
    assert estimate_tavily_cost(0, "basic") == 0.0


def test_estimate_tavily_cost_scales_linearly_with_search_count() -> None:
    assert estimate_tavily_cost(3, "basic") == 3 * TAVILY_PRICE_PER_CREDIT


def test_estimate_tavily_cost_advanced_search_costs_double_basic() -> None:
    assert estimate_tavily_cost(1, "advanced") == 2 * estimate_tavily_cost(1, "basic")


def test_estimate_tavily_cost_advanced_search_is_two_credits_per_search() -> None:
    assert estimate_tavily_cost(3, "advanced") == 3 * 2 * TAVILY_PRICE_PER_CREDIT


def test_format_groq_cost_flags_untracked_structured_output_calls() -> None:
    """successful_requests == 0 is how CrewAI's output_pydantic/InternalInstructor
    path (which never calls _track_token_usage_internal) looks from the outside —
    distinct from a real call that happened to use 0 tokens.
    """
    assert "unavailable" in format_groq_cost(UsageMetrics())


def test_format_groq_cost_renders_dollar_amount_for_tracked_calls() -> None:
    usage = UsageMetrics(prompt_tokens=1000, completion_tokens=500, successful_requests=1)

    assert format_groq_cost(usage) == f"~${estimate_groq_cost(usage):.4f}"
