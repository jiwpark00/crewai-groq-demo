from crewai.types.usage_metrics import UsageMetrics

from crewai_groq_demo.settings import (
    GROQ_INPUT_PRICE_PER_MILLION_TOKENS,
    GROQ_OUTPUT_PRICE_PER_MILLION_TOKENS,
    TAVILY_PRICE_PER_CREDIT,
)


def estimate_groq_cost(usage: UsageMetrics) -> float:
    return (
        usage.prompt_tokens * GROQ_INPUT_PRICE_PER_MILLION_TOKENS
        + usage.completion_tokens * GROQ_OUTPUT_PRICE_PER_MILLION_TOKENS
    ) / 1_000_000


def estimate_tavily_cost(search_count: int) -> float:
    return search_count * TAVILY_PRICE_PER_CREDIT


def format_groq_cost(usage: UsageMetrics) -> str:
    """Render a Groq call's estimated cost, or flag when it can't be known.

    CrewAI's `output_pydantic`/`response_model` tasks (e.g. the project
    advisor) go through an InternalInstructor path that never calls
    `_track_token_usage_internal` (confirmed in CrewAI's `llm.py`
    `_handle_non_streaming_response`), so `usage` comes back all-zero —
    including `successful_requests`, which is the one field a real tracked
    call always sets to at least 1. That all-zero pattern is how we tell
    "genuinely free" apart from "CrewAI didn't tell us."
    """
    if usage.successful_requests == 0:
        return "cost unavailable (CrewAI doesn't track tokens for structured/pydantic outputs)"
    return f"~${estimate_groq_cost(usage):.4f}"
