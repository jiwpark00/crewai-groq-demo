import pytest

from crewai_groq_demo.exceptions import (
    CrewDemoError,
    MissingAPIKeyError,
    RateLimitError,
    ResearchRetryExhaustedError,
)


def test_missing_api_key_error_message_and_attribute() -> None:
    error = MissingAPIKeyError("GROQ_API_KEY")

    assert error.var_name == "GROQ_API_KEY"
    assert str(error) == "Missing GROQ_API_KEY. Add it to your .env file."
    assert isinstance(error, CrewDemoError)


def test_research_retry_exhausted_error_message_and_attribute() -> None:
    error = ResearchRetryExhaustedError(3)

    assert error.attempts == 3
    assert str(error) == "Researcher failed tool calling 3 times in a row."
    assert isinstance(error, CrewDemoError)


def test_rate_limit_error_without_retry_after() -> None:
    error = RateLimitError(provider="groq")

    assert error.provider == "groq"
    assert error.retry_after is None
    assert str(error) == "Rate limited by groq."
    assert isinstance(error, CrewDemoError)


def test_rate_limit_error_with_retry_after() -> None:
    error = RateLimitError(provider="groq", retry_after=1.5)

    assert error.provider == "groq"
    assert error.retry_after == 1.5
    assert str(error) == "Rate limited by groq. Retry after 1.5s."


def test_crew_demo_error_is_plain_exception() -> None:
    error = CrewDemoError("boom")

    assert str(error) == "boom"
    assert isinstance(error, Exception)


@pytest.mark.parametrize(
    "exc_type",
    [MissingAPIKeyError, ResearchRetryExhaustedError, RateLimitError],
)
def test_all_custom_errors_are_crew_demo_errors(exc_type: type) -> None:
    assert issubclass(exc_type, CrewDemoError)
