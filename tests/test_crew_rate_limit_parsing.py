from crewai_groq_demo.crew import _parse_groq_rate_limit


def test_parses_retry_after_and_tpm_limit_type() -> None:
    error = Exception(
        "Rate limit reached for model `openai/gpt-oss-120b` in organization "
        "`org_x` service tier `on_demand` on tokens per minute (TPM): Limit "
        "8000, Used 7492, Requested 4458. Please try again in 29.625s. Need "
        "more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing"
    )

    retry_after, limit_type = _parse_groq_rate_limit(error)

    assert retry_after == 29.625
    assert limit_type == "TPM"


def test_parses_requests_per_day_limit_type() -> None:
    error = Exception("... on requests per day (RPD): ... Please try again in 3600s. ...")

    retry_after, limit_type = _parse_groq_rate_limit(error)

    assert retry_after == 3600.0
    assert limit_type == "RPD"


def test_returns_none_none_when_message_does_not_match() -> None:
    error = Exception("rate limited")

    retry_after, limit_type = _parse_groq_rate_limit(error)

    assert retry_after is None
    assert limit_type is None


def test_returns_retry_after_only_when_limit_type_not_present() -> None:
    error = Exception("Please try again in 5s.")

    retry_after, limit_type = _parse_groq_rate_limit(error)

    assert retry_after == 5.0
    assert limit_type is None
