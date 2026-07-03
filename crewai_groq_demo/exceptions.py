class CrewDemoError(Exception):
    pass


class MissingAPIKeyError(CrewDemoError):
    def __init__(self, var_name: str) -> None:
        self.var_name = var_name
        super().__init__(f"Missing {var_name}. Add it to your .env file.")


class ResearchRetryExhaustedError(CrewDemoError):
    def __init__(self, attempts: int) -> None:
        self.attempts = attempts
        super().__init__(f"Researcher failed tool calling {attempts} times in a row.")


class RateLimitError(CrewDemoError):
    def __init__(self, provider: str, retry_after: float | None = None) -> None:
        self.provider = provider
        self.retry_after = retry_after
        message = f"Rate limited by {provider}."
        if retry_after is not None:
            message += f" Retry after {retry_after}s."
        super().__init__(message)
