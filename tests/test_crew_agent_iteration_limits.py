from collections.abc import Iterator

import pytest

from crewai_groq_demo.crew import GroqDemoCrew
from crewai_groq_demo.settings import get_settings


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.parametrize(
    "agent_method_name", ["teacher", "project_advisor", "market_analyst", "builder"]
)
def test_toolless_agents_cap_max_iter_at_one(agent_method_name: str) -> None:
    """These agents have no tools, so they should resolve in a single pass.

    CrewAI's default (max_iter=25) means a response its executor doesn't
    recognize as "final" could silently re-prompt up to 25 times — each one
    a real, token-consuming Groq call — before ever reaching our own retry/
    error-handling layer. Confirmed live that only `researcher` had this
    constrained before; regression test to keep it that way for the rest.
    """
    crew_definition = GroqDemoCrew()
    agent = getattr(crew_definition, agent_method_name)()

    assert agent.max_iter == 1
