from collections.abc import Iterator
from pathlib import Path

import pytest

from crewai_groq_demo.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_cwd_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep settings tests from touching the real project .env or leaking lru_cache state.

    Settings.model_config points at ".env" relative to the process cwd, so without
    chdir-ing to an empty tmp_path these tests would load the real (gitignored)
    .env file and its real keys. get_settings() is also lru_cache-d at module
    scope, so every test must clear that cache before *and* after running.
    """
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_settings_defaults_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GROQ_TEMPERATURE", raising=False)
    monkeypatch.delenv("TAVILY_MAX_RESULTS", raising=False)
    monkeypatch.delenv("TAVILY_SEARCH_DEPTH", raising=False)

    settings = Settings()

    assert settings.groq_api_key == ""
    assert settings.tavily_api_key == ""
    assert settings.groq_model == "groq/llama-3.3-70b-versatile"
    assert settings.groq_temperature == 0.2
    assert settings.tavily_max_results == 5
    assert settings.tavily_search_depth == "basic"


def test_settings_load_from_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("GROQ_MODEL", "groq/some-other-model")
    monkeypatch.setenv("GROQ_TEMPERATURE", "0.7")
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "8")
    monkeypatch.setenv("TAVILY_SEARCH_DEPTH", "advanced")

    settings = Settings()

    assert settings.groq_api_key == "test-groq-key"
    assert settings.tavily_api_key == "test-tavily-key"
    assert settings.groq_model == "groq/some-other-model"
    assert settings.groq_temperature == 0.7
    assert settings.tavily_max_results == 8
    assert settings.tavily_search_depth == "advanced"


def test_settings_ignores_unknown_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_UNRELATED_VAR", "whatever")

    settings = Settings()

    assert not hasattr(settings, "some_unrelated_var")


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "first-key")

    first = get_settings()
    monkeypatch.setenv("GROQ_API_KEY", "second-key")
    second = get_settings()

    assert first is second
    assert second.groq_api_key == "first-key"


def test_get_settings_reflects_env_after_cache_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "first-key")
    get_settings()

    get_settings.cache_clear()
    monkeypatch.setenv("GROQ_API_KEY", "second-key")
    refreshed = get_settings()

    assert refreshed.groq_api_key == "second-key"
