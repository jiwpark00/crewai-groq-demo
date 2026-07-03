import time
from typing import Literal, NamedTuple

# Temporary workaround for CrewAI + Groq:
# CrewAI currently adds a `cache_breakpoint` field that Groq rejects.
# This monkey patch disables that cache marker by returning messages unchanged.
# Remove this once CrewAI/LiteLLM supports Groq without the cache_breakpoint error.
import crewai.llms.cache as _crewai_cache
from crewai import LLM, Agent, Crew, Task
from crewai.project import CrewBase, agent, task
from crewai.types.usage_metrics import UsageMetrics
from litellm.exceptions import RateLimitError as LiteLLMRateLimitError

from crewai_groq_demo.exceptions import (
    MissingAPIKeyError,
    RateLimitError,
    ResearchRetryExhaustedError,
)
from crewai_groq_demo.models import ProjectIdeaList
from crewai_groq_demo.settings import get_settings
from crewai_groq_demo.tools.counting_tavily_search_tool import CountingTavilySearchTool

_crewai_cache.mark_cache_breakpoint = lambda msg: msg


@CrewBase
class GroqDemoCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, search_depth: Literal["basic", "advanced"] | None = None) -> None:
        # Stashed here (rather than passed as a `researcher()` kwarg) because
        # CrewBase's metaclass calls `self.researcher()` with *no arguments*
        # immediately after __init__ returns, to resolve tasks.yaml's
        # `agent: researcher` string into a real Agent (see _map_task_variables
        # in crewai/project/crew_base.py). @agent's memoize decorator keys its
        # cache on the exact args passed, so a call shaped differently from
        # that internal zero-arg call (e.g. `researcher(search_depth=...)`)
        # builds a *second*, untracked Agent/tool instance — while the task
        # actually executes with the first one (CrewAI's sequential executor
        # uses `task.agent`, not whatever you later pass to `Crew(agents=...)`,
        # confirmed by reading `crew.py`'s `_get_agent_to_use`). Routing the
        # override through instance state read by a zero-arg `researcher()`
        # keeps every call to it args-identical, so it always resolves to the
        # one real instance CrewAI wires to the task.
        self._search_depth_override = search_depth

        settings = get_settings()
        if not settings.groq_api_key:
            raise MissingAPIKeyError("GROQ_API_KEY")

        self.llm = LLM(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.groq_temperature,
        )

    @agent
    def teacher(self) -> Agent:
        return Agent(config=self.agents_config["teacher"], llm=self.llm)

    @agent
    def project_advisor(self) -> Agent:
        return Agent(config=self.agents_config["project_advisor"], llm=self.llm)

    @agent
    def researcher(self) -> Agent:
        settings = get_settings()
        if not settings.tavily_api_key:
            raise MissingAPIKeyError("TAVILY_API_KEY")

        search_tool = CountingTavilySearchTool(
            max_results=settings.tavily_max_results,
            search_depth=self._search_depth_override or settings.tavily_search_depth,
        )
        return Agent(
            config=self.agents_config["researcher"],
            llm=self.llm,
            tools=[search_tool],
            max_iter=3,
        )

    @task
    def teaching_task(self) -> Task:
        return Task(config=self.tasks_config["teaching_task"])

    @task
    def project_task(self) -> Task:
        return Task(config=self.tasks_config["project_task"], output_pydantic=ProjectIdeaList)

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])


class TeachingResult(NamedTuple):
    text: str
    usage: UsageMetrics


def run_teaching(user_prompt: str, research_result: str) -> TeachingResult:
    """Run only the teacher agent and return its explanation."""
    crew_definition = GroqDemoCrew()
    crew = Crew(
        agents=[crew_definition.teacher()],
        tasks=[crew_definition.teaching_task()],
    )
    try:
        result = crew.kickoff(
            inputs={"user_prompt": user_prompt, "research_result": research_result}
        )
    except LiteLLMRateLimitError as error:
        raise RateLimitError(provider="groq") from error
    return TeachingResult(str(result), crew.usage_metrics or UsageMetrics())


NO_RESEARCH_TEXT = "(no research was run)"
"""Sentinel passed as `research_result` when the researcher stage was skipped.

Shared by main.py/app.py so both UIs agree on the exact string run_project()
checks for — and by the teacher, so it gets the same signal.
"""


class ProjectResult(NamedTuple):
    ideas: ProjectIdeaList
    usage: UsageMetrics


def run_project(
    user_prompt: str,
    teaching_result: str,
    research_result: str,
    output_path: str | None = None,
) -> ProjectResult:
    """Run the project advisor using the teacher's explanation and the raw
    research findings as input, so ideas can cite specific findings/URLs
    rather than whatever survived into the teacher's paraphrased summary.

    Only writes markdown to `output_path` when one is given — callers running
    in a server/UI context (Streamlit) shouldn't write files to disk on every run.
    """
    crew_definition = GroqDemoCrew()
    crew = Crew(
        agents=[crew_definition.project_advisor()],
        tasks=[crew_definition.project_task()],
    )
    try:
        result = crew.kickoff(
            inputs={
                "user_prompt": user_prompt,
                "teaching_result": teaching_result,
                "research_result": research_result,
            }
        )
    except LiteLLMRateLimitError as error:
        raise RateLimitError(provider="groq") from error
    project_ideas: ProjectIdeaList = result.pydantic

    if research_result.strip() == NO_RESEARCH_TEXT:
        # Don't trust the model to self-report low confidence when there's
        # nothing backing an idea — enforce it rather than risk a
        # fabricated-sounding "high confidence" idea with no evidence.
        for idea in project_ideas.ideas:
            idea.confidence = "low"

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(project_ideas.to_markdown())

    return ProjectResult(project_ideas, crew.usage_metrics or UsageMetrics())


class ResearchResult(NamedTuple):
    text: str
    successful_search_count: int
    """Searches made by the attempt whose result is actually returned."""
    total_search_count: int
    """Searches made across all attempts, including failed/discarded retries."""
    retries: int
    """Number of failed attempts before the one that succeeded."""
    queries: tuple[str, ...]
    """The actual search queries used by the successful attempt, in order."""
    usage: UsageMetrics
    """Groq token usage summed across all attempts (failed attempts raise before
    CrewAI populates their usage, so this may undercount tokens burned on retries)."""
    search_depth: Literal["basic", "advanced"]
    """The Tavily search_depth actually used for these searches (advanced costs
    2x the credits of basic — see cost.py's estimate_tavily_cost)."""
    from_cache: bool = False
    """True when this result was served from `_research_cache` instead of a
    fresh Tavily/Groq call — callers should treat it as $0 cost."""


_research_cache: dict[tuple[str, str], ResearchResult] = {}


def run_research(
    user_prompt: str, search_depth: Literal["basic", "advanced"] | None = None
) -> ResearchResult:
    """Run the researcher agent, retrying on malformed tool calls or rate limits.

    `search_depth` overrides the `TAVILY_SEARCH_DEPTH` setting for this call
    (e.g. a per-run UI/CLI choice); omit it to use the configured default.

    Repeat calls with the same `user_prompt` *and* `search_depth` (within this
    process) are served from `_research_cache` instead of re-running
    Tavily/Groq — a different depth is treated as a different request.

    Builds a fresh agent/crew per attempt so a failed attempt's conversation
    state and search count don't bleed into the retry. Retries back off
    exponentially (2s, 4s) so a rate-limited attempt doesn't immediately
    repeat into another rate limit.
    """
    effective_depth = search_depth or get_settings().tavily_search_depth
    cache_key = (user_prompt, effective_depth)
    cached = _research_cache.get(cache_key)
    if cached is not None:
        return cached._replace(from_cache=True)

    max_attempts = 3
    total_search_count = 0
    total_usage = UsageMetrics()
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        if attempt > 0:
            time.sleep(2**attempt)

        crew_definition = GroqDemoCrew(search_depth=search_depth)
        research_task = crew_definition.research_task()
        crew = Crew(
            agents=[crew_definition.researcher()],
            tasks=[research_task],
        )
        # Read the tool from research_task.agent (what CrewAI's sequential
        # executor actually runs — see _get_agent_to_use in crewai/crew.py),
        # not from crew_definition.researcher() above. They're the same
        # memoized instance as long as researcher() is always called with no
        # arguments (see the __init__ comment), but this is the version
        # that's true by construction rather than by memoization matching.
        search_tool = research_task.agent.tools[0]
        try:
            result = crew.kickoff(inputs={"user_prompt": user_prompt})
        except LiteLLMRateLimitError as error:
            total_search_count += search_tool.call_count
            last_error = error
            continue
        except Exception as error:
            total_search_count += search_tool.call_count
            if "tool_use_failed" not in str(error):
                raise
            last_error = error
            continue

        successful_search_count = search_tool.call_count
        total_search_count += successful_search_count
        if crew.usage_metrics:
            total_usage.add_usage_metrics(crew.usage_metrics)
        queries = tuple(search_tool.queries)
        research_result = ResearchResult(
            str(result),
            successful_search_count,
            total_search_count,
            attempt,
            queries,
            total_usage,
            search_tool.search_depth,
        )
        _research_cache[cache_key] = research_result
        return research_result

    if isinstance(last_error, LiteLLMRateLimitError):
        raise RateLimitError(provider="groq") from last_error
    raise ResearchRetryExhaustedError(attempts=max_attempts) from last_error
