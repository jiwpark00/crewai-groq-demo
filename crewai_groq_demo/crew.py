import time
from typing import NamedTuple

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

    def __init__(self) -> None:
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
            search_depth=settings.tavily_search_depth,
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


class ProjectResult(NamedTuple):
    ideas: ProjectIdeaList
    usage: UsageMetrics


def run_project(
    user_prompt: str, teaching_result: str, output_path: str | None = None
) -> ProjectResult:
    """Run the project advisor using the teacher's explanation as input.

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
            inputs={"user_prompt": user_prompt, "teaching_result": teaching_result}
        )
    except LiteLLMRateLimitError as error:
        raise RateLimitError(provider="groq") from error
    project_ideas: ProjectIdeaList = result.pydantic

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
    from_cache: bool = False
    """True when this result was served from `_research_cache` instead of a
    fresh Tavily/Groq call — callers should treat it as $0 cost."""


_research_cache: dict[str, ResearchResult] = {}


def run_research(user_prompt: str) -> ResearchResult:
    """Run the researcher agent, retrying on malformed tool calls or rate limits.

    Repeat calls with the same `user_prompt` (within this process) are served
    from `_research_cache` instead of re-running Tavily/Groq.

    Builds a fresh agent/crew per attempt so a failed attempt's conversation
    state and search count don't bleed into the retry. Retries back off
    exponentially (2s, 4s) so a rate-limited attempt doesn't immediately
    repeat into another rate limit.
    """
    cached = _research_cache.get(user_prompt)
    if cached is not None:
        return cached._replace(from_cache=True)

    max_attempts = 3
    total_search_count = 0
    total_usage = UsageMetrics()
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        if attempt > 0:
            time.sleep(2**attempt)

        crew_definition = GroqDemoCrew()
        researcher_agent = crew_definition.researcher()
        crew = Crew(
            agents=[researcher_agent],
            tasks=[crew_definition.research_task()],
        )
        try:
            result = crew.kickoff(inputs={"user_prompt": user_prompt})
        except LiteLLMRateLimitError as error:
            total_search_count += researcher_agent.tools[0].call_count
            last_error = error
            continue
        except Exception as error:
            total_search_count += researcher_agent.tools[0].call_count
            if "tool_use_failed" not in str(error):
                raise
            last_error = error
            continue

        successful_search_count = researcher_agent.tools[0].call_count
        total_search_count += successful_search_count
        if crew.usage_metrics:
            total_usage.add_usage_metrics(crew.usage_metrics)
        queries = tuple(researcher_agent.tools[0].queries)
        research_result = ResearchResult(
            str(result),
            successful_search_count,
            total_search_count,
            attempt,
            queries,
            total_usage,
        )
        _research_cache[user_prompt] = research_result
        return research_result

    if isinstance(last_error, LiteLLMRateLimitError):
        raise RateLimitError(provider="groq") from last_error
    raise ResearchRetryExhaustedError(attempts=max_attempts) from last_error
