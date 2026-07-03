import time
from typing import NamedTuple

# Temporary workaround for CrewAI + Groq:
# CrewAI currently adds a `cache_breakpoint` field that Groq rejects.
# This monkey patch disables that cache marker by returning messages unchanged.
# Remove this once CrewAI/LiteLLM supports Groq without the cache_breakpoint error.
import crewai.llms.cache as _crewai_cache
from crewai import LLM, Agent, Crew, Task
from crewai.project import CrewBase, agent, task
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
        if not get_settings().tavily_api_key:
            raise MissingAPIKeyError("TAVILY_API_KEY")

        search_tool = CountingTavilySearchTool(max_results=5, search_depth="basic")
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


def run_teaching(user_prompt: str, research_result: str) -> str:
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
    return str(result)


def run_project(user_prompt: str, teaching_result: str) -> ProjectIdeaList:
    """Run the project advisor using the teacher's explanation as input."""
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

    with open("output.md", "w", encoding="utf-8") as file:
        file.write(project_ideas.to_markdown())

    return project_ideas


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


def run_research(user_prompt: str) -> ResearchResult:
    """Run the researcher agent, retrying on malformed tool calls or rate limits.

    Builds a fresh agent/crew per attempt so a failed attempt's conversation
    state and search count don't bleed into the retry. Retries back off
    exponentially (2s, 4s) so a rate-limited attempt doesn't immediately
    repeat into another rate limit.
    """
    max_attempts = 3
    total_search_count = 0
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
        queries = tuple(researcher_agent.tools[0].queries)
        return ResearchResult(
            str(result), successful_search_count, total_search_count, attempt, queries
        )

    if isinstance(last_error, LiteLLMRateLimitError):
        raise RateLimitError(provider="groq") from last_error
    raise ResearchRetryExhaustedError(attempts=max_attempts) from last_error
