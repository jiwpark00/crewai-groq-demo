import re
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
from pydantic import BaseModel, ValidationError

from crewai_groq_demo.exceptions import (
    MissingAPIKeyError,
    RateLimitError,
    ResearchRetryExhaustedError,
    StructuredOutputParseError,
)
from crewai_groq_demo.models import BuildPlan, MarketAnalysis, ProjectIdeaList
from crewai_groq_demo.settings import get_settings
from crewai_groq_demo.structured_output import parse_structured_output
from crewai_groq_demo.tools.counting_tavily_search_tool import CountingTavilySearchTool

_crewai_cache.mark_cache_breakpoint = lambda msg: msg

_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)
_LIMIT_TYPE_RE = re.compile(r"\((TPM|RPM|TPD|RPD)\)")


def _parse_groq_rate_limit(error: Exception) -> tuple[float | None, str | None]:
    """Best-effort extraction of retry-after seconds and limit type (TPM/RPM/
    TPD/RPD) from Groq's rate-limit error text, e.g. "...on tokens per minute
    (TPM): Limit 8000, Used 7492, Requested 4458. Please try again in
    29.625s...". Groq's wording isn't a stable contract, so this returns
    (None, None) rather than raising if the message doesn't match — a worse
    error message beats a crash while handling a rate limit.
    """
    text = str(error)
    retry_match = _RETRY_AFTER_RE.search(text)
    retry_after = float(retry_match.group(1)) if retry_match else None
    limit_match = _LIMIT_TYPE_RE.search(text)
    limit_type = limit_match.group(1) if limit_match else None
    return retry_after, limit_type


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
        # max_iter=1: this agent has no tools, so it should resolve in a
        # single pass. CrewAI's default (25) means a model that produces a
        # response its executor doesn't recognize as "final" could silently
        # re-prompt up to 25 times — each one a real, token-consuming Groq
        # call — before ever reaching our own retry/error-handling layer.
        # Confirmed live: only `researcher` had this constrained before;
        # the other agents ran at the 25-iteration default.
        return Agent(config=self.agents_config["teacher"], llm=self.llm, max_iter=1)

    @agent
    def project_advisor(self) -> Agent:
        return Agent(config=self.agents_config["project_advisor"], llm=self.llm, max_iter=1)

    @agent
    def market_analyst(self) -> Agent:
        return Agent(config=self.agents_config["market_analyst"], llm=self.llm, max_iter=1)

    @agent
    def builder(self) -> Agent:
        return Agent(config=self.agents_config["builder"], llm=self.llm, max_iter=1)

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
        return Task(config=self.tasks_config["project_task"])

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def market_analysis_task(self) -> Task:
        return Task(config=self.tasks_config["market_analysis_task"])

    @task
    def builder_task(self) -> Task:
        return Task(config=self.tasks_config["builder_task"])


def _kickoff_with_structured_output[T: BaseModel](
    crew: Crew,
    inputs: dict[str, str],
    model: type[T],
    task_name: str,
    max_attempts: int = 3,
) -> tuple[T, UsageMetrics]:
    """Run `crew.kickoff` and parse its raw text output into `model`,
    retrying if the response doesn't parse or Groq rate-limits the request —
    one bad generation shouldn't be fatal, same rationale as run_research's
    tool_use_failed retry. Parse-failure retries back off exponentially (2s,
    4s); rate-limit retries instead wait however long Groq's own error says
    is needed (capped at 30s) — see run_research's docstring for why a fixed
    schedule doesn't work for real TPM cooldowns. A daily-quota limit
    (TPD/RPD) gives up immediately instead of retrying, since it won't clear
    within this call's lifetime.

    Tasks passed here must NOT use `output_pydantic`/`output_json` — see the
    module-level tasks in this file and `structured_output.py` for why.
    """
    total_usage = UsageMetrics()
    last_error: Exception | None = None
    next_sleep_seconds = 0.0
    for attempt in range(max_attempts):
        if attempt > 0:
            time.sleep(next_sleep_seconds)
        try:
            result = crew.kickoff(inputs=inputs)
        except LiteLLMRateLimitError as error:
            last_error = error
            retry_after, limit_type = _parse_groq_rate_limit(error)
            if limit_type in ("TPD", "RPD"):
                raise RateLimitError(
                    provider="groq", retry_after=retry_after, limit_type=limit_type
                ) from error
            next_sleep_seconds = (
                min(retry_after, 30.0) if retry_after is not None else 2 ** (attempt + 1)
            )
            continue
        if crew.usage_metrics:
            total_usage.add_usage_metrics(crew.usage_metrics)
        try:
            parsed = parse_structured_output(str(result), model)
        except ValidationError as error:
            last_error = error
            next_sleep_seconds = 2 ** (attempt + 1)
            continue
        return parsed, total_usage

    if isinstance(last_error, LiteLLMRateLimitError):
        retry_after, limit_type = _parse_groq_rate_limit(last_error)
        raise RateLimitError(
            provider="groq", retry_after=retry_after, limit_type=limit_type
        ) from last_error
    raise StructuredOutputParseError(task_name=task_name, attempts=max_attempts) from last_error


class TeachingResult(NamedTuple):
    text: str
    usage: UsageMetrics


def run_teaching(user_prompt: str, research_result: str) -> TeachingResult:
    """Run only the teacher agent and return its explanation.

    Retries once on a TPM/RPM rate limit, waiting however long Groq's own
    error says is needed (capped at 30s) — see run_research's docstring for
    why a fixed backoff doesn't work. Gives up immediately on a daily-quota
    limit (TPD/RPD), since no wait here will clear it.
    """
    crew_definition = GroqDemoCrew()
    crew = Crew(
        agents=[crew_definition.teacher()],
        tasks=[crew_definition.teaching_task()],
    )
    inputs = {"user_prompt": user_prompt, "research_result": research_result}
    try:
        result = crew.kickoff(inputs=inputs)
    except LiteLLMRateLimitError as error:
        retry_after, limit_type = _parse_groq_rate_limit(error)
        if limit_type in ("TPD", "RPD"):
            raise RateLimitError(
                provider="groq", retry_after=retry_after, limit_type=limit_type
            ) from error
        time.sleep(min(retry_after, 30.0) if retry_after is not None else 5.0)
        try:
            result = crew.kickoff(inputs=inputs)
        except LiteLLMRateLimitError as retry_error:
            retry_after, limit_type = _parse_groq_rate_limit(retry_error)
            raise RateLimitError(
                provider="groq", retry_after=retry_after, limit_type=limit_type
            ) from retry_error
    return TeachingResult(str(result), crew.usage_metrics or UsageMetrics())


NO_RESEARCH_TEXT = "(no research was run)"
"""Sentinel passed as `research_result` when the researcher stage was skipped.

Shared by main.py/app.py so both UIs agree on the exact string run_project()
checks for — and by the teacher, so it gets the same signal.
"""


class ProjectResult(NamedTuple):
    ideas: ProjectIdeaList
    usage: UsageMetrics


NO_BUILDER_TEXT = "(no builder assessment was run)"
"""Sentinel passed as `builder_result` when the builder stage was skipped —
mirrors NO_RESEARCH_TEXT/NO_MARKET_ANALYSIS_TEXT so project_task's shared
prompt template can signal "nothing to ground this in" consistently.
"""


def run_project(
    user_prompt: str,
    teaching_result: str,
    research_result: str,
    builder_result: str = NO_BUILDER_TEXT,
    output_path: str | None = None,
) -> ProjectResult:
    """Run the project advisor using the teacher's explanation, the raw
    research findings, and (optionally) the builder's differentiation
    assessment as input, so ideas can cite specific findings/URLs and avoid
    duplicating what the builder already flagged as commoditized.

    Only writes markdown to `output_path` when one is given — callers running
    in a server/UI context (Streamlit) shouldn't write files to disk on every run.
    """
    crew_definition = GroqDemoCrew()
    crew = Crew(
        agents=[crew_definition.project_advisor()],
        tasks=[crew_definition.project_task()],
    )
    project_ideas, usage = _kickoff_with_structured_output(
        crew,
        {
            "user_prompt": user_prompt,
            "teaching_result": teaching_result,
            "research_result": research_result,
            "builder_result": builder_result,
        },
        ProjectIdeaList,
        task_name="project_task",
    )

    if research_result.strip() == NO_RESEARCH_TEXT:
        # Don't trust the model to self-report low confidence when there's
        # nothing backing an idea — enforce it rather than risk a
        # fabricated-sounding "high confidence" idea with no evidence.
        for idea in project_ideas.ideas:
            idea.confidence = "low"

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(project_ideas.to_markdown())

    return ProjectResult(project_ideas, usage)


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
    state and search count don't bleed into the retry. Malformed-tool-call
    retries back off exponentially (2s, 4s). Rate-limit retries instead wait
    however long Groq's own error says is actually needed (capped at 30s) —
    confirmed live that Groq's real TPM cooldowns (20-30s) are much longer
    than the fixed exponential schedule, so retrying on the old schedule was
    guaranteed to hit the same wall again. A daily-quota limit (TPD/RPD)
    gives up immediately instead of retrying — it won't refill within this
    call's lifetime, so burning remaining attempts on it wastes tokens.
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
    next_sleep_seconds = 0.0

    for attempt in range(max_attempts):
        if attempt > 0:
            time.sleep(next_sleep_seconds)

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
            retry_after, limit_type = _parse_groq_rate_limit(error)
            if limit_type in ("TPD", "RPD"):
                break
            next_sleep_seconds = (
                min(retry_after, 30.0) if retry_after is not None else 2 ** (attempt + 1)
            )
            continue
        except Exception as error:
            total_search_count += search_tool.call_count
            if "tool_use_failed" not in str(error):
                raise
            last_error = error
            next_sleep_seconds = 2 ** (attempt + 1)
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
        retry_after, limit_type = _parse_groq_rate_limit(last_error)
        raise RateLimitError(
            provider="groq", retry_after=retry_after, limit_type=limit_type
        ) from last_error
    raise ResearchRetryExhaustedError(attempts=max_attempts) from last_error


class MarketAnalysisResult(NamedTuple):
    analysis: MarketAnalysis
    usage: UsageMetrics


def run_market_analysis(user_prompt: str, research_result: str) -> MarketAnalysisResult:
    """Run the market analyst using the researcher's raw findings directly —
    not the teacher's explanation. The market analyst's job is to spot niches
    in what was actually found, not to re-derive them from a paraphrase (same
    reasoning as project_task: paraphrasing loses the specific findings/URLs
    needed to ground each niche).
    """
    crew_definition = GroqDemoCrew()
    crew = Crew(
        agents=[crew_definition.market_analyst()],
        tasks=[crew_definition.market_analysis_task()],
    )
    analysis, usage = _kickoff_with_structured_output(
        crew,
        {"user_prompt": user_prompt, "research_result": research_result},
        MarketAnalysis,
        task_name="market_analysis_task",
    )

    if research_result.strip() == NO_RESEARCH_TEXT:
        # Same reasoning as run_project's confidence-forcing: don't trust the
        # model to self-report an empty list when there's nothing to ground
        # a niche in — enforce it rather than risk fabricated-looking niches.
        analysis.niches = []

    return MarketAnalysisResult(analysis, usage)


NO_MARKET_ANALYSIS_TEXT = "(no market analysis was run)"
"""Sentinel passed as `niches_text` to run_builder when Market Analyst was
skipped (it's optional, unlike Builder) — Builder falls back to assessing
the user's request directly from general knowledge rather than a specific,
evidence-backed niche.
"""


class BuilderResult(NamedTuple):
    plan: BuildPlan
    usage: UsageMetrics


def run_builder(user_prompt: str, niches_text: str) -> BuilderResult:
    """Run the builder agent to assess build differentiation.

    Takes pre-formatted `niches_text` (see `MarketAnalysis.to_niches_text()`)
    rather than the raw MarketAnalysis object or research_result — Builder
    only needs each niche's own compact evidence, not the researcher's full
    prose, and needs no research context at all when Market Analyst was
    skipped (falls back to general knowledge, signaled by
    `NO_MARKET_ANALYSIS_TEXT`).
    """
    crew_definition = GroqDemoCrew()
    crew = Crew(
        agents=[crew_definition.builder()],
        tasks=[crew_definition.builder_task()],
    )
    plan, usage = _kickoff_with_structured_output(
        crew,
        {"user_prompt": user_prompt, "niches_text": niches_text},
        BuildPlan,
        task_name="builder_task",
    )
    return BuilderResult(plan, usage)
