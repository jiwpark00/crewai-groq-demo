# CrewAI + Groq Demo

A project exploring agentic AI with [CrewAI](https://www.crewai.com/), using
[Groq](https://groq.com/) as the LLM provider to stay on free-tier usage
instead of paid APIs. It runs a small crew of agents — researcher, teacher,
and project advisor — that hand results to each other to turn a plain-English
question into a set of concrete agentic-AI project ideas.

## What it does

You ask a question like *"What can I build with agentic AI as an
entrepreneur?"* and three agents work through it in sequence:

1. **Researcher** — searches the web (via Tavily) for current, relevant
   findings and summarizes them with sources.
2. **Teacher** — explains the agentic-AI concepts relevant to your specific
   question, grounded in the research.
3. **Project Advisor** — turns that explanation into a structured list of
   project ideas, each with a goal, KPI, recommended package, and rationale.

Each stage is gated behind a manual step (a button in the UI, a y/N prompt in
the CLI) so you decide when to spend an API call, rather than the app
burning through them automatically.

## Demo

🚧 Under construction — no screenshots or demo GIF yet.

## Architecture

```mermaid
flowchart LR
    U[User prompt] --> R[Researcher agent<br/>Tavily search]
    R --> T[Teacher agent<br/>explains concepts]
    U --> T
    T --> P[Project Advisor agent<br/>structured project ideas]
    U --> P
    P --> O[output.md / CSV download]
```

Each agent runs as its own single-task `Crew` (see `run_research`,
`run_teaching`, `run_project` in
[crewai_groq_demo/crew.py](crewai_groq_demo/crew.py)), so a stage only runs
when explicitly triggered, and the caller decides what gets passed forward.

```
crewai_groq_demo/
  crew.py                 # LLM setup, per-agent Crew runners
  models.py                # Pydantic schema for project ideas
  settings.py               # Typed, validated env config (pydantic-settings)
  exceptions.py              # Typed exceptions (MissingAPIKeyError, RateLimitError, ...)
  config/
    agents.yaml            # role/goal/backstory per agent
    tasks.yaml              # description/expected_output per task
  tools/
    counting_tavily_search_tool.py  # Tavily search wrapper that tracks call count/queries
main.py                    # CLI entry point
app.py                     # Streamlit UI entry point
tests/                      # pytest suite (no live API calls — see Testing)
```

## Setup

This project is managed entirely with [`uv`](https://docs.astral.sh/uv/) —
no manual `.venv` activation needed.

```bash
uv sync
```

### Required environment variables

Create a `.env` file in the project root (not committed) with:

| Variable | Used by | Get one at |
|---|---|---|
| `GROQ_API_KEY` | teacher, project advisor, researcher LLM calls | [console.groq.com](https://console.groq.com/) |
| `TAVILY_API_KEY` | researcher's web search tool | [tavily.com](https://tavily.com/) |

Both are required — `crew.py` raises a `MissingAPIKeyError` immediately if
either is missing, caught in `app.py`/`main.py` and shown as a clean message
rather than a stack trace.

### Running

```bash
uv run main.py                    # CLI
uv run streamlit run app.py       # Streamlit UI
```

## Testing

```bash
uv run pytest      # run the test suite
uv run ruff check . # lint
uv run mypy crewai_groq_demo    # type-check
```

The test suite never calls Groq or Tavily — every test that touches
`run_research`/`run_teaching`/`run_project` mocks `Crew.kickoff` and scripts
its return value/exception, so tests run in milliseconds, for free, and
don't need a real `.env`. Coverage so far focuses on what's testable without
a live LLM in the loop:

- `models.py` — `ProjectIdeaList.to_markdown()` formatting
- `exceptions.py` — message formatting and attributes on each typed error
- `settings.py` — env var loading, defaults, and `get_settings()` caching
- `tools/counting_tavily_search_tool.py` — call-count/query tracking
- `crew.py`'s `run_research` retry loop — first-try success, retry-then-
  succeed, retries exhausted, Groq rate-limit handling, and the exponential
  backoff between retries

`ruff` and `mypy` are configured in `pyproject.toml`
(`select = ["E", "F", "I", "UP", "B"]` for ruff; fairly strict mypy with a
scoped `ignore_missing_imports` for `tavily.*`, the only direct dependency
without inline type stubs). Note: `crew.py` currently has a chunk of
pre-existing mypy noise from CrewAI's `@CrewBase`/`@agent`/`@task` decorator
magic (e.g. `self.agents_config["teacher"]` type-checking as indexing a
plain `str`) — not yet resolved, tracked as a follow-up rather than papered
over with blanket `# type: ignore`s.

## Example prompts

- "What can I build with agentic AI as an entrepreneur?"
- "Explain agentic AI to a solo SaaS founder in 3 points."
- "How could a small marketing agency use agentic AI? Give me 5 ideas."

The teacher and project advisor tailor depth, framing, and idea count to
whatever audience/domain/count you specify in the prompt.

## Example output

Given *"What can I build with agentic AI as an entrepreneur?"*, the project
advisor produces structured ideas like:

```markdown
## Client Onboarding Concierge

- **Goal:** Automatically gather, verify, and summarize new-client intake
  documents into a ready-to-review packet.
- **KPI:** Time from signed contract to fully verified client file.
- **Package:** CrewAI
- **Why this package:** Config-driven, sequential agent roles map cleanly
  onto a fixed intake checklist.
```

The Streamlit UI renders each idea as a card and also offers a CSV download;
the CLI writes the full set to `output.md`.

## Cost estimate per run

Both providers have usable free tiers, so a full run (research + teacher +
project advisor) typically costs **$0**:

- **Groq**: `llama-3.3-70b-versatile` is available on Groq's free tier,
  subject to requests/tokens-per-minute rate limits. Three short calls per
  run (one per agent) stays well within them for occasional use.
- **Tavily**: the researcher makes at most 2 searches per run (capped in
  `tasks.yaml`), well within Tavily's free monthly search quota.

If you exceed free-tier rate limits, Groq/Tavily calls will fail with a
429-style error rather than silently charging you — neither provider is
configured with a paid fallback here.

## Deployment

🚧 Under construction — no deployment target has been chosen yet.

## How it works

- **Config-driven agents/tasks**: roles, goals, and backstories live in
  [config/agents.yaml](crewai_groq_demo/config/agents.yaml); task prompts
  and expected outputs live in
  [config/tasks.yaml](crewai_groq_demo/config/tasks.yaml). `crew.py` wires
  them together rather than hardcoding prompts in Python.
- **Structured output**: the project advisor's task uses
  `output_pydantic=ProjectIdeaList`, so its output is validated, typed data
  (not markdown that needs parsing) — see
  [models.py](crewai_groq_demo/models.py).
- **Search accountability**: `CountingTavilySearchTool` wraps CrewAI's
  Tavily tool to track exactly how many searches ran and what was searched,
  so the UI can warn you if the researcher used more than one search or
  hallucinated findings without searching at all.
- **Retry on malformed tool calls or Groq rate limits**: `run_research`
  rebuilds the agent/crew from scratch and retries (up to 3 attempts, with
  exponential backoff) if Groq's tool-calling output is malformed or Groq
  rate-limits the request, without letting a failed attempt's search count
  or conversation state bleed into the retry. (Tavily rate limits aren't
  cleanly catchable here — CrewAI's tool-execution layer swallows them into
  agent-visible text before they'd reach this code.)
- **Typed config and errors**: `settings.py` (`pydantic-settings`) replaces
  scattered `os.getenv()` calls with a validated settings object;
  `exceptions.py` defines `MissingAPIKeyError`, `RateLimitError`, and
  `ResearchRetryExhaustedError` so callers handle failures by type instead
  of pattern-matching error strings.
- **Groq/CrewAI workaround**: `crew.py` monkey-patches CrewAI's
  `mark_cache_breakpoint` to a no-op, working around a CrewAI↔Groq
  incompatibility (Groq rejects a `cache_breakpoint` field CrewAI adds to
  messages). See the comment in `crew.py` before removing it.

## Roadmap

🚧 Under construction — no formal roadmap yet.

## Use cases

🚧 Under construction.
