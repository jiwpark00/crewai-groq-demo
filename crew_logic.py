import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# Temporary workaround for CrewAI + Groq:
# CrewAI currently adds a `cache_breakpoint` field that Groq rejects.
# This monkey patch disables that cache marker by returning messages unchanged.
# Remove this once CrewAI/LiteLLM supports Groq without the cache_breakpoint error.
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg


def run_crew(user_prompt: str) -> str:
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("Missing GROQ_API_KEY. Add it to your .env file.")

    llm = LLM(
        model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=groq_api_key,
    )

    teacher = Agent(
        role="Beginner Agentic AI Teacher",
        goal="Explain Agentic AI concepts clearly and simply",
        backstory="You are patient, practical, and good at giving concrete examples.",
        llm=llm,
    )

    project_advisor = Agent(
        role="Practical Project Advisor",
        goal="Turn agentic AI concepts into realistic business project ideas",
        backstory="You help entrepreneurs identify small projects they can actually build and finish.",
        llm=llm,
    )

    teaching_task = Task(
        description=(
            f"User request: {user_prompt}\n\n"
            "Explain the relevant agentic AI concepts in 5 beginner-friendly bullet points. "
            "When useful, mention packages like CrewAI, Google ADK, LangGraph, AutoGen, etc."
        ),
        expected_output="A short beginner-friendly bullet list with relevant packages.",
        agent=teacher,
    )

    project_task = Task(
        description=(
            "Using the previous explanation, suggest 3 entrepreneur-friendly projects. "
            "For each project, include the goal, KPI, recommended package "
            "(CrewAI, LangGraph, Google ADK, AutoGen, etc.), and why that package fits."
        ),
        expected_output="Three practical entrepreneur project ideas with recommended packages and KPI.",
        agent=project_advisor,
    )

    crew = Crew(
        agents=[teacher, project_advisor],
        tasks=[teaching_task, project_task],
    )

    result = crew.kickoff()
    result_text = str(result)

    with open("output.md", "w", encoding="utf-8") as file:
        file.write(result_text)

    return result_text