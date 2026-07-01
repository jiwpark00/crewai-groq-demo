import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
# Temporary workaround for CrewAI + Groq:
# CrewAI currently adds a `cache_breakpoint` field that Groq rejects.
# This monkey patch disables that cache marker by returning messages unchanged.
# Remove this once CrewAI/LiteLLM supports Groq without the cache_breakpoint error.
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY. Add it to your .env file.")

llm = LLM(
    model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key)

agent = Agent(
    role="Beginner Agentic AI Teacher",
    goal="Explain Agentic AI concepts clearly and simply",
    backstory="You are patient, practical, and good at giving concrete examples.",
    llm=llm,
    )

task = Task(
    description="Explain in 5 bullet points what I can build with agentic AI.",
    expected_output="A short beginner-friendly bullet list and specify packages, like CrewAI, ADK, LangGraph, etc.",
    agent=agent,
    )

crew = Crew(
    agents=[agent],
    tasks=[task],
)

result = crew.kickoff()

with open("output.md", "w", encoding="utf-8") as file:
    file.write(str(result))