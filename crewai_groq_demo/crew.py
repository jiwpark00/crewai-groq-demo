import os

from crewai import Agent, Crew, LLM, Task
from crewai.project import CrewBase, agent, task
from dotenv import load_dotenv

# Temporary workaround for CrewAI + Groq:
# CrewAI currently adds a `cache_breakpoint` field that Groq rejects.
# This monkey patch disables that cache marker by returning messages unchanged.
# Remove this once CrewAI/LiteLLM supports Groq without the cache_breakpoint error.
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg


@CrewBase
class GroqDemoCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self) -> None:
        load_dotenv()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY. Add it to your .env file.")

        self.llm = LLM(
            model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=groq_api_key,
        )

    @agent
    def teacher(self) -> Agent:
        return Agent(config=self.agents_config["teacher"], llm=self.llm)

    @agent
    def project_advisor(self) -> Agent:
        return Agent(config=self.agents_config["project_advisor"], llm=self.llm)

    @task
    def teaching_task(self) -> Task:
        return Task(config=self.tasks_config["teaching_task"])

    @task
    def project_task(self) -> Task:
        return Task(config=self.tasks_config["project_task"])


def run_teaching(user_prompt: str) -> str:
    """Run only the teacher agent and return its explanation."""
    crew_definition = GroqDemoCrew()
    crew = Crew(
        agents=[crew_definition.teacher()],
        tasks=[crew_definition.teaching_task()],
    )
    result = crew.kickoff(inputs={"user_prompt": user_prompt})
    return str(result)


def run_project(user_prompt: str, teaching_result: str) -> str:
    """Run the project advisor using the teacher's explanation as input."""
    crew_definition = GroqDemoCrew()
    crew = Crew(
        agents=[crew_definition.project_advisor()],
        tasks=[crew_definition.project_task()],
    )
    result = crew.kickoff(
        inputs={"user_prompt": user_prompt, "teaching_result": teaching_result}
    )
    result_text = str(result)

    with open("output.md", "w", encoding="utf-8") as file:
        file.write(result_text)

    return result_text
