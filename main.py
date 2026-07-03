from crewai_groq_demo.crew import run_project, run_research, run_teaching
from crewai_groq_demo.exceptions import CrewDemoError

user_prompt = "What can I build with agentic AI as an entrepreneur?"

try:
    run_research_choice = input(
        "Run web research first? This makes a Tavily API call. [y/N]: "
    ).strip().lower()

    if run_research_choice == "y":
        research = run_research(user_prompt)
        search_word = "search" if research.successful_search_count == 1 else "searches"
        print(
            f"\n=== Researcher's Findings ({research.successful_search_count} {search_word}) ==="
        )
        for i, query in enumerate(research.queries, start=1):
            print(f"  {i}. {query}")
        if research.retries > 0:
            print(
                f"(needed {research.retries} retry(ies) after malformed tool calls — "
                f"{research.total_search_count} total searches across all attempts)"
            )
        print(research.text)
        research_result = research.text
    else:
        research_result = "(no research was run)"

    teaching_result = run_teaching(user_prompt, research_result)
    print("\n=== Teacher's Explanation ===")
    print(teaching_result)

    proceed = input(
        "\nContinue to project ideas? This makes another Groq API call. [y/N]: "
    ).strip().lower()

    if proceed == "y":
        project_result = run_project(user_prompt, teaching_result)
        print("\n=== Project Ideas ===")
        print(project_result.to_markdown())
    else:
        print("Stopped before running the project advisor.")
except CrewDemoError as error:
    print(f"\nError: {error}")
    raise SystemExit(1) from error
