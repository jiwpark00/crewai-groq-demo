from crewai_groq_demo.crew import run_project, run_research, run_teaching

user_prompt = "What can I build with agentic AI as an entrepreneur?"

run_research_choice = input(
    "Run web research first? This makes a Tavily API call. [y/N]: "
).strip().lower()

if run_research_choice == "y":
    research_result, search_count = run_research(user_prompt)
    search_word = "search" if search_count == 1 else "searches"
    print(f"\n=== Researcher's Findings ({search_count} {search_word}) ===")
    print(research_result)
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
    print(project_result)
else:
    print("Stopped before running the project advisor.")
