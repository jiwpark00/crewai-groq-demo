from crewai_groq_demo.crew import run_project, run_teaching

user_prompt = "What can I build with agentic AI as an entrepreneur?"

teaching_result = run_teaching(user_prompt)
print("=== Teacher's Explanation ===")
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
