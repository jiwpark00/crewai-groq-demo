from crewai_groq_demo.crew import run_crew

user_prompt = "What can I build with agentic AI as an entrepreneur?"

result = run_crew(user_prompt)

print(result)