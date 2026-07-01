import streamlit as st
from crewai_groq_demo.crew import run_crew

st.set_page_config(
    page_title="CrewAI Groq Demo",
    page_icon="🤖",
    layout="centered",
)

st.title("CrewAI + Groq Demo")
st.write("Enter a business or agentic AI question, then run your two-agent CrewAI workflow.")

user_prompt = st.text_area(
    "What do you want the agents to help with?",
    value="What can I build with agentic AI as an entrepreneur?",
    height=120,
)

run_button = st.button("Run Crew")

if run_button:
    if not user_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Running CrewAI agents..."):
            result = run_crew(user_prompt)

        st.success("Done.")
        st.markdown("## Result")
        st.markdown(result)

        st.download_button(
            label="Download output.md",
            data=result,
            file_name="output.md",
            mime="text/markdown",
        )