import streamlit as st
from crewai_groq_demo.crew import run_project, run_teaching

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

if "teaching_result" not in st.session_state:
    st.session_state.teaching_result = None
    st.session_state.project_result = None
    st.session_state.gated_prompt = None

run_teacher_button = st.button("Run Teacher")

if run_teacher_button:
    if not user_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Running teaching agent..."):
            st.session_state.teaching_result = run_teaching(user_prompt)
        st.session_state.gated_prompt = user_prompt
        st.session_state.project_result = None

if st.session_state.teaching_result:
    st.markdown("## Teacher's Explanation")
    st.markdown(st.session_state.teaching_result)

    st.info("Review the explanation above before spending another Groq call on project ideas.")

    if st.button("Continue to Project Ideas"):
        with st.spinner("Running project advisor..."):
            st.session_state.project_result = run_project(
                st.session_state.gated_prompt, st.session_state.teaching_result
            )

if st.session_state.project_result:
    st.markdown("## Project Ideas")
    st.markdown(st.session_state.project_result)

    st.download_button(
        label="Download output.md",
        data=st.session_state.project_result,
        file_name="output.md",
        mime="text/markdown",
    )
