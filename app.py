import streamlit as st
from crewai_groq_demo.crew import run_project, run_teaching, run_research

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
    st.session_state.research_result = None
    st.session_state.research_prompt = None
    st.session_state.research_search_count = None

run_research_button = st.button("Run Researcher")

if run_research_button:
    if not user_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Searching the web..."):
            st.session_state.research_result, st.session_state.research_search_count = run_research(
                user_prompt
            )
        st.session_state.research_prompt = user_prompt
        st.session_state.teaching_result = None
        st.session_state.project_result = None

research_is_current = st.session_state.research_prompt == user_prompt

if st.session_state.research_result:
    st.markdown("## Researcher's Findings")
    st.markdown(st.session_state.research_result)

    search_count = st.session_state.research_search_count
    search_word = "search" if search_count == 1 else "searches"
    if search_count and search_count > 1:
        st.caption(f"⚠️ Ran {search_count} {search_word} for this request, not just one.")
    else:
        st.caption(f"Ran {search_count} {search_word} for this request.")

    if research_is_current:
        st.info("Review the sources above before spending a Groq call on the teacher.")
    else:
        st.warning(
            "This research was run for a different prompt. Re-run the researcher, "
            "or the teacher will proceed without it."
        )

run_teacher_button = st.button("Run Teacher")

if run_teacher_button:
    if not user_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        research_for_teacher = (
            st.session_state.research_result
            if research_is_current
            else "(no research was run for this prompt)"
        )
        with st.spinner("Running teaching agent..."):
            st.session_state.teaching_result = run_teaching(user_prompt, research_for_teacher)
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
