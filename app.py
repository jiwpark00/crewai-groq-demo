import csv
import io

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

for key in (
    "teaching_result",
    "project_result",
    "gated_prompt",
    "research_result",
    "research_prompt",
    "research_search_count",
):
    st.session_state.setdefault(key, None)

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
    if search_count == 0:
        st.error(
            "Ran 0 searches — these findings were not actually looked up and may be hallucinated."
        )
    elif search_count > 1:
        st.warning(f"⚠️ Ran {search_count} {search_word} for this request, not just one.")
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
    project_ideas = st.session_state.project_result

    for idea in project_ideas.ideas:
        with st.container(border=True):
            st.subheader(idea.name)
            st.markdown(f"**Goal:** {idea.goal}")
            st.markdown(f"**KPI:** {idea.kpi}")
            st.markdown(f"**Package:** {idea.package}")
            st.markdown(f"**Why this package:** {idea.rationale}")

    with st.expander("View as table"):
        st.dataframe(
            [idea.model_dump() for idea in project_ideas.ideas],
            use_container_width=True,
        )

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(
        csv_buffer, fieldnames=["name", "goal", "kpi", "package", "rationale"]
    )
    writer.writeheader()
    for idea in project_ideas.ideas:
        writer.writerow(idea.model_dump())

    st.download_button(
        label="Download ideas.csv",
        data=csv_buffer.getvalue(),
        file_name="project_ideas.csv",
        mime="text/csv",
    )
