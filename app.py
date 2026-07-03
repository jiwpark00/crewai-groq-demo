import csv
import io
from typing import Literal

import streamlit as st

from crewai_groq_demo.cost import estimate_groq_cost, estimate_tavily_cost, format_groq_cost
from crewai_groq_demo.crew import NO_RESEARCH_TEXT, run_project, run_research, run_teaching
from crewai_groq_demo.exceptions import CrewDemoError
from crewai_groq_demo.models import ProjectIdea
from crewai_groq_demo.settings import get_settings

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
    "research_for_project",
):
    st.session_state.setdefault(key, None)

search_depth_options: list[Literal["basic", "advanced"]] = ["basic", "advanced"]
search_depth = st.radio(
    "Search depth",
    options=search_depth_options,
    index=search_depth_options.index(get_settings().tavily_search_depth),
    horizontal=True,
    help="Advanced costs 2x the Tavily credits of basic.",
)

run_research_button = st.button("Run Researcher")

if run_research_button:
    if not user_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        try:
            with st.spinner("Searching the web..."):
                st.session_state.research_result = run_research(
                    user_prompt, search_depth=search_depth
                )
            st.session_state.research_prompt = user_prompt
            st.session_state.teaching_result = None
            st.session_state.project_result = None
        except CrewDemoError as error:
            st.error(str(error))

research_is_current = st.session_state.research_prompt == user_prompt

if st.session_state.research_result is not None:
    research = st.session_state.research_result
    st.markdown("## Researcher's Findings")
    st.markdown(research.text)

    search_word = "search" if research.successful_search_count == 1 else "searches"
    if research.successful_search_count == 0:
        st.error(
            "Ran 0 searches — these findings were not actually looked up and may be hallucinated."
        )
    elif research.successful_search_count > 1:
        st.warning(
            f"⚠️ Ran {research.successful_search_count} {search_word} for this "
            "request, not just one."
        )
    else:
        st.caption(f"Ran {research.successful_search_count} {search_word} for this request.")

    if research.queries:
        with st.expander("View search queries used"):
            for i, query in enumerate(research.queries, start=1):
                st.markdown(f"{i}. `{query}`")

    if research.retries > 0:
        discarded_searches = research.total_search_count - research.successful_search_count
        st.warning(
            f"Needed {research.retries} retry(ies) after malformed tool calls — "
            f"{research.total_search_count} total searches were made across all attempts "
            f"({discarded_searches} on discarded attempts)."
        )

    if research.from_cache:
        st.caption("Cost: $0 (served from cache — same prompt was already researched).")
    else:
        research_cost = estimate_groq_cost(research.usage) + estimate_tavily_cost(
            research.successful_search_count, research.search_depth
        )
        st.caption(f"Estimated cost: ~${research_cost:.4f}")

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
            st.session_state.research_result.text
            if research_is_current
            else NO_RESEARCH_TEXT
        )
        try:
            with st.spinner("Running teaching agent..."):
                st.session_state.teaching_result = run_teaching(
                    user_prompt, research_for_teacher
                )
            st.session_state.gated_prompt = user_prompt
            st.session_state.research_for_project = research_for_teacher
            st.session_state.project_result = None
        except CrewDemoError as error:
            st.error(str(error))

if st.session_state.teaching_result:
    st.markdown("## Teacher's Explanation")
    st.markdown(st.session_state.teaching_result.text)
    teaching_cost = estimate_groq_cost(st.session_state.teaching_result.usage)
    st.caption(f"Estimated cost: ~${teaching_cost:.4f}")

    st.info("Review the explanation above before spending another Groq call on project ideas.")

    if st.button("Continue to Project Ideas"):
        try:
            with st.spinner("Running project advisor..."):
                st.session_state.project_result = run_project(
                    st.session_state.gated_prompt,
                    st.session_state.teaching_result.text,
                    st.session_state.research_for_project,
                )
        except CrewDemoError as error:
            st.error(str(error))

if st.session_state.project_result:
    st.markdown("## Project Ideas")
    st.caption(format_groq_cost(st.session_state.project_result.usage))
    project_ideas = st.session_state.project_result.ideas

    for idea in project_ideas.ideas:
        with st.container(border=True):
            st.subheader(idea.name)
            st.markdown(f"**Goal:** {idea.goal}")
            st.markdown(f"**KPI:** {idea.kpi}")
            st.markdown(f"**Package:** {idea.package}")
            st.markdown(f"**Why this package:** {idea.rationale}")
            st.markdown(f"**Why this niche:** {idea.niche_rationale}")

            confidence_badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}[idea.confidence]
            st.markdown(f"**Confidence:** {confidence_badge} {idea.confidence}")

            if idea.evidence:
                with st.expander(f"Evidence ({len(idea.evidence)})"):
                    for item in idea.evidence:
                        st.markdown(f"- {item}")
            else:
                st.caption("No research backs this idea — treat it as speculative.")

            if idea.open_questions:
                with st.expander(f"Open questions ({len(idea.open_questions)})"):
                    for question in idea.open_questions:
                        st.markdown(f"- {question}")

    def _flatten_idea(idea: ProjectIdea) -> dict[str, object]:
        row = idea.model_dump()
        row["evidence"] = "; ".join(idea.evidence)
        row["open_questions"] = "; ".join(idea.open_questions)
        return row

    with st.expander("View as table"):
        st.dataframe(
            [_flatten_idea(idea) for idea in project_ideas.ideas],
            use_container_width=True,
        )

    csv_fieldnames = [
        "name",
        "goal",
        "kpi",
        "package",
        "rationale",
        "niche_rationale",
        "confidence",
        "evidence",
        "open_questions",
    ]
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=csv_fieldnames)
    writer.writeheader()
    for idea in project_ideas.ideas:
        writer.writerow(_flatten_idea(idea))

    st.download_button(
        label="Download ideas.csv",
        data=csv_buffer.getvalue(),
        file_name="project_ideas.csv",
        mime="text/csv",
    )
