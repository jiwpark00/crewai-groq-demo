import csv
import io
from typing import Literal

import streamlit as st

from crewai_groq_demo.cost import estimate_groq_cost, estimate_tavily_cost, format_groq_cost
from crewai_groq_demo.crew import (
    NO_BUILDER_TEXT,
    NO_MARKET_ANALYSIS_TEXT,
    NO_RESEARCH_TEXT,
    run_builder,
    run_market_analysis,
    run_project,
    run_research,
    run_teaching,
)
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
    "market_analysis_result",
    "market_analysis_prompt",
    "builder_result",
    "builder_prompt",
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
            # Market analysis/builder were derived from whatever research
            # existed before this call — same prompt text doesn't mean same
            # findings, so they're stale too, not just teaching/project.
            st.session_state.market_analysis_result = None
            st.session_state.builder_result = None
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

st.markdown("---")
st.markdown("### Market Analyst (optional)")
st.caption(
    "Screens the research findings above for specific, underserved niches — "
    "it's correct and expected for this to return 0 niches if nothing clears "
    "the bar. Skip this to let Builder reason from general knowledge instead."
)
run_market_analyst_button = st.button("Run Market Analyst")

if run_market_analyst_button:
    if not user_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        research_for_market_analysis = (
            st.session_state.research_result.text if research_is_current else NO_RESEARCH_TEXT
        )
        try:
            with st.spinner("Analyzing the market..."):
                st.session_state.market_analysis_result = run_market_analysis(
                    user_prompt, research_for_market_analysis
                )
            st.session_state.market_analysis_prompt = user_prompt
            # Builder and (transitively, through it) project ideas were
            # derived from whatever market analysis existed before this call.
            st.session_state.builder_result = None
            st.session_state.project_result = None
        except CrewDemoError as error:
            st.error(str(error))

market_analysis_is_current = st.session_state.market_analysis_prompt == user_prompt

if st.session_state.market_analysis_result is not None:
    analysis = st.session_state.market_analysis_result.analysis
    if analysis.niches:
        for niche in analysis.niches:
            with st.container(border=True):
                st.markdown(f"**{niche.niche}**")
                st.markdown(f"*Audience:* {niche.audience}")
                if niche.evidence:
                    with st.expander(f"Evidence ({len(niche.evidence)})"):
                        for item in niche.evidence:
                            st.markdown(f"- {item}")
    else:
        st.info(
            "No niches survived screening for this prompt — nothing here cleared the bar."
        )
    st.caption(format_groq_cost(st.session_state.market_analysis_result.usage))
    if not market_analysis_is_current:
        st.warning(
            "This market analysis was run for a different prompt. Re-run it, "
            "or Builder will proceed without it."
        )

st.markdown("---")
st.markdown("### Builder")
st.caption(
    "Assesses how differentiated a build would be from what already exists. "
    "Uses the market analyst's niches above if current, otherwise falls back "
    "to general knowledge about your request as a whole."
)
run_builder_button = st.button("Run Builder")

if run_builder_button:
    if not user_prompt.strip():
        st.warning("Please enter a prompt first.")
    elif (
        st.session_state.market_analysis_result is not None
        and market_analysis_is_current
        and not st.session_state.market_analysis_result.analysis.niches
    ):
        st.info(
            "Market analysis found nothing worth pursuing for this prompt, so "
            "the pipeline stops here rather than having Builder guess at "
            "niches that didn't survive screening. Try a different prompt, or "
            "skip Market Analyst entirely to let Builder reason from general "
            "knowledge instead."
        )
    else:
        niches_text = (
            st.session_state.market_analysis_result.analysis.to_niches_text()
            if st.session_state.market_analysis_result is not None
            and market_analysis_is_current
            else NO_MARKET_ANALYSIS_TEXT
        )
        try:
            with st.spinner("Assessing build differentiation..."):
                st.session_state.builder_result = run_builder(user_prompt, niches_text)
            st.session_state.builder_prompt = user_prompt
            # Project ideas (if any) were grounded in whatever build
            # assessment existed before this call.
            st.session_state.project_result = None
        except CrewDemoError as error:
            st.error(str(error))

builder_is_current = st.session_state.builder_prompt == user_prompt

if st.session_state.builder_result is not None:
    plan = st.session_state.builder_result.plan
    diff_badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    for assessment in plan.assessments:
        with st.container(border=True):
            st.markdown(f"**{assessment.niche}**")
            st.markdown(
                f"**Differentiation:** {diff_badge[assessment.differentiation]} "
                f"{assessment.differentiation}"
            )
            st.markdown(assessment.differentiation_rationale)
            st.markdown(f"**Recommended package:** {assessment.recommended_package}")
            if assessment.key_requirements:
                with st.expander(f"Key requirements ({len(assessment.key_requirements)})"):
                    for item in assessment.key_requirements:
                        st.markdown(f"- {item}")
            if assessment.risks:
                with st.expander(f"Risks ({len(assessment.risks)})"):
                    for item in assessment.risks:
                        st.markdown(f"- {item}")
    st.caption(format_groq_cost(st.session_state.builder_result.usage))
    if not builder_is_current:
        st.warning(
            "This build assessment was run for a different prompt. Re-run it, "
            "or the project advisor will proceed without it."
        )

st.markdown("---")
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

    builder_available = st.session_state.builder_result is not None and builder_is_current
    if builder_available:
        st.caption("Project ideas will be grounded in the build assessment above.")
    else:
        st.caption(
            "No current build assessment — project ideas will be generated without it. "
            "Run Builder above first if you want ideas grounded in its differentiation call."
        )

    if st.button("Continue to Project Ideas"):
        builder_result_for_project = (
            st.session_state.builder_result.plan.to_builder_text() if builder_available else ""
        ) or NO_BUILDER_TEXT
        try:
            with st.spinner("Running project advisor..."):
                st.session_state.project_result = run_project(
                    st.session_state.gated_prompt,
                    st.session_state.teaching_result.text,
                    st.session_state.research_for_project,
                    builder_result=builder_result_for_project,
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
