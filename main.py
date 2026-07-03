import argparse

from crewai_groq_demo.cost import estimate_groq_cost, estimate_tavily_cost, format_groq_cost
from crewai_groq_demo.crew import NO_RESEARCH_TEXT, run_project, run_research, run_teaching
from crewai_groq_demo.exceptions import CrewDemoError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CrewAI + Groq teacher/project-advisor pipeline."
    )
    parser.add_argument("prompt", help="What you want the agents to help with.")
    parser.add_argument(
        "--research",
        action="store_true",
        help="Run web research (Tavily) before teaching.",
    )
    parser.add_argument(
        "--search-depth",
        choices=["basic", "advanced"],
        default=None,
        help="Tavily search depth for --research (defaults to TAVILY_SEARCH_DEPTH setting). "
        "Advanced costs 2x the Tavily credits of basic.",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Write the project ideas as markdown to PATH.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_cost = 0.0

    try:
        research_result = NO_RESEARCH_TEXT
        if args.research:
            research = run_research(args.prompt, search_depth=args.search_depth)
            search_word = "search" if research.successful_search_count == 1 else "searches"
            print(
                f"\n=== Researcher's Findings "
                f"({research.successful_search_count} {search_word}) ==="
            )
            for i, query in enumerate(research.queries, start=1):
                print(f"  {i}. {query}")
            if research.retries > 0:
                print(
                    f"(needed {research.retries} retry(ies) after malformed tool calls — "
                    f"{research.total_search_count} total searches across all attempts)"
                )
            print(research.text)

            if research.from_cache:
                print("Research cost: $0 (cached)")
            else:
                research_cost = estimate_groq_cost(research.usage) + estimate_tavily_cost(
                    research.successful_search_count, research.search_depth
                )
                total_cost += research_cost
                print(f"Research cost: ~${research_cost:.4f}")
            research_result = research.text

        teaching = run_teaching(args.prompt, research_result)
        print("\n=== Teacher's Explanation ===")
        print(teaching.text)
        teaching_cost = estimate_groq_cost(teaching.usage)
        total_cost += teaching_cost
        print(f"Teaching cost: ~${teaching_cost:.4f}")

        project = run_project(
            args.prompt, teaching.text, research_result, output_path=args.output
        )
        print("\n=== Project Ideas ===")
        print(project.ideas.to_markdown())
        print(f"Project-advisor cost: {format_groq_cost(project.usage)}")
        if project.usage.successful_requests > 0:
            total_cost += estimate_groq_cost(project.usage)
        if args.output:
            print(f"Wrote project ideas to {args.output}")

        total_note = (
            "" if project.usage.successful_requests > 0 else " (excludes project-advisor call)"
        )
        print(f"\nTotal estimated cost: ~${total_cost:.4f}{total_note}")
    except CrewDemoError as error:
        print(f"\nError: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
