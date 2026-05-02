"""
main.py
-------
Entry point for the Software Code Review Multi-Agent System.

Pipeline (conditional):
  Coordinator → Code Analyzer → Security Scanner → Report Generator → END

Agents execute based on coordinator decisions.
"""

import sys
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.panel import Panel

from agents import (
    code_analyzer_node,
    coordinator_node,
    security_scanner_node,
    report_generator_node,
)

from observability import logger
from state.shared_state import CodeReviewState

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:

    graph = StateGraph(CodeReviewState)

    # Register agents
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("code_analyzer", code_analyzer_node)
    graph.add_node("security_scanner", security_scanner_node)
    # graph.add_node("report_generator", report_generator_node)

    # ─────────────────────────────────────────────
    # Routing logic
    # ─────────────────────────────────────────────

    def route_after_coordinator(state: CodeReviewState) -> str:
        assigned = state.get("assigned_agents", [])

        # priority order
        if "code_analyzer" in assigned:
            return "code_analyzer"
        if "security_scanner" in assigned:
            return "security_scanner"

        # fallback → still generate report
        return "report_generator"


    def route_after_code_analyzer(state: CodeReviewState) -> str:
        assigned = state.get("assigned_agents", [])

        if "security_scanner" in assigned:
            return "security_scanner"

        return "report_generator"


    def route_after_security_scanner(state: CodeReviewState) -> str:
        # ALWAYS go to report generation
        return "report_generator"


    def route_after_report(state: CodeReviewState) -> str:
        return END


    # ─────────────────────────────────────────────
    # Graph edges
    # ─────────────────────────────────────────────

    graph.add_edge(START, "coordinator")

    graph.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "code_analyzer": "code_analyzer",
            "security_scanner": "security_scanner",
            "report_generator": "report_generator",
        },
    )

    graph.add_conditional_edges(
        "code_analyzer",
        route_after_code_analyzer,
        {
            "security_scanner": "security_scanner",
            "report_generator": "report_generator",
        },
    )

    graph.add_conditional_edges(
        "security_scanner",
        route_after_security_scanner,
        {
            "report_generator": "report_generator",
        },
    )

    graph.add_edge("report_generator", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# RUN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

def run_review(code_file_path: str, language: str = "python"):

    if not Path(code_file_path).exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {code_file_path}")
        sys.exit(1)

    console.print(Panel(
        f"[bold]Software Code Review MAS[/bold]\n"
        f"File: [cyan]{code_file_path}[/cyan]\n"
        f"Session: {logger.session_id}",
        title="Starting Review",
        border_style="blue",
    ))

    # Initial state
    initial_state: CodeReviewState = {
        "code_file_path": code_file_path,
        "code_content": "",
        "language": language,
        "task_plan": "",
        "assigned_agents": [],

        "code_analysis_raw": {},
        "code_quality_score": 0.0,
        "code_issues": [],
        "code_analysis_summary": "",

        "security_vulnerabilities": [],
        "security_score": 0.0,
        "security_summary": "",

        "final_report": "",
        "report_file_path": "",

        "session_id": logger.session_id,
        "start_time": "",
        "current_agent": "",
        "messages": [],
        "errors": [],
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    # ─────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────

    console.print(Panel(
        f"[bold green]Review Complete![/bold green]\n\n"
        f"Quality Score   : [cyan]{final_state.get('code_quality_score', 'N/A')}/100[/cyan]\n"
        f"Security Score  : [cyan]{final_state.get('security_score', 'N/A')}/100[/cyan]\n"
        f"Final Score     : [cyan]{final_state.get('final_score', 'N/A')}/100[/cyan]\n"
        f"Report File     : [yellow]{final_state.get('report_file_path', 'N/A')}[/yellow]\n"
        f"Errors          : {len(final_state.get('errors', []))}",
        title="Results",
        border_style="green",
    ))

    return final_state


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) < 2:
        console.print("[yellow]Usage:[/yellow] python main.py <file>")
        sys.exit(0)

    target_file = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "python"

    run_review(target_file, lang)
