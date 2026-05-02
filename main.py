"""
main.py
-------
Entry point for the Software Code Review Multi-Agent System.

Builds a LangGraph StateGraph that chains four agents sequentially:
  Coordinator → Code Analyzer → Security Scanner → Report Generator

Usage:
    python main.py <path_to_python_file>

Example:
    python main.py sample_code/sample_buggy_code.py
"""

import sys
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.panel import Panel

from agents import (
    code_analyzer_node,
    coordinator_node,
    # report_generator_node,
    security_scanner_node,
)
from observability import logger
from state.shared_state import CodeReviewState

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Build the LangGraph workflow
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph for the Code Review MAS.

        The graph follows a conditional pipeline:
            START → coordinator → code_analyzer? → security_scanner? → END

    State is passed between nodes as a shared CodeReviewState TypedDict.
    Each node returns only the keys it modifies — LangGraph merges them.

    Returns:
        Compiled LangGraph graph ready to invoke.
    """
    graph = StateGraph(CodeReviewState)

    # Register nodes (each is an agent)
    graph.add_node("coordinator", coordinator_node)
    graph.add_node("code_analyzer", code_analyzer_node)
    graph.add_node("security_scanner", security_scanner_node)
    # graph.add_node("report_generator", report_generator_node)

    def route_after_coordinator(state: CodeReviewState) -> str:
        assigned_agents = state.get("assigned_agents", [])
        if "code_analyzer" in assigned_agents:
            return "code_analyzer"
        if "security_scanner" in assigned_agents:
            return "security_scanner"
        return "end"

    def route_after_code_analyzer(state: CodeReviewState) -> str:
        assigned_agents = state.get("assigned_agents", [])
        if "security_scanner" in assigned_agents:
            return "security_scanner"
        return "end"

    # Define the conditional pipeline
    graph.add_edge(START, "coordinator")
    graph.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "code_analyzer": "code_analyzer",
            "security_scanner": "security_scanner",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "code_analyzer",
        route_after_code_analyzer,
        {
            "security_scanner": "security_scanner",
            "end": END,
        },
    )
    graph.add_edge("security_scanner", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Run the system
# ─────────────────────────────────────────────────────────────────────────────

def run_review(code_file_path: str, language: str = "python") -> CodeReviewState:
    """
    Execute the full multi-agent code review pipeline.

    Args:
        code_file_path: Path to the Python file to review.
        language:       Programming language (default: "python").

    Returns:
        Final CodeReviewState after all agents have run.
    """
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

    # Initial state — only the fields the coordinator needs
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

    # Print summary
    console.print(Panel(
        f"[bold green]Review Complete![/bold green]\n"
        f"Quality Score  : [cyan]{final_state.get('code_quality_score', 'N/A')}/100[/cyan]\n"
        f"Security Score : [cyan]{final_state.get('security_score', 'N/A')}/100[/cyan]\n"
        f"Report saved to: [yellow]{final_state.get('report_file_path', 'N/A')}[/yellow]\n"
        f"Errors         : {len(final_state.get('errors', []))}",
        title="Results",
        border_style="green",
    ))

    return final_state


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[yellow]Usage:[/yellow] python main.py <path_to_python_file>")
        console.print("[yellow]Example:[/yellow] python main.py sample_code/sample_buggy_code.py")
        sys.exit(0)

    target_file = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "python"
    run_review(target_file, lang)
