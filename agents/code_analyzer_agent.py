"""
code_analyzer_agent.py
----------------------
Agent 2: Code Analyzer

This agent is responsible for performing static code quality analysis on the
submitted source file. It uses the custom `analyze_code_quality` tool to
examine the code without relying solely on the LLM's internal knowledge,
then uses the SLM (via Ollama) to produce a human-readable summary with
actionable recommendations.

"""

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from observability import logger
from state.shared_state import CodeReviewState
from tools.code_analysis_tool import analyze_code_quality


# ─────────────────────────────────────────────────────────────────────────────
# Agent Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Use phi3:mini — lightweight, runs well on 12GB RAM with integrated GPU.
# Fallback to qwen2:1.5b if phi3 is not pulled.
OLLAMA_MODEL = "phi3:mini"

# System prompt — carefully engineered for SLM capabilities.
# Key principles:
#   1. Constrained output format (JSON-like bullet list) prevents hallucination.
#   2. Role is specific: "senior software engineer", not "helpful assistant".
#   3. Explicit instruction NOT to invent issues not in the tool output.
#   4. Word limit prevents the SLM from rambling.
SYSTEM_PROMPT = """You are a senior software engineer performing a professional code quality review.

You will receive a JSON tool report from a static analysis tool. Your job is to:
1. Summarize the findings in clear, professional English (max 200 words).
2. List the TOP 3 most important issues to fix, ordered by severity (HIGH → MEDIUM → LOW).
3. Give one specific, actionable recommendation for each of the top 3 issues.
4. State the overall quality score out of 100.

STRICT RULES:
- Only reference issues that appear in the provided tool report. Do NOT invent new issues.
- Be concise. No introductory paragraphs. Start directly with the summary.
- Format your response exactly as shown:

SUMMARY: <2-3 sentence overview>

TOP ISSUES:
1. [SEVERITY] Line X — <issue> → Fix: <specific action>
2. [SEVERITY] Line X — <issue> → Fix: <specific action>
3. [SEVERITY] Line X — <issue> → Fix: <specific action>

QUALITY SCORE: <score>/100
RECOMMENDATION: <one overall recommendation>"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent node function (called by LangGraph)
# ─────────────────────────────────────────────────────────────────────────────


def code_analyzer_node(state: CodeReviewState) -> Dict[str, Any]:
    """
    LangGraph node for Agent 2: Code Analyzer.

    Workflow:
        1. Read code_content and file_path from shared state.
        2. Call the `analyze_code_quality` tool (no LLM knowledge used here).
        3. Pass the tool's JSON report to the SLM for human-readable summarisation.
        4. Return updated state keys.

    Args:
        state: The global CodeReviewState shared across all agents.

    Returns:
        Dict with updated state keys:
            - code_analysis_raw
            - code_quality_score
            - code_issues
            - code_analysis_summary
            - messages  (appended)
            - current_agent
            - errors    (appended on failure)
    """
    agent_name = "code_analyzer"
    logger.log_agent_start(
        agent_name,
        f"Analyzing file: {state.get('code_file_path', 'unknown')} "
        f"({len(state.get('code_content', ''))} characters)",
    )

    # ── Step 1: Run the static analysis tool ─────────────────────────────
    try:
        tool_inputs = {
            "code_content": state["code_content"],
            "file_path": state.get("code_file_path", "unknown.py"),
            "language": state.get("language", "python"),
        }

        analysis_result = analyze_code_quality(**tool_inputs)

        logger.log_tool_call(
            tool_name="analyze_code_quality",
            agent_name=agent_name,
            inputs={k: v for k, v in tool_inputs.items() if k != "code_content"},
            output=f"Score: {analysis_result['quality_score']}, Issues: {analysis_result['issue_counts']['TOTAL']}",
        )

    except (ValueError, KeyError) as exc:
        error_msg = f"Tool error in analyze_code_quality: {exc}"
        logger.log_error(agent_name, error_msg)
        return {
            "errors": state.get("errors", []) + [error_msg],
            "current_agent": agent_name,
        }

    # ── Step 2: Build prompt for SLM summarisation ────────────────────────
    # We feed only the essential data to the SLM to keep the context short
    # (important for small models with limited context windows).
    top_issues = sorted(
        analysis_result["issues"],
        key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x["severity"], 3),
    )[:5]  # send only top 5 to keep prompt short

    llm_input_data = {
        "file": analysis_result["file_path"],
        "quality_score": analysis_result["quality_score"],
        "issue_counts": analysis_result["issue_counts"],
        "metrics": {
            k: v for k, v in analysis_result["metrics"].items()
            if k in ["total_lines", "num_functions", "num_classes", "docstring_coverage_pct"]
        },
        "top_issues": top_issues,
        "avg_cyclomatic_complexity": analysis_result["cyclomatic_complexity"].get("average", "N/A"),
    }

    user_prompt = f"""Here is the static analysis tool report for the code file:

{json.dumps(llm_input_data, indent=2)}

Please provide your professional code quality assessment following the format in your instructions."""

    # ── Step 3: Call SLM via Ollama ───────────────────────────────────────
    summary = ""
    try:
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1, num_predict=400)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        summary = response.content.strip()

    except Exception as exc:  # noqa: BLE001
        # If Ollama is unavailable, generate a rule-based summary as fallback
        logger.log_error(agent_name, f"Ollama unavailable: {exc}. Using rule-based summary.")
        summary = _generate_fallback_summary(analysis_result)

    # ── Step 4: Update and return state ──────────────────────────────────
    updated_messages = state.get("messages", []) + [
        f"[code_analyzer] Analysis complete. Score: {analysis_result['quality_score']}/100, "
        f"Issues found: {analysis_result['issue_counts']['TOTAL']}"
    ]

    logger.log_agent_end(
        agent_name,
        f"Quality score: {analysis_result['quality_score']}/100 | "
        f"Issues: {analysis_result['issue_counts']}",
        score=analysis_result["quality_score"],
    )
    logger.log_state_update(
        agent_name,
        ["code_analysis_raw", "code_quality_score", "code_issues", "code_analysis_summary"],
    )

    return {
        "code_analysis_raw": analysis_result,
        "code_quality_score": analysis_result["quality_score"],
        "code_issues": analysis_result["issues"],
        "code_analysis_summary": summary,
        "messages": updated_messages,
        "current_agent": agent_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fallback summary (used when Ollama is offline)
# ─────────────────────────────────────────────────────────────────────────────


def _generate_fallback_summary(analysis: Dict[str, Any]) -> str:
    """
    Generate a rule-based summary when the Ollama LLM is unavailable.

    Args:
        analysis: The raw dict returned by analyze_code_quality().

    Returns:
        A formatted string summary.
    """
    score = analysis["quality_score"]
    counts = analysis["issue_counts"]
    top = sorted(
        analysis["issues"],
        key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x["severity"], 3),
    )[:3]

    lines = [f"SUMMARY: Code quality score is {score}/100. "
             f"Found {counts['HIGH']} HIGH, {counts['MEDIUM']} MEDIUM, "
             f"and {counts['LOW']} LOW severity issues.\n"]

    lines.append("TOP ISSUES:")
    for i, issue in enumerate(top, 1):
        lines.append(f"{i}. [{issue['severity']}] Line {issue['line']} — {issue['message']}")

    lines.append(f"\nQUALITY SCORE: {score}/100")
    lines.append("RECOMMENDATION: Address all HIGH severity issues before code review approval.")
    return "\n".join(lines)
