"""
report_generator_agent.py
--------------------------
Agent 4: Report Generator

This agent combines the outputs from:
    - Agent 2: Code Analyzer
    - Agent 3: Security Scanner

It generates:
    - Unified final score
    - Prioritized issues
    - Executive summary
    - Markdown report

Design principles:
    - Tool-first report generation
    - Structured professional output
    - Shared state integration
    - Fallback-safe summary generation
"""

from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from observability import logger
from state.shared_state import CodeReviewState

from tools.report_generator_tool import (
    calculate_final_score,
    prioritize_issues,
    generate_markdown_report,
    generate_severity_heatmap
)

# ─────────────────────────────────────────────────────────────────────────────
# Agent Configuration
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_MODEL = "phi3:mini"

SYSTEM_PROMPT = """
You are a senior technical review manager.

You will receive:
- Code quality analysis results
- Security scan results

Your task:
1. Generate a professional executive summary.
2. Explain the overall quality and security posture.
3. Highlight the most critical issues.
4. Provide final recommendations.

STRICT RULES:
- Only reference provided findings.
- Do not invent issues.
- Be concise and professional.
- Maximum 200 words.

FORMAT:

EXECUTIVE SUMMARY:
<summary>

OVERALL ASSESSMENT:
<overall assessment>

TOP PRIORITIES:
1. <priority 1>
2. <priority 2>
3. <priority 3>

FINAL RECOMMENDATION:
<recommendation>
"""


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph Node Function
# ─────────────────────────────────────────────────────────────────────────────

def report_generator_node(state: CodeReviewState) -> Dict[str, Any]:
    """
    LangGraph node for Agent 4: Report Generator.

    Workflow:
        1. Read outputs from Agent 2 and Agent 3
        2. Calculate unified score
        3. Prioritize issues
        4. Generate executive summary using Ollama
        5. Generate markdown report
        6. Return updated state

    Args:
        state: Global shared state

    Returns:
        Updated state dictionary
    """

    agent_name = "report_generator"

    logger.log_agent_start(
        agent_name,
        f"Generating final report for: "
        f"{state.get('code_file_path', 'unknown')}"
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Read Agent Outputs
    # ─────────────────────────────────────────────────────────────────────

    code_quality_score = state.get("code_quality_score", 0)

    security_score = state.get("security_score", 0)

    code_issues = state.get("code_issues", [])

    security_issues = state.get("security_vulnerabilities", [])

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Calculate Final Score
    # ─────────────────────────────────────────────────────────────────────

    final_score = calculate_final_score(
        code_quality_score=code_quality_score,
        security_score=security_score
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Prioritize Issues
    # ─────────────────────────────────────────────────────────────────────

    top_issues = prioritize_issues(
        code_issues=code_issues,
        security_issues=security_issues
    )

    all_issues = code_issues + security_issues

    heatmap = generate_severity_heatmap(all_issues)

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Build Prompt
    # ─────────────────────────────────────────────────────────────────────

    prompt = f"""
Code Quality Score:
{code_quality_score}

Security Score:
{security_score}

Code Analysis Summary:
{state.get("code_analysis_summary", "")}

Security Summary:
{state.get("security_summary", "")}

Top Issues:
{top_issues}
"""

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Generate Executive Summary
    # ─────────────────────────────────────────────────────────────────────

    try:

        llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.1,
            num_predict=400
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        response = llm.invoke(messages)

        executive_summary = response.content.strip()

    except Exception as exc:

        logger.log_error(
            agent_name,
            f"Ollama unavailable: {exc}. Using fallback summary."
        )

        executive_summary = _generate_fallback_summary(
            final_score,
            code_quality_score,
            security_score
        )

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Generate Markdown Report
    # ─────────────────────────────────────────────────────────────────────

    report_data = {
        "file_name": state.get("code_file_path"),
        "language": state.get("language"),

        "code_quality_score": code_quality_score,
        "security_score": security_score,
        "final_score": final_score,

        "summary": executive_summary,

        "top_issues": top_issues,

        "heatmap": heatmap
    }

    report_file_path = generate_markdown_report(report_data)

    logger.log_tool_call(
        tool_name="generate_markdown_report",
        agent_name=agent_name,
        inputs={"final_score": final_score},
        output=report_file_path
    )

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: Update State
    # ─────────────────────────────────────────────────────────────────────

    updated_messages = state.get("messages", []) + [
        f"[report_generator] Final report generated. "
        f"Final Score: {final_score}/100"
    ]

    logger.log_agent_end(
        agent_name,
        f"Final report generated successfully | "
        f"Final Score: {final_score}/100",
        score=final_score
    )

    logger.log_state_update(
        agent_name,
        [
            "final_score",
            "final_report_summary",
            "report_file_path",
            "top_issues"
        ]
    )

    return {
        "final_score": final_score,

        "final_report_summary": executive_summary,

        "report_file_path": report_file_path,

        "top_issues": top_issues,

        "messages": updated_messages,

        "current_agent": agent_name
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Summary
# ─────────────────────────────────────────────────────────────────────────────

def _generate_fallback_summary(
    final_score: float,
    code_quality_score: float,
    security_score: float
) -> str:
    """
    Rule-based fallback summary.
    """

    return f"""
EXECUTIVE SUMMARY:
The final review process has been completed successfully.

OVERALL ASSESSMENT:
Code Quality Score: {code_quality_score}/100
Security Score: {security_score}/100
Final Weighted Score: {final_score}/100

TOP PRIORITIES:
1. Resolve all HIGH severity vulnerabilities
2. Improve maintainability issues
3. Re-run analysis after fixes

FINAL RECOMMENDATION:
Address critical findings before deployment.
"""
