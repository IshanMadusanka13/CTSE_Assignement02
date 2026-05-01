"""
security_scanner_agent.py
--------------------------
Agent 3: Security Scanner

This agent is responsible for performing static security vulnerability analysis
on the submitted source file. It uses the custom `check_security` tool to detect
real vulnerabilities without relying on the LLM's internal knowledge, then uses
the SLM (via Ollama) to produce a professional, prioritised security report with
actionable remediation advice.

Design principles:
  - Tool-first: All vulnerability detection is done by check_security(), not the LLM.
  - Constrained output: System prompt forces a strict format to prevent hallucination.
  - Fallback safe: If Ollama is unavailable, rule-based summary is generated instead.
  - Context-efficient: Only top 5 findings are sent to the SLM to fit small context windows.
"""

import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from observability import logger
from state.shared_state import CodeReviewState
from tools.security_scan_tool import check_security


# ─────────────────────────────────────────────────────────────────────────────
# Agent Configuration
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_MODEL = "phi3:mini"

# System prompt — engineered to prevent the SLM from inventing vulnerabilities.
# Key constraints:
#   1. Role is specific: "security engineer", not "helpful assistant".
#   2. Explicit instruction to ONLY reference findings in the tool report.
#   3. Fixed output schema — prevents SLM from producing freeform text.
#   4. Word limit prevents rambling on small context-window models.
SYSTEM_PROMPT = """You are a senior application security engineer performing a professional security audit.

You will receive a JSON security scan report produced by a static analysis tool. Your job is to:
1. Write a 2-3 sentence executive summary of the overall security posture.
2. List the TOP 3 most critical vulnerabilities, ordered by severity (CRITICAL → HIGH → MEDIUM).
3. For each vulnerability: state the CWE identifier, the affected line, and one specific fix.
4. State the overall security risk level and score.

STRICT RULES:
- Only reference vulnerabilities that appear in the provided tool report. Do NOT invent new findings.
- Do NOT add general security advice not backed by a finding in the report.
- Be concise and professional. No introductory paragraphs. Start directly with SUMMARY.
- Format your response EXACTLY as shown below:

SUMMARY: <2-3 sentence executive summary of the security posture>

TOP VULNERABILITIES:
1. [SEVERITY] <CWE-ID> — Line X — <vulnerability description> → Fix: <specific remediation>
2. [SEVERITY] <CWE-ID> — Line X — <vulnerability description> → Fix: <specific remediation>
3. [SEVERITY] <CWE-ID> — Line X — <vulnerability description> → Fix: <specific remediation>

RISK LEVEL: <CRITICAL|HIGH|MEDIUM|LOW|SAFE>
SECURITY SCORE: <score>/100
RECOMMENDATION: <one overall remediation priority>"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent node function (called by LangGraph)
# ─────────────────────────────────────────────────────────────────────────────


def security_scanner_node(state: CodeReviewState) -> Dict[str, Any]:
    """
    LangGraph node for Agent 3: Security Scanner.

    Workflow:
        1. Read code_content and file_path from shared state (set by Coordinator).
        2. Call the `check_security` tool — no LLM knowledge used at this stage.
        3. Pass the top findings to the SLM (Ollama) for professional summarisation.
        4. Return updated state keys for the Report Generator.

    Args:
        state: The global CodeReviewState shared across all agents.

    Returns:
        Dict with updated state keys:
            - security_vulnerabilities  (list of vuln dicts from the tool)
            - security_score            (0-100 float)
            - security_risk_level       (CRITICAL | HIGH | MEDIUM | LOW | SAFE)
            - security_summary          (LLM-generated professional summary)
            - messages                  (appended)
            - current_agent
            - errors                    (appended on failure)
    """
    agent_name = "security_scanner"
    logger.log_agent_start(
        agent_name,
        f"Scanning file: {state.get('code_file_path', 'unknown')} "
        f"({len(state.get('code_content', ''))} characters)",
    )

    # ── Step 1: Run the security scan tool ───────────────────────────────
    try:
        tool_inputs = {
            "code_content": state["code_content"],
            "file_path": state.get("code_file_path", "unknown.py"),
            "language": state.get("language", "python"),
        }

        scan_result = check_security(**tool_inputs)

        logger.log_tool_call(
            tool_name="check_security",
            agent_name=agent_name,
            inputs={k: v for k, v in tool_inputs.items() if k != "code_content"},
            output=(
                f"Score: {scan_result['security_score']}, "
                f"Risk: {scan_result['risk_level']}, "
                f"Findings: {scan_result['vulnerability_counts']['TOTAL']}"
            ),
        )

    except (ValueError, KeyError) as exc:
        error_msg = f"Tool error in check_security: {exc}"
        logger.log_error(agent_name, error_msg)
        return {
            "errors": state.get("errors", []) + [error_msg],
            "current_agent": agent_name,
        }

    # ── Step 2: Build a compact prompt for the SLM ───────────────────────
    # Only send top 5 findings to keep prompt size small for SLMs.
    top_vulns = scan_result["vulnerabilities"][:5]

    llm_input_data = {
        "file": scan_result["file_path"],
        "security_score": scan_result["security_score"],
        "risk_level": scan_result["risk_level"],
        "vulnerability_counts": scan_result["vulnerability_counts"],
        "top_vulnerabilities": [
            {
                "type": v["type"],
                "severity": v["severity"],
                "line": v["line"],
                "cwe": v["cwe"],
                "message": v["message"],
                "fix": v["fix"],
            }
            for v in top_vulns
        ],
    }

    user_prompt = (
        "Here is the static security scan report for the code file:\n\n"
        f"{json.dumps(llm_input_data, indent=2)}\n\n"
        "Please provide your professional security assessment following the format in your instructions."
    )

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
        # Ollama unavailable — generate rule-based summary as fallback
        logger.log_error(agent_name, f"Ollama unavailable: {exc}. Using rule-based summary.")
        summary = _generate_fallback_summary(scan_result)

    # ── Step 4: Update and return state ──────────────────────────────────
    updated_messages = state.get("messages", []) + [
        f"[security_scanner] Scan complete. Score: {scan_result['security_score']}/100, "
        f"Risk: {scan_result['risk_level']}, "
        f"Findings: {scan_result['vulnerability_counts']['TOTAL']}"
    ]

    logger.log_agent_end(
        agent_name,
        f"Security score: {scan_result['security_score']}/100 | "
        f"Risk: {scan_result['risk_level']} | "
        f"Findings: {scan_result['vulnerability_counts']}",
        score=scan_result["security_score"],
    )
    logger.log_state_update(
        agent_name,
        ["security_vulnerabilities", "security_score", "security_risk_level", "security_summary"],
    )

    return {
        "security_vulnerabilities": scan_result["vulnerabilities"],
        "security_score": scan_result["security_score"],
        "security_risk_level": scan_result["risk_level"],
        "security_summary": summary,
        "messages": updated_messages,
        "current_agent": agent_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fallback summary (used when Ollama is offline)
# ─────────────────────────────────────────────────────────────────────────────


def _generate_fallback_summary(scan: Dict[str, Any]) -> str:
    """
    Generate a rule-based security summary when Ollama is unavailable.

    Args:
        scan: The raw dict returned by check_security().

    Returns:
        A formatted string summary matching the agent's output schema.
    """
    score = scan["security_score"]
    counts = scan["vulnerability_counts"]
    risk = scan["risk_level"]
    top = scan["vulnerabilities"][:3]

    lines = [
        f"SUMMARY: Security scan complete. Risk level is {risk} with a score of {score}/100. "
        f"Found {counts['CRITICAL']} CRITICAL, {counts['HIGH']} HIGH, "
        f"{counts['MEDIUM']} MEDIUM, and {counts['LOW']} LOW severity findings.\n"
    ]

    lines.append("TOP VULNERABILITIES:")
    for i, vuln in enumerate(top, 1):
        lines.append(
            f"{i}. [{vuln['severity']}] {vuln['cwe']} — "
            f"Line {vuln['line']} — {vuln['message']} → Fix: {vuln['fix']}"
        )

    if not top:
        lines.append("No vulnerabilities detected.")

    lines.append(f"\nRISK LEVEL: {risk}")
    lines.append(f"SECURITY SCORE: {score}/100")
    if counts["CRITICAL"] > 0:
        lines.append("RECOMMENDATION: Immediately address all CRITICAL findings before deployment.")
    elif counts["HIGH"] > 0:
        lines.append("RECOMMENDATION: Resolve all HIGH severity vulnerabilities before code review approval.")
    else:
        lines.append("RECOMMENDATION: Address remaining findings in the next sprint cycle.")

    return "\n".join(lines)
