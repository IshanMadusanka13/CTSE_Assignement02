"""
coordinator_agent.py
--------------------
Agent 1: Coordinator

This agent is the orchestrator of the Code Review MAS. It:
  1. Reads the input code file from the filesystem.
  2. Uses the coordinator_tool to analyze the file and determine routing.
  3. Routes the code to the appropriate analysis agents (Code Analyzer, Security Scanner).
  4. Manages the shared state throughout the workflow.
  5. Creates a detailed task plan with metrics and recommendations.

The Coordinator uses a tool-first approach:
  - coordinator_tool.py handles language detection, metrics extraction, and routing logic
  - Uses phi3:mini for intelligent agent selection confirmation
  - Maintains detailed analysis notes for downstream agents

Design principles:
  - Tool-driven: All file analysis done by coordinator_tool, not inline
  - Stateless analysis: Each analysis is independent and reproducible
  - Intelligent routing: Based on concrete metrics, not just LLM guessing
  - Comprehensive logging: Detailed audit trail for team review
"""

import json
import re
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from observability import logger, std_logger
from state.shared_state import CodeReviewState
from tools.coordinator_tool import analyze_file_for_routing


# ─────────────────────────────────────────────────────────────────────────────
# Agent Configuration
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_MODEL = "phi3:mini"

SYSTEM_PROMPT = """You are a code review orchestrator. Review the file analysis provided and 
confirm the routing decision.

You will receive a comprehensive analysis including:
- Language detection and confidence
- Code metrics (size, complexity, structure)
- Inferred code purpose
- Priority level assessment
- Suggested routing agents

Your job is to:
1. Review the analysis and confirm it makes sense
2. Agree or adjust the priority level if needed
3. Confirm the routing agents are appropriate
4. Add any additional concerns or observations

Respond with ONLY valid JSON, no other text.

JSON Response Format (REQUIRED):
{
  "analysis_confirmed": true,
  "language_ok": "python",
  "priority_adjusted": "MEDIUM",
  "agents_confirmed": ["code_analyzer", "security_scanner"],
  "additional_notes": "Any extra observations or concerns"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _read_code_file(file_path: str) -> str:
    """
    Read the contents of a code file safely.

    Args:
        file_path: Path to the code file.

    Returns:
        The file contents as a string.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If file cannot be read (e.g., binary file).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise ValueError(
            f"File {file_path} appears to be binary or non-UTF8. "
            f"Only text files are supported."
        ) from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Code file not found: {file_path}") from e


def _format_metrics_report(metrics: Dict[str, Any]) -> str:
    """
    Format code metrics into a human-readable report.

    Args:
        metrics: Code metrics dictionary from coordinator_tool

    Returns:
        Formatted string for display
    """
    report = f"""
CODE METRICS:
  • Total Lines: {metrics['total_lines']}
  • Code Lines: {metrics['code_lines']}
  • Comment Lines: {metrics['comment_lines']}
  • Blank Lines: {metrics['blank_lines']}
  • Functions: {metrics['function_count']}
  • Classes: {metrics['class_count']}
  • Imports: {metrics['import_count']}
  • Has Main: {metrics['has_main']}
  • Estimated Complexity: {metrics['estimated_complexity']}
  • Avg Function Length: {metrics['avg_function_length']} lines
"""
    return report.strip()


def _display_priority(priority: str) -> str:
    """Convert internal priority labels into user-facing text."""
    return "No issues found" if priority == "NONE" else priority


def _display_agents(agents: Any) -> str:
    """Convert assigned agents into a readable user-facing string."""
    if not agents:
        return "No downstream agents required"
    return ", ".join(agents)


def _parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM JSON response, with fallback to defaults.

    Args:
        response_text: Raw response from LLM

    Returns:
        Parsed JSON dict with confirmation data
    """
    if not isinstance(response_text, str):
        response_text = str(response_text)

    cleaned_text = response_text.strip()

    # Handle common LLM formatting patterns first: fenced JSON or JSON wrapped
    # in extra explanatory text.
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_text, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        # Fallback: extract the first JSON object embedded in the response.
        start_index = cleaned_text.find("{")
        end_index = cleaned_text.rfind("}")
        if start_index != -1 and end_index != -1 and end_index > start_index:
            candidate = cleaned_text[start_index : end_index + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Secondary fallback: scan line-by-line for a JSON object.
        for line in cleaned_text.split("\n"):
            stripped_line = line.strip()
            if stripped_line.startswith("{") and stripped_line.endswith("}"):
                try:
                    return json.loads(stripped_line)
                except json.JSONDecodeError:
                    continue

        # If parsing still fails, return safe default
        std_logger.warning("Coordinator: Failed to parse LLM response. Using defaults.")
        return {
            "analysis_confirmed": True,
            "language_ok": "unknown",
            "priority_adjusted": "MEDIUM",
            "agents_confirmed": ["code_analyzer", "security_scanner"],
            "additional_notes": "LLM response parsing failed",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Coordinator Node (called by LangGraph)
# ─────────────────────────────────────────────────────────────────────────────

def coordinator_node(state: CodeReviewState) -> Dict[str, Any]:
    """
    LangGraph node for Agent 1: Coordinator.

    Workflow:
        1. Read the input code file from disk.
        2. Use coordinator_tool to analyze the file comprehensively.
        3. Call LLM to confirm routing decisions.
        4. Initialize shared state with analysis results.
        5. Return updated state for downstream agents.

    Args:
        state: The global CodeReviewState shared across all agents.
               Should contain 'code_file_path' and 'language' (optional).

    Returns:
        Dict with updated state keys:
            - code_content              (str)          : The full source code
            - language                  (str)          : Detected/confirmed language
            - task_plan                 (str)          : Detailed plan with metrics
            - assigned_agents           (List[str])    : Agents to execute
            - start_time                (str)          : ISO timestamp
            - coordinator_analysis      (Dict)         : Full analysis from tool
            - current_agent             (str)          : "coordinator"
            - messages                  (List[Dict])   : Execution log
            - errors                    (List[str])    : Error messages if any
    """
    agent_name = "coordinator"

    logger.log_agent_start(
        agent_name,
        f"Starting orchestration for: {state.get('code_file_path', 'unknown')}",
    )

    # ── Step 1: Read the code file from disk ─────────────────────────────
    code_file_path = state.get("code_file_path", "")
    if not code_file_path:
        error_msg = "No code_file_path provided in initial state"
        logger.log_error(agent_name, error_msg)
        return {
            "errors": [error_msg],
            "current_agent": agent_name,
        }

    try:
        code_content = _read_code_file(code_file_path)
        logger.log_tool_call(
            tool_name="read_code_file",
            agent_name=agent_name,
            inputs={"file_path": code_file_path},
            output=f"Successfully read {len(code_content)} characters",
        )
    except (FileNotFoundError, ValueError) as exc:
        error_msg = f"Failed to read code file: {exc}"
        logger.log_error(agent_name, error_msg)
        return {
            "errors": [error_msg],
            "current_agent": agent_name,
        }

    # ── Step 2: Use coordinator_tool for comprehensive analysis ──────────
    try:
        tool_analysis = analyze_file_for_routing(
            file_path=code_file_path,
            code_content=code_content,
        )

        logger.log_tool_call(
            tool_name="analyze_file_for_routing",
            agent_name=agent_name,
            inputs={"file_path": code_file_path},
            output=f"Language: {tool_analysis['language']}, "
            f"Priority: {tool_analysis['priority']}, "
            f"Agents: {tool_analysis['routing_agents']}",
        )

    except Exception as exc:
        error_msg = f"Tool error in coordinator_tool: {exc}"
        logger.log_error(agent_name, error_msg)
        return {
            "errors": [error_msg],
            "current_agent": agent_name,
        }

    # ── Step 3: Use LLM to confirm routing ──────────────────────────
    metrics_report = _format_metrics_report(tool_analysis["metrics"])

    llm_input = f"""File Analysis Complete. Please review and confirm routing decision:

FILE: {tool_analysis['file_path']}
FILE SIZE: {tool_analysis['file_size_bytes']} bytes
LANGUAGE: {tool_analysis['language']} (confidence: {tool_analysis['confidence']})

{metrics_report}

CODE PURPOSE: {tool_analysis['code_purpose']}
PRIORITY LEVEL: {tool_analysis['priority']}
COMPLEXITY ASSESSMENT: {tool_analysis['complexity_assessment']}

SPECIAL NOTES: {tool_analysis['analysis_notes']}

RECOMMENDED ROUTING: {", ".join(tool_analysis['routing_agents'])}

Please confirm this analysis and routing decision using the JSON format in your instructions."""

    try:
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url="http://localhost:11434",
            temperature=0.3,  # Low temp for deterministic confirmation
        )

        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=llm_input),
            ]
        )

        llm_confirmation = _parse_llm_response(response.content)

        logger.log_tool_call(
            tool_name="ollama",
            agent_name=agent_name,
            inputs={"model": OLLAMA_MODEL, "task": "confirm_routing"},
            output=f"Routing confirmed: {llm_confirmation.get('agents_confirmed', [])}",
        )

    except Exception as exc:
        std_logger.warning(
            "Coordinator: Ollama unavailable (%s). Using tool analysis directly.",
            exc,
        )
        # Fallback: use tool analysis directly
        llm_confirmation = {
            "analysis_confirmed": True,
            "language_ok": tool_analysis["language"],
            "priority_adjusted": tool_analysis["priority"],
            "agents_confirmed": tool_analysis["routing_agents"],
            "additional_notes": "Using tool analysis only (Ollama unavailable)",
        }

    # ── Step 4: Merge tool analysis with LLM confirmation ────────────────
    final_language = llm_confirmation.get("language_ok", tool_analysis["language"])
    if final_language == "unknown" and tool_analysis.get("language") != "unknown":
        final_language = tool_analysis["language"]
    final_priority = llm_confirmation.get("priority_adjusted", tool_analysis["priority"])
    assigned_agents = tool_analysis["routing_agents"]

    if not assigned_agents:
        final_priority = "NONE"

    display_priority = _display_priority(final_priority)
    display_quality_priority = _display_priority(tool_analysis["code_quality_priority"])
    display_security_priority = _display_priority(tool_analysis["security_priority"])

    confirmed_agents = llm_confirmation.get("agents_confirmed", [])
    # Intentionally do not log LLM vs tool routing mismatch to avoid
    # user-facing messages; coordinator trusts tool routing by design.

    # ── Step 5: Build comprehensive task plan ─────────────────────────────
    task_plan = f"""═══════════════════════════════════════════════════════════════════════════════════
CODE REVIEW TASK PLAN
═══════════════════════════════════════════════════════════════════════════════════

FILE INFORMATION:
  Path: {tool_analysis['file_path']}
  Size: {tool_analysis['file_size_bytes']} bytes
  Language: {final_language} (confidence: {tool_analysis['confidence']})

CODE ANALYSIS:
  Purpose: {tool_analysis['code_purpose']}
        Overall Priority: {display_priority}
        Code Quality Priority: {display_quality_priority} → code_analyzer
    Security Priority: {display_security_priority} → security_scanner
  {metrics_report}

COMPLEXITY ASSESSMENT:
  {tool_analysis['complexity_assessment']}

ROUTING RATIONALE:
    {tool_analysis['routing_rationale']}

ROUTING DECISION:
    Agents Assigned: {_display_agents(assigned_agents)}
"""

    if not assigned_agents:
        task_plan += """

NO ISSUES FOUND:
  - The coordinator detected no quality or security issues.
  - The review ends here with no downstream agents.
"""

    for agent in assigned_agents:
        if agent == "code_analyzer":
            task_plan += """
  • CODE ANALYZER:
      - Perform static code quality analysis
      - Identify code smells and maintainability issues
      - Assess architectural patterns
      - Calculate quality metrics and scores"""
        elif agent == "security_scanner":
            task_plan += """
  • SECURITY SCANNER:
      - Identify security vulnerabilities
      - Scan for CWE violations
      - Assess cryptographic practices
      - Check for injection/XSS/CSRF risks"""

    additional_notes = llm_confirmation.get("additional_notes", "")
    if additional_notes and additional_notes != "LLM response parsing failed":
        task_plan += f"\n\nADDITIONAL NOTES FROM COORDINATOR:\n  {additional_notes}"

    if tool_analysis["analysis_notes"] != "No special notes":
        task_plan += f"\n\nANALYSIS NOTES:\n  {tool_analysis['analysis_notes']}"

    task_plan += "\n" + "═" * 83

    # ── Step 6: Return updated state ──────────────────────────────────────
    start_time = datetime.now().isoformat()

    # Present coordinator completion using the AgentTracer boxed UI
    output_summary = (
        f"Language: {final_language} | Code Quality Priority: {display_quality_priority} | "
        f"Security Priority: {display_security_priority} | Agents: {_display_agents(assigned_agents)}\n"
        f"{tool_analysis['routing_rationale']}"
    )
    logger.log_agent_end(agent_name, output_summary)

    return {
        "code_content": code_content,
        "language": final_language,
        "task_plan": task_plan,
        "assigned_agents": assigned_agents,
        "start_time": start_time,
        "coordinator_analysis": tool_analysis,  # Store full analysis for reference
        "current_agent": agent_name,
        "messages": state.get("messages", [])
        + [
            {
                "agent": agent_name,
                "action": "coordinate",
                "language": final_language,
                "priority": final_priority,
                "agents": assigned_agents,
                "timestamp": start_time,
            }
        ],
    }


