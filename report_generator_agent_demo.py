"""
report_generator_agent_demo.py
------------------------------
Standalone demo for Report Generator Agent.

Purpose:
- Test Report Generator independently
- Simulate outputs from Code Analyzer + Security Scanner
- Verify scoring + report generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.report_generator_agent import report_generator_node


# ─────────────────────────────────────────────────────────────
# Mock full upstream state (from other agents)
# ─────────────────────────────────────────────────────────────

state = {
    "code_file_path": "sample_code/sample_buggy_code.py",
    "language": "python",

    # Code Analyzer outputs
    "code_quality_score": 82.0,
    "code_analysis_summary": "Moderate code quality with maintainability issues.",
    "code_issues": [
        {
            "message": "Long function",
            "severity": "MEDIUM",
            "line": 20,
            "type": "Maintainability"
        }
    ],

    # Security Scanner outputs
    "security_score": 68.0,
    "security_summary": "Critical SQL Injection vulnerability detected.",
    "security_vulnerabilities": [
        {
            "message": "SQL Injection",
            "severity": "HIGH",
            "line": 50,
            "type": "Injection",
            "cwe": "CWE-89",
            "fix": "Use parameterized queries"
        }
    ],

    # Required shared fields
    "messages": [],
    "errors": [],
    "session_id": "demo-report-001",
    "start_time": "",
    "current_agent": "",
    "task_plan": "",
    "assigned_agents": [],
}


# ─────────────────────────────────────────────────────────────
# Run agent
# ─────────────────────────────────────────────────────────────

print("\n🚀 Running Report Generator Agent Demo...\n")

result = report_generator_node(state)


# ─────────────────────────────────────────────────────────────
# Output results
# ─────────────────────────────────────────────────────────────

print("📊 FINAL SCORE:", result.get("final_score"))

print("\n📄 EXECUTIVE SUMMARY:\n")
print(result.get("final_report_summary"))

print("\n📌 TOP PRIORITY ISSUES:")
for i, issue in enumerate(result.get("top_issues", []), 1):
    print(f"\n{i}. {issue.get('message')}")
    print(f"   Category : {issue.get('category')}")
    print(f"   Severity : {issue.get('severity')}")
    print(f"   Line     : {issue.get('line')}")

print("\n📁 Report File:", result.get("report_file_path"))

print("\n✅ Demo Completed\n")