"""
report_generator_demo.py
------------------------
Standalone demo for Report Generator Agent.

Purpose:
- Tests report generation independently
- Simulates full pipeline outputs (Analyzer + Security Scanner)
- Validates scoring + markdown report generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.report_generator_agent import report_generator_node


# ─────────────────────────────────────────────
# Mock pipeline state
# ─────────────────────────────────────────────

state = {
    "code_file_path": "sample_code/sample_buggy_code.py",
    "language": "python",

    # Code Analyzer output
    "code_quality_score": 80.0,
    "code_analysis_summary": "Moderate maintainability issues detected.",
    "code_issues": [
        {
            "message": "Long function",
            "severity": "MEDIUM",
            "line": 20,
            "type": "Maintainability"
        }
    ],

    # Security Scanner output
    "security_score": 70.0,
    "security_summary": "Critical SQL injection risk identified.",
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

    # Required system fields
    "messages": [],
    "errors": [],
    "session_id": "demo-session",
    "start_time": "",
    "current_agent": "",
    "assigned_agents": [],
}


# ─────────────────────────────────────────────
# Execute agent
# ─────────────────────────────────────────────

print("\n🚀 Running Report Generator Demo...\n")

result = report_generator_node(state)


# ─────────────────────────────────────────────
# Output results
# ─────────────────────────────────────────────

print("📊 FINAL SCORE:", result.get("final_score"))

print("\n📄 EXECUTIVE SUMMARY:\n")
print(result.get("final_report_summary"))

print("\n📌 TOP PRIORITY ISSUES:")

for i, issue in enumerate(result.get("top_issues", []), 1):
    print(f"\n{i}. {issue.get('message')}")
    print(f"   Category : {issue.get('category')}")
    print(f"   Severity : {issue.get('severity')}")
    print(f"   Line     : {issue.get('line')}")

print("\n📁 Report saved at:", result.get("report_file_path"))

print("\n✅ Report Generator Demo Completed")