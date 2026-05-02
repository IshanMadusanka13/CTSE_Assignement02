import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.report_generator_agent import report_generator_node

state = {
    "code_file_path": "sample.py",
    "language": "python",

    "code_quality_score": 80,
    "security_score": 70,

    "code_analysis_summary": "Code quality is moderate.",

    "security_summary": "Security risks detected.",

    "code_issues": [
        {
            "message": "Long function",
            "severity": "MEDIUM",
            "line": 20
        }
    ],

    "security_vulnerabilities": [
        {
            "message": "SQL Injection",
            "severity": "HIGH",
            "line": 50
        }
    ],

    "messages": []
}

result = report_generator_node(state)

print(result)