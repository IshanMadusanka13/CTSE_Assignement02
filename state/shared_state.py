"""
shared_state.py
---------------
Shared state TypedDict for the Code Review MAS.

This module defines the CodeReviewState TypedDict that is passed between all agents
in the LangGraph workflow. Each agent reads from and updates this shared state.

State Flow:
  1. Coordinator: Initializes state, reads code file, creates task plan
  2. Code Analyzer: Analyzes code quality, updates code_analysis_* fields
  3. Security Scanner: Performs security scan, updates security_* fields
  4. Report Generator: Compiles final report using all previous findings
"""

from typing import Any, Dict, List, TypedDict


class CodeReviewState(TypedDict, total=False):
    """
    Shared state passed through the LangGraph workflow.

    All fields are optional (total=False) to allow agents to return only the keys
    they modify. LangGraph automatically merges returned dicts into the state.

    Fields:
        # Input & Metadata
        code_file_path          (str)          : Path to the code file to review
        code_content            (str)          : Full source code content
        language                (str)          : Programming language ("python", etc.)
        session_id              (str)          : Unique session identifier
        start_time              (str)          : ISO timestamp when review started

        # Coordinator Output
        task_plan               (str)          : Coordinator's analysis of what needs to be done
        assigned_agents         (List[str])    : List of agents to run (["code_analyzer", "security_scanner"])

        # Code Analyzer Output
        code_analysis_raw       (Dict[str,Any]): Raw output from analyze_code_quality tool
        code_quality_score      (float)        : Quality score 0-100
        code_issues             (List[Dict])   : List of code quality issues found
        code_analysis_summary   (str)          : Human-readable summary from SLM

        # Security Scanner Output
        security_vulnerabilities (List[Dict])  : List of vulnerability dicts from check_security
        security_score          (float)        : Security score 0-100
        security_risk_level     (str)          : CRITICAL | HIGH | MEDIUM | LOW | SAFE
        security_summary        (str)          : Human-readable summary from SLM

        # Report Generator Output
        final_report            (str)          : Compiled final review report
        report_file_path        (str)          : Path where report was saved

        # Execution Flow
        current_agent           (str)          : Name of agent currently executing
        messages                (List[Dict])   : Chat/execution messages for audit trail
        errors                  (List[str])    : Accumulated error messages
    """

    # Input & Metadata
    code_file_path: str
    code_content: str
    language: str
    session_id: str
    start_time: str

    # Coordinator Output
    task_plan: str
    assigned_agents: List[str]

    # Code Analyzer Output
    code_analysis_raw: Dict[str, Any]
    code_quality_score: float
    code_issues: List[Dict[str, Any]]
    code_analysis_summary: str

    # Security Scanner Output
    security_vulnerabilities: List[Dict[str, Any]]
    security_score: float
    security_risk_level: str
    security_summary: str

    # Report Generator Output
    final_report: str
    report_file_path: str

    # Execution Flow
    current_agent: str
    messages: List[Dict[str, Any]]
    errors: List[str]

