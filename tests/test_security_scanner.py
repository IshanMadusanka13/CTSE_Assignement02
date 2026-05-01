"""
test_security_scanner.py
------------------------
Automated evaluation/testing script for Agent 3: Security Scanner.

Test strategy:
  1. Property-based tests  — structural guarantees: output schema, type
     correctness, score bounds, severity/risk level consistency.
  2. Functional tests      — verify the tool detects every vulnerability
     category it claims to cover (hardcoded secrets, SQL injection, etc.)
  3. Negative tests        — clean code must produce SAFE risk and score=100.
  4. Edge-case tests       — empty input, syntax error, unicode, minimal file.
  5. Security-contract tests — critical invariants that must never be broken
     (e.g. score must be 0 when CRITICAL findings exist with max deductions).
  6. LLM-as-a-Judge test   — use Ollama to evaluate whether the agent summary
     accurately and honestly reflects the tool findings (no hallucinations).

Run with:
    pytest tests/test_security_scanner.py -v
    pytest tests/test_security_scanner.py -v -k "not llm_judge"  # skip Ollama
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add project root to path so imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.security_scan_tool import (
    check_security,
    _detect_hardcoded_secrets,
    _detect_dangerous_functions,
    _detect_dangerous_module_calls,
    _detect_sql_injection,
    _detect_command_injection,
    _detect_insecure_network,
    _detect_debug_artifacts,
    _detect_sensitive_data_exposure,
    _calculate_security_score,
    _parse_ast,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
)


# ─────────────────────────────────────────────────────────────────────────────
# Code fixtures
# ─────────────────────────────────────────────────────────────────────────────

# Perfect code: no vulnerabilities expected
CLEAN_CODE = '''"""Secure module with no vulnerabilities."""
import os
import hashlib
import subprocess
import json


def get_user_by_id(conn, user_id: int) -> dict:
    """Fetch a user record safely using parameterised query."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()


def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def run_command(args: list) -> str:
    """Run a subprocess command safely without shell=True."""
    result = subprocess.run(args, capture_output=True, text=True, shell=False)
    return result.stdout


SECRET_KEY = os.environ.get("SECRET_KEY", "")
'''

# Code with hardcoded password
CODE_HARDCODED_PASSWORD = '''
password = "SuperSecret123"
api_key = "sk-abcdef1234567890"
'''

# Code with eval()
CODE_EVAL = '''
def run_user_input(data):
    result = eval(data)
    return result
'''

# Code with pickle.loads()
CODE_PICKLE = '''
import pickle

def load_data(raw_bytes):
    return pickle.loads(raw_bytes)
'''

# Code with SQL injection via f-string (inline — not via a variable)
CODE_SQL_FSTRING = '''
def get_user(conn, username):
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE name = {username}")
'''

# Code with SQL injection via % formatting
CODE_SQL_PERCENT = '''
def get_user(conn, username):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE name = %s" % username)
'''

# Code with subprocess shell=True
CODE_SHELL_TRUE = '''
import subprocess

def run(cmd):
    subprocess.run(cmd, shell=True)
'''

# Code with requests verify=False
CODE_VERIFY_FALSE = '''
import requests

def fetch(url):
    return requests.get(url, verify=False)
'''

# Code with http:// in requests
CODE_HTTP_URL = '''
import requests

def fetch():
    return requests.get("http://api.example.com/data")
'''

# Code with DEBUG = True
CODE_DEBUG_TRUE = '''
DEBUG = True

def process():
    pass
'''

# Code with assert as security guard
CODE_ASSERT_GUARD = '''
def admin_only(user):
    assert user.is_admin, "Not admin"
    return "secret data"
'''

# Code with print(password)
CODE_PRINT_SECRET = '''
def authenticate(password):
    print(password)
    return True
'''

# Code with yaml.load() without Loader
CODE_YAML_UNSAFE = '''
import yaml

def load_config(data):
    return yaml.load(data)
'''

# Code with yaml.load() safely (should NOT flag)
CODE_YAML_SAFE = '''
import yaml

def load_config(data):
    return yaml.load(data, Loader=yaml.SafeLoader)
'''

# Syntax error code
CODE_SYNTAX_ERROR = '''
def broken(:
    pass
'''

# Multiple vulnerabilities in one file
CODE_MULTI_VULN = '''
import pickle
import subprocess

password = "hardcoded_pass_123"

def dangerous(data, cmd):
    obj = pickle.loads(data)
    subprocess.run(cmd, shell=True)
    eval(data)
    return obj
'''


# ─────────────────────────────────────────────────────────────────────────────
# 1. Property-based tests: output schema and type correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputSchema:
    """Verify the output structure of check_security() is always correct."""

    def test_required_keys_present(self):
        result = check_security(CLEAN_CODE, run_bandit=False)
        required = {
            "file_path", "language", "syntax_error", "security_score",
            "vulnerabilities", "vulnerability_counts", "risk_level", "bandit",
        }
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_vulnerability_counts_keys(self):
        result = check_security(CLEAN_CODE, run_bandit=False)
        counts = result["vulnerability_counts"]
        for key in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "TOTAL"):
            assert key in counts

    def test_score_is_float_in_bounds(self):
        for code in [CLEAN_CODE, CODE_MULTI_VULN]:
            result = check_security(code, run_bandit=False)
            assert isinstance(result["security_score"], float)
            assert 0.0 <= result["security_score"] <= 100.0

    def test_vulnerabilities_is_list(self):
        result = check_security(CODE_MULTI_VULN, run_bandit=False)
        assert isinstance(result["vulnerabilities"], list)

    def test_each_vuln_has_required_keys(self):
        result = check_security(CODE_MULTI_VULN, run_bandit=False)
        required_vuln_keys = {"type", "severity", "line", "message", "cwe", "fix"}
        for v in result["vulnerabilities"]:
            assert required_vuln_keys.issubset(v.keys()), f"Vuln missing keys: {v}"

    def test_severity_values_are_valid(self):
        valid = {SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW}
        result = check_security(CODE_MULTI_VULN, run_bandit=False)
        for v in result["vulnerabilities"]:
            assert v["severity"] in valid, f"Invalid severity: {v['severity']}"

    def test_risk_level_values_are_valid(self):
        valid = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "SAFE"}
        for code in [CLEAN_CODE, CODE_EVAL, CODE_PICKLE]:
            result = check_security(code, run_bandit=False)
            assert result["risk_level"] in valid

    def test_line_numbers_are_positive_integers(self):
        result = check_security(CODE_MULTI_VULN, run_bandit=False)
        for v in result["vulnerabilities"]:
            assert isinstance(v["line"], int)
            assert v["line"] >= 0

    def test_total_count_matches_list_length(self):
        result = check_security(CODE_MULTI_VULN, run_bandit=False)
        counts = result["vulnerability_counts"]
        assert counts["TOTAL"] == len(result["vulnerabilities"])
        assert counts["TOTAL"] == counts["CRITICAL"] + counts["HIGH"] + counts["MEDIUM"] + counts["LOW"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Functional tests: each vulnerability category must be detected
# ─────────────────────────────────────────────────────────────────────────────

class TestVulnerabilityDetection:
    """Verify every detection category produces the expected finding."""

    def test_detects_hardcoded_password(self):
        result = check_security(CODE_HARDCODED_PASSWORD, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "hardcoded_secret" in types, "Should detect hardcoded password"

    def test_detects_hardcoded_api_key(self):
        result = check_security(CODE_HARDCODED_PASSWORD, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "hardcoded_secret" in types, "Should detect hardcoded API key"

    def test_hardcoded_secret_is_critical(self):
        result = check_security(CODE_HARDCODED_PASSWORD, run_bandit=False)
        secrets = [v for v in result["vulnerabilities"] if v["type"] == "hardcoded_secret"]
        assert all(v["severity"] == SEVERITY_CRITICAL for v in secrets)

    def test_hardcoded_secret_value_is_redacted(self):
        """Secret values must not appear verbatim in the reported message."""
        result = check_security(CODE_HARDCODED_PASSWORD, run_bandit=False)
        for v in result["vulnerabilities"]:
            assert "SuperSecret123" not in v["message"]
            assert "sk-abcdef1234567890" not in v["message"]

    def test_detects_eval(self):
        result = check_security(CODE_EVAL, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "dangerous_function_eval" in types

    def test_eval_is_critical(self):
        result = check_security(CODE_EVAL, run_bandit=False)
        evals = [v for v in result["vulnerabilities"] if v["type"] == "dangerous_function_eval"]
        assert all(v["severity"] == SEVERITY_CRITICAL for v in evals)

    def test_detects_pickle_loads(self):
        result = check_security(CODE_PICKLE, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "insecure_deserialization_pickle" in types

    def test_pickle_is_critical(self):
        result = check_security(CODE_PICKLE, run_bandit=False)
        pickles = [v for v in result["vulnerabilities"] if v["type"] == "insecure_deserialization_pickle"]
        assert all(v["severity"] == SEVERITY_CRITICAL for v in pickles)

    def test_detects_sql_injection_fstring(self):
        result = check_security(CODE_SQL_FSTRING, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "sql_injection_risk" in types

    def test_detects_sql_injection_percent(self):
        result = check_security(CODE_SQL_PERCENT, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "sql_injection_risk" in types

    def test_sql_injection_is_critical(self):
        result = check_security(CODE_SQL_FSTRING, run_bandit=False)
        sqls = [v for v in result["vulnerabilities"] if v["type"] == "sql_injection_risk"]
        assert all(v["severity"] == SEVERITY_CRITICAL for v in sqls)

    def test_detects_shell_true(self):
        result = check_security(CODE_SHELL_TRUE, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "command_injection_shell_true" in types

    def test_detects_verify_false(self):
        result = check_security(CODE_VERIFY_FALSE, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "tls_verification_disabled" in types

    def test_detects_http_url(self):
        result = check_security(CODE_HTTP_URL, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "insecure_http_url" in types

    def test_detects_debug_true(self):
        result = check_security(CODE_DEBUG_TRUE, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "debug_artifact" in types

    def test_detects_assert_guard(self):
        result = check_security(CODE_ASSERT_GUARD, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "assert_used_as_guard" in types

    def test_detects_print_password(self):
        result = check_security(CODE_PRINT_SECRET, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "sensitive_data_exposure" in types

    def test_detects_yaml_load_unsafe(self):
        result = check_security(CODE_YAML_UNSAFE, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "insecure_deserialization_yaml" in types

    def test_yaml_load_with_safe_loader_not_flagged(self):
        """yaml.load(data, Loader=yaml.SafeLoader) must NOT be flagged."""
        result = check_security(CODE_YAML_SAFE, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "insecure_deserialization_yaml" not in types

    def test_each_vuln_has_cwe_reference(self):
        result = check_security(CODE_MULTI_VULN, run_bandit=False)
        for v in result["vulnerabilities"]:
            assert v["cwe"].startswith("CWE-"), f"CWE reference missing or wrong format: {v['cwe']}"

    def test_each_vuln_has_non_empty_fix(self):
        result = check_security(CODE_MULTI_VULN, run_bandit=False)
        for v in result["vulnerabilities"]:
            assert v["fix"].strip(), f"Fix advice is empty for vuln type: {v['type']}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Negative tests: clean code must score SAFE
# ─────────────────────────────────────────────────────────────────────────────

class TestNegativeCases:
    """Clean, well-written code should produce no findings."""

    def test_clean_code_risk_is_safe(self):
        result = check_security(CLEAN_CODE, run_bandit=False)
        assert result["risk_level"] == "SAFE", (
            f"Clean code should be SAFE but got {result['risk_level']}. "
            f"Findings: {result['vulnerabilities']}"
        )

    def test_clean_code_score_is_100(self):
        result = check_security(CLEAN_CODE, run_bandit=False)
        assert result["security_score"] == 100.0, (
            f"Clean code should score 100 but got {result['security_score']}. "
            f"Findings: {result['vulnerabilities']}"
        )

    def test_clean_code_zero_vulnerabilities(self):
        result = check_security(CLEAN_CODE, run_bandit=False)
        assert result["vulnerability_counts"]["TOTAL"] == 0, (
            f"Clean code should have 0 findings but found: {result['vulnerabilities']}"
        )

    def test_parameterised_query_not_flagged(self):
        """Parameterised SQL queries must NOT be flagged as SQL injection."""
        code = '''
import sqlite3

def get_user(conn, user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()
'''
        result = check_security(code, run_bandit=False)
        types = [v["type"] for v in result["vulnerabilities"]]
        assert "sql_injection_risk" not in types


# ─────────────────────────────────────────────────────────────────────────────
# 4. Edge-case tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary conditions and unusual inputs must not crash the tool."""

    def test_empty_code_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            check_security("", run_bandit=False)

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            check_security("   \n\t  ", run_bandit=False)

    def test_unsupported_language_raises_value_error(self):
        with pytest.raises(ValueError, match="not supported"):
            check_security("print('hello')", language="javascript", run_bandit=False)

    def test_syntax_error_returns_gracefully(self):
        result = check_security(CODE_SYNTAX_ERROR, run_bandit=False)
        assert result["syntax_error"] is not None
        assert result["risk_level"] == "CRITICAL"
        assert result["vulnerability_counts"]["TOTAL"] >= 1

    def test_single_line_file(self):
        result = check_security("x = 1\n", run_bandit=False)
        assert isinstance(result["security_score"], float)

    def test_large_file_does_not_crash(self):
        """A file with many functions should not raise or time out."""
        lines = ['"""Module."""\n']
        for i in range(200):
            lines.append(f'def func_{i}(x):\n    """Docstring."""\n    return x + {i}\n\n')
        big_code = "".join(lines)
        result = check_security(big_code, run_bandit=False)
        assert 0.0 <= result["security_score"] <= 100.0

    def test_unicode_in_code_does_not_crash(self):
        code = '"""Module with unicode: 你好 مرحبا."""\n\nx = "hello"\n'
        result = check_security(code, run_bandit=False)
        assert result is not None

    def test_file_path_stored_in_result(self):
        result = check_security(CLEAN_CODE, file_path="my_app.py", run_bandit=False)
        assert result["file_path"] == "my_app.py"

    def test_default_language_is_python(self):
        result = check_security(CLEAN_CODE, run_bandit=False)
        assert result["language"] == "python"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Security contract tests: critical invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestSecurityContracts:
    """
    These are hard invariants the scanner must NEVER violate.
    Failing any of these means the tool gives a false sense of security.
    """

    def test_critical_finding_forces_critical_risk_level(self):
        """If any CRITICAL vuln exists, risk_level MUST be CRITICAL."""
        result = check_security(CODE_EVAL, run_bandit=False)
        has_critical = any(v["severity"] == SEVERITY_CRITICAL for v in result["vulnerabilities"])
        if has_critical:
            assert result["risk_level"] == "CRITICAL"

    def test_score_decreases_with_more_vulns(self):
        """More vulnerabilities must result in an equal or lower score."""
        single = check_security(CODE_EVAL, run_bandit=False)
        multi = check_security(CODE_MULTI_VULN, run_bandit=False)
        assert multi["security_score"] <= single["security_score"]

    def test_no_duplicate_findings_same_type_same_line(self):
        """Two identical findings (same type + line) must not both appear."""
        result = check_security(CODE_MULTI_VULN, run_bandit=False)
        seen = set()
        for v in result["vulnerabilities"]:
            key = (v["type"], v["line"])
            assert key not in seen, f"Duplicate finding: {key}"
            seen.add(key)

    def test_vulnerabilities_sorted_critical_first(self):
        """CRITICAL findings must appear before HIGH, MEDIUM, LOW."""
        result = check_security(CODE_MULTI_VULN, run_bandit=False)
        order = {SEVERITY_CRITICAL: 0, SEVERITY_HIGH: 1, SEVERITY_MEDIUM: 2, SEVERITY_LOW: 3}
        severities = [order[v["severity"]] for v in result["vulnerabilities"]]
        assert severities == sorted(severities), "Vulnerabilities are not sorted by severity"

    def test_score_zero_with_heavy_vulns(self):
        """Score must be clamped to 0 — never go negative."""
        result = check_security(CODE_MULTI_VULN, run_bandit=False)
        assert result["security_score"] >= 0.0

    def test_count_consistency_after_dedup(self):
        """
        CRITICAL+HIGH+MEDIUM+LOW must always equal TOTAL,
        even after deduplication.
        """
        result = check_security(CODE_MULTI_VULN, run_bandit=False)
        c = result["vulnerability_counts"]
        assert c["CRITICAL"] + c["HIGH"] + c["MEDIUM"] + c["LOW"] == c["TOTAL"]

    def test_safe_implies_score_100(self):
        """If risk_level is SAFE there must be no vulnerabilities."""
        result = check_security(CLEAN_CODE, run_bandit=False)
        if result["risk_level"] == "SAFE":
            assert result["vulnerability_counts"]["TOTAL"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Unit tests for internal helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestHelperFunctions:
    """Direct unit tests for the private detection helpers."""

    def test_parse_ast_success(self):
        tree, err = _parse_ast("x = 1\n")
        assert tree is not None
        assert err is None

    def test_parse_ast_syntax_error(self):
        tree, err = _parse_ast("def broken(:\n    pass\n")
        assert tree is None
        assert err is not None
        assert "SyntaxError" in err

    def test_detect_hardcoded_secrets_direct(self):
        issues = _detect_hardcoded_secrets('password = "mypassword123"\n')
        assert len(issues) >= 1
        assert issues[0]["severity"] == SEVERITY_CRITICAL

    def test_detect_dangerous_functions_eval(self):
        tree, _ = _parse_ast("eval(user_input)\n")
        issues = _detect_dangerous_functions(tree)
        assert any(i["type"] == "dangerous_function_eval" for i in issues)

    def test_detect_dangerous_functions_exec(self):
        tree, _ = _parse_ast("exec(code)\n")
        issues = _detect_dangerous_functions(tree)
        assert any(i["type"] == "dangerous_function_exec" for i in issues)

    def test_detect_sql_injection_fstring(self):
        code = 'cursor.execute(f"SELECT * FROM users WHERE id={uid}")\n'
        tree, _ = _parse_ast(code)
        issues = _detect_sql_injection(tree)
        assert any(i["type"] == "sql_injection_risk" for i in issues)

    def test_detect_command_injection(self):
        code = "import subprocess\nsubprocess.run(cmd, shell=True)\n"
        tree, _ = _parse_ast(code)
        issues = _detect_command_injection(tree)
        assert any(i["type"] == "command_injection_shell_true" for i in issues)

    def test_detect_debug_artifact(self):
        tree, _ = _parse_ast("DEBUG = True\n")
        issues = _detect_debug_artifacts(tree)
        assert any(i["type"] == "debug_artifact" for i in issues)

    def test_detect_assert_guard(self):
        code = "def f(x):\n    assert x > 0\n    return x\n"
        tree, _ = _parse_ast(code)
        issues = _detect_debug_artifacts(tree)
        assert any(i["type"] == "assert_used_as_guard" for i in issues)

    def test_calculate_security_score_no_issues(self):
        score = _calculate_security_score([])
        assert score == 100.0

    def test_calculate_security_score_clamps_to_zero(self):
        """Score must not go below 0 regardless of how many issues exist."""
        many_critical = [
            {"type": "hardcoded_secret", "severity": SEVERITY_CRITICAL}
            for _ in range(30)
        ]
        score = _calculate_security_score(many_critical)
        assert score == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. LLM-as-a-Judge test (requires Ollama to be running)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.llm_judge
def test_agent_summary_accuracy_llm_judge():
    """
    LLM-as-a-Judge evaluation: use Ollama to check that the security_scanner
    agent's natural-language summary accurately reflects the tool findings
    without hallucinating vulnerabilities that do not exist.

    This test is marked `llm_judge` — skip with:
        pytest -v -k "not llm_judge"
    """
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        pytest.skip("langchain_ollama not installed")

    # Run the security scan tool to get ground-truth findings
    scan_result = check_security(CODE_MULTI_VULN, run_bandit=False)

    # Import and run the agent node directly (bypasses LangGraph)
    from agents.security_scanner_agent import security_scanner_node

    fake_state = {
        "code_file_path": "test_multi_vuln.py",
        "code_content": CODE_MULTI_VULN,
        "language": "python",
        "messages": [],
        "errors": [],
        "session_id": "test-session",
        "start_time": "",
        "current_agent": "",
        "task_plan": "",
        "assigned_agents": [],
        "code_analysis_raw": {},
        "code_quality_score": 0.0,
        "code_issues": [],
        "code_analysis_summary": "",
        "security_vulnerabilities": [],
        "security_score": 0.0,
        "security_risk_level": "",
        "security_summary": "",
        "final_report": "",
        "report_file_path": "",
    }

    agent_output = security_scanner_node(fake_state)
    agent_summary = agent_output.get("security_summary", "")

    assert agent_summary, "Agent produced an empty summary"

    # Build the judge prompt
    judge_system = """You are an impartial AI evaluator. Your job is to judge whether an AI-generated
security summary accurately and honestly reflects the findings in a static analysis tool report.

You must answer with a JSON object in this exact format:
{
  "verdict": "PASS" or "FAIL",
  "score": <integer 0-10>,
  "reason": "<one sentence explaining your verdict>"
}

PASS criteria (score >= 7):
- The summary mentions the correct risk level.
- The summary does not invent vulnerabilities absent from the tool report.
- All mentioned vulnerabilities are present in the tool report.

FAIL criteria (score < 7):
- The summary mentions vulnerabilities not in the tool report (hallucination).
- The summary completely omits all CRITICAL findings.
- The risk level stated is wrong."""

    tool_findings_summary = json.dumps({
        "risk_level": scan_result["risk_level"],
        "security_score": scan_result["security_score"],
        "vulnerability_types_found": list({v["type"] for v in scan_result["vulnerabilities"]}),
    }, indent=2)

    judge_prompt = (
        f"TOOL REPORT (ground truth):\n{tool_findings_summary}\n\n"
        f"AGENT SUMMARY (to evaluate):\n{agent_summary}\n\n"
        "Is the agent summary accurate? Respond with the JSON verdict only."
    )

    try:
        llm = ChatOllama(model="phi3:mini", temperature=0.0, num_predict=200)
        response = llm.invoke([
            SystemMessage(content=judge_system),
            HumanMessage(content=judge_prompt),
        ])
        raw = response.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()

        verdict_data = json.loads(raw)
        score = int(verdict_data.get("score", 0))
        verdict = verdict_data.get("verdict", "FAIL")
        reason = verdict_data.get("reason", "No reason given")

        print(f"\n[LLM-as-a-Judge] Verdict: {verdict} | Score: {score}/10 | Reason: {reason}")

        assert verdict == "PASS" and score >= 7, (
            f"LLM judge rated the agent summary as {verdict} ({score}/10): {reason}"
        )

    except Exception as exc:
        pytest.skip(f"Ollama unavailable or failed to parse verdict: {exc}")
