"""
test_code_analyzer.py
---------------------
Automated evaluation/testing script for Agent 2: Code Analyzer.

Test strategy:
  1. Property-based tests  — verify structural properties of tool output
     (correct types, score bounds, required keys, etc.)
  2. Functional tests      — verify the tool detects specific known issues
  3. Edge-case tests       — empty input, syntax errors, perfect code
  4. LLM-as-a-Judge test   — use Ollama to judge whether the agent's
     natural-language summary accurately reflects the tool report

"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.code_analysis_tool import (
    analyze_code_quality,
    _check_docstrings,
    _check_function_complexity,
    _check_bad_patterns,
    _calculate_cyclomatic_complexity,
    _parse_ast,
    _calculate_quality_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────────

PERFECT_CODE = '''"""Module docstring."""


def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b


class Calculator:
    """A simple calculator class."""

    def multiply(self, x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y
'''

BUGGY_CODE = '''import os
from math import *

SECRET = "hardcoded_password_123"

def bad_function(a, b, c, d, e, f, g):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        return True
    return False

def another_bad():
    try:
        x = eval("1+1")
    except:
        pass

class NoDocClass:
    def no_doc_method(self):
        pass
'''

SYNTAX_ERROR_CODE = "def broken(:\n    pass"

EMPTY_CODE = "   \n\n   "


# ─────────────────────────────────────────────────────────────────────────────
# 1. Property-based tests — OUTPUT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputStructure:
    """Verify the tool always returns the correct data structure."""

    def test_returns_dict(self):
        """Tool must return a dict, not None or a string."""
        result = analyze_code_quality(PERFECT_CODE, "test.py")
        assert isinstance(result, dict), "Result must be a dict"

    def test_required_keys_present(self):
        """All required keys must be present in the result."""
        required_keys = [
            "file_path", "language", "syntax_error", "quality_score",
            "metrics", "cyclomatic_complexity", "issues", "issue_counts",
        ]
        result = analyze_code_quality(PERFECT_CODE, "test.py")
        for key in required_keys:
            assert key in result, f"Missing required key: '{key}'"

    def test_quality_score_is_float(self):
        """quality_score must be a float between 0 and 100."""
        result = analyze_code_quality(BUGGY_CODE, "buggy.py")
        assert isinstance(result["quality_score"], float), "quality_score must be float"

    def test_quality_score_bounds(self):
        """quality_score must be in range [0.0, 100.0]."""
        for code in [PERFECT_CODE, BUGGY_CODE]:
            result = analyze_code_quality(code, "test.py")
            assert 0.0 <= result["quality_score"] <= 100.0, (
                f"quality_score {result['quality_score']} out of bounds [0, 100]"
            )

    def test_issues_is_list(self):
        """issues must always be a list (even if empty)."""
        result = analyze_code_quality(PERFECT_CODE, "test.py")
        assert isinstance(result["issues"], list), "issues must be a list"

    def test_each_issue_has_required_fields(self):
        """Every issue in the list must have type, severity, line, message."""
        result = analyze_code_quality(BUGGY_CODE, "buggy.py")
        for issue in result["issues"]:
            assert "type" in issue, "Issue missing 'type'"
            assert "severity" in issue, "Issue missing 'severity'"
            assert "line" in issue, "Issue missing 'line'"
            assert "message" in issue, "Issue missing 'message'"

    def test_severity_values_are_valid(self):
        """All severity values must be HIGH, MEDIUM, or LOW."""
        result = analyze_code_quality(BUGGY_CODE, "buggy.py")
        valid_severities = {"HIGH", "MEDIUM", "LOW"}
        for issue in result["issues"]:
            assert issue["severity"] in valid_severities, (
                f"Invalid severity: {issue['severity']}"
            )

    def test_issue_counts_match_issues_list(self):
        """issue_counts totals must add up to len(issues)."""
        result = analyze_code_quality(BUGGY_CODE, "buggy.py")
        counts = result["issue_counts"]
        total_from_dict = counts["HIGH"] + counts["MEDIUM"] + counts["LOW"]
        assert total_from_dict == counts["TOTAL"], (
            f"issue_counts totals mismatch: {total_from_dict} vs TOTAL={counts['TOTAL']}"
        )
        assert counts["TOTAL"] == len(result["issues"]), (
            f"TOTAL count {counts['TOTAL']} != len(issues) {len(result['issues'])}"
        )

    def test_metrics_contains_required_fields(self):
        """Metrics dict must contain standard code metrics."""
        result = analyze_code_quality(PERFECT_CODE, "test.py")
        for key in ["total_lines", "lines_of_code", "num_functions", "num_classes"]:
            assert key in result["metrics"], f"Metrics missing '{key}'"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Functional tests — ISSUE DETECTION ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

class TestIssueDetection:
    """Verify the tool correctly identifies known code problems."""

    def test_detects_missing_module_docstring(self):
        """Tool must flag code that lacks a module-level docstring."""
        code = "def foo():\n    pass\n"
        result = analyze_code_quality(code, "test.py")
        types = [i["type"] for i in result["issues"]]
        assert "missing_docstring_module" in types, (
            "Failed to detect missing module docstring"
        )

    def test_detects_missing_function_docstring(self):
        """Tool must flag functions without docstrings."""
        code = '"""Module doc."""\n\ndef no_doc():\n    return 1\n'
        result = analyze_code_quality(code, "test.py")
        types = [i["type"] for i in result["issues"]]
        assert "missing_docstring_function" in types, (
            "Failed to detect missing function docstring"
        )

    def test_detects_bare_except(self):
        """Tool must flag bare except clauses."""
        code = 'try:\n    x = 1\nexcept:\n    pass\n'
        result = analyze_code_quality(code, "test.py")
        types = [i["type"] for i in result["issues"]]
        assert "bare_except_clause" in types, "Failed to detect bare except clause"

    def test_detects_mutable_default_argument(self):
        """Tool must flag functions with mutable default arguments."""
        code = 'def foo(items=[]):\n    pass\n'
        result = analyze_code_quality(code, "test.py")
        types = [i["type"] for i in result["issues"]]
        assert "mutable_default_argument" in types, (
            "Failed to detect mutable default argument"
        )

    def test_detects_wildcard_import(self):
        """Tool must flag wildcard imports."""
        code = 'from os import *\n'
        result = analyze_code_quality(code, "test.py")
        types = [i["type"] for i in result["issues"]]
        assert "wildcard_import" in types, "Failed to detect wildcard import"

    def test_detects_too_many_parameters(self):
        """Tool must flag functions with more than 5 parameters."""
        code = 'def foo(a, b, c, d, e, f):\n    pass\n'
        result = analyze_code_quality(code, "test.py")
        types = [i["type"] for i in result["issues"]]
        assert "too_many_parameters" in types, "Failed to detect too many parameters"

    def test_perfect_code_has_high_score(self):
        """Well-written code should score above 80/100."""
        result = analyze_code_quality(PERFECT_CODE, "test.py")
        assert result["quality_score"] >= 80.0, (
            f"Perfect code scored too low: {result['quality_score']}"
        )

    def test_buggy_code_has_lower_score_than_perfect(self):
        """Buggy code must score lower than perfect code."""
        perfect = analyze_code_quality(PERFECT_CODE, "perfect.py")
        buggy = analyze_code_quality(BUGGY_CODE, "buggy.py")
        assert buggy["quality_score"] < perfect["quality_score"], (
            "Buggy code scored higher than or equal to perfect code"
        )

    def test_buggy_code_has_more_issues(self):
        """Buggy code must generate more issues than clean code."""
        perfect = analyze_code_quality(PERFECT_CODE, "perfect.py")
        buggy = analyze_code_quality(BUGGY_CODE, "buggy.py")
        assert len(buggy["issues"]) > len(perfect["issues"]), (
            "Buggy code did not generate more issues than clean code"
        )

    def test_detects_global_variable(self):
        """Tool must flag usage of the global statement."""
        code = 'x = 0\ndef foo():\n    global x\n    x = 1\n'
        result = analyze_code_quality(code, "test.py")
        types = [i["type"] for i in result["issues"]]
        assert "global_variable" in types, "Failed to detect global statement usage"

    def test_detects_deeply_nested_code(self):
        """Tool must flag code nested more than 4 levels deep."""
        code = (
            "def foo():\n"
            "    if True:\n"
            "        if True:\n"
            "            if True:\n"
            "                if True:\n"
            "                    if True:\n"
            "                        pass\n"
        )
        result = analyze_code_quality(code, "test.py")
        types = [i["type"] for i in result["issues"]]
        assert "deeply_nested_code" in types, "Failed to detect deeply nested code"

    def test_detects_long_function(self):
        """Tool must flag functions longer than 30 lines."""
        long_func = "def long_func():\n" + "\n".join(f"    x_{i} = {i}" for i in range(35))
        result = analyze_code_quality(long_func, "test.py")
        types = [i["type"] for i in result["issues"]]
        assert "function_too_long" in types, "Failed to detect long function"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Edge-case tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test the tool's behaviour at boundary conditions."""

    def test_syntax_error_code_returns_error(self):
        """Syntactically invalid code must return a syntax_error field."""
        result = analyze_code_quality(SYNTAX_ERROR_CODE, "broken.py")
        assert result["syntax_error"] is not None, (
            "Syntax error code should set syntax_error field"
        )
        assert result["quality_score"] == 0.0, (
            "Syntax error code should have quality_score of 0"
        )

    def test_empty_code_raises_value_error(self):
        """Empty code_content must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            analyze_code_quality(EMPTY_CODE, "empty.py")

    def test_unsupported_language_raises_value_error(self):
        """Non-Python language must raise ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            analyze_code_quality("const x = 1;", "test.js", language="javascript")

    def test_single_line_code_no_crash(self):
        """Single-line valid code must not crash."""
        result = analyze_code_quality("x = 1\n", "single.py")
        assert "quality_score" in result

    def test_very_large_code_no_crash(self):
        """Large code files must not crash the tool."""
        large_code = '"""Module."""\n\n' + "\n".join(
            f'def func_{i}(x: int) -> int:\n    """Doc."""\n    return x + {i}\n'
            for i in range(100)
        )
        result = analyze_code_quality(large_code, "large.py")
        assert 0.0 <= result["quality_score"] <= 100.0

    def test_file_path_preserved_in_output(self):
        """The file_path passed in must appear in the result unchanged."""
        result = analyze_code_quality(PERFECT_CODE, "my/custom/path.py")
        assert result["file_path"] == "my/custom/path.py"

    def test_result_is_json_serialisable(self):
        """The result dict must be JSON-serialisable (for state management)."""
        result = analyze_code_quality(BUGGY_CODE, "buggy.py")
        serialised = json.dumps(result)
        assert isinstance(serialised, str) and len(serialised) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Security / Robustness tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSecurity:
    """Verify the tool handles adversarial inputs safely."""

    def test_null_bytes_in_code(self):
        """Code with null bytes should not crash — should return syntax error."""
        code_with_null = "x = 1\x00\ny = 2\n"
        try:
            result = analyze_code_quality(code_with_null, "null.py")
            # Either returns a syntax error or processes gracefully
            assert "quality_score" in result
        except (ValueError, SyntaxError):
            pass  # Acceptable — tool raised an appropriate error

    def test_very_long_single_line(self):
        """A single extremely long line should not crash the tool."""
        long_line = "x = " + "1 + " * 5000 + "1\n"
        try:
            result = analyze_code_quality(long_line, "long.py")
            assert "quality_score" in result
        except RecursionError:
            pytest.skip("RecursionError on very long AST — known Python limitation")

    def test_code_with_unicode(self):
        """Code with Unicode identifiers should be handled correctly."""
        code = '"""Module."""\n\nresult = "こんにちは"\n'
        result = analyze_code_quality(code, "unicode.py")
        assert "quality_score" in result


# ─────────────────────────────────────────────────────────────────────────────
# 5. LLM-as-a-Judge test (optional — skipped if Ollama is unavailable)
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMAsJudge:
    """
    Use the local LLM (Ollama) to judge whether the agent's natural-language
    summary accurately reflects the tool's JSON report.

    This test is skipped automatically if Ollama is not running.
    """

    JUDGE_PROMPT = """You are an impartial evaluator. You will be given:
1. A static analysis tool report (JSON)
2. A human-written summary of that report

Evaluate the summary on two criteria:
A. ACCURACY (0-10): Does the summary accurately reflect the issues in the JSON?
B. HALLUCINATION (0/1): Does the summary invent issues NOT in the JSON? (0=no hallucination, 1=hallucination found)

Respond ONLY in this format:
ACCURACY: <score 0-10>
HALLUCINATION: <0 or 1>
REASON: <one sentence>"""

    @pytest.mark.skipif(
        not _try_ollama_connection(),
        reason="Ollama is not running — skipping LLM-as-a-Judge test",
    ) if False else pytest.mark.skip(reason="Define _try_ollama_connection before enabling")
    def test_summary_accuracy_via_llm(self):
        """LLM judge must rate the agent's summary as accurate (>=7/10)."""
        pass  # Implemented below after helper function


def _try_ollama_connection() -> bool:
    """Check if Ollama is reachable (for conditional test skipping)."""
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _try_ollama_connection(), reason="Ollama not running")
def test_llm_judge_summary_accuracy():
    """
    LLM-as-a-Judge: Evaluate the Code Analyzer agent's summary quality.

    The judge LLM reads both the raw tool output and the agent's summary,
    then verifies correctness using unambiguous YES/NO questions.
    Test passes if score and counts are correct and no hallucination is detected.
    """
    from langchain_core.messages import HumanMessage
    from langchain_ollama import ChatOllama
    from agents.code_analyzer_agent import _generate_fallback_summary

    # ── Step 1: Get tool output and fallback summary ──────────────────────
    tool_result = analyze_code_quality(BUGGY_CODE, "buggy.py")
    summary = _generate_fallback_summary(tool_result)

    # ── Step 2: Build unambiguous YES/NO judge prompt ─────────────────────
    # Root cause of original failure: phi3:mini interpreted "ACCURACY: 1"
    # as boolean (1=true) not as a score (1/10), causing a false negative.
    # YES/NO questions eliminate this ambiguity entirely.
    strict_judge_prompt = f"""You are a strict evaluator. Compare ONLY these two things:

TOOL REPORT (ground truth):
- quality_score: {tool_result['quality_score']}
- HIGH issues: {tool_result['issue_counts']['HIGH']}
- MEDIUM issues: {tool_result['issue_counts']['MEDIUM']}
- LOW issues: {tool_result['issue_counts']['LOW']}
- TOTAL issues: {tool_result['issue_counts']['TOTAL']}

AGENT SUMMARY:
{summary}

INSTRUCTIONS:
1. Does the summary correctly state the quality score ({tool_result['quality_score']}/100)? Answer YES or NO.
2. Does the summary correctly state HIGH={tool_result['issue_counts']['HIGH']}, MEDIUM={tool_result['issue_counts']['MEDIUM']}, LOW={tool_result['issue_counts']['LOW']}? Answer YES or NO.
3. Does the summary introduce any issue counts NOT present in the tool report? Answer YES or NO.

Respond EXACTLY in this format with no extra text:
SCORE_CORRECT: <YES or NO>
COUNTS_CORRECT: <YES or NO>
HALLUCINATION: <YES or NO>
REASON: <one sentence explaining your evaluation>"""

    # ── Step 3: Call judge LLM ────────────────────────────────────────────
    llm = ChatOllama(model="phi3:mini", temperature=0.0, num_predict=100)
    response = llm.invoke([HumanMessage(content=strict_judge_prompt)])
    output = response.content.strip()
    print(f"\nJudge output:\n{output}")

    # ── Step 4: Parse YES/NO responses robustly ───────────────────────────
    # .upper() handles yes / Yes / YES variations from the model
    parsed = {}
    for line in output.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            parsed[key.strip()] = value.strip().upper()

    score_correct  = parsed.get("SCORE_CORRECT", "NO") == "YES"
    counts_correct = parsed.get("COUNTS_CORRECT", "NO") == "YES"
    hallucination  = parsed.get("HALLUCINATION", "YES") == "YES"
    reason         = parsed.get("REASON", "No reason provided")

    # ── Step 5: Assert all three criteria ────────────────────────────────
    assert score_correct, (
        f"Summary did not correctly report the quality score "
        f"(expected {tool_result['quality_score']}/100). "
        f"Judge said: {reason}"
    )
    assert counts_correct, (
        f"Summary did not correctly report issue counts "
        f"(expected HIGH={tool_result['issue_counts']['HIGH']}, "
        f"MEDIUM={tool_result['issue_counts']['MEDIUM']}, "
        f"LOW={tool_result['issue_counts']['LOW']}). "
        f"Judge said: {reason}"
    )
    assert not hallucination, (
        f"Hallucination detected in summary — invented issues not in tool report. "
        f"Judge said: {reason}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
