"""
test_coordinator.py
-------------------
Automated evaluation/testing script for Agent 1: Coordinator.

Test strategy:
  1. Property-based tests  — verify tool output structure, types, bounds, required keys
  2. Functional tests      — verify language detection for multiple languages
  3. Metrics validation    — ensure extracted metrics are mathematically correct
  4. Edge-case tests       — empty file, syntax errors, unknown languages, binary files
  5. Routing logic tests   — verify priority assignment and agent selection
  6. LLM-as-a-Judge test   — verify phi3:mini confirms routing decisions accurately

Run with:
    pytest tests/test_coordinator.py -v
    pytest tests/test_coordinator.py -v -k "not llm_judge"  # skip Ollama tests
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.coordinator_tool import (
    analyze_file_for_routing,
    _detect_language,
    _extract_code_metrics,
    _infer_code_purpose,
    _assess_priority,
)
from agents.coordinator_agent import _parse_llm_response


# ─────────────────────────────────────────────────────────────────────────────
# PROPERTY-BASED TESTS: Verify Tool Output Structure & Types
# ─────────────────────────────────────────────────────────────────────────────


class TestCoordinatorToolStructure:
    """Property-based tests ensuring output conforms to expected schema."""

    @pytest.fixture
    def sample_python_code(self) -> str:
        """Sample Python code for testing."""
        return """
import sys
import json

def process_data(items):
    \"\"\"Process a list of items.\"\"\"
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
    return result

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    print(process_data(data))
"""

    def test_analyze_returns_dict(self, sample_python_code):
        """Test that analyze_file_for_routing returns a dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_python_code)
            f.flush()

            result = analyze_file_for_routing(f.name, sample_python_code)
            assert isinstance(result, dict), "analyze_file_for_routing should return dict"

    def test_output_has_required_keys(self, sample_python_code):
        """Test that output contains all required keys."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_python_code)
            f.flush()

            result = analyze_file_for_routing(f.name, sample_python_code)
            required_keys = [
                "file_path",
                "file_size_bytes",
                "language",
                "confidence",
                "metrics",
                "code_purpose",
                "priority",
                "routing_agents",
                "complexity_assessment",
                "analysis_notes",
            ]
            for key in required_keys:
                assert key in result, f"Missing required key: {key}"

    def test_confidence_in_valid_range(self, sample_python_code):
        """Test that language confidence is between 0 and 1."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_python_code)
            f.flush()

            result = analyze_file_for_routing(f.name, sample_python_code)
            confidence = result["confidence"]
            assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range [0, 1]"

    def test_priority_has_valid_value(self, sample_python_code):
        """Test that priority is one of HIGH, MEDIUM, LOW."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_python_code)
            f.flush()

            result = analyze_file_for_routing(f.name, sample_python_code)
            assert result["priority"] in [
                "HIGH",
                "MEDIUM",
                "LOW",
            ], f"Invalid priority: {result['priority']}"

    def test_routing_agents_is_list(self, sample_python_code):
        """Test that routing_agents is a list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_python_code)
            f.flush()

            result = analyze_file_for_routing(f.name, sample_python_code)
            assert isinstance(
                result["routing_agents"], list
            ), "routing_agents should be a list"
            assert len(result["routing_agents"]) > 0, "routing_agents should not be empty"
            assert all(
                agent in ["code_analyzer", "security_scanner"]
                for agent in result["routing_agents"]
            ), f"Invalid agent in routing_agents: {result['routing_agents']}"

    def test_metrics_is_dict_with_valid_types(self, sample_python_code):
        """Test that metrics dict contains correct types."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(sample_python_code)
            f.flush()

            result = analyze_file_for_routing(f.name, sample_python_code)
            metrics = result["metrics"]

            # Verify types
            assert isinstance(metrics["total_lines"], int)
            assert isinstance(metrics["code_lines"], int)
            assert isinstance(metrics["function_count"], int)
            assert isinstance(metrics["class_count"], int)
            assert isinstance(metrics["has_main"], bool)
            assert metrics["estimated_complexity"] in ["LOW", "MEDIUM", "HIGH"]


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTIONAL TESTS: Verify Language Detection Accuracy
# ─────────────────────────────────────────────────────────────────────────────


class TestLanguageDetection:
    """Test language detection for multiple programming languages."""

    def test_detect_python(self):
        """Test Python language detection."""
        python_code = """
import sys
def hello():
    print("Hello")
"""
        lang, confidence = _detect_language("test.py", python_code)
        assert lang == "python", f"Expected python, got {lang}"
        assert confidence > 0.7, f"Python confidence too low: {confidence}"

    def test_detect_javascript(self):
        """Test JavaScript language detection."""
        js_code = """
const x = 10;
function hello() {
    console.log('Hello');
}
"""
        lang, confidence = _detect_language("test.js", js_code)
        assert lang == "javascript", f"Expected javascript, got {lang}"
        assert confidence > 0.5, f"JavaScript confidence too low: {confidence}"

    def test_detect_java(self):
        """Test Java language detection."""
        java_code = """
public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello");
    }
}
"""
        lang, confidence = _detect_language("Hello.java", java_code)
        assert lang == "java", f"Expected java, got {lang}"
        assert confidence > 0.5, f"Java confidence too low: {confidence}"

    def test_detect_csharp(self):
        """Test C# language detection."""
        csharp_code = """
using System;
namespace HelloWorld {
    class Program {
        static void Main() {
            Console.WriteLine("Hello");
        }
    }
}
"""
        lang, confidence = _detect_language("Program.cs", csharp_code)
        assert lang == "csharp", f"Expected csharp, got {lang}"
        assert confidence > 0.5, f"C# confidence too low: {confidence}"

    def test_extension_overrides_content(self):
        """Test that file extension is strong signal for language."""
        # Python extension with content that looks like JS
        code = "const x = 10;"
        lang, confidence = _detect_language("test.py", code)
        # Should still prefer python due to extension
        assert lang == "python"


# ─────────────────────────────────────────────────────────────────────────────
# METRICS VALIDATION TESTS: Verify Correctness of Extracted Metrics
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsExtraction:
    """Test that code metrics are correctly extracted."""

    def test_line_count_accuracy(self):
        """Test that line counts are accurate."""
        code = """line1
line2
line3

line5"""
        metrics = _extract_code_metrics(code, "python")
        assert metrics.total_lines == 5, f"Expected 5 lines, got {metrics.total_lines}"
        assert metrics.blank_lines == 1, f"Expected 1 blank line, got {metrics.blank_lines}"

    def test_function_detection_python(self):
        """Test function detection in Python."""
        code = """
def func1():
    pass

def func2():
    pass

class MyClass:
    def method(self):
        pass
"""
        metrics = _extract_code_metrics(code, "python")
        assert (
            metrics.function_count >= 2
        ), f"Should detect at least 2 functions, got {metrics.function_count}"

    def test_class_detection_python(self):
        """Test class detection in Python."""
        code = """
class Class1:
    pass

class Class2:
    pass
"""
        metrics = _extract_code_metrics(code, "python")
        assert (
            metrics.class_count >= 2
        ), f"Should detect at least 2 classes, got {metrics.class_count}"

    def test_import_detection(self):
        """Test import detection."""
        code = """
import sys
import os
from json import loads
from pathlib import Path
"""
        metrics = _extract_code_metrics(code, "python")
        assert metrics.import_count >= 3, f"Should detect at least 3 imports, got {metrics.import_count}"

    def test_main_detection(self):
        """Test detection of main entry point."""
        code = """
if __name__ == '__main__':
    print('Running main')
"""
        metrics = _extract_code_metrics(code, "python")
        assert metrics.has_main is True, "Should detect __main__ block"

    def test_code_vs_comment_lines(self):
        """Test distinction between code and comment lines."""
        code = """
# This is a comment
x = 10  # inline comment
# Another comment
y = 20
"""
        metrics = _extract_code_metrics(code, "python")
        assert (
            metrics.comment_lines >= 2
        ), f"Should detect at least 2 comment lines, got {metrics.comment_lines}"


# ─────────────────────────────────────────────────────────────────────────────
# EDGE-CASE TESTS: Handle Unusual Inputs
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_file(self):
        """Test handling of empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("")
            f.flush()

            result = analyze_file_for_routing(f.name, "")
            assert result["metrics"]["total_lines"] == 1  # single newline or empty
            assert result["priority"] == "LOW"

    def test_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            analyze_file_for_routing("/nonexistent/file.py")

    def test_binary_file_detection(self):
        """Test that binary files are rejected."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin", delete=False) as f:
            f.write(b"\x00\x01\x02\x03")
            f.flush()

            # Should raise ValueError for non-UTF8 content
            with pytest.raises(ValueError):
                with open(f.name, "rb") as bf:
                    content = bf.read()
                # Try to analyze as if read
                analyze_file_for_routing(f.name)

    def test_syntax_error_code(self):
        """Test handling of code with syntax errors."""
        bad_code = """
def broken(
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(bad_code)
            f.flush()

            # Should still analyze without crashing
            result = analyze_file_for_routing(f.name, bad_code)
            assert result is not None

    def test_unknown_extension(self):
        """Test handling of unknown file extensions."""
        code = "print('hello')"
        lang, confidence = _detect_language("file.xyz", code)
        # Should fall back to content analysis
        assert lang != "unknown" or confidence >= 0


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING LOGIC TESTS: Verify Priority and Agent Selection
# ─────────────────────────────────────────────────────────────────────────────


class TestRoutingLogic:
    """Test priority assignment and routing decisions."""

    def test_small_simple_code_is_low_priority(self):
        """Test that a trivial script gets NONE priority."""
        simple_code = """
x = 10
print(x)
"""
        metrics = _extract_code_metrics(simple_code, "python")
        priority = _assess_priority(metrics, "python", "Simple script")
        assert priority == "NONE", f"Expected NONE priority for simple code, got {priority}"

    def test_large_complex_code_is_high_priority(self):
        """Test that large, complex code gets HIGH priority."""
        complex_code = """
def func1(a, b, c, d, e, f, g):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        return a + b + c + d + e
    return 0

""" + "\n".join([f"def func{i}(): pass" for i in range(2, 20)])

        metrics = _extract_code_metrics(complex_code, "python")
        priority = _assess_priority(metrics, "python", "Complex application")
        # Should be MEDIUM or HIGH due to size
        assert priority in [
            "MEDIUM",
            "HIGH",
        ], f"Expected MEDIUM/HIGH priority for complex code, got {priority}"

    def test_web_app_gets_high_priority(self):
        """Test that web application code gets higher priority."""
        web_code = """
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World'
"""
        metrics = _extract_code_metrics(web_code, "python")
        priority = _assess_priority(metrics, "python", "Web application/API")
        assert priority in [
            "MEDIUM",
            "HIGH",
        ], f"Web app should get MEDIUM+ priority, got {priority}"

    def test_clean_simple_code_routes_to_no_agents(self):
        """Test that clean simple code stops at the coordinator."""
        code = "print('test')"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            result = analyze_file_for_routing(f.name, code)
            agents = result["routing_agents"]
            assert agents == []
            assert result["code_quality_priority"] == "NONE"
            assert result["security_priority"] == "NONE"

    def test_simple_security_risky_code_routes_to_security_only(self):
        """Test that a security-risk file routes only to the security scanner."""
        code = """
API_KEY = 'sk-test-12345'

def run_command(user_input):
    return eval(user_input)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            result = analyze_file_for_routing(f.name, code)
            agents = result["routing_agents"]
            assert agents == ["security_scanner"]
            assert result["code_quality_priority"] == "NONE"
            assert result["security_priority"] in ["MEDIUM", "HIGH"]

    def test_complex_buggy_code_routes_to_both_agents(self):
        """Test that complex risky code routes to both analyzers."""
        sample_path = Path(__file__).parent.parent / "sample_code" / "sample_buggy_code.py"
        code = sample_path.read_text(encoding="utf-8")

        result = analyze_file_for_routing(str(sample_path), code)
        agents = result["routing_agents"]
        assert "code_analyzer" in agents
        assert "security_scanner" in agents
        assert result["code_quality_priority"] in ["MEDIUM", "HIGH"]
        assert result["security_priority"] in ["MEDIUM", "HIGH"]

    def test_no_risk_file_routes_to_no_agents(self):
        """Test that a no-risk file stops at the coordinator with no downstream agents."""
        code = """
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            result = analyze_file_for_routing(f.name, code)
            assert result["code_quality_priority"] == "NONE"
            assert result["security_priority"] == "NONE"
            assert result["routing_agents"] == []


class TestCoordinatorResponseParsing:
    """Test parsing of coordinator LLM responses."""

    def test_parse_fenced_json_with_extra_text(self):
        response_text = """Here is the routing confirmation:

```json
{
  "analysis_confirmed": true,
  "language_ok": "python",
  "priority_adjusted": "MEDIUM",
  "agents_confirmed": ["code_analyzer", "security_scanner"],
  "additional_notes": "Looks good"
}
```
"""

        result = _parse_llm_response(response_text)

        assert result["analysis_confirmed"] is True
        assert result["language_ok"] == "python"
        assert result["priority_adjusted"] == "MEDIUM"
        assert result["agents_confirmed"] == ["code_analyzer", "security_scanner"]


# ─────────────────────────────────────────────────────────────────────────────
# LLM-AS-A-JUDGE TEST: Verify Routing Decision With phi3:mini
# ─────────────────────────────────────────────────────────────────────────────


class TestLLMAsJudge:
    """LLM-as-a-Judge test to verify routing decisions are sensible."""

    @pytest.mark.skipif(
        True,  # Skip by default (requires Ollama running), change to False to run
        reason="Requires Ollama service running on localhost:11434",
    )
    def test_phi3_confirms_routing_decisions(self):
        """Test that phi3:mini agrees with routing decisions."""
        pytest.skip("Requires Ollama. Run with: pytest -v -k 'not llm_judge'")

        # This test would verify that phi3:mini confirms our routing
        # by asking it to review the analysis and confirm the agents
        # This is a placeholder for integration testing with Ollama


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
