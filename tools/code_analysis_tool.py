"""
code_analysis_tool.py
---------------------
Custom Python tool for Agent 2: Code Analyzer.

Performs static analysis of Python source code using the built-in `ast`
module (zero extra dependencies) plus optional pylint/pyflakes/radon
if they are installed.

"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Helper dataclasses / constants
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"

# Score deductions per issue type
SCORE_DEDUCTIONS: Dict[str, float] = {
    "missing_docstring_module": 5.0,
    "missing_docstring_function": 3.0,
    "missing_docstring_class": 3.0,
    "function_too_long": 5.0,
    "too_many_parameters": 4.0,
    "deeply_nested_code": 4.0,
    "bare_except_clause": 5.0,
    "mutable_default_argument": 4.0,
    "unused_import": 2.0,
    "wildcard_import": 3.0,
    "global_variable": 3.0,
    "high_cyclomatic_complexity": 5.0,
    "duplicate_code_smell": 3.0,
    "magic_number": 2.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis functions
# ─────────────────────────────────────────────────────────────────────────────


def _parse_ast(code_content: str) -> Tuple[Optional[ast.Module], Optional[str]]:
    """
    Parse source code into an AST tree.

    Args:
        code_content: Raw Python source code as a string.

    Returns:
        A tuple of (ast.Module, None) on success or (None, error_message) on
        syntax error.
    """
    try:
        tree = ast.parse(code_content)
        return tree, None
    except SyntaxError as exc:
        return None, f"SyntaxError at line {exc.lineno}: {exc.msg}"


def _check_docstrings(tree: ast.Module) -> List[Dict[str, Any]]:
    """
    Walk the AST and flag any module, class, or function that lacks a docstring.

    Args:
        tree: Parsed AST module object.

    Returns:
        List of issue dicts, each with keys: type, severity, line, message.
    """
    issues: List[Dict[str, Any]] = []

    # Module-level docstring
    if not (tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)):
        issues.append({
            "type": "missing_docstring_module",
            "severity": SEVERITY_MEDIUM,
            "line": 1,
            "message": "Module is missing a top-level docstring.",
        })

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)):
                issues.append({
                    "type": "missing_docstring_function",
                    "severity": SEVERITY_LOW,
                    "line": node.lineno,
                    "message": f"Function '{node.name}' at line {node.lineno} is missing a docstring.",
                })
        elif isinstance(node, ast.ClassDef):
            if not (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)):
                issues.append({
                    "type": "missing_docstring_class",
                    "severity": SEVERITY_MEDIUM,
                    "line": node.lineno,
                    "message": f"Class '{node.name}' at line {node.lineno} is missing a docstring.",
                })

    return issues


def _check_function_complexity(tree: ast.Module, max_lines: int = 30, max_params: int = 5) -> List[Dict[str, Any]]:
    """
    Detect functions that are too long or have too many parameters.

    Args:
        tree:       Parsed AST module object.
        max_lines:  Maximum allowed lines per function (default 30).
        max_params: Maximum allowed parameters per function (default 5).

    Returns:
        List of issue dicts.
    """
    issues: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Function length check
            if hasattr(node, "end_lineno") and node.end_lineno is not None:
                length = node.end_lineno - node.lineno + 1
                if length > max_lines:
                    issues.append({
                        "type": "function_too_long",
                        "severity": SEVERITY_MEDIUM,
                        "line": node.lineno,
                        "message": (
                            f"Function '{node.name}' is {length} lines long "
                            f"(max allowed: {max_lines}). Consider breaking it into smaller functions."
                        ),
                    })

            # Parameter count check
            param_count = len(node.args.args)
            if param_count > max_params:
                issues.append({
                    "type": "too_many_parameters",
                    "severity": SEVERITY_MEDIUM,
                    "line": node.lineno,
                    "message": (
                        f"Function '{node.name}' has {param_count} parameters "
                        f"(max allowed: {max_params}). Consider using a config object."
                    ),
                })

    return issues


def _check_nesting_depth(tree: ast.Module, max_depth: int = 4) -> List[Dict[str, Any]]:
    """
    Detect deeply nested code blocks that reduce readability.

    Args:
        tree:      Parsed AST module object.
        max_depth: Maximum allowed nesting depth (default 4).

    Returns:
        List of issue dicts.
    """
    issues: List[Dict[str, Any]] = []
    NESTING_NODES = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)

    def _depth(node: ast.AST, current: int) -> None:
        if current > max_depth:
            issues.append({
                "type": "deeply_nested_code",
                "severity": SEVERITY_MEDIUM,
                "line": getattr(node, "lineno", 0),
                "message": (
                    f"Code is nested {current} levels deep at line "
                    f"{getattr(node, 'lineno', '?')} (max allowed: {max_depth})."
                ),
            })
            return  # Don't recurse further — avoid duplicate reports

        for child in ast.iter_child_nodes(node):
            if isinstance(child, NESTING_NODES):
                _depth(child, current + 1)
            else:
                _depth(child, current)

    _depth(tree, 0)
    return issues


def _check_bad_patterns(tree: ast.Module) -> List[Dict[str, Any]]:
    """
    Detect common Python anti-patterns:
      - Bare except clauses
      - Mutable default arguments (list/dict/set as default value)
      - Wildcard imports (from x import *)
      - Global variable usage

    Args:
        tree: Parsed AST module object.

    Returns:
        List of issue dicts.
    """
    issues: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        # Bare except: ExceptHandler with no type specified
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append({
                "type": "bare_except_clause",
                "severity": SEVERITY_HIGH,
                "line": node.lineno,
                "message": (
                    f"Bare 'except:' clause at line {node.lineno}. "
                    "Always specify the exception type (e.g., 'except ValueError:')."
                ),
            })

        # Mutable default arguments
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    issues.append({
                        "type": "mutable_default_argument",
                        "severity": SEVERITY_HIGH,
                        "line": node.lineno,
                        "message": (
                            f"Function '{node.name}' uses a mutable default argument "
                            f"at line {node.lineno}. Use None and assign inside the function."
                        ),
                    })

        # Wildcard import: from module import *
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            issues.append({
                "type": "wildcard_import",
                "severity": SEVERITY_MEDIUM,
                "line": node.lineno,
                "message": (
                    f"Wildcard import 'from {node.module} import *' at line {node.lineno}. "
                    "Explicitly import only what you need."
                ),
            })

        # Global statement usage
        if isinstance(node, ast.Global):
            issues.append({
                "type": "global_variable",
                "severity": SEVERITY_MEDIUM,
                "line": node.lineno,
                "message": (
                    f"'global' statement at line {node.lineno}: {', '.join(node.names)}. "
                    "Avoid global state — use function parameters or class attributes."
                ),
            })

    return issues


def _calculate_cyclomatic_complexity(tree: ast.Module) -> Dict[str, Any]:
    """
    Calculate a simplified cyclomatic complexity score per function.

    Cyclomatic complexity = 1 + number of decision points (if, for, while,
    and, or, except, with, assert, comprehensions).

    Args:
        tree: Parsed AST module object.

    Returns:
        Dict with keys:
          - "per_function": {func_name: complexity_score}
          - "average": float
          - "max": int
          - "issues": list of high-complexity warnings
    """
    DECISION_NODES = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
        ast.BoolOp,  # covers 'and' / 'or'
    )

    per_function: Dict[str, int] = {}
    issues: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = 1  # base complexity
            for child in ast.walk(node):
                if isinstance(child, DECISION_NODES):
                    complexity += 1
            per_function[node.name] = complexity

            if complexity > 10:
                issues.append({
                    "type": "high_cyclomatic_complexity",
                    "severity": SEVERITY_HIGH,
                    "line": node.lineno,
                    "message": (
                        f"Function '{node.name}' has cyclomatic complexity of {complexity} "
                        f"(threshold: 10). High complexity increases bug risk."
                    ),
                })

    scores = list(per_function.values()) or [1]
    return {
        "per_function": per_function,
        "average": round(sum(scores) / len(scores), 2),
        "max": max(scores),
        "issues": issues,
    }


def _collect_metrics(tree: ast.Module, code_content: str) -> Dict[str, Any]:
    """
    Collect high-level code metrics.

    Args:
        tree:         Parsed AST module object.
        code_content: Raw source code string.

    Returns:
        Dict of numeric metrics: lines_of_code, num_functions, num_classes,
        num_imports, blank_lines, comment_lines, docstring_coverage_pct.
    """
    lines = code_content.splitlines()
    total_lines = len(lines)
    blank_lines = sum(1 for l in lines if l.strip() == "")
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))

    functions: List[ast.FunctionDef] = [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    classes: List[ast.ClassDef] = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    imports: List[ast.stmt] = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]

    # Docstring coverage
    documented = sum(
        1 for f in functions
        if f.body and isinstance(f.body[0], ast.Expr) and isinstance(f.body[0].value, ast.Constant)
    )
    coverage_pct = (documented / len(functions) * 100) if functions else 100.0

    return {
        "total_lines": total_lines,
        "lines_of_code": total_lines - blank_lines - comment_lines,
        "blank_lines": blank_lines,
        "comment_lines": comment_lines,
        "num_functions": len(functions),
        "num_classes": len(classes),
        "num_imports": len(imports),
        "docstring_coverage_pct": round(coverage_pct, 1),
    }


def _calculate_quality_score(issues: List[Dict[str, Any]]) -> float:
    """
    Calculate a 0–100 quality score by deducting points for each issue.

    Args:
        issues: Combined list of all detected issues.

    Returns:
        Float score clamped between 0.0 and 100.0.
    """
    score = 100.0
    for issue in issues:
        deduction = SCORE_DEDUCTIONS.get(issue["type"], 2.0)
        score -= deduction
    return round(max(0.0, min(100.0, score)), 2)


def _run_pylint(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Run pylint on the target file and return parsed results.

    Returns None if pylint is not installed or the file doesn't exist.

    Args:
        file_path: Absolute or relative path to the Python file.

    Returns:
        Dict with "score" and "messages", or None on failure.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pylint", file_path, "--output-format=json", "--score=yes"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        messages = json.loads(result.stdout) if result.stdout.strip() else []
        # Extract score from stderr (pylint prints "Your code has been rated at X/10")
        score_line = [l for l in result.stderr.splitlines() if "rated at" in l]
        score_str = score_line[0].split("rated at")[1].split("/")[0].strip() if score_line else "N/A"
        return {"score": score_str, "messages": messages[:20]}  # cap at 20 messages
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC TOOL FUNCTION  ← This is what the LangGraph agent calls
# ─────────────────────────────────────────────────────────────────────────────


def analyze_code_quality(
    code_content: str,
    file_path: str = "unknown.py",
    language: str = "python",
    run_pylint: bool = True,
) -> Dict[str, Any]:
    """
    Perform comprehensive static analysis of Python source code.

    This is the primary tool used by Agent 2 (Code Analyzer). It combines
    AST-based analysis with optional pylint integration to produce a
    structured quality report.

    Analysis dimensions:
      1. Docstring coverage  — Are all functions/classes/modules documented?
      2. Function complexity — Are functions too long or have too many params?
      3. Nesting depth       — Is the code too deeply nested to follow?
      4. Anti-patterns       — Bare excepts, mutable defaults, wildcard imports, globals
      5. Cyclomatic complexity — Decision-point count per function
      6. Code metrics        — LOC, blank lines, class/function counts
      7. Quality score       — Composite 0-100 score (100 = perfect)
      8. Pylint integration  — Optional deeper lint check if pylint is installed

    Args:
        code_content: Raw Python source code as a plain string.
        file_path:    Path to the source file (used for pylint and reporting).
        language:     Programming language (currently only "python" is supported).
        run_pylint:   If True, attempt to run pylint for additional checks.

    Returns:
        A dict with the following structure::

            {
                "file_path": str,
                "language": str,
                "syntax_error": str | None,
                "quality_score": float,          # 0–100
                "metrics": {
                    "total_lines": int,
                    "lines_of_code": int,
                    "num_functions": int,
                    "num_classes": int,
                    "docstring_coverage_pct": float,
                    ...
                },
                "cyclomatic_complexity": {
                    "per_function": {name: score},
                    "average": float,
                    "max": int,
                },
                "issues": [
                    {
                        "type": str,
                        "severity": "HIGH" | "MEDIUM" | "LOW",
                        "line": int,
                        "message": str,
                    },
                    ...
                ],
                "issue_counts": {
                    "HIGH": int,
                    "MEDIUM": int,
                    "LOW": int,
                    "TOTAL": int,
                },
                "pylint": {
                    "score": str,
                    "messages": [...],
                } | None,
            }

    Raises:
        ValueError: If code_content is empty or language is not "python".

    Example::

        result = analyze_code_quality(
            code_content=open("my_script.py").read(),
            file_path="my_script.py",
        )
        print(f"Quality score: {result['quality_score']}/100")
        for issue in result['issues']:
            print(f"  [{issue['severity']}] Line {issue['line']}: {issue['message']}")
    """
    # ── Input validation ──────────────────────────────────────────────────
    if not code_content or not code_content.strip():
        raise ValueError("code_content must not be empty.")
    if language.lower() != "python":
        raise ValueError(f"Language '{language}' is not supported. Only 'python' is currently supported.")

    result: Dict[str, Any] = {
        "file_path": file_path,
        "language": language,
        "syntax_error": None,
        "quality_score": 0.0,
        "metrics": {},
        "cyclomatic_complexity": {},
        "issues": [],
        "issue_counts": {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "TOTAL": 0},
        "pylint": None,
    }

    # ── Parse AST ─────────────────────────────────────────────────────────
    tree, parse_error = _parse_ast(code_content)
    if parse_error:
        result["syntax_error"] = parse_error
        result["quality_score"] = 0.0
        result["issues"] = [{
            "type": "syntax_error",
            "severity": SEVERITY_HIGH,
            "line": 0,
            "message": parse_error,
        }]
        result["issue_counts"] = {"HIGH": 1, "MEDIUM": 0, "LOW": 0, "TOTAL": 1}
        return result

    # ── Run all checks ────────────────────────────────────────────────────
    all_issues: List[Dict[str, Any]] = []
    all_issues.extend(_check_docstrings(tree))
    all_issues.extend(_check_function_complexity(tree))
    all_issues.extend(_check_nesting_depth(tree))
    all_issues.extend(_check_bad_patterns(tree))

    complexity_result = _calculate_cyclomatic_complexity(tree)
    all_issues.extend(complexity_result.pop("issues"))

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = _collect_metrics(tree, code_content)

    # ── Quality score ─────────────────────────────────────────────────────
    quality_score = _calculate_quality_score(all_issues)

    # ── Issue counts ──────────────────────────────────────────────────────
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "TOTAL": len(all_issues)}
    for issue in all_issues:
        counts[issue["severity"]] += 1

    # ── Optional pylint ───────────────────────────────────────────────────
    pylint_result = None
    if run_pylint and Path(file_path).exists():
        pylint_result = _run_pylint(file_path)

    # ── Assemble final result ─────────────────────────────────────────────
    result.update({
        "quality_score": quality_score,
        "metrics": metrics,
        "cyclomatic_complexity": complexity_result,
        "issues": all_issues,
        "issue_counts": counts,
        "pylint": pylint_result,
    })

    return result
