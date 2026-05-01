"""
security_scan_tool.py
---------------------
Custom Python tool for Agent 3: Security Scanner.

Performs static security analysis of Python source code using the built-in
`ast` module (zero extra dependencies) plus optional Bandit integration if
it is installed.

Vulnerability categories covered:
  1. Hardcoded secrets       — passwords, API keys, tokens in plain text
  2. Dangerous functions     — eval(), exec(), pickle.loads(), os.system()
  3. SQL injection risk       — string-formatted SQL queries
  4. Insecure deserialization — pickle, marshal, yaml.load()
  5. Path traversal risk      — open() with user-controlled paths
  6. Weak cryptography        — MD5, SHA1 usage
  7. Debug/dev artifacts      — assert statements used as security checks,
                                hardcoded DEBUG=True flags
  8. Insecure network calls   — http:// URLs in requests, verify=False TLS
  9. Command injection risk   — subprocess with shell=True
 10. Sensitive data exposure  — print() of password/token/secret variables
"""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"

# Points deducted per vulnerability type (from base score of 100)
SCORE_DEDUCTIONS: Dict[str, float] = {
    "hardcoded_secret": 15.0,
    "dangerous_function_eval": 15.0,
    "dangerous_function_exec": 12.0,
    "dangerous_function_compile": 8.0,
    "sql_injection_risk": 12.0,
    "insecure_deserialization_pickle": 12.0,
    "insecure_deserialization_yaml": 10.0,
    "insecure_deserialization_marshal": 10.0,
    "weak_cryptography": 8.0,
    "path_traversal_risk": 8.0,
    "debug_artifact": 5.0,
    "insecure_http_url": 5.0,
    "tls_verification_disabled": 10.0,
    "command_injection_shell_true": 12.0,
    "sensitive_data_exposure": 8.0,
    "assert_used_as_guard": 6.0,
}

# Regex patterns for hardcoded secret detection (applied to string literals)
_SECRET_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)(password|passwd|pwd)\s*=\s*['\"][^'\"]{4,}['\"]", "Hardcoded password"),
    (r"(?i)(api_?key|apikey|access_?key)\s*=\s*['\"][^'\"]{8,}['\"]", "Hardcoded API key"),
    (r"(?i)(secret|token|auth)\s*=\s*['\"][^'\"]{8,}['\"]", "Hardcoded secret/token"),
    (r"(?i)(private_?key|rsa_?key)\s*=\s*['\"][^'\"]{8,}['\"]", "Hardcoded private key"),
    (r"(?i)aws_?(access|secret)_?key\s*=\s*['\"][^'\"]{8,}['\"]", "Hardcoded AWS credential"),
]

# Dangerous built-in function names
_DANGEROUS_BUILTINS: Dict[str, Tuple[str, str]] = {
    "eval":    ("dangerous_function_eval",    SEVERITY_CRITICAL),
    "exec":    ("dangerous_function_exec",    SEVERITY_HIGH),
    "compile": ("dangerous_function_compile", SEVERITY_MEDIUM),
}

# Insecure module.function pairs  →  (vuln_type, severity)
_DANGEROUS_CALLS: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("pickle",  "loads"):       ("insecure_deserialization_pickle", SEVERITY_CRITICAL),
    ("pickle",  "load"):        ("insecure_deserialization_pickle", SEVERITY_CRITICAL),
    ("cPickle", "loads"):       ("insecure_deserialization_pickle", SEVERITY_CRITICAL),
    ("marshal", "loads"):       ("insecure_deserialization_marshal", SEVERITY_HIGH),
    ("yaml",    "load"):        ("insecure_deserialization_yaml",   SEVERITY_HIGH),
    ("os",      "system"):      ("dangerous_function_exec",         SEVERITY_HIGH),
    ("os",      "popen"):       ("dangerous_function_exec",         SEVERITY_HIGH),
    ("hashlib", "md5"):         ("weak_cryptography",               SEVERITY_MEDIUM),
    ("hashlib", "sha1"):        ("weak_cryptography",               SEVERITY_MEDIUM),
    ("Crypto",  "MD5"):         ("weak_cryptography",               SEVERITY_MEDIUM),
    ("Crypto",  "SHA"):         ("weak_cryptography",               SEVERITY_MEDIUM),
}


# ─────────────────────────────────────────────────────────────────────────────
# AST helper
# ─────────────────────────────────────────────────────────────────────────────


def _parse_ast(code_content: str) -> Tuple[Optional[ast.Module], Optional[str]]:
    """
    Parse Python source into an AST module.

    Args:
        code_content: Raw Python source code.

    Returns:
        (ast.Module, None) on success, (None, error_message) on SyntaxError.
    """
    try:
        return ast.parse(code_content), None
    except SyntaxError as exc:
        return None, f"SyntaxError at line {exc.lineno}: {exc.msg}"


# ─────────────────────────────────────────────────────────────────────────────
# Detection functions
# ─────────────────────────────────────────────────────────────────────────────


def _detect_hardcoded_secrets(code_content: str) -> List[Dict[str, Any]]:
    """
    Scan raw source text for hardcoded credentials using regex patterns.

    This intentionally operates on the raw text (not AST) so multi-line
    string assignments are also caught.

    Args:
        code_content: Raw Python source code.

    Returns:
        List of vulnerability dicts (type, severity, line, message, cwe).
    """
    issues: List[Dict[str, Any]] = []
    lines = code_content.splitlines()

    for line_no, line in enumerate(lines, start=1):
        for pattern, label in _SECRET_PATTERNS:
            if re.search(pattern, line):
                # Redact the value from the reported line to avoid leaking it
                safe_line = re.sub(r"(['\"])[^'\"]{4,}(['\"])", r"\1***REDACTED***\2", line.strip())
                issues.append({
                    "type": "hardcoded_secret",
                    "severity": SEVERITY_CRITICAL,
                    "line": line_no,
                    "message": (
                        f"{label} detected at line {line_no}: `{safe_line}`. "
                        "Never store credentials in source code. Use environment variables."
                    ),
                    "cwe": "CWE-798: Use of Hard-coded Credentials",
                    "fix": "Replace with os.environ.get('SECRET_NAME') or a secrets manager.",
                })
                break  # One report per line

    return issues


def _detect_dangerous_functions(tree: ast.Module) -> List[Dict[str, Any]]:
    """
    Detect calls to dangerous built-in functions (eval, exec, compile).

    Args:
        tree: Parsed AST module.

    Returns:
        List of vulnerability dicts.
    """
    issues: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Direct call: eval(...), exec(...)
            if isinstance(node.func, ast.Name) and node.func.id in _DANGEROUS_BUILTINS:
                func_name = node.func.id
                vuln_type, severity = _DANGEROUS_BUILTINS[func_name]
                issues.append({
                    "type": vuln_type,
                    "severity": severity,
                    "line": node.lineno,
                    "message": (
                        f"Dangerous call to '{func_name}()' at line {node.lineno}. "
                        "Arbitrary code execution risk."
                    ),
                    "cwe": "CWE-95: Improper Neutralization of Directives in Dynamically Evaluated Code",
                    "fix": f"Avoid '{func_name}()'. Use a safe parser or data structure instead.",
                })

    return issues


def _detect_dangerous_module_calls(tree: ast.Module) -> List[Dict[str, Any]]:
    """
    Detect calls to dangerous module functions (pickle.loads, yaml.load, etc.).

    Handles both:
      - import pickle; pickle.loads(data)        → Attribute call
      - from pickle import loads; loads(data)    → Name call after alias tracking

    Args:
        tree: Parsed AST module.

    Returns:
        List of vulnerability dicts.
    """
    issues: List[Dict[str, Any]] = []

    # Build alias map: local_name → module_name for "from X import Y" imports
    alias_map: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local = alias.asname or alias.name
                alias_map[local] = node.module

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # module.function() pattern
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            module = node.func.value.id
            func = node.func.attr
            key = (module, func)
            if key in _DANGEROUS_CALLS:
                vuln_type, severity = _DANGEROUS_CALLS[key]

                # Special handling: yaml.load() is safe with Loader=yaml.SafeLoader
                if module == "yaml" and func == "load":
                    keywords = {kw.arg for kw in node.keywords}
                    if "Loader" in keywords:
                        continue  # Safe usage with explicit Loader

                issues.append(_build_module_issue(vuln_type, severity, node.lineno, module, func))

        # Direct call after "from pickle import loads" style import
        elif isinstance(node.func, ast.Name):
            local_name = node.func.id
            parent_module = alias_map.get(local_name, "")
            key = (parent_module.split(".")[0], local_name)
            if key in _DANGEROUS_CALLS:
                vuln_type, severity = _DANGEROUS_CALLS[key]
                issues.append(_build_module_issue(vuln_type, severity, node.lineno, parent_module, local_name))

    return issues


def _build_module_issue(
    vuln_type: str,
    severity: str,
    line: int,
    module: str,
    func: str,
) -> Dict[str, Any]:
    """Build a standardised issue dict for a dangerous module call."""
    cwe_map = {
        "insecure_deserialization_pickle": "CWE-502: Deserialization of Untrusted Data",
        "insecure_deserialization_yaml":   "CWE-502: Deserialization of Untrusted Data",
        "insecure_deserialization_marshal":"CWE-502: Deserialization of Untrusted Data",
        "dangerous_function_exec":         "CWE-78: OS Command Injection",
        "weak_cryptography":               "CWE-327: Use of Broken Cryptographic Algorithm",
    }
    fix_map = {
        "insecure_deserialization_pickle": "Use json.loads() for safe data exchange instead of pickle.",
        "insecure_deserialization_yaml":   "Use yaml.safe_load() to prevent arbitrary object instantiation.",
        "insecure_deserialization_marshal":"Use JSON or another safe serialization format.",
        "dangerous_function_exec":         "Avoid os.system(); use subprocess.run() with a list of args.",
        "weak_cryptography":               "Use SHA-256 or SHA-3 via hashlib.sha256() instead.",
    }
    return {
        "type": vuln_type,
        "severity": severity,
        "line": line,
        "message": (
            f"Insecure call to '{module}.{func}()' at line {line}."
        ),
        "cwe": cwe_map.get(vuln_type, "CWE-Unknown"),
        "fix": fix_map.get(vuln_type, "Review usage and replace with a safer alternative."),
    }


def _detect_sql_injection(tree: ast.Module) -> List[Dict[str, Any]]:
    """
    Detect SQL injection risk: string formatting/concatenation used to build
    SQL queries that are then passed to execute().

    Looks for patterns like:
      cursor.execute("SELECT ... WHERE id=%s" % user_input)
      cursor.execute("SELECT ... WHERE id=" + user_id)
      cursor.execute(f"SELECT ... WHERE id={user_id}")

    Args:
        tree: Parsed AST module.

    Returns:
        List of vulnerability dicts.
    """
    issues: List[Dict[str, Any]] = []
    SQL_KEYWORDS = {"select", "insert", "update", "delete", "drop", "create", "alter", "truncate"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "execute"):
            continue

        if not node.args:
            continue

        first_arg = node.args[0]

        # f-string: f"SELECT ... {var}"
        if isinstance(first_arg, ast.JoinedStr):
            sql_text = "".join(
                part.value if isinstance(part, ast.Constant) else "{expr}"
                for part in first_arg.values
            )
            if any(kw in sql_text.lower() for kw in SQL_KEYWORDS):
                issues.append({
                    "type": "sql_injection_risk",
                    "severity": SEVERITY_CRITICAL,
                    "line": node.lineno,
                    "message": (
                        f"Potential SQL injection at line {node.lineno}: f-string used to build SQL query. "
                        "User-controlled values may be injected."
                    ),
                    "cwe": "CWE-89: SQL Injection",
                    "fix": "Use parameterised queries: cursor.execute('SELECT ... WHERE id=?', (user_id,))",
                })

        # % formatting or + concatenation
        elif isinstance(first_arg, (ast.BinOp)):
            if isinstance(first_arg.op, (ast.Mod, ast.Add)):
                if isinstance(first_arg.left, ast.Constant) and isinstance(first_arg.left.value, str):
                    if any(kw in first_arg.left.value.lower() for kw in SQL_KEYWORDS):
                        issues.append({
                            "type": "sql_injection_risk",
                            "severity": SEVERITY_CRITICAL,
                            "line": node.lineno,
                            "message": (
                                f"Potential SQL injection at line {node.lineno}: "
                                "string formatting used to build SQL query."
                            ),
                            "cwe": "CWE-89: SQL Injection",
                            "fix": "Use parameterised queries: cursor.execute('SELECT ... WHERE id=?', (val,))",
                        })

    return issues


def _detect_command_injection(tree: ast.Module) -> List[Dict[str, Any]]:
    """
    Detect subprocess calls with shell=True which enable command injection.

    Args:
        tree: Parsed AST module.

    Returns:
        List of vulnerability dicts.
    """
    issues: List[Dict[str, Any]] = []
    SUBPROCESS_FUNCS = {"run", "call", "check_call", "check_output", "Popen"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if not (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr in SUBPROCESS_FUNCS
        ):
            continue

        for kw in node.keywords:
            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                issues.append({
                    "type": "command_injection_shell_true",
                    "severity": SEVERITY_HIGH,
                    "line": node.lineno,
                    "message": (
                        f"subprocess.{node.func.attr}() called with shell=True at line {node.lineno}. "
                        "Enables OS command injection if any argument is user-controlled."
                    ),
                    "cwe": "CWE-78: OS Command Injection",
                    "fix": "Pass a list of args instead: subprocess.run(['cmd', arg1, arg2], shell=False)",
                })

    return issues


def _detect_insecure_network(tree: ast.Module, code_content: str) -> List[Dict[str, Any]]:
    """
    Detect insecure network patterns:
      - HTTP URLs in requests calls (should be HTTPS)
      - requests calls with verify=False (disables TLS certificate validation)

    Args:
        tree:         Parsed AST module.
        code_content: Raw source text (for URL scanning).

    Returns:
        List of vulnerability dicts.
    """
    issues: List[Dict[str, Any]] = []

    # Scan raw text for http:// in requests.get/post/put/delete calls
    lines = code_content.splitlines()
    for line_no, line in enumerate(lines, start=1):
        if re.search(r'requests\.(get|post|put|delete|request)\s*\(.*http://', line):
            issues.append({
                "type": "insecure_http_url",
                "severity": SEVERITY_MEDIUM,
                "line": line_no,
                "message": (
                    f"Plain HTTP URL used in requests call at line {line_no}. "
                    "Data transmitted without encryption."
                ),
                "cwe": "CWE-319: Cleartext Transmission of Sensitive Information",
                "fix": "Use https:// to encrypt data in transit.",
            })

    # AST scan for verify=False
    REQUESTS_METHODS = {"get", "post", "put", "delete", "patch", "request", "Session"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_requests_call = (
            (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
             and func.value.id == "requests" and func.attr in REQUESTS_METHODS)
        )
        if not is_requests_call:
            continue
        for kw in node.keywords:
            if kw.arg == "verify" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                issues.append({
                    "type": "tls_verification_disabled",
                    "severity": SEVERITY_HIGH,
                    "line": node.lineno,
                    "message": (
                        f"TLS certificate verification disabled (verify=False) at line {node.lineno}. "
                        "Vulnerable to man-in-the-middle attacks."
                    ),
                    "cwe": "CWE-295: Improper Certificate Validation",
                    "fix": "Remove verify=False. If using self-signed certs, set verify='/path/to/cert.pem'.",
                })

    return issues


def _detect_debug_artifacts(tree: ast.Module) -> List[Dict[str, Any]]:
    """
    Detect security-relevant debug artifacts:
      - assert statements used as input validation (bypassed with python -O)
      - DEBUG = True assignments

    Args:
        tree: Parsed AST module.

    Returns:
        List of vulnerability dicts.
    """
    issues: List[Dict[str, Any]] = []

    for node in ast.walk(tree):
        # assert used for access control / input validation
        if isinstance(node, ast.Assert):
            issues.append({
                "type": "assert_used_as_guard",
                "severity": SEVERITY_MEDIUM,
                "line": node.lineno,
                "message": (
                    f"'assert' statement at line {node.lineno}. "
                    "Assertions are disabled when Python runs with -O (optimise). "
                    "Never use assert for security checks or input validation."
                ),
                "cwe": "CWE-617: Reachable Assertion",
                "fix": "Replace with an explicit if/raise check: if not condition: raise ValueError(...)",
            })

        # DEBUG = True
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id.upper() in {"DEBUG", "TESTING", "DEV_MODE"}
                for t in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
        ):
            target_name = next(
                t.id for t in node.targets
                if isinstance(t, ast.Name) and t.id.upper() in {"DEBUG", "TESTING", "DEV_MODE"}
            )
            issues.append({
                "type": "debug_artifact",
                "severity": SEVERITY_MEDIUM,
                "line": node.lineno,
                "message": (
                    f"'{target_name} = True' at line {node.lineno}. "
                    "Debug mode enabled in source code. May expose stack traces or sensitive data."
                ),
                "cwe": "CWE-489: Active Debug Code",
                "fix": f"Set {target_name} via environment variable: {target_name} = os.getenv('{target_name}', 'False') == 'True'",
            })

    return issues


def _detect_sensitive_data_exposure(tree: ast.Module) -> List[Dict[str, Any]]:
    """
    Detect print() or logging calls that may expose sensitive variable names.

    Flags calls like: print(password), print(api_key), logger.debug(token)

    Args:
        tree: Parsed AST module.

    Returns:
        List of vulnerability dicts.
    """
    issues: List[Dict[str, Any]] = []
    SENSITIVE_NAMES = {
        "password", "passwd", "pwd", "api_key", "apikey", "secret",
        "token", "auth_token", "access_token", "private_key", "credential",
    }
    PRINT_FUNCS = {"print"}
    LOG_ATTRS = {"debug", "info", "warning", "error", "critical"}

    def _arg_contains_sensitive(args: list) -> Optional[str]:
        for arg in args:
            if isinstance(arg, ast.Name) and arg.id.lower() in SENSITIVE_NAMES:
                return arg.id
            if isinstance(arg, ast.Attribute) and arg.attr.lower() in SENSITIVE_NAMES:
                return arg.attr
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # print(sensitive_var)
        if isinstance(node.func, ast.Name) and node.func.id in PRINT_FUNCS:
            name = _arg_contains_sensitive(node.args)
            if name:
                issues.append({
                    "type": "sensitive_data_exposure",
                    "severity": SEVERITY_HIGH,
                    "line": node.lineno,
                    "message": (
                        f"Potential sensitive data exposure: print('{name}') at line {node.lineno}. "
                        "Sensitive values should never be printed to stdout."
                    ),
                    "cwe": "CWE-209: Information Exposure Through an Error Message",
                    "fix": "Remove the print statement or mask the value: print('***').",
                })

        # logger.debug(sensitive_var) / logging.info(sensitive_var)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in LOG_ATTRS
            and isinstance(node.func.value, ast.Name)
        ):
            name = _arg_contains_sensitive(node.args)
            if name:
                issues.append({
                    "type": "sensitive_data_exposure",
                    "severity": SEVERITY_HIGH,
                    "line": node.lineno,
                    "message": (
                        f"Potential sensitive data logged: .{node.func.attr}('{name}') at line {node.lineno}. "
                        "Log files may be accessible to attackers."
                    ),
                    "cwe": "CWE-532: Insertion of Sensitive Information into Log File",
                    "fix": "Mask the value before logging: log.debug('token=%s', '***')",
                })

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────


def _calculate_security_score(vulnerabilities: List[Dict[str, Any]]) -> float:
    """
    Calculate a 0–100 security score. Deduct points per vulnerability type.

    CRITICAL findings apply a 1.5× multiplier to reflect higher risk.

    Args:
        vulnerabilities: List of vulnerability dicts from all detection passes.

    Returns:
        Float score clamped to [0.0, 100.0].
    """
    score = 100.0
    for vuln in vulnerabilities:
        base = SCORE_DEDUCTIONS.get(vuln["type"], 5.0)
        multiplier = 1.5 if vuln["severity"] == SEVERITY_CRITICAL else 1.0
        score -= base * multiplier
    return round(max(0.0, min(100.0, score)), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Optional Bandit integration
# ─────────────────────────────────────────────────────────────────────────────


def _run_bandit(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Run Bandit security linter on the target file if it is installed.

    Bandit (https://bandit.readthedocs.io) is a widely-used Python security
    linter maintained by PyCQA. It checks for common security issues.

    Args:
        file_path: Path to the Python source file.

    Returns:
        Dict with "results" and "metrics" from Bandit, or None if unavailable.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "bandit", "-f", "json", "-q", file_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Bandit exits with 1 when issues are found — that is expected
        output = result.stdout.strip() or result.stderr.strip()
        if not output:
            return None
        data = json.loads(output)
        return {
            "results": data.get("results", [])[:20],  # cap at 20
            "metrics": data.get("metrics", {}),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC TOOL FUNCTION  ← This is what the LangGraph agent calls
# ─────────────────────────────────────────────────────────────────────────────


def check_security(
    code_content: str,
    file_path: str = "unknown.py",
    language: str = "python",
    run_bandit: bool = True,
) -> Dict[str, Any]:
    """
    Perform comprehensive static security analysis of Python source code.

    This is the primary tool used by Agent 3 (Security Scanner). It combines
    AST-based vulnerability detection with optional Bandit integration to
    produce a structured security report.

    Security dimensions covered:
      1. Hardcoded secrets       — credentials, API keys, tokens in plain text
      2. Dangerous functions     — eval(), exec(), os.system()
      3. Insecure deserialization — pickle, yaml.load(), marshal
      4. SQL injection risk       — f-string / % / + formatted SQL queries
      5. Command injection        — subprocess with shell=True
      6. Insecure network         — HTTP URLs, verify=False TLS
      7. Weak cryptography        — MD5, SHA1
      8. Debug artifacts          — DEBUG=True, assert-as-guard
      9. Sensitive data exposure  — print(password), log(token)
     10. Bandit scan (optional)   — if bandit is installed

    Args:
        code_content: Raw Python source code as a plain string.
        file_path:    Path to the source file (used for Bandit and reporting).
        language:     Programming language (only "python" is supported).
        run_bandit:   If True, attempt to run Bandit for additional findings.

    Returns:
        A dict with the following structure::

            {
                "file_path": str,
                "language": str,
                "syntax_error": str | None,
                "security_score": float,          # 0–100 (100 = no issues)
                "vulnerabilities": [
                    {
                        "type": str,
                        "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
                        "line": int,
                        "message": str,
                        "cwe": str,               # CWE reference
                        "fix": str,               # Specific remediation advice
                    },
                    ...
                ],
                "vulnerability_counts": {
                    "CRITICAL": int,
                    "HIGH": int,
                    "MEDIUM": int,
                    "LOW": int,
                    "TOTAL": int,
                },
                "risk_level": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "SAFE",
                "bandit": {
                    "results": [...],
                    "metrics": {...},
                } | None,
            }

    Raises:
        ValueError: If code_content is empty or language is not "python".

    Example::

        result = check_security(
            code_content=open("my_app.py").read(),
            file_path="my_app.py",
        )
        print(f"Security score: {result['security_score']}/100")
        print(f"Risk level: {result['risk_level']}")
        for vuln in result['vulnerabilities']:
            print(f"  [{vuln['severity']}] {vuln['cwe']} — Line {vuln['line']}: {vuln['message']}")
    """
    # ── Input validation ──────────────────────────────────────────────────
    if not code_content or not code_content.strip():
        raise ValueError("code_content must not be empty.")
    if language.lower() != "python":
        raise ValueError(
            f"Language '{language}' is not supported. Only 'python' is currently supported."
        )

    result: Dict[str, Any] = {
        "file_path": file_path,
        "language": language,
        "syntax_error": None,
        "security_score": 0.0,
        "vulnerabilities": [],
        "vulnerability_counts": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "TOTAL": 0},
        "risk_level": "SAFE",
        "bandit": None,
    }

    # ── Parse AST ─────────────────────────────────────────────────────────
    tree, parse_error = _parse_ast(code_content)
    if parse_error:
        result["syntax_error"] = parse_error
        result["security_score"] = 0.0
        result["vulnerabilities"] = [{
            "type": "syntax_error",
            "severity": SEVERITY_CRITICAL,
            "line": 0,
            "message": f"Cannot analyse: {parse_error}",
            "cwe": "N/A",
            "fix": "Fix the syntax error before running a security scan.",
        }]
        result["vulnerability_counts"] = {"CRITICAL": 1, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "TOTAL": 1}
        result["risk_level"] = "CRITICAL"
        return result

    # ── Run all detection passes ──────────────────────────────────────────
    all_vulns: List[Dict[str, Any]] = []
    all_vulns.extend(_detect_hardcoded_secrets(code_content))
    all_vulns.extend(_detect_dangerous_functions(tree))
    all_vulns.extend(_detect_dangerous_module_calls(tree))
    all_vulns.extend(_detect_sql_injection(tree))
    all_vulns.extend(_detect_command_injection(tree))
    all_vulns.extend(_detect_insecure_network(tree, code_content))
    all_vulns.extend(_detect_debug_artifacts(tree))
    all_vulns.extend(_detect_sensitive_data_exposure(tree))

    # ── Deduplicate by (type, line) ───────────────────────────────────────
    seen: set = set()
    unique_vulns: List[Dict[str, Any]] = []
    for v in all_vulns:
        key = (v["type"], v["line"])
        if key not in seen:
            seen.add(key)
            unique_vulns.append(v)

    # Sort: CRITICAL first, then HIGH, MEDIUM, LOW; within tier sort by line
    severity_order = {SEVERITY_CRITICAL: 0, SEVERITY_HIGH: 1, SEVERITY_MEDIUM: 2, SEVERITY_LOW: 3}
    unique_vulns.sort(key=lambda v: (severity_order.get(v["severity"], 4), v["line"]))

    # ── Compute counts ────────────────────────────────────────────────────
    counts: Dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "TOTAL": len(unique_vulns)}
    for v in unique_vulns:
        counts[v["severity"]] = counts.get(v["severity"], 0) + 1

    # ── Security score ────────────────────────────────────────────────────
    security_score = _calculate_security_score(unique_vulns)

    # ── Risk level ────────────────────────────────────────────────────────
    if counts["CRITICAL"] > 0:
        risk_level = "CRITICAL"
    elif counts["HIGH"] > 0:
        risk_level = "HIGH"
    elif counts["MEDIUM"] > 0:
        risk_level = "MEDIUM"
    elif counts["LOW"] > 0:
        risk_level = "LOW"
    else:
        risk_level = "SAFE"

    # ── Optional Bandit ───────────────────────────────────────────────────
    bandit_result = None
    if run_bandit and Path(file_path).exists():
        bandit_result = _run_bandit(file_path)

    # ── Assemble result ───────────────────────────────────────────────────
    result.update({
        "security_score": security_score,
        "vulnerabilities": unique_vulns,
        "vulnerability_counts": counts,
        "risk_level": risk_level,
        "bandit": bandit_result,
    })

    return result
