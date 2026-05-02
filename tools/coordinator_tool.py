"""
coordinator_tool.py
-------------------
Coordinator Analysis Tool

This tool performs the heavy lifting for the Coordinator Agent:
  1. Validates and analyzes input code files
  2. Extracts file metadata and structure
  3. Detects programming language with confidence scoring
  4. Analyzes code complexity and patterns
  5. Determines which agents should process the code
  6. Assigns priority levels based on code characteristics

This is a tool-first approach, allowing the coordinator to make intelligent
routing decisions based on concrete analysis rather than LLM guessing.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class CodeMetrics:
    """Container for code metrics extracted from source file."""
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    function_count: int
    class_count: int
    import_count: int
    has_main: bool
    estimated_complexity: str  # "LOW", "MEDIUM", "HIGH"
    max_function_length: int
    avg_function_length: float


@dataclass
class FileAnalysis:
    """Complete file analysis results."""
    file_path: str
    file_size_bytes: int
    language: str
    confidence: float
    metrics: CodeMetrics
    code_purpose: str
    priority: str  # "HIGH", "MEDIUM", "LOW"
    routing_agents: List[str]
    analysis_notes: str
    complexity_assessment: str


# ─────────────────────────────────────────────────────────────────────────────
# Language Detection
# ─────────────────────────────────────────────────────────────────────────────

LANGUAGE_PATTERNS = {
    "python": {
        "extensions": [".py"],
        "signatures": [
            (r"^import\s+", "STRONG"),
            (r"^from\s+\w+\s+import\s+", "STRONG"),
            (r"^def\s+\w+\s*\(", "STRONG"),
            (r"^class\s+\w+", "STRONG"),
            (r"^\s*@\w+", "MEDIUM"),  # decorators
            (r"print\(", "MEDIUM"),
            (r"if\s+__name__\s*==\s*['\"]__main__['\"]", "STRONG"),
            (r":\s*$", "WEAK"),  # Python indentation style
        ],
    },
    "javascript": {
        "extensions": [".js"],
        "signatures": [
            (r"^const\s+\w+\s*=", "STRONG"),
            (r"^let\s+\w+\s*=", "STRONG"),
            (r"^var\s+\w+\s*=", "STRONG"),
            (r"^function\s+\w+\s*\(", "STRONG"),
            (r"=>\s*{", "STRONG"),  # arrow function
            (r"console\.log\(", "MEDIUM"),
            (r"require\(['\"]", "STRONG"),
            (r"import\s+.*from\s+['\"]", "STRONG"),
        ],
    },
    "java": {
        "extensions": [".java"],
        "signatures": [
            (r"^public\s+class\s+\w+", "STRONG"),
            (r"^private\s+\w+\s+\w+", "STRONG"),
            (r"^public\s+static\s+void\s+main", "STRONG"),
            (r"import\s+java\.", "STRONG"),
            (r"new\s+\w+\(", "MEDIUM"),
            (r"@Override", "MEDIUM"),
        ],
    },
    "cpp": {
        "extensions": [".cpp", ".cc", ".cxx", ".h", ".hpp"],
        "signatures": [
            (r"^#include\s+[<\"]", "STRONG"),
            (r"^using\s+namespace\s+", "STRONG"),
            (r"^class\s+\w+", "STRONG"),
            (r"std::", "STRONG"),
            (r"->", "MEDIUM"),
            (r"::", "MEDIUM"),
        ],
    },
    "csharp": {
        "extensions": [".cs"],
        "signatures": [
            (r"^using\s+", "STRONG"),
            (r"^namespace\s+", "STRONG"),
            (r"^public\s+class\s+", "STRONG"),
            (r"async\s+Task", "STRONG"),
            (r"=>", "MEDIUM"),
        ],
    },
}


def _detect_language(file_path: str, code_content: str) -> Tuple[str, float]:
    """
    Detect programming language from file extension and content analysis.

    Scoring system:
    - STRONG signature: +1.0 confidence per occurrence
    - MEDIUM signature: +0.5 confidence per occurrence
    - WEAK signature: +0.2 confidence per occurrence

    Args:
        file_path: Path to the code file
        code_content: Full content of the code file

    Returns:
        Tuple of (language_name, confidence_score_0_to_1)
    """
    ext = Path(file_path).suffix.lower()
    scores: Dict[str, float] = {}

    # First pass: check extension
    for lang, patterns in LANGUAGE_PATTERNS.items():
        if ext in patterns["extensions"]:
            scores[lang] = scores.get(lang, 0) + 2.0  # Strong extension match

    # Second pass: scan content for language signatures
    first_500_chars = code_content[:500]
    for line in first_500_chars.split("\n"):
        for lang, patterns in LANGUAGE_PATTERNS.items():
            for pattern, strength in patterns["signatures"]:
                if re.search(pattern, line, re.IGNORECASE):
                    weight = {"STRONG": 1.0, "MEDIUM": 0.5, "WEAK": 0.2}[strength]
                    scores[lang] = scores.get(lang, 0) + weight

    if not scores:
        return "unknown", 0.0

    best_lang = max(scores, key=scores.get)
    max_score = scores[best_lang]

    # Normalize confidence to 0-1 range
    confidence = min(max_score / 5.0, 1.0)

    return best_lang, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Code Metrics Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_code_metrics(code_content: str, language: str) -> CodeMetrics:
    """
    Extract structural metrics from code content.

    Metrics extracted:
    - Line counts (code, comments, blank)
    - Function/class counts
    - Import count
    - Presence of main entry point
    - Estimated complexity (based on nesting and control flow)
    - Function length statistics

    Args:
        code_content: Full source code
        language: Programming language detected

    Returns:
        CodeMetrics dataclass with extracted metrics
    """
    lines = code_content.split("\n")
    total_lines = len(lines)

    # Count different line types
    code_lines = 0
    comment_lines = 0
    blank_lines = 0
    function_count = 0
    class_count = 0
    import_count = 0
    has_main = False
    control_flow_count = 0  # for complexity
    function_lengths: List[int] = []
    current_function_lines = 0

    for line in lines:
        stripped = line.strip()

        # Blank line
        if not stripped:
            blank_lines += 1
            continue

        # Comment line
        if language == "python" and stripped.startswith("#"):
            comment_lines += 1
            continue
        elif language in ["java", "csharp", "cpp"] and (
            stripped.startswith("//") or stripped.startswith("/*")
        ):
            comment_lines += 1
            continue
        elif language == "javascript" and (
            stripped.startswith("//") or stripped.startswith("/*")
        ):
            comment_lines += 1
            continue

        code_lines += 1

        # Detect structures based on language
        if language == "python":
            if re.match(r"^def\s+\w+", stripped):
                function_count += 1
            elif re.match(r"^class\s+\w+", stripped):
                class_count += 1
            elif re.match(r"^import\s+|^from\s+\w+\s+import", stripped):
                import_count += 1
            elif stripped == "if __name__ == '__main__':":
                has_main = True

        elif language in ["java", "csharp"]:
            if re.search(r"\bvoid\s+\w+\s*\(|\bpublic\s+\w+\s+\w+\s*\(", stripped):
                function_count += 1
            elif re.match(r"^(public\s+)?class\s+\w+", stripped):
                class_count += 1
            elif re.match(r"^import\s+", stripped):
                import_count += 1
            elif re.search(r"public\s+static\s+void\s+main", stripped):
                has_main = True

        elif language == "javascript":
            if re.match(r"^function\s+\w+|^const\s+\w+\s*=\s*\(.*\)\s*=>", stripped):
                function_count += 1
            elif re.match(r"^class\s+\w+", stripped):
                class_count += 1
            elif re.match(r"^import\s+|^require\(", stripped):
                import_count += 1

        # Count control flow for complexity
        if re.search(r"\b(if|else|for|while|switch|case|catch)\b", stripped):
            control_flow_count += 1

    # Estimate complexity
    avg_function_length = code_lines / max(function_count, 1)
    max_function_length = int(avg_function_length * 2)  # rough estimate

    if code_lines > 500 or control_flow_count > 30:
        complexity = "HIGH"
    elif code_lines > 200 or control_flow_count > 10:
        complexity = "MEDIUM"
    else:
        complexity = "LOW"

    return CodeMetrics(
        total_lines=total_lines,
        code_lines=code_lines,
        comment_lines=comment_lines,
        blank_lines=blank_lines,
        function_count=function_count,
        class_count=class_count,
        import_count=import_count,
        has_main=has_main,
        estimated_complexity=complexity,
        max_function_length=max_function_length,
        avg_function_length=round(avg_function_length, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Code Purpose & Priority Assessment
# ─────────────────────────────────────────────────────────────────────────────

def _infer_code_purpose(code_content: str, language: str, metrics: CodeMetrics) -> str:
    """
    Infer the purpose/domain of the code from content and structure.

    Looks for patterns indicating:
    - Web/API code (Flask, Django, Express, etc.)
    - Data processing (pandas, numpy, data operations)
    - System/utility code
    - Game/graphics code
    - Testing code

    Args:
        code_content: Full source code
        language: Detected language
        metrics: Extracted code metrics

    Returns:
        Brief description of code purpose
    """
    first_1000_chars = code_content[:1000].lower()

    if language == "python":
        if any(
            lib in first_1000_chars
            for lib in ["flask", "django", "fastapi", "tornado", "bottle"]
        ):
            return "Web application/API"
        elif any(
            lib in first_1000_chars for lib in ["pandas", "numpy", "scipy", "sklearn"]
        ):
            return "Data processing/ML application"
        elif any(
            lib in first_1000_chars for lib in ["pygame", "turtle", "pyglet"]
        ):
            return "Game/Graphics application"
        elif "unittest" in first_1000_chars or "pytest" in first_1000_chars:
            return "Test suite"
        elif "asyncio" in first_1000_chars:
            return "Asynchronous/concurrent application"
        else:
            return "Utility/General script"

    elif language == "javascript":
        if any(
            lib in first_1000_chars
            for lib in ["express", "react", "vue", "angular", "next"]
        ):
            return "Web/Frontend application"
        elif "node" in first_1000_chars or "require" in first_1000_chars:
            return "Node.js backend application"
        else:
            return "JavaScript application"

    elif language == "java":
        if any(lib in first_1000_chars for lib in ["spring", "hibernate", "junit"]):
            return "Enterprise Java application"
        else:
            return "Java application"

    return "General application"


def _assess_priority(metrics: CodeMetrics, language: str, code_purpose: str) -> str:
    """
    Assess code priority based on size, complexity, and purpose.

    HIGH priority:
    - Large codebases (>500 lines)
    - HIGH estimated complexity
    - Contains main entry point
    - Web/enterprise code

    MEDIUM priority:
    - Medium size (200-500 lines)
    - MEDIUM complexity
    - Standard applications

    LOW priority:
    - Small utility scripts (<200 lines)
    - LOW complexity
    - Test files

    Args:
        metrics: Code metrics
        language: Detected language
        code_purpose: Inferred purpose

    Returns:
        Priority level: "HIGH", "MEDIUM", or "LOW"
    """
    score = 0

    # Size factor
    if metrics.total_lines > 500:
        score += 2
    elif metrics.total_lines > 200:
        score += 1

    # Complexity factor
    if metrics.estimated_complexity == "HIGH":
        score += 2
    elif metrics.estimated_complexity == "MEDIUM":
        score += 1

    # Structure factor
    if metrics.has_main:
        score += 1
    if metrics.class_count > 3:
        score += 1
    elif metrics.class_count > 0 and metrics.function_count > 3:
        score += 1
    if metrics.function_count > 10:
        score += 2
    elif metrics.function_count > 5:
        score += 1
    if metrics.import_count > 3:
        score += 1

    # Purpose factor
    if any(p in code_purpose for p in ["Enterprise", "Web", "ML", "Async"]):
        score += 2

    if score >= 4:
        return "HIGH"
    elif score >= 2:
        return "MEDIUM"
    elif score == 0:
        return "NONE"
    else:
        return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Main Tool Function
# ─────────────────────────────────────────────────────────────────────────────


SECURITY_RISK_PATTERNS = [
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"subprocess\.",
    r"os\.system\s*\(",
    r"pickle\.loads\s*\(",
    r"yaml\.load\s*\(",
    r"hashlib\.md5\s*\(",
    r"password\s*=",
    r"secret\s*=",
    r"api[_-]?key\s*=",
    r"token\s*=",
    r"private[_-]?key\s*=",
]

HIGH_SECURITY_PATTERNS = {
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"subprocess\.",
    r"os\.system\s*\(",
    r"pickle\.loads\s*\(",
    r"yaml\.load\s*\(",
    r"password\s*=",
    r"secret\s*=",
    r"api[_-]?key\s*=",
    r"token\s*=",
    r"private[_-]?key\s*=",
}

MEDIUM_SECURITY_PATTERNS = {
    r"http://",
    r"except:",
    r"from\s+math\s+import\s+\*",
    r"hashlib\.md5\s*\(",
}


def _detect_security_hotspots(code_content: str) -> List[str]:
    """
    Detect security-sensitive patterns that should influence routing.

    Args:
        code_content: Raw source code content.

    Returns:
        A list of matched hotspot labels.
    """
    matched_hotspots = []

    for pattern in SECURITY_RISK_PATTERNS:
        if re.search(pattern, code_content, re.IGNORECASE | re.MULTILINE):
            matched_hotspots.append(pattern)

    return matched_hotspots


def _assess_security_priority(code_content: str) -> str:
    """
    Assess the security priority separately from code-quality priority.

    Args:
        code_content: Raw source code content.

    Returns:
        Security priority as LOW, MEDIUM, or HIGH.
    """
    matched_high = sum(
        1 for pattern in HIGH_SECURITY_PATTERNS if re.search(pattern, code_content, re.IGNORECASE | re.MULTILINE)
    )
    matched_medium = sum(
        1 for pattern in MEDIUM_SECURITY_PATTERNS if re.search(pattern, code_content, re.IGNORECASE | re.MULTILINE)
    )

    if matched_high >= 2 or (matched_high >= 1 and matched_medium >= 1):
        return "HIGH"
    if matched_high >= 1 or matched_medium >= 1:
        return "MEDIUM"
    return "NONE"


def _determine_routing_agents(
    metrics: CodeMetrics,
    language: str,
    code_purpose: str,
    code_content: str,
    code_quality_priority: str,
    security_priority: str,
) -> List[str]:
    """
    Decide which agents should process the file.

    Routing rules:
    - Simple code with no security hotspots -> code analyzer only
    - Simple code with security hotspots -> security scanner only
    - Medium/high complexity or web/enterprise style code -> both agents

    Args:
        metrics: Extracted code metrics.
        language: Detected programming language.
        code_purpose: Inferred purpose of the file.
        code_content: Raw source code content.

    Returns:
        Ordered list of agent names.
    """
    if code_quality_priority == "NONE" and security_priority == "NONE":
        return []

    if code_quality_priority == "NONE" and security_priority in {"MEDIUM", "HIGH"}:
        return ["security_scanner"]

    if code_quality_priority in {"LOW", "MEDIUM"} and security_priority == "NONE":
        return ["code_analyzer"]

    return ["code_analyzer", "security_scanner"]


def _build_routing_rationale(
    metrics: CodeMetrics,
    code_purpose: str,
    routing_agents: List[str],
    code_content: str,
    code_quality_priority: str,
    security_priority: str,
) -> str:
    """
    Explain why the coordinator selected a given routing path.

    Args:
        metrics: Extracted code metrics.
        code_purpose: Inferred purpose of the file.
        routing_agents: Final selected agents.
        code_content: Raw source code content.

    Returns:
        Short human-readable rationale for the routing decision.
    """
    route_set = set(routing_agents)

    def _user_friendly_priority(p: str) -> str:
        return "No issues found" if p == "NONE" else p

    display_cq = _user_friendly_priority(code_quality_priority)
    display_sec = _user_friendly_priority(security_priority)

    if not route_set:
        return (
            "No issues found in the file: both code quality and security priorities "
            f"are {display_cq}, so the coordinator can finish without downstream agents."
        )

    if route_set == {"security_scanner"}:
        return (
            f"Security-only routing: code quality priority is {display_cq} "
            f"and security priority is {display_sec}, so only the security "
            "scanner is needed."
        )

    if route_set == {"code_analyzer"}:
        return (
            f"Code-quality-only routing: code quality priority is {display_cq} "
            f"and security priority is {display_sec}, so only the code analyzer "
            "is needed."
        )

    return (
        f"Dual routing: code quality priority is {display_cq} and security "
        f"priority is {display_sec}, so both analyzers should review the file."
    )

def analyze_file_for_routing(
    file_path: str,
    code_content: str = None,
) -> Dict[str, Any]:
    """
    Comprehensive file analysis tool for the Coordinator Agent.

    This tool performs all analysis needed for intelligent routing:
    1. Validates the file
    2. Detects language with confidence scoring
    3. Extracts code metrics
    4. Infers code purpose
    5. Assesses priority
    6. Determines which agents to route to

    Args:
        file_path: Path to the code file
        code_content: Optional pre-read code content. If None, file is read here.

    Returns:
        Dict containing:
            - language (str): Detected language
            - confidence (float): Language detection confidence (0-1)
            - metrics (dict): Code metrics
            - code_purpose (str): Inferred purpose
            - priority (str): Priority level (HIGH/MEDIUM/LOW)
            - routing_agents (list): ["code_analyzer", "security_scanner", ...]
            - complexity_assessment (str): Human-readable complexity description
            - analysis_notes (str): Additional insights

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file cannot be read or is binary
    """
    # ── Step 1: Validate file ─────────────────────────────────────────────
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)

    # Read file if not provided
    if code_content is None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code_content = f.read()
        except UnicodeDecodeError:
            raise ValueError(
                f"File {file_path} appears to be binary. Only text files are supported."
            )

    # ── Step 2: Detect language ───────────────────────────────────────────
    language, confidence = _detect_language(file_path, code_content)

    # ── Step 3: Extract metrics ───────────────────────────────────────────
    metrics = _extract_code_metrics(code_content, language)

    # ── Step 4: Infer purpose ─────────────────────────────────────────────
    code_purpose = _infer_code_purpose(code_content, language, metrics)

    # ── Step 5: Assess priority ───────────────────────────────────────────
    code_quality_priority = _assess_priority(metrics, language, code_purpose)
    security_priority = _assess_security_priority(code_content)
    priority = max(
        [code_quality_priority, security_priority],
        key=lambda level: {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(level, 0),
    )

    # ── Step 6: Determine routing ─────────────────────────────────────────
    routing_agents = _determine_routing_agents(
        metrics,
        language,
        code_purpose,
        code_content,
        code_quality_priority,
        security_priority,
    )
    routing_rationale = _build_routing_rationale(
        metrics=metrics,
        code_purpose=code_purpose,
        routing_agents=routing_agents,
        code_content=code_content,
        code_quality_priority=code_quality_priority,
        security_priority=security_priority,
    )

    # ── Step 7: Build complexity assessment ───────────────────────────────
    complexity_parts = [
        f"Estimated Complexity: {metrics.estimated_complexity}",
        f"Functions: {metrics.function_count}, Classes: {metrics.class_count}",
        f"Average Function Length: {metrics.avg_function_length} lines",
    ]

    if metrics.has_main:
        complexity_parts.append("Entry point detected (runnable)")

    if metrics.estimated_complexity == "HIGH":
        complexity_parts.append(
            "⚠️ High complexity detected - careful review recommended"
        )

    complexity_assessment = " | ".join(complexity_parts)

    # ── Step 8: Build analysis notes ──────────────────────────────────────
    notes = []

    if confidence < 0.6:
        notes.append(f"⚠️ Language detection uncertain ({confidence:.1%})")

    if metrics.code_lines > 1000:
        notes.append("📊 Large codebase - may require extended analysis time")

    if metrics.total_lines > 0:
        comment_ratio = metrics.comment_lines / metrics.total_lines
        if comment_ratio < 0.05:
            notes.append("📝 Low comment ratio - documentation could be improved")

    if metrics.import_count == 0 and language == "python":
        notes.append("🔧 No imports detected - possible script fragment")

    analysis_notes = " | ".join(notes) if notes else "No special notes"

    # ── Return comprehensive analysis ─────────────────────────────────────
    return {
        "file_path": file_path,
        "file_size_bytes": file_size,
        "language": language,
        "confidence": round(confidence, 2),
        "metrics": asdict(metrics),
        "code_purpose": code_purpose,
        "code_quality_priority": code_quality_priority,
        "security_priority": security_priority,
        "priority": priority,
        "routing_agents": routing_agents,
        "routing_rationale": routing_rationale,
        "complexity_assessment": complexity_assessment,
        "analysis_notes": analysis_notes,
    }
