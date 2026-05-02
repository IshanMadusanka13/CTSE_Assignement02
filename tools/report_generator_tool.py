from datetime import datetime
from typing import Dict, List, Any
import os


# ─────────────────────────────────────────────────────────────────────────────
# Final Score Calculation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_final_score(
    code_quality_score: float,
    security_score: float,
    code_weight: float = 0.5,
    security_weight: float = 0.5
) -> float:
    """
    Calculate weighted final score using:
      - Code quality score from analyze_code_quality()
      - Security score from check_security()

    Args:
        code_quality_score: Score from code analyzer (0-100)
        security_score:     Score from security scanner (0-100)
        code_weight:        Weight for code quality score
        security_weight:    Weight for security score

    Returns:
        Final weighted score rounded to 2 decimals.
    """

    final_score = (
        (code_quality_score * code_weight) +
        (security_score * security_weight)
    )

    return round(final_score, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Issue Prioritization
# ─────────────────────────────────────────────────────────────────────────────

#def prioritize_issues(
   # code_issues: List[Dict[str, Any]],
   # security_issues: List[Dict[str, Any]],
   # limit: int = 10
#) -> List[Dict[str, Any]]:
    
def prioritize_issues(
    code_issues: List[Dict],
    security_issues: List[Dict],
    limit: int = 10
) -> List[Dict]:

    severity_map = {
        "CRITICAL": 5,
        "HIGH": 4,
        "MEDIUM": 3,
        "LOW": 2
    }

    combined = []

    # ✅ Preserve ALL fields for code issues
    for issue in code_issues:
        combined.append({
            "category": "Code Quality",
            "type": issue.get("type"),
            "severity": issue.get("severity"),
            "line": issue.get("line"),
            "message": issue.get("message"),
        })

    # ✅ Preserve ALL fields for security issues
    for vuln in security_issues:
        combined.append({
            "category": "Security",
            "type": vuln.get("type"),
            "severity": vuln.get("severity"),
            "line": vuln.get("line"),
            "message": vuln.get("message"),
            "cwe": vuln.get("cwe"),
            "fix": vuln.get("fix"),
        })

    # ✅ Sort correctly (severity DESC, then line ASC)
    combined.sort(
        key=lambda x: (
            severity_map.get(x.get("severity", "LOW").upper(), 1),
            -x.get("line", 0)
        ),
        reverse=True
    )

    return combined[:limit]

# ─────────────────────────────────────────────────────────────────────────────
# Severity Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def generate_severity_heatmap(all_issues: List[Dict[str, Any]]) -> str:
    """
    Generate text-based issue density heatmap by line range.

    Args:
        all_issues: Combined issues/vulnerabilities list

    Returns:
        Multi-line heatmap string.
    """

    ranges = {
        "1-50": 0,
        "51-100": 0,
        "101-150": 0,
        "151-200": 0,
        "201+": 0,
    }

    for issue in all_issues:

        line = issue.get("line", 1)

        if line <= 50:
            ranges["1-50"] += 1

        elif line <= 100:
            ranges["51-100"] += 1

        elif line <= 150:
            ranges["101-150"] += 1

        elif line <= 200:
            ranges["151-200"] += 1

        else:
            ranges["201+"] += 1

    heatmap = ""

    for key, value in ranges.items():
        heatmap += f"{key:<10} : {'█' * value} ({value} issues)\n"

    return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Markdown Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_markdown_report(data: Dict[str, Any]) -> str:
    """
    Generate a professional markdown report from:
      - analyze_code_quality()
      - check_security()

    Saves report into /reports directory.

    Args:
        data: Combined report dictionary

    Returns:
        Path to generated markdown report.
    """

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_path = os.path.join(
        reports_dir,
        f"report_{timestamp}.md"
    )

    markdown = f"""
# AI Code Review & Security Report

## File Information

| Field | Value |
|---|---|
| File Name | {data.get("file_name")} |
| Language | {data.get("language")} |
| Generated At | {datetime.now()} |

---

# Executive Summary

{data.get("summary")}

---

# Overall Scores

| Metric | Score |
|---|---|
| Code Quality Score | {data.get("code_quality_score")} / 100 |
| Security Score | {data.get("security_score")} / 100 |
| Final Weighted Score | {data.get("final_score")} / 100 |

---

# Code Metrics

| Metric | Value |
|---|---|
| Total Lines | {data.get("metrics", {}).get("total_lines")} |
| Lines of Code | {data.get("metrics", {}).get("lines_of_code")} |
| Functions | {data.get("metrics", {}).get("num_functions")} |
| Classes | {data.get("metrics", {}).get("num_classes")} |
| Imports | {data.get("metrics", {}).get("num_imports")} |
| Docstring Coverage | {data.get("metrics", {}).get("docstring_coverage_pct")}% |

---

# Cyclomatic Complexity

| Metric | Value |
|---|---|
| Average Complexity | {data.get("cyclomatic_complexity", {}).get("average")} |
| Maximum Complexity | {data.get("cyclomatic_complexity", {}).get("max")} |

---

# Issue Statistics

## Code Quality Issues

| Severity | Count |
|---|---|
| HIGH | {data.get("code_issue_counts", {}).get("HIGH", 0)} |
| MEDIUM | {data.get("code_issue_counts", {}).get("MEDIUM", 0)} |
| LOW | {data.get("code_issue_counts", {}).get("LOW", 0)} |
| TOTAL | {data.get("code_issue_counts", {}).get("TOTAL", 0)} |

## Security Vulnerabilities

| Severity | Count |
|---|---|
| CRITICAL | {data.get("security_counts", {}).get("CRITICAL", 0)} |
| HIGH | {data.get("security_counts", {}).get("HIGH", 0)} |
| MEDIUM | {data.get("security_counts", {}).get("MEDIUM", 0)} |
| LOW | {data.get("security_counts", {}).get("LOW", 0)} |
| TOTAL | {data.get("security_counts", {}).get("TOTAL", 0)} |

---

# Top Prioritized Issues
"""

    # Top Issues Section
    for idx, issue in enumerate(data.get("top_issues", []), start=1):

        markdown += f"""

## {idx}. {issue.get("message")}

| Field | Value |
|---|---|
| Category | {issue.get("category")} |
| Type | {issue.get("type")} |
| Severity | {issue.get("severity")} |
| Line | {issue.get("line")} |
"""

        if issue.get("category") == "Security":
            markdown += f"""| CWE | {issue.get("cwe", "N/A")} |
| Recommended Fix | {issue.get("fix", "N/A")} |
"""

    markdown += f"""

---

# Severity Heatmap

```text
{data.get("heatmap")}
```
"""

    with open(file_path, 'w', encoding='utf-8') as report_file:
        report_file.write(markdown)

    return file_path

