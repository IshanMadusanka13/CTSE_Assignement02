"""
observability.py
----------------
LLMOps / AgentOps logging and tracing for the Code Review MAS.

Every agent call, tool invocation, and state transition is recorded to:
  - Console  (rich-formatted, colour-coded)
  - logs/agent_trace.jsonl  (structured JSON Lines file for analysis)

Usage:
    from observability import logger
    logger.log_agent_start("code_analyzer", state)
    logger.log_tool_call("analyze_code_quality", inputs, output)
    logger.log_agent_end("code_analyzer", state)
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ── Setup ─────────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
TRACE_FILE = LOG_DIR / "agent_trace.jsonl"

console = Console()

# Standard Python logger (also writes to logs/system.log)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "system.log"),
        logging.StreamHandler(),
    ],
)
std_logger = logging.getLogger("code_review_mas")


# ── Tracer class ──────────────────────────────────────────────────────────────
class AgentTracer:
    """
    Structured tracer that records every agent/tool event to JSONL.

    Each log entry is a self-contained JSON object with:
      - event_type  : "agent_start" | "agent_end" | "tool_call" | "state_update" | "error"
      - agent_name  : which agent fired the event
      - timestamp   : ISO-8601 UTC
      - session_id  : groups all events for one run
      - payload     : event-specific data
    """

    def __init__(self) -> None:
        self.session_id: str = str(uuid.uuid4())[:8]

    # ── Internal writer ───────────────────────────────────────────────────
    def _write(self, event_type: str, agent_name: str, payload: Dict[str, Any]) -> None:
        """Append a single event record to the JSONL trace file."""
        record = {
            "event_type": event_type,
            "agent_name": agent_name,
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "payload": payload,
        }
        with TRACE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ── Public API ────────────────────────────────────────────────────────
    def log_agent_start(self, agent_name: str, input_summary: str) -> None:
        """Log when an agent begins processing."""
        self._write("agent_start", agent_name, {"input_summary": input_summary})
        console.print(
            Panel(
                f"[bold cyan]▶ {agent_name.upper()} STARTED[/bold cyan]\n{input_summary}",
                border_style="cyan",
            )
        )

    def log_agent_end(self, agent_name: str, output_summary: str, score: Optional[float] = None) -> None:
        """Log when an agent finishes processing."""
        payload: Dict[str, Any] = {"output_summary": output_summary}
        if score is not None:
            payload["score"] = score
        self._write("agent_end", agent_name, payload)
        console.print(
            Panel(
                f"[bold green]✔ {agent_name.upper()} COMPLETED[/bold green]\n{output_summary}",
                border_style="green",
            )
        )

    def log_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        inputs: Dict[str, Any],
        output: Any,
    ) -> None:
        """Log a tool invocation with its inputs and outputs."""
        self._write(
            "tool_call",
            agent_name,
            {"tool_name": tool_name, "inputs": inputs, "output_preview": str(output)[:500]},
        )
        console.print(
            f"  [yellow]🔧 TOOL:[/yellow] [bold]{tool_name}[/bold] called by [italic]{agent_name}[/italic]"
        )

    def log_state_update(self, agent_name: str, updated_keys: list) -> None:
        """Log which state keys were modified by an agent."""
        self._write("state_update", agent_name, {"updated_keys": updated_keys})
        console.print(f"  [magenta]📦 STATE UPDATED:[/magenta] {updated_keys}")

    def log_error(self, agent_name: str, error: str) -> None:
        """Log an error that occurred during agent execution."""
        self._write("error", agent_name, {"error": error})
        console.print(f"  [bold red]✖ ERROR in {agent_name}:[/bold red] {error}")
        std_logger.error("Agent: %s | Error: %s", agent_name, error)


# ── Singleton instance used throughout the project ────────────────────────────
logger = AgentTracer()
