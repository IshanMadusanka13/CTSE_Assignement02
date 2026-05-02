"""
Small security-sensitive sample used to demonstrate security-only routing.

The code is intentionally simple, but it contains risky patterns that should
send it to the security scanner without needing the code quality analyzer.
"""

import os

API_TOKEN = "sk-demo-token-12345"


def execute_command(command_text):
    """Run a command string for demonstration purposes."""
    return os.system(command_text)


def unsafe_parse(user_input):
    return eval(user_input)