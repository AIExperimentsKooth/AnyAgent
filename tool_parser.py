"""
tool_parser — Parse tool/function calls from LLM text output.

Supports two calling conventions designed for small models:

  1. TOOL name(param=value, ...)      — compact, single-line
  2. FINISH: result text              — termination signal

Parameters support string literals (quoted), bare numbers, and booleans:
    TOOL read_file(path="/etc/hostname")
    TOOL shell(command="ls -la", timeout=30)

Returns TookCall namedtuples with tool_name + dict of parsed kwargs.
"""

import re
import shlex
from dataclasses import dataclass


@dataclass
class ToolCall:
    name: str
    kwargs: dict


@dataclass
class FinishSignal:
    """Agent is done — contains final message."""
    message: str


ParseResult = ToolCall | FinishSignal | None


def _parse_kv(text: str) -> dict:
    """Parse key=value pairs from a parameter string.

    Handles quoted strings, bare words, ints, floats, booleans, None.
    """
    kwargs = {}
    # Tokenize respecting quotes
    # Simple approach: split on commas not inside parens/quotes
    tokens = _tokenize_params(text)
    for token in tokens:
        token = token.strip()
        if not token or "=" not in token:
            continue
        key, _, raw = token.partition("=")
        key = key.strip()
        raw = raw.strip()
        kwargs[key] = _coerce_value(raw)
    return kwargs


def _tokenize_params(text: str) -> list[str]:
    """Split comma-separated params while respecting nested parens/quotes."""
    tokens = []
    depth = 0
    in_single = False
    in_double = False
    current: list[str] = []

    for ch in text:
        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
        elif ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
        elif ch == "(" and not (in_single or in_double):
            depth += 1
            current.append(ch)
        elif ch == ")" and not (in_single or in_double):
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0 and not (in_single or in_double):
            tokens.append("".join(current))
            current = []
        else:
            current.append(ch)

    remaining = "".join(current).strip()
    if remaining:
        tokens.append(remaining)
    return tokens


def _coerce_value(raw: str):
    """Turn a string into a Python value."""
    if not raw:
        return ""

    # Quoted strings
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        try:
            return shlex.split(raw)[0]
        except ValueError:
            return raw[1:-1]

    # Special literals
    if raw == "True":
        return True
    if raw == "False":
        return False
    if raw == "None":
        return None

    # Numbers
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass

    # Fallback: bare string
    return raw


def parse(text: str) -> list[ParseResult]:
    """Parse a response text for tool calls and finish signals.

    Returns a list because an LLM may emit multiple calls.
    """
    results: list[ParseResult] = []
    lines = text.strip().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # FINISH: <message>
        finish_match = re.match(r"^FINISH\s*:\s*(.*)", line)
        if finish_match:
            message = finish_match.group(1).strip()
            results.append(FinishSignal(message=message))
            i += 1
            continue

        # TOOL name(args, ...)
        tool_match = re.match(r"^TOOL\s+(\w[\w_]*)\s*\((.*)\)\s*$", line)
        if tool_match:
            name = tool_match.group(1)
            args_text = tool_match.group(2).strip()
            kwargs = _parse_kv(args_text) if args_text else {}
            results.append(ToolCall(name=name, kwargs=kwargs))
            i += 1
            continue

        # Multi-line: TOOL name on its own line, then key=val lines
        ml_match = re.match(r"^TOOL\s+(\w[\w_]*)\s*$", line)
        if ml_match:
            name = ml_match.group(1)
            param_lines = []
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line or next_line.startswith("TOOL ") or next_line.startswith("FINISH"):
                    break
                param_lines.append(next_line)
                i += 1
            kwargs = _parse_kv(", ".join(param_lines)) if param_lines else {}
            results.append(ToolCall(name=name, kwargs=kwargs))
            continue

        i += 1

    return results
