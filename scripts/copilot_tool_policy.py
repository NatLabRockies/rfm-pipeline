#!/usr/bin/env python3
"""Copilot CLI policy hook for repo-local agent workflow.

The hook is intentionally stdlib-only so it can run before Pixi is installed. It
enforces repo policy for shell commands. Project code/tests/builds must still use
repo-local Pixi.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentPolicy:
    """Small policy model parsed from config/agent_policy.yaml.

    The parser intentionally supports only simple `key: value` booleans/strings so this
    toolkit does not require PyYAML.
    """

    autonomy_tier: int = 1
    require_pixi: bool = True
    allow_git_execution: bool = False
    allow_destructive_git: bool = False
    allow_pr_execution: bool = False
    allow_branch_deletion: bool = False
    allow_dependency_changes: bool = False


BOOLEANS = {"true": True, "false": False, "yes": True, "no": False}

# Sensitive command matching intentionally scans the entire shell string rather than
# only the first executable. Agents sometimes wrap commands in `pixi run`, command
# substitution, `bash -c`, chaining, or subshells. These prefixes are broad by design:
# false positives are cheaper than allowing unsafe tool execution.
COMMAND_PREFIX = r"(?<![A-Za-z0-9_./-])"
EXECUTABLE_PATH = r"(?:\S*/)?"
GIT_PREFIX = (
    COMMAND_PREFIX
    + EXECUTABLE_PATH
    + r"git(?:\s+(?:(?:-C|-c|--git-dir|--work-tree|--namespace|--config-env)"
    + r"\s+\S+|--[^\s]+|-[A-Za-z]+))*\s+"
)
GH_PR_PREFIX = (
    COMMAND_PREFIX
    + EXECUTABLE_PATH
    + r"gh(?:\s+(?:(?:-R|--repo)\s+\S+|--[^\s]+|-[A-Za-z]+))*\s+pr\s+"
)
TOOL_PREFIX = COMMAND_PREFIX + EXECUTABLE_PATH
GIT_WRITE_SUBCOMMANDS = (
    r"(?:add|commit|push|merge|tag|switch|checkout|branch|rebase|reset|clean|stash|restore)"
)
PR_WRITE_SUBCOMMANDS = r"(?:create|merge|close|edit|reopen)"

GIT_WRITE_PATTERNS = (
    GIT_PREFIX + r"add\b",
    GIT_PREFIX + r"commit\b",
    GIT_PREFIX + r"push\b",
    GIT_PREFIX + r"merge\b",
    GIT_PREFIX + r"tag\b",
    GIT_PREFIX + r"switch\b",
    GIT_PREFIX + r"checkout\s+[^-]",
    GIT_PREFIX + r"checkout\s+-[bcB]\b",
    GH_PR_PREFIX + r"create\b",
    GH_PR_PREFIX + r"merge\b",
    GH_PR_PREFIX + r"close\b",
    GH_PR_PREFIX + r"edit\b",
    GH_PR_PREFIX + r"reopen\b",
    GIT_PREFIX + r".*-c\s+alias\.[^\s=]+=[\'\"]?!?(?:git\s+)?" + GIT_WRITE_SUBCOMMANDS + r"\b",
    r"[\'\"]git[\'\"]\s*,\s*[\'\"]" + GIT_WRITE_SUBCOMMANDS + r"[\'\"]",
    r"[\'\"]gh[\'\"]\s*,\s*[\'\"]pr[\'\"]\s*,\s*[\'\"]" + PR_WRITE_SUBCOMMANDS + r"[\'\"]",
    r"(?:^|[;&|]\s*)alias\s+\w+=[\'\"]?git\b",
    r"(?:^|[;&|]\s*)\w+=[\'\"]?git[\'\"]?\s*(?:;|&&|\|\|).*\$\w+\s+"
    + GIT_WRITE_SUBCOMMANDS
    + r"\b",
)

DESTRUCTIVE_PATTERNS = (
    GIT_PREFIX + r"reset\b",
    GIT_PREFIX + r"clean\b",
    GIT_PREFIX + r"rebase\b",
    GIT_PREFIX + r"stash\b",
    GIT_PREFIX + r"push\b.*\s--force",
    GIT_PREFIX + r"push\b.*\s--force-with-lease",
    GIT_PREFIX + r"restore\b",
    GIT_PREFIX + r"checkout\s+--\b",
    TOOL_PREFIX + r"rm\s+-rf\b",
)

BRANCH_DELETION_PATTERNS = (
    GIT_PREFIX + r"branch\s+-d\b",
    GIT_PREFIX + r"branch\s+-D\b",
    GIT_PREFIX + r"push\b.*\s--delete\b",
    GIT_PREFIX + r"push\s+origin\s+:.+",
)

DEPENDENCY_PATTERNS = (
    TOOL_PREFIX + r"pixi\s+add\b",
    TOOL_PREFIX + r"pixi\s+remove\b",
    TOOL_PREFIX + r"pixi\s+upgrade\b",
    TOOL_PREFIX + r"pip[0-9.]*\s+install\b",
    TOOL_PREFIX + r"pipx\s+install\b",
    r"[\'\"]pip[0-9.]*[\'\"]\s*,\s*[\'\"]install[\'\"]",
    TOOL_PREFIX + r"uv\s+add\b",
    TOOL_PREFIX + r"uv\s+pip\s+install\b",
    TOOL_PREFIX + r"poetry\s+add\b",
    TOOL_PREFIX + r"conda\s+install\b",
    TOOL_PREFIX + r"npm\s+(install|i|add|update)\b",
    TOOL_PREFIX + r"pnpm\s+(install|add|update)\b",
    TOOL_PREFIX + r"yarn\s+(install|add|upgrade)\b",
    TOOL_PREFIX + r"cargo\s+(add|install|update)\b",
    TOOL_PREFIX + r"brew\s+(install|upgrade)\b",
    TOOL_PREFIX + r"gem\s+(install|update)\b",
    TOOL_PREFIX + r"go\s+(get|install)\b",
)

DANGEROUS_EXEC_INTERPRETER = r"(?:bash|sh|zsh|python[0-9.]*|node|ruby|perl|source|\.)"

DANGEROUS_EXEC_PATTERNS = (
    rf"\b(curl|wget)\b.*\|\s*{DANGEROUS_EXEC_INTERPRETER}\b",
    rf"\b(wget)\b.*(?:-O\s*-|-O-|-qO\s*-|-qO-).*\|\s*{DANGEROUS_EXEC_INTERPRETER}\b",
    rf"\b{DANGEROUS_EXEC_INTERPRETER}\s*<\s*\(",
    rf"\b(curl|wget)\b.*(?:&&|;|\|\|)\s*{DANGEROUS_EXEC_INTERPRETER}\b",
    r"\b(curl|wget)\b.*(?:&&|;|\|\|).*\bchmod\s+(?:\+x|[0-7]{3,4})\b"
    r".*(?:&&|;|\|\|)\s*(?:\./|/tmp/|/var/tmp/)",
    r"\b(curl|wget)\b.*(?:&&|;|\|\|)\s*(?:\./|/tmp/|/var/tmp/)\S+",
    r"\b(curl|wget)\b.*(?:&&|;|\|\|)\s*\.\s+\S+",
)

GLOBAL_PIXI_PATTERNS = (
    TOOL_PREFIX + r"pixi\s+global\b",
    TOOL_PREFIX + r"pixi\s+shell\b",
    TOOL_PREFIX + r"pixi\s+exec\b",
)

PIXI_MANAGED_TOOLS = {
    "python",
    "python3",
    "pytest",
    "ruff",
    "mypy",
    "pre-commit",
    "pip",
    "pip3",
    "pipx",
    "uv",
    "poetry",
    "jupyter",
    "sphinx-build",
    "mkdocs",
}

PIXI_RUN_FLAGS_WITH_VALUE = {
    "--environment",
    "--feature",
    "--manifest-path",
    "--cwd",
    "-e",
    "-f",
    "-m",
    "-C",
}

PIXI_RUN_FLAGS_NO_VALUE = {
    "--clean-env",
    "--locked",
    "--frozen",
    "--skip-deps-install",
}


def parse_policy(path: str | Path) -> AgentPolicy:
    """Parse a minimal YAML-like policy file."""
    p = Path(path)
    values: dict[str, object] = {}
    if p.exists():
        for raw_line in p.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            low = value.lower()
            if low in BOOLEANS:
                values[key] = BOOLEANS[low]
            elif low.isdigit():
                values[key] = int(low)
            else:
                values[key] = value
    fields = AgentPolicy.__dataclass_fields__
    filtered = {key: value for key, value in values.items() if key in fields}
    return AgentPolicy(**filtered)


def matches_any(command: str, patterns: tuple[str, ...]) -> bool:
    """Return whether a command matches any regex pattern."""
    return any(re.search(pattern, command) for pattern in patterns)


def shell_words(command: str) -> list[str]:
    """Parse shell words or return an empty list when parsing fails."""
    try:
        return shlex.split(command)
    except ValueError:
        return []


def first_word(command: str) -> str:
    """Return a shell command's executable basename when it can be parsed."""
    parts = shell_words(command)
    if not parts:
        return ""
    return Path(parts[0]).name


def is_pixi_managed_tool(tool: str) -> bool:
    """Return whether an executable should be run through repo-local Pixi."""
    return tool in PIXI_MANAGED_TOOLS or bool(
        re.fullmatch(r"python[0-9.]*", tool) or re.fullmatch(r"pip[0-9.]*", tool)
    )


def command_starts_with(command: str, prefix: str) -> bool:
    """Return whether command starts with a command token/prefix."""
    stripped = command.strip()
    return stripped == prefix or stripped.startswith(prefix + " ")


def extract_pixi_run_inner_command(command: str) -> str | None:
    """Return the command executed by `pixi run`, if command starts with pixi run.

    This intentionally handles common `pixi run` flags so policy checks cannot be
    bypassed by wrapping a forbidden operation in `pixi run`.
    """
    parts = shell_words(command)
    if len(parts) < 3 or Path(parts[0]).name != "pixi" or parts[1] != "run":
        return None

    idx = 2
    while idx < len(parts):
        part = parts[idx]
        if part == "--":
            idx += 1
            break
        if part in PIXI_RUN_FLAGS_WITH_VALUE:
            idx += 2
            continue
        if any(part.startswith(flag + "=") for flag in PIXI_RUN_FLAGS_WITH_VALUE):
            idx += 1
            continue
        if part in PIXI_RUN_FLAGS_NO_VALUE:
            idx += 1
            continue
        break

    if idx >= len(parts):
        return ""
    return shlex.join(parts[idx:])


def evaluate_command(command: str, policy: AgentPolicy, pixi_repo: bool = True) -> tuple[bool, str]:
    """Evaluate whether a shell command should be allowed.

    Returns `(allowed, reason)`.
    """
    cmd = command.strip()
    if not cmd:
        return True, "empty command"

    if matches_any(cmd, DANGEROUS_EXEC_PATTERNS):
        return False, "Pipe-to-shell and process-substitution execution are blocked."

    if matches_any(cmd, GLOBAL_PIXI_PATTERNS):
        return False, "Use repo-local Pixi, not pixi global/shell/exec."

    pixi_inner = extract_pixi_run_inner_command(cmd)
    if pixi_inner is not None:
        if not pixi_inner:
            return True, "pixi run without inner command"
        allowed, reason = evaluate_command(pixi_inner, policy, pixi_repo=False)
        if not allowed:
            return False, f"Blocked inside pixi run: {reason}"

    if matches_any(cmd, DEPENDENCY_PATTERNS) and not policy.allow_dependency_changes:
        return False, "Dependency changes require explicit approval."

    if matches_any(cmd, BRANCH_DELETION_PATTERNS) and not policy.allow_branch_deletion:
        return False, "Branch deletion requires explicit approval or autonomous policy."

    if matches_any(cmd, DESTRUCTIVE_PATTERNS) and not policy.allow_destructive_git:
        return False, "Destructive commands require explicit approval or autonomous policy."

    if matches_any(cmd, GIT_WRITE_PATTERNS):
        is_pr_write = bool(re.search(GH_PR_PREFIX, cmd))
        if (is_pr_write and not policy.allow_pr_execution) or not policy.allow_git_execution:
            return False, "Git/PR write commands are disabled by this repo policy."

    if pixi_repo and policy.require_pixi:
        tool = first_word(cmd)
        direct_ok = (
            command_starts_with(cmd, "pixi run")
            or command_starts_with(cmd, "pixi install --locked")
            or command_starts_with(cmd, "./test_repo.sh")
            or command_starts_with(cmd, "bash test_repo.sh")
            or command_starts_with(cmd, "git ")
            or command_starts_with(cmd, "gh ")
            or command_starts_with(cmd, "pwd")
            or command_starts_with(cmd, "ls")
            or command_starts_with(cmd, "find ")
            or command_starts_with(cmd, "rg ")
            or command_starts_with(cmd, "grep ")
            or command_starts_with(cmd, "sed ")
            or command_starts_with(cmd, "cat ")
            or command_starts_with(cmd, "jq ")
            or command_starts_with(cmd, "chmod +x")
            or command_starts_with(cmd, "mkdir ")
        )
        if is_pixi_managed_tool(tool) and not direct_ok:
            return False, "This repo has pixi.toml. Use `pixi run ...` or `./test_repo.sh`."

    return True, "allowed"


def decision(kind: str, reason: str) -> None:
    """Emit a Copilot permission decision."""
    print(
        json.dumps(
            {"permissionDecision": kind, "permissionDecisionReason": reason},
            separators=(",", ":"),
        )
    )


def parse_tool_args(value: Any) -> dict[str, Any]:
    """Parse Copilot hook tool arguments."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"command": value}
        if isinstance(parsed, dict):
            return parsed
    return {}


def extract_command(payload: dict[str, Any]) -> str:
    """Extract a shell command from a Copilot hook payload."""
    tool_args = (
        payload.get("toolArgs")
        or payload.get("tool_args")
        or payload.get("toolInput")
        or payload.get("tool_input")
        or {}
    )
    args = parse_tool_args(tool_args)
    command = args.get("command") or args.get("cmd") or ""
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return str(command)


def is_shell_tool(payload: dict[str, Any]) -> bool:
    """Return whether a Copilot hook payload represents a shell tool."""
    tool_name = str(
        payload.get("toolName") or payload.get("tool_name") or payload.get("tool") or ""
    )
    return tool_name in {"bash", "shell"} or tool_name.endswith(".bash")


def main() -> int:
    """Run the hook."""
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0
    if not is_shell_tool(payload):
        return 0
    command = extract_command(payload)
    policy = parse_policy(Path("config/agent_policy.yaml"))
    allowed, reason = evaluate_command(command, policy, pixi_repo=Path("pixi.toml").exists())
    if not allowed:
        decision("deny", reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
