#!/usr/bin/env python3
"""Regenerate the machine-local paths in .mcp.json.

.mcp.json registers the project's MCP servers (including
"geno-thermal-targeting") with absolute paths to the interpreter and entry
point. Those paths are specific to the machine that wrote the file, and
.mcp.json is gitignored (not committed), so there is no shared template a
teammate or a new machine can copy from.

Run this script from anywhere (no arguments) after cloning the repo onto a
new machine, or whenever .mcp.json's command/args stop matching the local
filesystem layout:

    python scripts/regen_mcp_json.py

It rewrites only the "command" and the mcp_geno_thermal.py entry in "args"
for the "geno-thermal-targeting" server, computed from this machine's actual
project root. Everything else already present in .mcp.json (other servers
such as "brightdata", extra env vars, etc.) is left untouched. If .mcp.json
does not exist yet, a minimal file containing just the
"geno-thermal-targeting" server is created.

Stdlib only, no third-party dependencies. Safe to run multiple times
(idempotent).
"""

import json
import os
import sys

SERVER_NAME = "geno-thermal-targeting"
ENTRY_POINT_NAME = "mcp_geno_thermal.py"


def get_project_root():
    """Absolute path to the project root (one level up from scripts/)."""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(scripts_dir)


def get_venv_python(root):
    """Absolute path to the project's .venv-flash interpreter.

    Cross-platform aware: macOS/Linux venvs put the interpreter under
    bin/, Windows venvs put it under Scripts/.
    """
    if sys.platform.startswith("win"):
        return os.path.join(root, ".venv-flash", "Scripts", "python.exe")
    return os.path.join(root, ".venv-flash", "bin", "python")


def get_entry_point(root):
    """Absolute path to mcp_geno_thermal.py."""
    return os.path.join(root, ENTRY_POINT_NAME)


def load_existing_config(mcp_json_path):
    """Load the existing .mcp.json if present, else return None."""
    if not os.path.exists(mcp_json_path):
        return None
    with open(mcp_json_path, "r") as f:
        return json.load(f)


def build_minimal_config(venv_python, entry_point):
    """A full minimal .mcp.json with just the geno-thermal-targeting server."""
    return {
        "mcpServers": {
            SERVER_NAME: {
                "command": venv_python,
                "args": [entry_point],
                "env": {"GENOTHERMAL_LOG_LEVEL": "WARNING"},
            }
        }
    }


def update_args_entry_point(args, entry_point):
    """Return a copy of args with the mcp_geno_thermal.py path replaced.

    Looks for an existing element that already points at the entry point
    (by basename), and replaces just that element so any other CLI flags
    in args are preserved. Falls back to replacing the last element, or
    appending the entry point if args is empty.
    """
    new_args = list(args)
    for i, arg in enumerate(new_args):
        if os.path.basename(str(arg)) == ENTRY_POINT_NAME:
            new_args[i] = entry_point
            return new_args
    if new_args:
        new_args[-1] = entry_point
        return new_args
    return [entry_point]


def regenerate(mcp_json_path=None):
    """Compute machine-correct paths and (re)write .mcp.json.

    Returns a dict describing what changed, for printing/inspection.
    """
    root = get_project_root()
    if mcp_json_path is None:
        mcp_json_path = os.path.join(root, ".mcp.json")

    venv_python = get_venv_python(root)
    entry_point = get_entry_point(root)

    existing = load_existing_config(mcp_json_path)

    changes = []

    if existing is None:
        config = build_minimal_config(venv_python, entry_point)
        changes.append("created new .mcp.json with '%s' server" % SERVER_NAME)
    else:
        config = existing
        servers = config.setdefault("mcpServers", {})
        server = servers.get(SERVER_NAME)

        if server is None:
            servers[SERVER_NAME] = {
                "command": venv_python,
                "args": [entry_point],
                "env": {"GENOTHERMAL_LOG_LEVEL": "WARNING"},
            }
            changes.append("added missing '%s' server" % SERVER_NAME)
        else:
            old_command = server.get("command")
            if old_command != venv_python:
                server["command"] = venv_python
                changes.append(
                    "command: %r -> %r" % (old_command, venv_python)
                )

            old_args = server.get("args", [])
            new_args = update_args_entry_point(old_args, entry_point)
            if new_args != old_args:
                server["args"] = new_args
                changes.append("args: %r -> %r" % (old_args, new_args))

    with open(mcp_json_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    return {
        "mcp_json_path": mcp_json_path,
        "venv_python": venv_python,
        "entry_point": entry_point,
        "changes": changes,
    }


def main():
    result = regenerate()
    print("Project root resolved via:", get_project_root())
    print("Target file:", result["mcp_json_path"])
    print("Resolved interpreter:", result["venv_python"])
    print("Resolved entry point:", result["entry_point"])
    if result["changes"]:
        print("Changes made:")
        for change in result["changes"]:
            print("  -", change)
    else:
        print("No changes needed; .mcp.json already matches this machine.")


if __name__ == "__main__":
    main()
