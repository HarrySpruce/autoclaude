#!/usr/bin/env python3
"""
Autonomous Task Executor for Claude Code
=========================================
Accepts a high-level task with a list of objectives, then drives Claude Code
in a loop until every objective is complete.

Features
--------
* Persistent state (executor_state.json) – survives crashes / manual restarts.
* Context-window exhaustion detection – starts a fresh Claude Code session
  (with summarised history in the prompt) so work continues uninterrupted.
* Rate-limit detection – reads the reset timestamp from the RateLimitEvent
  and sleeps until the limit clears, then resumes the same session.
* Planner LLM – a separate Anthropic API call produces each focused prompt
  and evaluates which objectives were completed after each session.

Usage
-----
  # Inline
  python autonomous_executor.py \\
    --task "Build a REST API for a todo app" \\
    --objectives "Scaffold Flask project" "Add GET /todos" "Add POST /todos" "Write tests"

  # JSON task file  ({"task": "...", "objectives": [...]})
  python autonomous_executor.py --task-file task.json

  # Resume from a previous run's saved state
  python autonomous_executor.py --resume

  # Interactive (prompts for task / objectives)
  python autonomous_executor.py
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import anyio
from claude_agent_sdk import (
    CLIConnectionError,
    CLINotFoundError,
    ClaudeAgentOptions,
    RateLimitEvent,
    ResultMessage,
    SystemMessage,
    query,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STATE_FILE = Path("executor_state.json")
DEFAULT_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]

# Stop reasons that indicate the context / turn budget has been exhausted.
# A fresh session (with summarised history) should be started on the next turn.
CONTEXT_EXHAUSTED_REASONS = {"max_turns", "max_tokens"}

# Error substrings that suggest the context window was exceeded.
CONTEXT_LIMIT_KEYWORDS = (
    "context_length_exceeded",
    "context window",
    "too many tokens",
    "token limit",
    "context limit",
    "input too long",
)


# ─────────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Planner (uses Anthropic API directly)
# ─────────────────────────────────────────────────────────────────────────────

async def plan_next_prompt(
    client: anthropic.AsyncAnthropic,
    task: str,
    objectives: list[str],
    completed: list[str],
    history: list[str],
) -> str:
    """Generate a focused, actionable prompt for the next Claude Code session."""
    remaining = [o for o in objectives if o not in completed]
    history_block = (
        "\n".join(f"  • {h}" for h in history[-10:])
        if history
        else "  (none yet)"
    )

    response = await client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": (
                    "You are orchestrating an autonomous coding assistant.\n\n"
                    f"Overall task:\n  {task}\n\n"
                    "Remaining objectives (in order):\n"
                    + "\n".join(f"  {i+1}. {o}" for i, o in enumerate(remaining))
                    + f"\n\nPrevious sessions:\n{history_block}\n\n"
                    "Write a concise, actionable prompt for the coding assistant to work on "
                    "the next 1–2 remaining objectives. Be specific. "
                    "At the end of the prompt, ask the assistant to summarise what it accomplished.\n\n"
                    "Reply with ONLY the prompt text, no preamble or explanation."
                ),
            }
        ],
    )

    for block in response.content:
        if block.type == "text":
            return block.text.strip()

    return (
        f"Work on the following for the task '{task}': "
        + (remaining[0] if remaining else "review overall progress")
        + ". Summarise what you accomplished at the end."
    )


async def evaluate_progress(
    client: anthropic.AsyncAnthropic,
    task: str,
    objectives: list[str],
    completed: list[str],
    session_output: str,
) -> list[str]:
    """
    Given the output of the latest session, return the updated list of
    completed objectives (superset of the previously completed ones).
    """
    snippet = session_output[-3000:] if len(session_output) > 3000 else session_output

    response = await client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Task: {task}\n\n"
                    "All objectives:\n"
                    + "\n".join(f"  - {o}" for o in objectives)
                    + "\n\nAlready confirmed complete:\n"
                    + (
                        "\n".join(f"  - {o}" for o in completed)
                        if completed
                        else "  (none)"
                    )
                    + f"\n\nLatest assistant output:\n{snippet}\n\n"
                    "Which objectives from 'All objectives' are NOW complete? "
                    "Include previously completed ones. "
                    "Only mark something complete if there is clear evidence it was done. "
                    "Reply with ONLY a JSON array of the exact objective strings."
                ),
            }
        ],
    )

    for block in response.content:
        if block.type == "text":
            m = re.search(r"\[.*?\]", block.text, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group())
                    # Only return items that are known objectives.
                    return [o for o in result if o in objectives]
                except json.JSONDecodeError:
                    pass

    return completed  # Safe fallback: keep what was already done.


# ─────────────────────────────────────────────────────────────────────────────
# Claude Code session runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_claude_session(
    prompt: str,
    cwd: str,
    resume_id: Optional[str],
    tools: list[str],
) -> tuple[str, Optional[str], bool, Optional[str]]:
    """
    Run one Claude Code session and collect results.

    Returns
    -------
    output            : Combined text produced during the session.
    session_id        : Session ID (usable for resumption on the next call).
    context_exhausted : True when the session hit a context / turn-budget limit.
    rate_limit_reset  : ISO-8601 timestamp when the rate limit clears, or None.
    """
    output_parts: list[str] = []
    session_id: Optional[str] = resume_id
    context_exhausted = False
    rate_limit_reset: Optional[str] = None

    # Build options.  When resuming we pass only `resume`; the existing session
    # already carries cwd, tool list, etc.  For a fresh start we supply all opts.
    if resume_id:
        opts = ClaudeAgentOptions(resume=resume_id)
    else:
        opts = ClaudeAgentOptions(
            cwd=cwd,
            allowed_tools=tools,
            permission_mode="acceptEdits",
            max_turns=80,
        )

    try:
        async for msg in query(prompt=prompt, options=opts):
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                new_id = msg.data.get("session_id")
                if new_id:
                    session_id = new_id

            elif isinstance(msg, ResultMessage):
                if msg.result:
                    output_parts.append(msg.result)
                if msg.stop_reason in CONTEXT_EXHAUSTED_REASONS:
                    print(f"  [!] Session ended: stop_reason={msg.stop_reason!r} "
                          "(context / turn limit reached)")
                    context_exhausted = True

            elif isinstance(msg, RateLimitEvent):
                info = msg.rate_limit_info
                print(f"  [!] Rate limit event: status={info.status!r}")
                if info.status == "rejected" and info.resets_at:
                    rate_limit_reset = (
                        info.resets_at.isoformat()
                        if hasattr(info.resets_at, "isoformat")
                        else str(info.resets_at)
                    )

    except (CLINotFoundError, CLIConnectionError) as exc:
        print(f"  [!] CLI error: {exc}")
        output_parts.append(f"[CLI error: {exc}]")

    except Exception as exc:  # noqa: BLE001
        err_lower = str(exc).lower()
        print(f"  [!] Session error: {exc}")
        output_parts.append(f"[Error: {exc}]")
        if any(kw in err_lower for kw in CONTEXT_LIMIT_KEYWORDS):
            print("  [!] Context-limit keyword detected in error.")
            context_exhausted = True

    return "\n".join(output_parts), session_id, context_exhausted, rate_limit_reset


# ─────────────────────────────────────────────────────────────────────────────
# Main execution loop
# ─────────────────────────────────────────────────────────────────────────────

async def run(
    task: str,
    objectives: list[str],
    cwd: str = ".",
    tools: Optional[list[str]] = None,
) -> None:
    """Drive Claude Code in a loop until all objectives are met."""
    tools = tools or DEFAULT_TOOLS
    planner = anthropic.AsyncAnthropic()

    # ── Load / initialise state ──────────────────────────────────────────────
    state = load_state()

    if state.get("task") != task:
        print(f"\nStarting new task: {task}")
        state = {
            "task": task,
            "objectives": objectives,
            "completed_objectives": [],
            "history": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    else:
        done = len(state.get("completed_objectives", []))
        print(f"\nResuming task ({done}/{len(objectives)} objectives done): {task}")

    save_state(state)

    # ── Main loop ────────────────────────────────────────────────────────────
    while True:
        completed: list[str] = state.get("completed_objectives", [])
        remaining = [o for o in objectives if o not in completed]

        # ── Check for completion ─────────────────────────────────────────────
        if not remaining:
            print("\n" + "═" * 60)
            print("  ✓ All objectives completed!")
            print("═" * 60)
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
            state.pop("session_id", None)
            save_state(state)
            break

        # ── Honour any pending rate-limit wait ───────────────────────────────
        rl_reset = state.pop("rate_limit_resets_at", None)
        if rl_reset:
            try:
                reset_dt = datetime.fromisoformat(rl_reset)
                if reset_dt.tzinfo is None:
                    reset_dt = reset_dt.replace(tzinfo=timezone.utc)
                wait_secs = (reset_dt - datetime.now(timezone.utc)).total_seconds() + 5
                if wait_secs > 0:
                    print(f"\n⏳  Rate limited — sleeping {wait_secs:.0f}s (until {rl_reset})")
                    await asyncio.sleep(wait_secs)
            except (ValueError, TypeError):
                pass
            save_state(state)

        # ── Session header ───────────────────────────────────────────────────
        session_num = len(state.get("history", [])) + 1
        print(f"\n{'─' * 60}")
        print(f"  Session #{session_num}  |  {len(completed)}/{len(objectives)} objectives done")
        print(f"  Remaining: {remaining}")
        print(f"{'─' * 60}")

        # ── Plan the next prompt ─────────────────────────────────────────────
        prompt = await plan_next_prompt(
            planner, task, objectives, completed, state.get("history", [])
        )
        print(f"\n[Prompt → Claude Code]\n{prompt}\n")

        # ── Run Claude Code ──────────────────────────────────────────────────
        resume_id: Optional[str] = state.get("session_id")

        output, new_session_id, context_exhausted, rl_time = await run_claude_session(
            prompt=prompt,
            cwd=cwd,
            resume_id=resume_id,
            tools=tools,
        )

        # ── Handle context exhaustion (start fresh next turn) ────────────────
        if context_exhausted:
            print(
                "\n⚠  Context window exhausted — dropping session ID. "
                "Next session will start fresh with history-aware prompt."
            )
            state.pop("session_id", None)
        elif new_session_id:
            state["session_id"] = new_session_id

        # ── Handle rate limit (sleep next iteration) ─────────────────────────
        if rl_time:
            print(f"\n⚠  Rate limit active — will wait until {rl_time}")
            state["rate_limit_resets_at"] = rl_time

        # ── Evaluate which objectives are now done ───────────────────────────
        if output.strip():
            prev_count = len(completed)
            updated = await evaluate_progress(planner, task, objectives, completed, output)
            state["completed_objectives"] = updated
            newly_done = [o for o in updated if o not in completed]

            if newly_done:
                print(f"\n✓  Newly completed ({len(newly_done)}): {newly_done}")
            else:
                print("\n  No new objectives completed this session.")
                # Brief back-off if stuck.
                await asyncio.sleep(8)
        else:
            print("\n  (No output from session.)")
            await asyncio.sleep(8)

        # ── Update history ───────────────────────────────────────────────────
        completed_now = state.get("completed_objectives", [])
        state.setdefault("history", []).append(
            f"Session {session_num}: {prompt[:100]}… "
            f"[{len(completed_now)}/{len(objectives)} done]"
        )
        save_state(state)

        await asyncio.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Autonomous Task Executor — drives Claude Code in a loop until "
            "every objective is met, auto-restarting on context / rate limits."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Inline task and objectives
python autonomous_executor.py \\
  --task "Build a REST API for a todo app" \\
  --objectives "Scaffold Flask project" \\
               "Add GET /todos endpoint" \\
               "Add POST /todos endpoint" \\
               "Write pytest tests" \\
               "Add README"

# JSON task file  {\"task\": \"...\", \"objectives\": [...]}
python autonomous_executor.py --task-file task.json

# Resume from last saved state
python autonomous_executor.py --resume

# Interactive mode (no flags)
python autonomous_executor.py
""",
    )
    p.add_argument("--task", help="High-level task description")
    p.add_argument(
        "--objectives",
        nargs="+",
        metavar="OBJ",
        help="Ordered list of objectives to complete",
    )
    p.add_argument(
        "--task-file",
        metavar="FILE",
        help='JSON file with keys "task" (str) and "objectives" (list[str])',
    )
    p.add_argument(
        "--cwd",
        default=".",
        metavar="DIR",
        help="Working directory for Claude Code (default: current directory)",
    )
    p.add_argument(
        "--tools",
        nargs="+",
        metavar="TOOL",
        help=f"Claude Code tools to allow (default: {' '.join(DEFAULT_TOOLS)})",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the saved executor_state.json",
    )
    return p


async def async_main() -> None:
    args = build_parser().parse_args()

    # ── Determine task & objectives ──────────────────────────────────────────
    if args.resume:
        state = load_state()
        if not state:
            print("No saved state found. Start a new task with --task and --objectives.")
            sys.exit(1)
        task: str = state["task"]
        objectives: list[str] = state["objectives"]

    elif args.task_file:
        data = json.loads(Path(args.task_file).read_text())
        task = data["task"]
        objectives = data["objectives"]

    elif args.task and args.objectives:
        task = args.task
        objectives = list(args.objectives)

    else:
        # Interactive mode
        print("═" * 55)
        print("  Autonomous Task Executor for Claude Code")
        print("═" * 55)
        task = input("\nTask description: ").strip()
        if not task:
            print("No task provided.")
            sys.exit(1)
        print("\nEnter objectives one per line (blank line to finish):")
        objectives = []
        while True:
            obj = input(f"  {len(objectives) + 1}. ").strip()
            if not obj:
                break
            objectives.append(obj)
        if not objectives:
            print("No objectives provided.")
            sys.exit(1)

    cwd = str(Path(args.cwd).resolve())
    tools = list(args.tools) if args.tools else DEFAULT_TOOLS

    # ── Summary header ───────────────────────────────────────────────────────
    print("\nTask      :", task)
    print("Objectives:")
    for i, o in enumerate(objectives, 1):
        print(f"  {i:2d}. {o}")
    print("CWD       :", cwd)
    print("Tools     :", ", ".join(tools))
    print("State file:", STATE_FILE.resolve())

    await run(task, objectives, cwd, tools)


if __name__ == "__main__":
    anyio.run(async_main)
