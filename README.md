# auto-claude

Autonomous task executor that drives Claude Code in a loop until all objectives are met.

## Features

- Accepts a high-level task with ordered objectives
- Drives Claude Code session after session until every objective is complete
- Persists state to `executor_state.json` — survives crashes and manual restarts
- Auto-restarts with summarised history when the context window is exhausted
- Detects rate limits and sleeps with exponential backoff before resuming
- Uses Claude Code's own session auth — no `ANTHROPIC_API_KEY` required

## Usage

```bash
pip install -r requirements.txt

# Inline
python autonomous_executor.py \
  --task "Build a REST API for a todo app" \
  --objectives "Scaffold Flask project" "Add GET /todos" "Add POST /todos" "Write tests"

# JSON task file
python autonomous_executor.py --task-file task.json

# Resume from last saved state
python autonomous_executor.py --resume

# Interactive mode
python autonomous_executor.py
```

## Requirements

- Python 3.11+
- Claude Code CLI installed and authenticated (`claude login`)
