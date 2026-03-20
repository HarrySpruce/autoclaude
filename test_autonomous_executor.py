"""
Tests for autonomous_executor.py — load_state and save_state functions.
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Patch autonomous_executor.query before importing the module so that no real
# API calls are ever made, even at module-import time side effects.
# ---------------------------------------------------------------------------
_mock_query = AsyncMock()

with patch.dict(sys.modules, {}):
    with patch("claude_agent_sdk.query", _mock_query):
        import autonomous_executor


# ---------------------------------------------------------------------------
# Module-level autouse fixture: ensure autonomous_executor.query is always
# replaced with an AsyncMock for every test in this file.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_query():
    with patch.object(autonomous_executor, "query", new_callable=AsyncMock) as mock_q:
        yield mock_q


# ---------------------------------------------------------------------------
# load_state tests
# ---------------------------------------------------------------------------

class TestLoadState:
    def test_load_state_no_file_returns_empty_dict(self, monkeypatch, tmp_path):
        """When STATE_FILE does not exist, load_state() returns {}."""
        non_existent = tmp_path / "does_not_exist.json"
        monkeypatch.setattr(autonomous_executor, "STATE_FILE", non_existent)

        result = autonomous_executor.load_state()

        assert result == {}

    def test_load_state_valid_json_returns_contents(self, monkeypatch, tmp_path):
        """When STATE_FILE contains valid JSON, load_state() returns its contents."""
        state_file = tmp_path / "executor_state.json"
        expected = {
            "task": "Build a REST API",
            "objectives": ["Scaffold Flask project", "Add GET /todos"],
            "completed_objectives": ["Scaffold Flask project"],
            "history": ["Session 1: Scaffold Flask project... [1/2 done]"],
            "started_at": "2026-03-20T10:00:00+00:00",
        }
        state_file.write_text(json.dumps(expected))
        monkeypatch.setattr(autonomous_executor, "STATE_FILE", state_file)

        result = autonomous_executor.load_state()

        assert result == expected

    def test_load_state_invalid_json_raises_decode_error(self, monkeypatch, tmp_path):
        """When STATE_FILE contains invalid JSON, load_state() raises json.JSONDecodeError."""
        state_file = tmp_path / "executor_state.json"
        state_file.write_text("not valid json {{{")
        monkeypatch.setattr(autonomous_executor, "STATE_FILE", state_file)

        with pytest.raises(json.JSONDecodeError):
            autonomous_executor.load_state()


# ---------------------------------------------------------------------------
# save_state tests
# ---------------------------------------------------------------------------

class TestSaveState:
    def test_save_state_writes_valid_json_to_file(self, monkeypatch, tmp_path):
        """save_state writes the dict as valid, readable JSON to STATE_FILE."""
        state_file = tmp_path / "executor_state.json"
        monkeypatch.setattr(autonomous_executor, "STATE_FILE", state_file)

        original = {
            "task": "Build a REST API",
            "objectives": ["Scaffold Flask project", "Add POST /todos"],
            "completed_objectives": [],
            "history": [],
            "started_at": "2026-03-20T10:00:00+00:00",
        }

        autonomous_executor.save_state(original)

        assert state_file.exists()
        assert json.loads(state_file.read_text()) == original

    def test_save_state_round_trips_through_load_state(self, monkeypatch, tmp_path):
        """Data written by save_state is returned unchanged by load_state."""
        state_file = tmp_path / "executor_state.json"
        monkeypatch.setattr(autonomous_executor, "STATE_FILE", state_file)

        original = {
            "task": "Write tests",
            "objectives": ["Create test file", "Run pytest"],
            "completed_objectives": ["Create test file"],
            "history": ["Session 1: Create test file... [1/2 done]"],
            "started_at": "2026-03-20T09:00:00+00:00",
        }

        autonomous_executor.save_state(original)
        result = autonomous_executor.load_state()

        assert result == original


# ---------------------------------------------------------------------------
# evaluate_progress tests
# ---------------------------------------------------------------------------

TASK = "Build a todo app"
OBJECTIVES = ["Create project", "Add endpoints", "Write tests"]


class TestEvaluateProgress:
    @pytest.mark.asyncio
    async def test_returns_matching_completed_objectives(self, monkeypatch):
        """When the planner returns valid JSON with known objectives, those are returned."""
        async def fake_planner(prompt: str) -> str:
            return '["Create project", "Add endpoints"]'

        monkeypatch.setattr(autonomous_executor, "_planner_query", fake_planner)

        result = await autonomous_executor.evaluate_progress(
            TASK, OBJECTIVES, [], "session output showing project and endpoints done"
        )

        assert result == ["Create project", "Add endpoints"]

    @pytest.mark.asyncio
    async def test_json_parse_fallback_returns_existing_completed(self, monkeypatch):
        """When the planner returns non-JSON, the existing completed list is returned unchanged."""
        async def fake_planner(prompt: str) -> str:
            return "Sorry, I cannot determine which objectives are complete."

        monkeypatch.setattr(autonomous_executor, "_planner_query", fake_planner)

        existing = ["Create project"]
        result = await autonomous_executor.evaluate_progress(
            TASK, OBJECTIVES, existing, "some output"
        )

        assert result == existing

    @pytest.mark.asyncio
    async def test_unknown_objectives_are_filtered_out(self, monkeypatch):
        """Objectives returned by the planner that are not in the known list are filtered."""
        async def fake_planner(prompt: str) -> str:
            return '["Create project", "Deploy to production", "Add endpoints"]'

        monkeypatch.setattr(autonomous_executor, "_planner_query", fake_planner)

        result = await autonomous_executor.evaluate_progress(
            TASK, OBJECTIVES, [], "output"
        )

        assert "Deploy to production" not in result
        assert "Create project" in result
        assert "Add endpoints" in result

    @pytest.mark.asyncio
    async def test_empty_planner_output_returns_existing_completed(self, monkeypatch):
        """When the planner returns an empty string, the existing list is unchanged."""
        async def fake_planner(prompt: str) -> str:
            return ""

        monkeypatch.setattr(autonomous_executor, "_planner_query", fake_planner)

        existing = ["Create project"]
        result = await autonomous_executor.evaluate_progress(
            TASK, OBJECTIVES, existing, ""
        )

        assert result == existing


# ---------------------------------------------------------------------------
# run_claude_session tests
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock  # noqa: E402

from claude_agent_sdk import CLINotFoundError  # noqa: E402


def _mock_result(result_text: str, stop_reason: str = "end_turn") -> MagicMock:
    """Build a MagicMock that looks like a ResultMessage."""
    msg = MagicMock()
    msg.__class__ = autonomous_executor.ResultMessage
    msg.result = result_text
    msg.stop_reason = stop_reason
    msg.is_error = False
    return msg


class TestRunClaudeSession:
    @pytest.mark.asyncio
    async def test_context_exhaustion_stop_reason_sets_flag(self, monkeypatch):
        """stop_reason in CONTEXT_EXHAUSTED_REASONS causes context_exhausted=True."""
        msg = _mock_result("done", stop_reason="max_turns")

        async def fake_query(prompt, options):
            yield msg

        monkeypatch.setattr(autonomous_executor, "query", fake_query)

        output, _, context_exhausted, rate_limited = await autonomous_executor.run_claude_session(
            prompt="do something", cwd=".", resume_id=None, tools=[]
        )

        assert context_exhausted is True
        assert rate_limited is False

    @pytest.mark.asyncio
    async def test_rate_limit_keyword_in_exception_sets_flag(self, monkeypatch):
        """An exception whose message contains a rate-limit keyword sets rate_limited=True."""
        async def fake_query(prompt, options):
            raise Exception("Request failed: 429 rate limit exceeded")
            yield  # makes this an async generator

        monkeypatch.setattr(autonomous_executor, "query", fake_query)

        output, _, context_exhausted, rate_limited = await autonomous_executor.run_claude_session(
            prompt="do something", cwd=".", resume_id=None, tools=[]
        )

        assert rate_limited is True
        assert context_exhausted is False
        assert "Error:" in output

    @pytest.mark.asyncio
    async def test_cli_not_found_error_is_caught(self, monkeypatch):
        """CLINotFoundError is caught; output contains the error message."""
        async def fake_query(prompt, options):
            raise CLINotFoundError("claude binary not found")
            yield  # makes this an async generator

        monkeypatch.setattr(autonomous_executor, "query", fake_query)

        output, _, context_exhausted, rate_limited = await autonomous_executor.run_claude_session(
            prompt="do something", cwd=".", resume_id=None, tools=[]
        )

        assert "claude binary not found" in output
        assert context_exhausted is False
        assert rate_limited is False

    @pytest.mark.asyncio
    async def test_normal_result_captured_in_output(self, monkeypatch):
        """ResultMessage text is collected and returned in output."""
        msg = _mock_result("All done!", stop_reason="end_turn")

        async def fake_query(prompt, options):
            yield msg

        monkeypatch.setattr(autonomous_executor, "query", fake_query)

        output, _, context_exhausted, rate_limited = await autonomous_executor.run_claude_session(
            prompt="do something", cwd=".", resume_id=None, tools=[]
        )

        assert "All done!" in output
        assert context_exhausted is False
        assert rate_limited is False
