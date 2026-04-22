# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ML Intern — an autonomous ML agent that researches, writes, and ships ML code using the Hugging Face ecosystem. Provides a CLI (interactive + headless) and a web UI (FastAPI backend + React frontend).

## Commands

```bash
# Install
uv sync                    # Python deps
uv tool install -e .       # Install CLI globally

# Run CLI
ml-intern                  # Interactive mode
ml-intern "prompt"         # Headless mode (auto-approve)

# Web UI (development)
cd backend && python main.py          # FastAPI on :7860
cd frontend && npm install && npm run dev  # Vite on :5173

# Frontend
cd frontend
npm run build              # tsc -b && vite build
npm run lint               # eslint

# Tests
uv run pytest              # Python (requires [dev] extra: uv sync --extra dev)

# Docker (full stack)
docker build -t ml-intern .
```

## Architecture

### Core flow: Queue-based async agent loop

```
User → submission_queue → submission_loop (agent_loop.py) → event_queue → UI
```

- **`agent/main.py`** — CLI entry point (`cli()`). Interactive (prompt-toolkit + rich) and headless modes.
- **`agent/core/agent_loop.py`** — `submission_loop()` consumes operations from queue, `run_agent()` runs the agentic loop (up to 300 iterations): LLM call → parse tool_calls → approval check → ToolRouter.execute → add results → repeat.
- **`agent/core/session.py`** — `Session` holds config, context_manager, hf_token, sandbox ref, event_queue. `OpType` enum defines operation types (USER_INPUT, EXEC_APPROVAL, INTERRUPT, UNDO, COMPACT, SHUTDOWN).
- **`agent/core/tools.py`** — `ToolSpec` (name, description, parameters, handler) and `ToolRouter` (registers built-in + MCP tools, dispatches execution).
- **`agent/context_manager/manager.py`** — Message history, system prompt loading (YAML + Jinja2), auto-compaction at 170k tokens.
- **`agent/core/doom_loop.py`** — Detects repeated tool call patterns, injects corrective prompts.

### Tools (`agent/tools/`)

19 tool modules. Key ones: `sandbox_tool.py` (bash/read/write/edit on HF Spaces sandbox), `docs_tools.py` (HF docs), `dataset_tools.py`, `papers_tool.py`, `github_*.py`, `hf_repo_*.py`, `jobs_tool.py` (HF Jobs), `research_tool.py` (subagent spawning), `plan_tool.py`, `local_tools.py` (local bash/read/write for CLI mode).

### Web backend (`backend/`)

- **`main.py`** — FastAPI app, CORS, static file serving, port 7860.
- **`routes/agent.py`** — REST + SSE: session CRUD, submit operations, event streaming.
- **`routes/auth.py`** — HF OAuth login/logout with session cookies.
- **`session_manager.py`** — `SessionManager` (max 10 concurrent sessions), `EventBroadcaster` fans events to SSE subscribers.

### Web frontend (`frontend/`)

React 18 + TypeScript + Vite + MUI 6. State via Zustand. Key: `SessionChat.tsx` (chat interface), `AppLayout.tsx` (layout), `api.ts` (centralized fetch).

### Config

- **`configs/main_agent_config.json`** — Model, MCP servers, approval settings. Env vars in MCP headers auto-substituted from `.env`.
- **System prompts** — `agent/prompts/system_prompt_v3.yaml` (current), Jinja2-templated.

### LLM integration

Uses `litellm` for provider-agnostic LLM calls. Model IDs use litellm format (e.g., `anthropic/claude-opus-4-6`). HF inference router support for HF-hosted models.

## Environment Variables

```
ANTHROPIC_API_KEY   # Claude API key
HF_TOKEN            # Hugging Face token (prompted on first launch if missing)
GITHUB_TOKEN        # GitHub PAT for code search tools
INFERENCE_TOKEN     # Optional: shared token for HF Spaces inference router
```

## Key Patterns

- Async-first: `asyncio.Queue` for both submissions and events.
- Tool approval: certain tools (sandbox creation, HF Jobs, file uploads) require user confirmation via `_needs_approval()` in agent_loop.py.
- Adding tools: create handler in `agent/tools/`, register `ToolSpec` in `agent/core/tools.py:create_builtin_tools()`.
- Adding MCP servers: add entry to `configs/main_agent_config.json` under `mcpServers`.
