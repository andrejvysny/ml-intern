# AGENTS.md

Operational guide for coding agents working in this repository.
Scope: whole repo (`agent/`, `backend/`, `frontend/`, `configs/`).

## 1) Project at a glance

- Name: **ML Intern**.
- Purpose: autonomous ML agent + CLI + web UI.
- Backend: Python/FastAPI.
- Core agent runtime: async Python (`asyncio`, `litellm`, queue/event architecture).
- Frontend: React 18 + TypeScript + Vite + MUI + Zustand.
- Python package manager/runtime: **uv**.

## 2) Environment + setup commands

- Install Python deps:
  - `uv sync`
- Install with dev extras (needed for tests):
  - `uv sync --extra dev`
- Install CLI entrypoint globally:
  - `uv tool install -e .`
- Optional frontend deps:
  - `cd frontend && npm install`

## 3) Run commands (dev)

- CLI interactive:
  - `ml-intern`
- CLI headless:
  - `ml-intern "your prompt"`
- Backend dev server:
  - `cd backend && python main.py`
- Frontend dev server:
  - `cd frontend && npm run dev`

## 4) Build commands

- Frontend production build:
  - `cd frontend && npm run build`
- Explicit TypeScript build/type-check:
  - `cd frontend && npx tsc -b`
- Docker full-stack build:
  - `docker build -t ml-intern .`

## 5) Lint commands

- Frontend lint:
  - `cd frontend && npm run lint`
- Python lint:
  - No dedicated repo linter config found (`ruff`, `black`, `flake8`, `mypy` not configured in repo).
  - Keep changes minimal and style-consistent; rely on tests + runtime checks.

## 6) Test commands

- Run all Python tests:
  - `uv run pytest`
- Run a single test file:
  - `uv run pytest path/to/test_file.py`
- Run a single test case:
  - `uv run pytest path/to/test_file.py::test_name`
- Run a single test method in class:
  - `uv run pytest path/to/test_file.py::TestClass::test_name`
- Run by keyword expression:
  - `uv run pytest -k "keyword_expr"`
- Verbose + fail fast (useful while iterating):
  - `uv run pytest -x -vv`

Notes:
- At time of writing, few/no committed test files may exist; still use the above pytest selectors when adding tests.
- No frontend test runner scripts are currently defined in `frontend/package.json`.

## 7) Architecture-sensitive workflow rules

- Respect async/evented design:
  - `submission_queue` in, `event_queue` out.
  - Do not block the event loop with long sync work.
- Prefer incremental changes in core loop files (`agent/core/agent_loop.py`, `agent/main.py`).
- Preserve approval/cancellation behavior for sensitive tools.
- Avoid changing public API payload shapes unless updating frontend accordingly.

## 8) Python style guidelines (repo-specific)

- Use Python 3.11+ features and typing syntax (`str | None`, `list[dict]`, etc.).
- Add type hints on all new/changed functions.
- Keep functions focused; split large logic into helpers where practical.
- Imports:
  - Standard library first, then third-party, then local imports.
  - Prefer absolute imports from `agent.*` in agent package.
- Naming:
  - `snake_case` for functions/variables.
  - `PascalCase` for classes/dataclasses.
  - `UPPER_SNAKE_CASE` for constants.
- Data models:
  - Use `pydantic.BaseModel` for API schemas.
  - Use dataclasses for lightweight internal structs where appropriate.
- Error handling:
  - Fail with actionable errors.
  - Catch broad exceptions only at boundaries (LLM/tool/network/UI boundaries).
  - Log exceptions with enough context; avoid swallowing silently.
- Logging:
  - Use module logger (`logging.getLogger(__name__)`).
  - Keep user-facing vs debug logs distinct.
- Paths/files:
  - Prefer `pathlib.Path` over `os.path` in new code.

## 9) TypeScript/React style guidelines (repo-specific)

- TypeScript is strict; do not introduce `any` unless unavoidable.
- Prefer explicit interfaces/types for API contracts and store state.
- Imports:
  - Use alias `@/` for `frontend/src/*`.
  - Keep import ordering consistent with surrounding file style.
- Components:
  - Functional components only.
  - Hooks follow `useXxx` naming.
  - Component files use `PascalCase.tsx`.
- State:
  - Zustand stores in `frontend/src/store`.
  - Preserve per-session state semantics (important in chat flow).
- Formatting conventions observed:
  - 2-space indentation.
  - Semicolons present.
  - Single quotes in TS/TSX.
- Lint compliance:
  - Run `npm run lint` after frontend edits.

## 10) Naming + API conventions

- Backend routes in `backend/routes/*.py` are authoritative for frontend API calls.
- Keep REST and SSE event names stable (`processing`, `tool_call`, `tool_output`, `turn_complete`, etc.).
- If adding events, update both backend emitter + frontend consumers.
- Keep model IDs in litellm format (`provider/model` or HF router form).

## 11) Error handling + resilience expectations

- Preserve retry logic for transient LLM/network failures.
- Preserve context-window handling and compaction behavior.
- Maintain cancellation cleanup paths (sandbox/job cleanup).
- Prefer user-readable error messages at external boundaries (CLI/API).

## 12) Security + secrets

- Never commit `.env`, tokens, credentials, or secret-bearing logs.
- Treat `HF_TOKEN`, `ANTHROPIC_API_KEY`, `GITHUB_TOKEN`, `INFERENCE_TOKEN` as sensitive.
- Redact sensitive values in logs and examples.

## 13) Git + change hygiene for agents

- Do **not** auto-commit, auto-push, or auto-pull unless explicitly asked.
- Keep diffs targeted; avoid broad refactors unless requested.
- Update docs when behavior/commands/interfaces change.
- Validate changed surface area with relevant commands before finishing.

## 14) Discovered Cursor/Copilot rules

Checked for:
- `.cursor/rules/`
- `.cursorrules`
- `.github/copilot-instructions.md`

Result at time of scan:
- No Cursor rules found.
- No Copilot instructions file found.

If these files are added later, treat them as higher-priority agent instructions and merge into this guide.
