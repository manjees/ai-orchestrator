# AI Orchestrator

Telegram bot that turns a Mac Mini into a remote AI-powered development server. Control system monitoring, run shell commands, and **auto-solve GitHub issues** — all from your phone.

## Architecture

```
Telegram  ──>  Bot (python-telegram-bot)
                ├── System Monitor (psutil, pmset)
                ├── Shell Executor (async subprocess)
                ├── tmux Viewer
                ├── Service Controller (launchd)
                └── AI Pipeline
                    ├── /solve  ──>  Dual-Check Pipeline
                    │                ├── DeepSeek Design (Ollama)
                    │                ├── Claude Implement (CLI)
                    │                ├── Claude Review (API)
                    │                └── DeepSeek Audit (Ollama)
                    └── /rebase ──>  Auto conflict resolution (Claude CLI)
```

## Features

### System Control
- `/status` — CPU, RAM, Disk, Thermal pressure, Ollama status, tmux sessions
- `/cmd <command>` — Remote shell execution with `--long` and `--stream` modes
- `/view` — Capture tmux pane output (secrets auto-masked)
- `/service` — launchd service control (start/stop/restart/logs)

### AI-Powered Development
- `/issues <project>` — List open GitHub issues with inline Solve buttons
- `/solve <project> <#> [#...]` — Auto-solve issues with a 4-step dual-check pipeline
- `/rebase <project> <pr#>` — Rebase PR onto main, auto-resolve conflicts with Claude

### Dual-Check Pipeline (`/solve`)

A 4-step quality gate that uses two different AI models to cross-check each other:

1. **DeepSeek Design** (Ollama, local) — Analyzes the issue and creates an implementation plan
2. **Claude Implement** (Claude CLI) — Implements the solution following the design
3. **Claude Review** (Anthropic API) — Code review with PASS/FAIL verdict; auto-retries on FAIL
4. **DeepSeek Audit** (Ollama, local) — Final audit with APPROVED/REJECTED verdict

Each issue runs in an isolated **git worktree**, enabling parallel solves without interference.

### PR Rebase (`/rebase`)

Automatically rebases a PR branch onto `main`. When conflicts occur, Claude CLI resolves them — no manual intervention needed.

## Tech Stack

- **Python 3.11+** with full `asyncio` concurrency
- **python-telegram-bot** — Telegram Bot API framework
- **Claude CLI** — Agentic code generation and conflict resolution
- **Anthropic API** — Direct API calls for code review
- **Ollama** — Local LLM inference (DeepSeek R1)
- **psutil** — Cross-platform system monitoring
- **Git worktrees** — Isolated parallel workspaces
- **launchd** — macOS service management

## Setup

### Prerequisites
- macOS with [Homebrew](https://brew.sh)
- Python 3.11+, [uv](https://github.com/astral-sh/uv)
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-cli) installed
- [GitHub CLI](https://cli.github.com/) (`gh`) authenticated
- [Ollama](https://ollama.ai) (optional, for dual-check pipeline)

### Install

```bash
git clone https://github.com/yourusername/ai-orchestrator.git
cd ai-orchestrator
uv sync
```

### Configure

```bash
cp .env.example .env
# Edit .env with your tokens:
#   TELEGRAM_BOT_TOKEN=...
#   TELEGRAM_ALLOWED_USER_ID=...

cp orchestrator/projects.json.example orchestrator/projects.json
# Edit projects.json with your project paths
```

### Run

```bash
# Direct
uv run python -m orchestrator

# Or as a macOS launchd service (see COMMANDS.md for details)
```

## Project Structure

```
orchestrator/
├── __main__.py          # Entrypoint
├── bot.py               # Telegram app factory & handler registration
├── config.py            # Pydantic settings (.env)
├── handlers.py          # Command handlers (/status, /cmd, /solve, /rebase, ...)
├── pipeline.py          # Dual-check pipeline (DeepSeek ↔ Claude)
├── security.py          # Auth filter & secret masking
├── system_monitor.py    # CPU, RAM, Disk, Thermal via psutil
├── tmux_manager.py      # tmux session capture
└── ai/
    ├── base.py          # Abstract AI provider interface
    ├── anthropic_provider.py
    └── ollama_provider.py
```

## Security

- **User whitelist** — Only the configured Telegram user ID can interact with the bot
- **Secret masking** — API keys and tokens in command output are auto-replaced with `[MASKED]`
- **No secrets in repo** — `.env` and `projects.json` are gitignored

## License

MIT
