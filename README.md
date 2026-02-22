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
                    ├── /solve  ──>  Triple-Model Pipeline
                    │                ├── DeepSeek Design (Ollama)
                    │                ├── Qwen Pre-Implement (Ollama)
                    │                ├── Claude Implement (CLI)
                    │                ├── Claude Review (API)
                    │                ├── DeepSeek Audit (Ollama)
                    │                └── Data Mining (Ollama, conditional)
                    ├── /extract ──>  Training data generation (Qwen)
                    └── /rebase  ──>  Auto conflict resolution (Claude CLI)
```

## Features

### System Control
- `/status` — CPU, RAM, Disk, Thermal pressure, Ollama status, tmux sessions
- `/cmd <command>` — Remote shell execution with `--long` and `--stream` modes
- `/view` — Capture tmux pane output (secrets auto-masked)
- `/service` — launchd service control (start/stop/restart/logs)

### AI-Powered Development
- `/issues <project>` — List open GitHub issues with inline Solve buttons
- `/solve <project> <#> [#...]` — Auto-solve issues with a 5-step triple-model pipeline
- `/rebase <project> <pr#>` — Rebase PR onto main, auto-resolve conflicts with Claude
- `/extract <project> <file>` — Generate JSONL training data from a source file

### Triple-Model Pipeline (`/solve`)

A 5-step quality gate that uses three AI models (DeepSeek R1, Qwen2.5-Coder, Claude) to cross-check each other:

1. **DeepSeek Design** (Ollama) — Analyzes the issue and creates an implementation plan
2. **Qwen Pre-Implement** (Ollama) — Generates code-only implementation hints for Claude
3. **Claude Implement** (CLI) — Implements the solution using the design + Qwen hints
4. **Claude Review** (Anthropic API) — Code review with PASS/FAIL verdict; auto-retries on FAIL
5. **DeepSeek Audit** (Ollama) — Final audit with APPROVED/REJECTED verdict
6. **Data Mining** (Ollama, conditional) — On success, generates JSONL training pairs from the solve

Steps 2 and 6 are non-fatal — the pipeline continues if they fail. Each issue runs in an isolated **git worktree**, enabling parallel solves.

### Training Data (`/extract`)

Generate instruction-output JSONL pairs from any source file using Qwen2.5-Coder. Results under 4KB are sent inline; larger outputs are sent as downloadable `.jsonl` files.

### PR Rebase (`/rebase`)

Automatically rebases a PR branch onto `main`. When conflicts occur, Claude CLI resolves them — no manual intervention needed.

## Tech Stack

- **Python 3.11+** with full `asyncio` concurrency
- **python-telegram-bot** — Telegram Bot API framework
- **Claude CLI** — Agentic code generation and conflict resolution
- **Anthropic API** — Direct API calls for code review
- **Ollama** — Local LLM inference (DeepSeek R1, Qwen2.5-Coder-32B)
- **psutil** — Cross-platform system monitoring
- **Git worktrees** — Isolated parallel workspaces
- **launchd** — macOS service management

## Setup

### Prerequisites
- macOS with [Homebrew](https://brew.sh)
- Python 3.11+, [uv](https://github.com/astral-sh/uv)
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-cli) installed
- [GitHub CLI](https://cli.github.com/) (`gh`) authenticated
- [Ollama](https://ollama.ai) with DeepSeek R1 and Qwen2.5-Coder-32B models

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
├── pipeline.py          # Triple-model pipeline (DeepSeek ↔ Qwen ↔ Claude)
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
