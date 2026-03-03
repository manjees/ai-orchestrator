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
                    ├── /solve  ──>  Five-brid Pipeline (9-step, 5 models)
                    │                ├── Haiku Research (Claude CLI)
                    │                ├── Opus Design (Claude CLI)
                    │                ├── Gemini Critique (Gemini CLI, loop)
                    │                ├── Qwen Hints (Ollama)
                    │                ├── Sonnet Implement (Claude CLI)
                    │                ├── Sonnet Self-Review (Claude CLI, safe-fail)
                    │                ├── Gemini Cross-Review (Gemini CLI)
                    │                ├── DeepSeek Audit (Ollama)
                    │                └── Data Mining (Ollama, conditional)
                    ├── /solve  ──>  Legacy Triple-Model Pipeline (5-step)
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
- `/solve <project> <#> [#...]` — Auto-solve issues via configurable pipeline
- `/rebase <project> <pr#>` — Rebase PR onto main, auto-resolve conflicts with Claude
- `/extract <project> <file>` — Generate JSONL training data from a source file

### Five-brid Pipeline (`PIPELINE_MODE=fivebrid`)

A 9-step pipeline using **5 models** (Haiku, Opus, Gemini, Sonnet, DeepSeek/Qwen) for maximum quality:

| Step | Model | Role | Fatal? |
|------|-------|------|--------|
| 0 | **Haiku** (Claude CLI) | Issue research + code pattern exploration | Fatal |
| 1 | **Opus** (Claude CLI) | Detailed design document | Fatal |
| 2 | **Gemini** (Gemini CLI) | Design critique — loops with Opus (max N retries) | Non-fatal |
| 3 | **Qwen** (Ollama) | Code implementation hints | Non-fatal |
| 4 | **Sonnet** (Claude CLI) | Full implementation | Fatal |
| 5 | **Sonnet** (Claude CLI) | Self-review + fix (safe-fail with snapshot recovery) | Safe-fail |
| 6 | **Gemini** (Gemini CLI) | Cross-review from different perspective | Non-fatal |
| 7 | **DeepSeek** (Ollama) | Final security/logic audit (APPROVED/REJECTED) | Fatal |
| 8 | **Qwen** (Ollama) | Enhanced training data extraction (conditional) | Non-fatal |

Key features:
- **Design Loop** — Opus and Gemini iterate on the design up to `MAX_DESIGN_RETRIES` times
- **Safe-fail Self-Review** — Step 4 snapshot is saved; if Step 5 breaks the code, it auto-recovers
- **Enhanced Data Mining** — Bundles research background + design intent + final code as JSONL training data
- **No API costs for Claude** — Haiku/Opus/Sonnet all run via `claude -p --model` CLI

### Legacy Triple-Model Pipeline (`PIPELINE_MODE=legacy`)

The original 5-step pipeline with DeepSeek R1, Qwen2.5-Coder, and Claude:

1. **DeepSeek Design** (Ollama) — Implementation plan
2. **Qwen Pre-Implement** (Ollama) — Code hints (non-fatal)
3. **Claude Implement** (CLI) — Full implementation with auto-retry on review failure
4. **Claude Review** (API/CLI) — Code review with PASS/FAIL verdict
5. **DeepSeek Audit** (Ollama) — Final APPROVED/REJECTED verdict
6. **Data Mining** (Ollama, conditional) — JSONL training data generation

### Pipeline Mode Selection

Set `PIPELINE_MODE` in `.env`:

| Mode | Value | Description |
|------|-------|-------------|
| Five-brid | `fivebrid` | 9-step, 5 models (recommended) |
| Legacy | `legacy` | 5-step, 3 models (default) |
| Direct | `legacy` + `DUAL_CHECK_ENABLED=false` | Claude-only, no review gates |

### Training Data (`/extract`)

Generate instruction-output JSONL pairs from any source file using Qwen2.5-Coder. Results under 4KB are sent inline; larger outputs are sent as downloadable `.jsonl` files.

### PR Rebase (`/rebase`)

Automatically rebases a PR branch onto `main`. When conflicts occur, Claude CLI resolves them — no manual intervention needed.

## Tech Stack

- **Python 3.11+** with full `asyncio` concurrency
- **python-telegram-bot** — Telegram Bot API framework
- **Claude CLI** — Haiku/Opus/Sonnet via `claude -p --model`
- **Gemini CLI** — Design critique and cross-review
- **Anthropic API** — Direct API calls for legacy code review
- **Ollama** — Local LLM inference (DeepSeek R1, Qwen2.5-Coder-32B)
- **psutil** — Cross-platform system monitoring
- **Git worktrees** — Isolated parallel workspaces
- **launchd** — macOS service management

## Setup

### Prerequisites
- macOS with [Homebrew](https://brew.sh)
- Python 3.11+, [uv](https://github.com/astral-sh/uv)
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-cli) installed
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) installed (for fivebrid mode)
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
#   PIPELINE_MODE=fivebrid  (or legacy)

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
├── bot.py               # Telegram app factory & provider init
├── config.py            # Pydantic settings (.env)
├── handlers.py          # Command handlers (/status, /cmd, /solve, /rebase, ...)
├── pipeline.py          # Five-brid + legacy pipelines
├── security.py          # Auth filter & secret masking
├── system_monitor.py    # CPU, RAM, Disk, Thermal via psutil
├── tmux_manager.py      # tmux session capture
└── ai/
    ├── base.py              # Abstract AI provider interface
    ├── anthropic_provider.py  # Anthropic SDK wrapper
    ├── gemini_provider.py     # Gemini CLI subprocess wrapper
    └── ollama_provider.py     # Ollama REST API wrapper
```

## Security

- **User whitelist** — Only the configured Telegram user ID can interact with the bot
- **Secret masking** — API keys and tokens in command output are auto-replaced with `[MASKED]`
- **No secrets in repo** — `.env` and `projects.json` are gitignored

## License

MIT
