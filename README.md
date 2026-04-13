# AI Orchestrator v9.5 — Supreme Commander

Telegram bot that turns a Mac Mini into a remote AI-powered development server. **Bootstrap new projects**, auto-solve GitHub issues with staggered parallelism, and control system monitoring — all from your phone.

## Architecture

```
Telegram  ──>  Bot (python-telegram-bot)
                ├── System Monitor (psutil, pmset)
                ├── Shell Executor (async subprocess)
                ├── tmux Viewer
                ├── Service Controller (launchd)
                └── AI Pipeline
                    ├── /init   ──>  Project Bootstrap Pipeline (5-step)
                    │                ├── Stack Scout (Haiku CLI)
                    │                ├── Architecting (Opus CLI)
                    │                ├── Execution (Sonnet CLI)
                    │                ├── CI Watch + Auto-Fix (Sonnet CLI)
                    │                └── Issue Planning (Opus CLI)
                    ├── /solve  ──>  Adaptive Pipeline (express/standard/full)
                    │                ├── Triage + Split Detection (Haiku)
                    │                ├── Haiku Research (Claude CLI)
                    │                ├── Opus Design (Claude CLI) + State Sync
                    │                ├── Gemini Critique (Gemini CLI, loop)
                    │                ├── Qwen Hints (Ollama)
                    │                ├── Sonnet Implement (Claude CLI)
                    │                ├── Local CI Check + Auto-Fix
                    │                ├── Sonnet Self-Review (Claude CLI)
                    │                ├── Gemini Cross-Review (Gemini CLI)
                    │                ├── AI Audit (DeepSeek R1, strict Why policy)
                    │                ├── Supreme Court (Gemini, on conflict)
                    │                └── Data Mining (Ollama, conditional)
                    ├── /solve --parallel ──>  Staggered Parallel Solve
                    │                ├── Dependency Triage (Haiku, all issues)
                    │                ├── File Conflict Detection
                    │                ├── Staggered Execution (Wait-Gates)
                    │                └── State Sync (project_summary.json)
                    ├── /extract ──>  Training data generation (Qwen)
                    └── /rebase  ──>  Auto conflict resolution (Claude CLI)
```

## End-to-End Flow (Staggered Parallel)

```
/solve project 1 2 3 --parallel
  │
  ├─ 1. Dependency-aware Triage (Haiku → all issues analyzed together)
  │     → Issue #1: independent | Issue #2: depends on #1 | Issue #3: independent
  │
  ├─ 2. Staggered Execution
  │     Issue #1: Research → Design → Implement ─── Audit ─── Done
  │     Issue #3: Research → Design → Implement ─┐ (Wait-Gate) ─── Audit ─── Done
  │     Issue #2:                    (Wait for #1 Audit)─── Research → Design → ...
  │
  ├─ 3. Audit Phase
  │     Sonnet Self-Review: PASS ─┐
  │     DeepSeek R1 Audit:  FAIL ─┤ → Supreme Court (Gemini) → User Decision
  │
  ├─ 4. State Sync
  │     → project_summary.json accumulates design decisions
  │     → Injected into subsequent issues' Design/Implement prompts
  │
  └─ 5. PR Creation + Results
```

## Features

### System Control
- `/status` — CPU, RAM, Disk, Thermal pressure, Ollama status, tmux sessions
- `/cmd <command>` — Remote shell execution with `--long` and `--stream` modes
- `/view` — Capture tmux pane output (secrets auto-masked)
- `/service` — launchd service control (start/stop/restart/logs)

### AI-Powered Development
- `/init [--public|--private] <name> <description>` — Bootstrap a new project
- `/issues <project>` — List open GitHub issues with inline Solve buttons
- `/solve <project> <#> [#...] [--fast|--std|--full] [--parallel]` — Auto-solve issues
- `/retry [project] [#]` — Resume failed solve from checkpoint
- `/rebase <project> <pr#>` — Rebase PR onto main, auto-resolve conflicts
- `/plan <project>` — Plan new development issues
- `/discuss <project> <question>` — Technical consultation with Opus
- `/extract <project> <file>` — Generate JSONL training data

### Adaptive Pipeline Modes

| Mode | Flag | Steps | Design Retries | CI Retries | AI Audit | ETA |
|------|------|-------|----------------|------------|----------|-----|
| Express | `--fast` | 3-Step | 0 | 2 | No | 5-15 min |
| Standard | `--std` | 6-Step | 2 | 5 | Yes (3x) | 15-40 min |
| Full | `--full` | 9-Step | 3 | 7 | Yes (5x) | 30-90 min |
| Auto | (default) | Haiku decides | Per mode | Per mode | Per mode | Varies |

### Staggered Parallel Solve (`--parallel`)

When solving multiple issues with `--parallel`, the scheduler:

1. **Dependency Triage** — Haiku analyzes all issues together for inter-dependencies
2. **File Conflict Detection** — Issues modifying the same files are automatically sequenced
3. **Staggered Execution** — Independent issues run in parallel; dependent issues wait at gates
4. **State Sync** — Design decisions accumulate in `project_summary.json` and are injected into subsequent issues

### Supreme Court (Conflict Resolution)

When Self-Review (Sonnet) returns PASS but AI Audit (DeepSeek R1) returns FAIL:

1. **Gemini mediates** — Analyzes both reviews and issues a ruling
2. **User decides** — Telegram buttons for Accept/Uphold/Overturn
3. **Rulings**: UPHOLD (re-implement), OVERTURN (proceed), REDESIGN (fail pipeline)
4. **Timeout** — Auto-accepts Gemini's ruling after 5 minutes

### Zero-Defect Shield

- **Local CI Check** — Auto-detect and run build/lint/test commands
- **AI Audit** — DeepSeek R1 adversarial audit with strict "Why" comment policy
- **Test-Gate** — CI failures trigger Sonnet auto-fix loop
- **Checkpoint & Resume** — `/retry` resumes from last successful step

### Init Pipeline (`/init`)

```
/init [--public|--private] my-app A KMP mobile app for task management

Step 0: [Haiku]   Stack Scout — tech stack + latest versions
Step 1: [Opus]    Architecting — CLAUDE.md + agents.md generation
Step 2: [Sonnet]  Execution — directory/file creation, git init, gh repo create
Step 3: [Sonnet]  CI Watch — wait for CI, auto-fix failures
Step 4: [Opus]    Issue Planning — 5-10 GitHub issues auto-created
```

### Five-brid Pipeline Steps

| Step | Model | Role | Fatal? |
|------|-------|------|--------|
| 0 | **Haiku** (Claude CLI) | Issue research + code pattern exploration | Fatal |
| 1 | **Opus** (Claude CLI) | Detailed design document + state context | Fatal |
| 2 | **Gemini** (Gemini CLI) | Design critique — loops with Opus | Non-fatal |
| 3 | **Qwen** (Ollama) | Code implementation hints | Non-fatal |
| 4 | **Sonnet** (Claude CLI) | Full implementation | Fatal |
| 5 | **Sonnet** (Claude CLI) | Self-review (safe-fail with snapshot) | Safe-fail |
| 6 | **Gemini** (Gemini CLI) | Cross-review | Non-fatal |
| 7 | **DeepSeek** (Ollama) | Adversarial audit (strict Why policy) | Fatal |
| 7.5 | **Gemini** (Gemini CLI) | Supreme Court (on Self-Review/Audit conflict) | Conditional |
| 8 | **Qwen** (Ollama) | Training data extraction | Non-fatal |

### State Sync (`project_summary.json`)

After each successful solve, key architectural decisions are extracted and stored in `.claude/project_summary.json`. The last 5 issues' decisions are injected into Design and Implement prompts of subsequent solves, ensuring continuity across issues.

## Tech Stack

- **Python 3.11+** with full `asyncio` concurrency
- **python-telegram-bot** — Telegram Bot API framework
- **Claude CLI** — Haiku/Opus/Sonnet via `claude -p --model`
- **Gemini CLI** — Design critique, cross-review, and Supreme Court mediation
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
- [GitHub CLI](https://cli.github.com/) (`gh`) authenticated
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) installed (for fivebrid mode)
- [Ollama](https://ollama.ai) with DeepSeek R1 and Qwen2.5-Coder-32B models

### Install

```bash
git clone https://github.com/manjees/ai-orchestrator.git
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
#   GITHUB_USER=your-github-username
#   PROJECTS_BASE_DIR=~/Desktop/dev

cp orchestrator/projects.json.example orchestrator/projects.json
# Edit projects.json with your project paths
```

### Run

```bash
# Direct
uv run python -m orchestrator

# Or as a macOS launchd service
```

## Project Structure

```
orchestrator/
├── __main__.py          # Entrypoint
├── bot.py               # Telegram app factory & provider init
├── config.py            # Pydantic settings (.env)
├── handlers.py          # Command handlers (/status, /cmd, /init, /solve, ...)
├── pipeline.py          # Init + Five-brid + legacy pipelines
├── scheduler.py         # Staggered parallel scheduler + dependency triage
├── state_sync.py        # Project summary persistence (decisions across issues)
├── checkpoint.py        # Pipeline checkpoint + resume (/retry)
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
