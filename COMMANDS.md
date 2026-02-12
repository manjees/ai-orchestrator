# AI Orchestrator — Telegram 명령어 레퍼런스

## 시스템 모니터링

### `/status`
RAM, CPU, Thermal, Disk, Ollama, tmux 세션 상태를 한눈에 확인.
프로세스 Pause/Resume 버튼도 함께 표시.

```
/status
```

---

## 명령 실행

### `/cmd <command>`
Mac Mini에서 셸 명령 실행. 기본 30초 타임아웃.

```
/cmd uptime
/cmd ls -la ~/Desktop
/cmd brew update
```

### `/cmd --long <command>`
빌드, 테스트 등 오래 걸리는 작업용. 600초(10분) 타임아웃.

```
/cmd --long cd ~/project && ./gradlew assembleDebug
/cmd --long cd ~/project && npm run build
```

### `/cmd --stream <command>`
실시간 진행 확인. 10초마다 경과 시간 + 출력 업데이트.
완료 시 `[done in Xm Xs, exit=0]` 표시.

```
/cmd --stream ping -c 20 google.com
/cmd --stream cd ~/project && npm test
```

---

## Claude Code 원격 실행

### 기본 패턴: 파이프 입력 + `-p`

```
/cmd --stream cd <project-dir> && <input-command> | claude -p "<prompt>"
```

```
# GitHub 이슈 읽고 분석
/cmd --stream cd ~/project && gh issue view 4 | claude -p "Analyze this issue and suggest implementation plan"

# PR diff 리뷰
/cmd --stream cd ~/project && gh pr diff 12 | claude -p "Review this PR for bugs and improvements"

# 파일 내용 기반 작업
/cmd --stream cd ~/project && cat src/main.kt | claude -p "Add error handling to this file"
```

### 파이프 없이 단독 실행 (`-p` 필수)

```
/cmd --stream cd ~/project && claude -p "List all TODO comments in the codebase"
/cmd --stream cd ~/project && claude -p "Explain the architecture of this project"
```

> **주의**: `-p` 없이 `claude "..."` 단독 실행하면 interactive 모드로 진입 → TTY 없어서 hang.
> 파이프(`|`)로 stdin 연결 시에는 자동 non-interactive.

### 이전 대화 이어서 작업 (`-c`)

```
# 직전 세션의 Plan을 이어서 구현 시작
/cmd --stream cd ~/project && claude -c -p "Plan approved. Proceed with implementation."
```

### 자율 실행 모드 (파일 수정 권한 자동 승인)

```
/cmd --stream cd ~/project && claude -c -p --dangerously-skip-permissions "Implement all planned changes"
```

> **`--dangerously-skip-permissions`**: Edit, Write, Bash 등 도구를 확인 없이 실행.
> 신뢰하는 프로젝트에서만 사용할 것.

### 유용한 Claude CLI 옵션 조합

| 옵션 | 설명 |
|------|------|
| `-p` | Non-interactive 모드 (print & exit) |
| `-c` | 직전 대화 이어서 계속 |
| `--model opus` | 모델 지정 (sonnet, opus, haiku) |
| `--allowedTools "Edit Bash"` | 특정 도구만 허용 |
| `--dangerously-skip-permissions` | 모든 도구 자동 승인 |
| `--max-budget-usd 1.0` | API 비용 상한 설정 |

---

## tmux 세션 보기

### `/view`
`ai_factory` tmux 세션의 마지막 20줄 캡처.
API 키 등 시크릿은 자동 `[MASKED]` 처리.

```
/view
```

---

## 서비스 제어 (launchd)

### `/service`
사용 가능한 서브 명령어 안내.

### `/service status`
launchd 서비스 상태 (PID, 마지막 종료 코드).

### `/service restart`
봇 즉시 재시작 (`launchctl kickstart -k`).

### `/service stop`
봇 중지 + KeepAlive 비활성화 (`launchctl unload`).

### `/service start`
중지된 봇 다시 시작 (`launchctl load`).

### `/service logs`
최근 stderr 로그 40줄.

### `/service logs stdout`
최근 stdout 로그 40줄.

---

## 실전 워크플로우 예시

### 1. GitHub 이슈 → 계획 → 구현 (3단계)

```
# Step 1: 이슈 분석 + 계획
/cmd --stream cd ~/project && gh issue view 4 | claude -p "Read this issue. Plan the implementation. Show which files to modify."

# Step 2: 계획 승인 + 구현 시작
/cmd --stream cd ~/project && claude -c -p --dangerously-skip-permissions "Plan approved. Implement all changes."

# Step 3: 결과 확인
/cmd cd ~/project && git diff --stat
```

### 2. PR 리뷰 자동화

```
/cmd --stream cd ~/project && gh pr diff 15 | claude -p "Review this PR. Check for bugs, security issues, and style problems."
```

### 3. 빌드 + 테스트

```
# 빌드
/cmd --long cd ~/project && ./gradlew build

# 테스트 (실시간 확인)
/cmd --stream cd ~/project && ./gradlew test
```

### 4. 시스템 점검 루틴

```
/status
/cmd df -h
/cmd top -l 1 -n 5
/view
```
