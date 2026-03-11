#!/bin/bash
cd ~/claude/open-mind
LOG="build-log.txt"
CC=~/.npm-global/bin/claude

run_feature() {
  local NUM="$1"
  local NAME="$2"
  echo "===================================================" | tee -a "$LOG"
  echo "STARTING: $NUM - $NAME at $(date)" | tee -a "$LOG"
  echo "===================================================" | tee -a "$LOG"

  cat > /tmp/om-prompt.txt << PROMPT
You are implementing features for the Open Mind project.

READ THESE FILES FIRST before writing any code:
1. CLAUDE.md - full project context, architecture, schema, endpoints
2. v3-spec.md - the feature spec you are implementing

TASK: Implement Feature $NUM - $NAME from v3-spec.md.

RULES:
- The server is om-server.py. The frontend is om-viz.html. Both are in this directory.
- Read both files fully before making changes.
- Do NOT break existing features or endpoints.
- Use subagents to parallelize: one for server changes, one for frontend changes. Coordinate at the end.
- SAVE FREQUENTLY: After each major change write the files to disk immediately. Do not accumulate large changes in memory.
- Use existing patterns: db_lock for SQLite, asyncio.to_thread for blocking calls, sanitize_fts for search.
- For new database tables, add CREATE TABLE statements to the init_db function.
- For frontend changes, add to the existing view system.
- TESTING: Production runs on port 8250. Do NOT kill it. Test on port 9250 with a throwaway DB: OPENMIND_PORT=9250 OPENMIND_DB=/tmp/openmind-test.db python3 om-server.py then curl localhost:9250/api/stats then kill the test process. Always clean up.
- Do not ask questions. Just implement.
PROMPT

  cat /tmp/om-prompt.txt | $CC -p - --dangerously-skip-permissions --max-turns 50 --output-format text 2>&1 | tee -a "$LOG"
  echo "FINISHED: $NUM - $NAME at $(date)" | tee -a "$LOG"
  sleep 5
}

run_feature "1" "Ask Your Knowledge Base"
run_feature "2" "LLM Edge Enrichment"
run_feature "3" "Resurface Forgotten Nodes"
run_feature "4" "Multiple Named Canvases"
run_feature "5" "Chat With Selection"
run_feature "6" "Structured Type Fields"
run_feature "7" "Canvas Chat"

echo "ALL FEATURES COMPLETE at $(date)" | tee -a "$LOG"
