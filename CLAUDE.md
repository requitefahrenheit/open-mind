# CLAUDE.md — OpenMind

> Last updated: 2026-03-06. Server: 3238 lines. Frontend: 9968 lines.

## What is this?

OpenMind is a personal knowledge workspace — a force-directed hyperbolic graph that captures ideas, URLs, images, notes, diary entries, papers and tasks from multiple input channels and auto-links them by semantic similarity. Think Obsidian graph view meets Heptabase spatial canvas, with a universal natural language interface, MCP tool access, and voice/SMS input.

Live at: **https://openmind.fahrenheitrequited.dev**

---

## Architecture Overview

```
Input Channels
  Web UI · SMS (Twilio) · MCP tools · Voice (mic button / voice-server) · iOS Shortcut · Image paste
             ↓
  POST /api/nl  (universal NL endpoint) or direct API endpoints
             ↓
  FastAPI Server (port 8250) — om-server.py
    SQLite + WAL · all-MiniLM-L6-v2 embeddings · FTS5 · GPT-4o-mini intent parser
             ↓
  om-viz.html — single-file frontend (9968 lines, vanilla JS + 2D canvas)
```

**Companion services (all on same server):**

| Service | Port | File | Tunnel |
|---------|------|------|--------|
| OpenMind HTTP server | 8250 | `~/claude/open-mind/om-server.py` | openmind.fahrenheitrequited.dev |
| OpenMind MCP server | 8254 | `~/claude/openmind-mcp/om-mcp-server.py` | om.fahrenheitrequited.dev |
| Voice remote server | 8255 | `~/claude/voice-remote/voice-server.py` | voice.fahrenheitrequited.dev |
| Voice poller | — | `~/claude/voice-remote/voice-poller.py` | (polls Cortex, no tunnel) |

All tunneled via single Cloudflare tunnel `5382c123-1ceb-4b5f-9200-94974a8f6ee9`.
Config: `~/.cloudflared/config.yml`

---

## Tech Stack

- **Backend:** Python 3, FastAPI, uvicorn, SQLite (WAL mode)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (same model as Cortex)
- **LLM:** OpenAI `gpt-4o-mini` via httpx (intent parsing, enrichment, digest, captions)
- **OCR:** tesseract (optional, for image text extraction)
- **Frontend:** Single HTML file (`om-viz.html`), vanilla JS, 2D canvas — NO framework
- **MCP:** FastMCP server exposing 13 tools to Claude.ai
- **Voice:** Dedicated voice-server.py (FastAPI + ElevenLabs TTS + Claude API + MCP)

---

## Running Processes

```bash
# Check all running OM processes
ps aux | grep -E 'om-server|om-mcp|voice' | grep -v grep

# om-server.py — started by kick-off.sh or directly:
nohup python3 -u ~/claude/open-mind/om-server.py >> /tmp/om.log 2>&1 &

# om-mcp-server.py — runs in screen session 'om-mcp'
screen -r om-mcp

# voice-server.py — runs in screen session 'voice'
screen -r voice

# voice-poller.py — runs in screen session 'voice-poller'
screen -r voice-poller

# Restart om-server only:
pkill -f om-server.py && sleep 2
cd ~/claude/open-mind && python3 -u om-server.py >> /tmp/om.log 2>&1 &
```

**IMPORTANT:** Do NOT `pkill -f cloudflared` — it kills the tunnel for ALL services (Cortex, Autonomous, RWX, Therapy, OpenMind, Voice). Kill specific pids only.

---

## File Structure

```
~/claude/open-mind/          ← working source (edit here)
├── CLAUDE.md                ← this file
├── OPEN-MIND-SPEC.md        ← user-facing spec
├── om-server.py             ← FastAPI backend (3238 lines)
├── om-viz.html              ← single-file frontend (9968 lines)
├── kick-off.sh              ← start server + set env vars
└── backlog.md               ← feature backlog

~/claude/openmind-mcp/
├── om-mcp-server.py         ← MCP server (234 lines, FastMCP)
└── om-mcp.log               ← MCP server logs

~/claude/voice-remote/
├── voice-server.py          ← Voice HTTP server (248 lines)
├── voice-poller.py          ← Cortex voice command poller (116 lines)
└── om-voice.html            ← Voice UI frontend

~/openmind/
└── openmind.db              ← SQLite database (LIVE — edit cautiously)
```

---

## Database Schema

### nodes
| Column | Type | Notes |
|--------|------|-------|
| id | TEXT PK | 12-char hex |
| content | TEXT | full text |
| label | TEXT | short display label (~60 chars) |
| node_type | TEXT | `note`, `idea`, `url`, `paper`, `project`, `task`, `image`, `appointment`, `daily`, `diary`, `chore` |
| status | TEXT | `inbox`, `permanent`, `active` |
| pinned | INT | manually positioned |
| starred | INT | favorited |
| x, y | REAL | canvas coords (null = auto-layout) |
| image_path | TEXT | filename in uploads/ |
| url | TEXT | source URL |
| due_date | TEXT | ISO date |
| temperature | REAL | 0.1–2.0, decays over time |
| visit_count | INT | access counter |
| last_visited | TEXT | ISO datetime |
| embedding | BLOB | float32 vector |
| metadata | TEXT | JSON blob |
| created_at, updated_at | TEXT | ISO datetimes |

### edges
id, source_id, target_id, label, weight (cosine sim), auto_created, UNIQUE(source_id, target_id)

### daily_nodes
One row per calendar day, linked to a `daily` type node.

### nodes_fts
FTS5 virtual table over id, content, label, node_type.

---

## Key API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Frontend HTML |
| GET | `/api/graph` | Full graph (nodes + edges) |
| POST | `/api/nl` | Universal NL → LLM routes to add/search/digest |
| POST | `/api/add` | Direct add (optional LLM parsing) |
| POST | `/api/add-image` | Multipart image + OCR + GPT-4o caption |
| POST | `/api/add-pdf` | PDF upload + text extraction + thumbnail |
| GET | `/api/node/{id}` | Full node detail + connections + records visit |
| PATCH | `/api/node/{id}` | Update fields |
| DELETE | `/api/node/{id}` | Delete + cascade edges |
| POST | `/api/link` | Create manual edge |
| GET | `/api/search?q=` | Semantic + FTS search |
| GET | `/api/stats` | Node/edge/inbox counts |
| POST | `/api/sms` | Twilio webhook |
| POST | `/api/transcribe` | Whisper audio → NL pipeline |
| POST | `/api/digest` | LLM summary of recent activity |
| POST | `/api/positions` | Batch save x,y positions |
| GET | `/api/inbox` | 20 most recent inbox nodes |
| GET | `/api/canvases` | List canvases |
| **POST** | **`/api/remote-command`** | **Queue browser navigation command** |
| **GET** | **`/api/remote-command/poll`** | **Browser polls for queued commands (250ms)** |
| GET | `/uploads/{file}` | Serve uploads |

---

## Remote Control System (Voice → Browser)

The browser polls `/api/remote-command/poll` every **250ms**.
Anything that POSTs to `/api/remote-command` makes the browser move within ~250ms.

### POST /api/remote-command
```json
{ "action": "navigate", "query": "nukes" }       // fly to best matching node
{ "action": "switch_view", "view": "focus" }      // switch view
{ "action": "search", "query": "consciousness" }  // open search
{ "action": "toast", "query": "hello" }           // show toast message
```

**navigate** immediately queues the command and starts background semantic search to resolve `node_id`. The browser receives the command within 250ms; `node_id` is typically resolved within ~85ms (well before next poll).

### Three paths that POST to /api/remote-command:
1. **MCP tools** — `openmind:focus_on` and `openmind:switch_view` in Claude text sessions
2. **voice-server.py** — dedicated voice endpoint at `voice.fahrenheitrequited.dev`
3. **Zapier webhook** — `OM Voice` Zapier action → `https://openmind.fahrenheitrequited.dev/api/remote-command`

### Voice mode status (as of 2026-03-06):
- **Mic button in Claude.ai text chat** → WORKS perfectly. MCP tools fire, browser moves.
- **Advanced Voice Mode (phone icon)** → tool calls hang/timeout. Does NOT work reliably.
- **voice.fahrenheitrequited.dev** → dedicated voice UI, not yet fully tested from phone.
- **Root cause of AVM failure:** MCP tool round-trip latency causes voice mode to time out before tool executes.

---

## MCP Server (om-mcp-server.py)

FastMCP server at port 8254, tunneled at `om.fahrenheitrequited.dev/mcp?token=emc2ymmv`.

**13 tools:**
`add_node`, `natural_language`, `search`, `ask`, `get_node`, `edit_node`, `delete_node`,
`add_image`, `stats`, `recent_nodes`, `focus_on`, `switch_view`, `digest`

`focus_on` and `switch_view` POST to `/api/remote-command` on the HTTP server.

Token auth: `?token=emc2ymmv` or `Authorization: Bearer emc2ymmv`

---

## Voice Server (voice-server.py)

FastAPI at port 8255, tunneled at `voice.fahrenheitrequited.dev`.
- Accepts POST `/api/voice` with `{"text": "...", "session_id": "..."}`
- Calls Claude API (`claude-sonnet-4-5-20250929`) with OpenMind MCP attached
- Maintains per-session conversation history (up to 20 turns)
- Returns ElevenLabs TTS audio (base64 mp3, `eleven_flash_v2_5` low-latency model)
- Falls back to local regex parsing if Claude API fails
- Frontend: `om-voice.html` served at root

---

## Voice Poller (voice-poller.py)

Python script polling Cortex every 2 seconds for entries tagged `voice-command`.
Parses command text → POST to `/api/remote-command` → tags entry `voice-processed`.
Designed as a Cortex-based async command queue (not actively used for primary voice flow).

---

## Frontend Views (om-viz.html)

Five views, tab-switched, persisted in localStorage:

| View | Description |
|------|-------------|
| **Stream** | Reverse-chronological card feed. Filter chips by type, text filter, temporal slider. |
| **Focus** | Poincaré hyperbolic disk — the primary work area. Möbius fly-through animations, summary rail with Bézier connectors, primary/secondary/tertiary node layers, inline edit, history nav (⌫/⌦), 2-hop toggle. |
| **Map** | 2D force-directed canvas. Full physics simulation. Pinning, drag, zoom/pan. |
| **Semantic** | Embedding-layout view. Nodes positioned by semantic similarity. |
| **Triage** | Inbox processing. Swipe gestures, keyboard shortcuts (p=permanent, d=delete, e=edit, u=undo), undo bar. |

### Remote command handler (in frontend):
```javascript
setInterval(async () => {
  const data = await fetch('/api/remote-command/poll').then(r => r.json());
  for (const cmd of data.commands) {
    if (cmd.action === 'navigate') { enterFocusView(cmd.node_id || resolveLocal(cmd.query)); }
    else if (cmd.action === 'switch_view') { switchView(cmd.view); }
    else if (cmd.action === 'search') { /* populates semantic search */ }
    else if (cmd.action === 'toast') { toast(cmd.query, 'remote'); }
  }
}, 250); // 250ms — changed from 2000ms on 2026-03-06
```

---

## Current State (2026-03-06)

- 245 nodes, 496 edges, 216 inbox items
- Types: 99 diary, 49 image, 26 paper, 22 chore, 11 daily, 6 appointment, 4 idea, 3 note, 25 url
- All services running (om-server, om-mcp-server, voice-server, voice-poller, cloudflared)
- Remote command poll interval: **250ms** (changed from 2000ms today)
- Remote command endpoint: now returns **instantly** (background async node_id resolution)

---

## Environment Variables

| Var | Purpose |
|-----|---------|
| `OPENAI_API_KEY` | GPT-4o-mini for intent parsing + Vision captions |
| `OPENMIND_PORT` | Default 8250 |
| `OPENMIND_DB` | Default `~/openmind/openmind.db` |
| `OPENMIND_UPLOADS` | Default `~/openmind/uploads` |

---

## Thread Safety

All SQLite ops wrapped in `threading.Lock` (`db_lock`). Blocking ops use `asyncio.to_thread()`.
FTS queries sanitized via `sanitize_fts()` (each token double-quoted to prevent injection).

---

## Relationship to Cortex

Cortex (`cortex.fahrenheitrequited.dev`) is the AI conversation memory store — structured key-value + semantic search, used by Claude sessions for persistent memory. OpenMind is the spatial knowledge graph for active thinking. Same embedding model. Planned: cross-reference between the two.

---

## Quick Commands

```bash
# Stats
curl -s http://localhost:8250/api/stats | python3 -m json.tool

# Test remote command
curl -s -X POST http://localhost:8250/api/remote-command \
  -H 'Content-Type: application/json' \
  -d '{"action":"navigate","query":"nukes"}'

# Check poll
curl -s http://localhost:8250/api/remote-command/poll

# Logs
tail -f /tmp/om.log | grep -v 'remote-command/poll'
tail -f ~/claude/openmind-mcp/om-mcp.log

# Screen sessions
screen -ls
screen -r om-mcp
screen -r voice
screen -r voice-poller
```
