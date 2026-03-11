# CLAUDE.md — Open Mind

## What is this?

Open Mind is a personal knowledge workspace — a force-directed graph that captures ideas, URLs, images, notes, and tasks from multiple input channels and auto-links them by semantic similarity. Think Obsidian graph view meets Heptabase spatial canvas, but with a universal natural language interface and SMS/voice input.

## Naming Convention

All project files use the `om-` prefix: `om-server.py`, `om-viz.html`, `om-kickoff.sh`. Follow this pattern for new files.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Input Channels                                  │
│  ┌──────┐ ┌──────┐ ┌───────┐ ┌─────┐ ┌───────┐ │
│  │ Web  │ │ SMS  │ │ Voice │ │ iOS │ │ Image │ │
│  │  UI  │ │Twilio│ │Whisper│ │Short│ │Paste/ │ │
│  │      │ │      │ │      │ │ cut │ │ Drop  │ │
│  └──┬───┘ └──┬───┘ └──┬────┘ └──┬──┘ └──┬────┘ │
│     └────────┴────────┴─────────┴───────┘       │
│                    ▼                             │
│          POST /api/nl  (universal NL endpoint)   │
│                    ▼                             │
│         GPT-4o-mini Intent Parser                │
│         (add / search / digest / link)           │
│                    ▼                             │
│     ┌──────────────────────────┐                 │
│     │  FastAPI Server (8250)   │                 │
│     │  • SQLite + WAL          │                 │
│     │  • all-MiniLM-L6-v2     │                 │
│     │  • Auto-link (cosine)    │                 │
│     │  • FTS5 full-text        │                 │
│     │  • Temperature decay     │                 │
│     │  • OCR (tesseract)       │                 │
│     └──────────────────────────┘                 │
│                    ▼                             │
│     ┌──────────────────────────┐                 │
│     │  openmind.html           │                 │
│     │  Single-file 2D canvas   │                 │
│     │  Force-directed graph    │                 │
│     └──────────────────────────┘                 │
└─────────────────────────────────────────────────┘
```

## Tech Stack

- **Backend:** Python 3, FastAPI, uvicorn, SQLite (WAL mode)
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (same model as Cortex)
- **LLM:** OpenAI `gpt-4o-mini` via httpx (intent parsing, enrichment)
- **OCR:** tesseract (optional, for image text extraction)
- **Frontend:** Single HTML file, vanilla JS, 2D canvas (no framework)
- **Server:** c-jfischer3, port 8250, cloudflared tunnel
- **SMS:** Twilio webhook at `/api/sms`

## File Structure

```
~/openmind/
├── CLAUDE.md              ← you are here
├── om-server.py           ← FastAPI backend (~1480 lines, all server logic)
├── om-viz.html            ← frontend (~2420 lines, single HTML file)
├── om-ios-shortcut.md     ← iOS Shortcut setup guide
├── kick-off.sh            ← env vars + start server + tunnel
├── setup.sh               ← install deps
├── openmind.db            ← SQLite database (created on first run)
└── uploads/               ← stored files:
    ├── *.jpg/png/webp     ← uploaded images
    ├── thumb_*.png/jpg    ← thumbnails (og:image, PDF first page)
    ├── *.pdf              ← uploaded PDFs
    └── archive_*.html     ← archived URL pages
```

## Database Schema

### nodes
| Column | Type | Notes |
|--------|------|-------|
| id | TEXT PK | 12-char uuid hex |
| content | TEXT | full text content |
| label | TEXT | short display label (max ~60 chars) |
| node_type | TEXT | `note`, `idea`, `url`, `paper`, `project`, `task`, `image`, `appointment`, `daily` |
| status | TEXT | `inbox`, `permanent`, `active` |
| pinned | INT | manually positioned (spatial memory) |
| starred | INT | favorited |
| x, y | REAL | canvas coordinates (null = auto-layout) |
| image_path | TEXT | filename in uploads/ dir |
| url | TEXT | source URL if applicable |
| due_date | TEXT | ISO date for time-sensitive nodes |
| temperature | REAL | 0.1–2.0, decays over time, boosted on visit |
| visit_count | INT | access counter |
| last_visited | TEXT | ISO datetime |
| embedding | BLOB | float32 vector from sentence-transformers |
| metadata | TEXT | JSON blob for extensible data (url_meta, etc) |
| created_at, updated_at | TEXT | ISO datetimes |

### edges
| Column | Type | Notes |
|--------|------|-------|
| id | TEXT PK | 12-char uuid hex |
| source_id, target_id | TEXT FK | → nodes.id, CASCADE delete |
| label | TEXT | `related`, `daily`, or custom |
| weight | REAL | cosine similarity score for auto-edges |
| auto_created | INT | 1 if system-generated, 0 if manual |
| UNIQUE(source_id, target_id) | | no duplicate edges |

### daily_nodes
One row per day, links to a "daily" type node. New nodes auto-attach here.

### nodes_fts
FTS5 virtual table mirroring id, content, label, node_type. Rebuilt on startup.

## Key API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve frontend HTML |
| GET | `/api/graph` | Full graph export (nodes + edges) |
| POST | `/api/nl` | **Universal NL endpoint** — LLM parses intent, routes to add/search/digest |
| POST | `/api/add` | Direct add (with optional LLM parsing) |
| POST | `/api/add-image` | Multipart image upload + OCR + caption + detailed description |
| POST | `/api/add-pdf` | PDF upload + text extraction + thumbnail |
| GET | `/api/node/{id}` | Full node detail + connections (also records visit) |
| PATCH | `/api/node/{id}` | Update node fields |
| DELETE | `/api/node/{id}` | Delete node + cascade edges |
| POST | `/api/link` | Create manual edge |
| DELETE | `/api/edge/{id}` | Delete edge |
| POST | `/api/positions` | Batch save x,y positions from frontend |
| GET | `/api/search?q=` | Combined semantic + FTS search |
| GET | `/api/stats` | Node/edge/inbox counts |
| GET | `/api/similar-images/{id}` | Find semantically similar image nodes |
| POST | `/api/sms` | Twilio webhook (bidirectional SMS) |
| POST | `/api/transcribe` | Whisper audio transcription → NL pipeline |
| POST | `/api/digest` | LLM summary of recent graph activity |
| GET | `/uploads/{file}` | Serve uploaded files (images, PDFs, HTML archives) |

## Core Concepts

### Node Lifecycle
`inbox` → `permanent` → decay
- New nodes arrive as `inbox` (pulsing dashed ring in UI)
- User promotes to `permanent` via detail panel
- Temperature decays hourly (~10%/day), boosted +0.3 on each visit
- Hot nodes (temp > 1.2) glow brighter, pull toward center
- Cold nodes fade and drift to periphery

### Auto-Linking
On every node add:
1. Compute embedding of `"{label}. {content[:500]}"`
2. Compare cosine similarity against all existing node embeddings
3. Create edges for top 5 matches above threshold (0.45)
4. Also attach to today's daily node

### Natural Language Interface
Every input (typed, texted, spoken) hits `/api/nl` which:
1. Sends text + recent node context to GPT-4o-mini
2. LLM returns structured JSON: action, content, label, type, search_query, etc.
3. Server routes to appropriate handler (create_node, semantic_search, digest)
4. Returns response text (shown as toast in web, SMS reply for Twilio)

### Spatial Memory
- Dragging a node pins it (`_pinned = true`, position saved to DB)
- Pinned nodes don't move during force simulation
- Unpinned nodes auto-layout via force-directed physics
- Users build spatial memory: "that idea was bottom-left near the project stuff"

### Physics Simulation (Frontend)
- Repulsion: all-pairs inverse-square, `800 * alpha / dist²`
- Attraction: edge spring force, `(dist - 120) * 0.005 * alpha * weight`
- Center gravity: `0.0005 * alpha * temperature` (hot nodes pull to center)
- Damping: 0.85 per tick
- Alpha decays at 0.998/frame, simulation stops below 0.001

## Thread Safety

All SQLite operations are wrapped in `threading.Lock` (`db_lock`). Blocking operations (LLM calls, embeddings, URL fetches) use `asyncio.to_thread()` to avoid stalling the event loop.

## FTS Safety

User queries are sanitized through `sanitize_fts()` before hitting FTS5 MATCH — each word is individually double-quoted to prevent operator injection (`NEAR`, `NOT`, `*`, etc).

## Environment Variables

| Var | Default | Purpose |
|-----|---------|---------|
| `OPENMIND_PORT` | `8250` | Server port |
| `OPENMIND_DB` | `~/openmind/openmind.db` | Database path |
| `OPENMIND_UPLOADS` | `~/openmind/uploads` | Image storage dir |
| `OPENAI_API_KEY` | (none) | Required for LLM features |

## Deployment (Live)

- **Server:** c-jfischer3 (Linux, `192.168.129.205`)
- **Port:** 8250
- **URL:** `https://openmind.fahrenheitrequited.dev`
- **Tunnel:** Cloudflare tunnel ID `5382c123-1ceb-4b5f-9200-94974a8f6ee9`, shared with Cortex
- **Tunnel config:** `~/.cloudflared/config.yml` — both cortex (8080) and openmind (8250) ingress rules
- **Project dir on server:** `~/openmind/` (files deployed here by setup.sh)
- **Source dir:** `~/claude/open-mind/` (working copy)
- **Kickoff:** `cd ~/openmind && sh kick-off.sh` (sets OPENAI_API_KEY, starts server + tunnel)
- **Logs:** `~/openmind/openmind.log` (server), `~/openmind/tunnel-om.log` (tunnel), `~/openmind/tunnel.log` (shared tunnel)
- **Python:** `~/miniconda3/bin/python3` (conda base env)
- **Cloudflared binary:** `~/cloudflared`
- **No systemd** — D-Bus unavailable on this box, use kick-off.sh instead
- **Cortex tunnel:** runs on same tunnel ID, do NOT `pkill -f cloudflared` blindly — scope kills to specific processes

## Current State (Feb 27, 2026)

- Server running, frontend accessible at public URL
- DB has ~23 nodes, 28 edges, 5 inbox items
- Embedder loaded (all-MiniLM-L6-v2 on CPU)
- OpenAI API key set (GPT-4o-mini for intent parsing + Vision for image descriptions)
- Tesseract installed (OCR enabled)
- Twilio credentials configured in kick-off.sh
- om-server.py: ~1674 lines, om-viz.html: ~4193 lines
- 5-view architecture: Stream (default), Focus, Map, Semantic, Triage

## What's Done

- [x] Core server (FastAPI + SQLite + embeddings + LLM parsing)
- [x] Force-directed graph frontend (2D canvas)
- [x] Cloudflare tunnel + public URL
- [x] Auto-linking via cosine similarity
- [x] Daily node auto-creation
- [x] Temperature system + decay
- [x] Natural language input (add/search/digest)
- [x] Whisper transcription endpoint (`POST /api/transcribe`)
- [x] Twilio SMS webhook (`POST /api/sms`)
- [x] iOS Shortcut guide (`om-ios-shortcut.md`)
- [x] Tesseract OCR
- [x] 3D toggle (Three.js r128 + OrbitControls)
- [x] Expand/collapse node clusters (union-find)
- [x] Labeled edge types from LLM enrichment
- [x] Daily digest (`POST /api/digest`)
- [x] Image captioning (OpenAI Vision, short + detailed descriptions)
- [x] URL archival (full HTML saved to uploads/, "View Archived Page" in detail panel)
- [x] PDF uploads with text extraction + first-page thumbnail
- [x] PDF download button in detail panel
- [x] Semantic image descriptions (`detailed_image_description()`, stored in metadata)
- [x] 5-view architecture (Stream, Focus, Map, Semantic, Triage)
- [x] Stream view (filter chips, cards, pagination, reverse-chronological)
- [x] Focus view (radial neighbors, SVG lines, history navigation, inline edit, 2-hop toggle)
- [x] Triage view (replaces old process mode — swipe gestures, keyboard shortcuts, undo bar, edit form)
- [x] View switcher with tab navigation + localStorage persistence
- [x] Inbox badge with count + click-to-triage
- [x] Canvas performance skip when DOM views are active
- [x] Similar images endpoint + grid (`GET /api/similar-images/{id}`)

## What's Next / Ideas

- [ ] Scheduled daily digest (cron or background task)
- [ ] Voice input via iOS Shortcut (record → transcribe → NL)
- [ ] Cross-reference with Cortex knowledge store
- [ ] Tag/folder system for manual organization
- [ ] Export/import (JSON, markdown)
- [ ] Multi-user support / sharing

## Quick Commands

```bash
# Start everything
cd ~/openmind && sh kick-off.sh

# Check status
curl -s http://localhost:8250/api/stats | python3 -m json.tool

# View logs
tail -f ~/openmind/openmind.log

# Restart just the server
pkill -f om-server.py
~/miniconda3/bin/python3 -u ~/openmind/om-server.py > ~/openmind/openmind.log 2>&1 &

# Install deps (first time only)
pip install --break-system-packages fastapi uvicorn[standard] sentence-transformers numpy httpx python-multipart
```

## Relationship to Cortex

Open Mind uses the same embedding model (`all-MiniLM-L6-v2`) as the Cortex project. They're complementary — Cortex is a backend knowledge store for AI conversation history, Open Mind is a spatial interface for active thinking and capture. Eventually they could share data or cross-referenceo
