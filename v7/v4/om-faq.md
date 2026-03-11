# Open Mind — FAQ & Capabilities Guide

## What is Open Mind?

Open Mind is a personal knowledge workspace that captures ideas, URLs, images, notes, and tasks from multiple input channels and auto-links them by semantic similarity. It combines a force-directed graph visualization with a natural language interface, accessible via web, SMS, and voice.

**Live at:** `https://openmind.fahrenheitrequited.dev`

---

## How do I add things?

### Text / Ideas / Notes
Type anything into the input bar at the bottom and hit Enter. Open Mind's LLM parser will figure out what you mean — whether it's a note, idea, task, or search query.

**Examples:**
- `remember to buy groceries` → creates a task node
- `idea: what if we used embeddings for search` → creates an idea node
- `meeting with Sarah on Friday` → creates an appointment node

### URLs
Paste or type a URL. Open Mind will:
- Fetch the page title, description, and og:image
- Archive the full HTML for offline viewing
- Scrape the page text into the node content
- Generate a thumbnail from og:image or PDF first page

You can also click the **+** button next to the input bar to open the Add dialog with a dedicated URL field.

### Images
- **Drag & drop** an image onto the page
- **Paste** from clipboard (Ctrl+V)
- **Click +** → Choose image or PDF
- Open Mind will OCR the image (Tesseract), generate a short caption and a detailed description (GPT-4o Vision), and create a node with all of that

### PDFs
Upload via the + dialog or drag & drop. Open Mind extracts text, generates a first-page thumbnail, and stores the PDF for download.

### Voice
Click **+** → Record audio. Open Mind transcribes via Whisper and processes the text through the NL pipeline — so you can speak a note, a search query, or a command.

### SMS (Twilio)
Text your thoughts to the configured Twilio number. Supports text and MMS (images). Replies come back via SMS.

### iOS Shortcut
See `om-ios-shortcut.md` for setup. Allows quick capture from the iOS share sheet or a home screen shortcut.

---

## Views

Open Mind has 5 views, switchable via tabs in the top bar:

### Stream (default)
A reverse-chronological card list of all your nodes. Features:
- **Filter chips:** All, Inbox, Ideas, URLs, Images, Papers, Tasks, Today, This Week
- **Cards show:** type badge, label, 2-line preview, thumbnail, relative time, link count
- **Inbox nodes** have a green left border
- **Temperature-based opacity** — cold/old nodes fade slightly
- **Click any card** → opens Focus view
- **Load more** pagination (30 cards per page)

### Focus
A radial detail view centered on one node with its connections radiating outward. Features:
- **Center panel:** full content, image, URL, archive link, PDF download, metadata
- **Radial neighbors:** positioned in a circle with SVG connection lines
- **2-hop toggle:** show connections-of-connections in an outer ring
- **Navigation history:** back button + breadcrumb trail
- **Actions:** Edit (inline), Promote, Star, Link (opens command palette), Show in Map, Delete
- Click any neighbor to navigate to it

### Map
The 2D canvas with saved positions. Nodes stay where you drag them (positions persist to DB). Good for spatial organization — "that idea was bottom-left near the project stuff."

### Semantic
The force-directed physics simulation. Nodes arrange themselves by connection strength — closely related nodes cluster together. Features:
- Repulsion (inverse-square), attraction (spring force on edges), center gravity (temperature-based)
- Double-click a cluster boundary to collapse/expand it
- Drag nodes to pin them; they'll stay put during simulation

### Triage
Inbox processing mode. Presents inbox items one at a time for quick triage:
- **Promote (P):** mark as permanent
- **Star + Promote (F):** favorite and promote
- **Edit (E):** change label, content, type before promoting
- **Skip (S):** leave in inbox for later
- **Delete (D):** remove permanently
- **Swipe gestures** on mobile (right=promote, left=skip, up=delete)
- **Undo bar** appears for 4 seconds after each action
- **Progress bar** shows how far through the inbox you are
- **Auto-suggested connections** shown as chips on each card
- Click the **inbox badge** (top bar) or type `process inbox` to enter

---

## Searching

### Natural Language
Type search queries naturally:
- `find papers about embeddings`
- `show me everything from today`
- `what's related to the ML project?`

Results appear in an overlay. Click any result to open it in Focus view.

### Command Palette
Press `/` to open. Type to fuzzy-search all nodes. Prefix with `+` to quick-add. Prefix with `>` to link nodes.

### Semantic + Full-Text
The `/api/search?q=` endpoint combines embedding-based semantic search with FTS5 full-text search for comprehensive results.

---

## How does auto-linking work?

Every time you add a node, Open Mind:
1. Computes a 384-dimensional embedding using `all-MiniLM-L6-v2`
2. Compares cosine similarity against all existing nodes
3. Creates edges to the top 5 matches above 0.45 similarity
4. Also links to today's daily node

This means your knowledge graph builds itself — related ideas cluster together automatically.

---

## What's the temperature system?

Each node has a temperature (0.1–2.0):
- **New nodes** start warm
- **Temperature decays** ~10% per day
- **Visiting a node** boosts temperature by +0.3
- **Hot nodes** (>1.2) glow brighter and pull toward the center in Semantic view
- **Cold nodes** fade and drift to the periphery

This creates a natural "attention decay" — frequently-visited nodes stay prominent, while forgotten ones fade.

---

## Node Types

| Type | Color | Use Case |
|------|-------|----------|
| note | green | General text, thoughts, observations |
| idea | purple | Creative concepts, brainstorms |
| url | blue | Web links, articles, references |
| paper | orange | Academic papers, research |
| project | green | Project nodes for grouping |
| task | pink | To-dos, action items |
| image | magenta | Photos, screenshots, diagrams |
| appointment | yellow | Calendar events, meetings |
| daily | gray | Auto-created daily summary nodes |

---

## Node Lifecycle

```
inbox → permanent → (temperature decay over time)
```

- New nodes arrive as **inbox** (pulsing ring in canvas views, green border in Stream)
- Promote to **permanent** via Focus view actions or Triage
- Temperature decays hourly; visit nodes to keep them warm

---

## Keyboard Shortcuts

| Key | Context | Action |
|-----|---------|--------|
| `/` | Global | Open command palette |
| `Escape` | Global | Close current panel/view |
| `Tab` | Global | Focus the input bar |
| `P` | Triage | Promote current item |
| `F` | Triage | Star + promote |
| `E` | Triage | Edit current item |
| `S` | Triage | Skip current item |
| `D` | Triage | Delete current item |

---

## Daily Digest

Type `digest` or `summarize today` in the input bar. Open Mind uses GPT-4o-mini to generate a summary of recent graph activity — new nodes, connections, and themes.

---

## Similar Images

When viewing an image node in the detail panel, Open Mind shows semantically similar images based on their Vision-generated descriptions. Useful for finding related screenshots, photos, or diagrams.

---

## Use Cases

### Personal Knowledge Management
Capture ideas, articles, and notes throughout the day. Let auto-linking surface unexpected connections between your thoughts.

### Research
Add papers, URLs, and notes. The semantic graph reveals how concepts relate across different sources. Use the command palette to manually link related items.

### Daily Journaling
Each day gets a daily node. Everything you add auto-links to it, creating a natural daily log. Use the digest feature to summarize your day.

### Inbox Zero for Thoughts
Capture everything quickly (via text, SMS, voice, or paste). Then use Triage view to process your inbox — promote what matters, delete what doesn't.

### Visual Organization
Use Map view to spatially organize your knowledge. Drag related nodes together. Your spatial memory helps you find things later.

### Mobile Capture
Text ideas to the Twilio number from anywhere. Use the iOS Shortcut for quick capture from your phone. Voice record when typing is inconvenient.

### Project Planning
Create project nodes and manually link related ideas, tasks, and references. The Focus view shows everything connected to a project at a glance.

---

## Technical Details

- **Backend:** Python 3, FastAPI, SQLite (WAL mode), uvicorn
- **Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (384-dim, runs on CPU)
- **LLM:** OpenAI GPT-4o-mini (intent parsing, enrichment, digests, image descriptions)
- **OCR:** Tesseract
- **Frontend:** Single HTML file (~4200 lines), vanilla JS, 2D canvas + DOM views
- **Server:** Port 8250, Cloudflare tunnel
- **SMS:** Twilio webhook
- **Voice:** OpenAI Whisper API
