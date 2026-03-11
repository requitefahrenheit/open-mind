# Open Mind — PKM Feature Spec

## Overview

Seven features to transform Open Mind from a capture-and-display system into an active thinking tool. These are prioritized by impact but can be built in any order. Each section is self-contained with API changes, frontend changes, and data model changes.

The core insight: the graph is plumbing. These features make the plumbing *do work* — synthesizing, resurfacing, structuring, and prompting the user to think, not just store.

-----

## 1. Ask Your Knowledge Base

**What:** Natural language Q&A over your own nodes. Instead of search results, you get a synthesized answer grounded in your captured knowledge.

**Why:** This is the single highest-impact feature. It turns passive storage into an active thinking partner. Every other PKM tool that added this (Mem.ai, Saner.ai, Heptabase’s chat) saw it become the most-used feature overnight.

**User experience:**

- In the input bar, type a question: “What do I know about embedding models?”
- Open Mind runs semantic search against all nodes
- Top 10-15 relevant nodes are assembled as context
- GPT-4o-mini receives: system prompt (“Answer using ONLY the provided knowledge base entries. Cite node labels when referencing specific entries. If the knowledge base doesn’t contain relevant info, say so honestly.”) + the node contents + the user’s question
- Response appears in a chat-style overlay panel, with cited node labels as clickable links that open Focus view
- The response is NOT saved as a node by default — it’s ephemeral. But a “Save as note” button lets the user capture the synthesis if it’s valuable.

**API changes:**

```
POST /api/ask
Body: { "question": "What do I know about embedding models?" }

Response: {
  "answer": "Based on your notes, you've explored embedding models in several contexts...",
  "sources": [
    { "id": "abc123", "label": "all-MiniLM-L6-v2 notes", "relevance": 0.82 },
    { "id": "def456", "label": "Cortex embedding pipeline", "relevance": 0.76 }
  ]
}
```

**Implementation:**

1. Reuse existing `semantic_search()` to find top 15 nodes by cosine similarity to the question embedding
1. Also run FTS search on the question terms, merge results (dedup by ID, take union)
1. Build context string: for each source node, format as `[{label}] (id:{id}): {content[:500]}`
1. Send to GPT-4o-mini with system prompt enforcing grounded answers and citation format
1. Parse response, extract any `[Node Label]` citations, match to source node IDs
1. Return answer + sources array

**Frontend:**

- New overlay panel (similar to search results but styled as a chat bubble)
- Question shown at top, answer below with markdown rendering
- Cited node labels are highlighted chips — click to open Focus view
- “Save as note” button at bottom creates a new node of type `note` with the synthesis
- “Ask follow-up” input at bottom for multi-turn conversation (append previous Q&A to context)
- Keyboard shortcut: `?` opens ask mode (input bar gets “Ask your knowledge…” placeholder)

**LLM prompt template:**

```
You are an assistant helping a user query their personal knowledge base.
Answer the question using ONLY the knowledge base entries provided below.
When referencing specific entries, cite them as [Entry Label].
If the knowledge base doesn't contain enough information, say so.
Be concise and direct. Synthesize across entries when relevant.
Do not make up information not present in the entries.

KNOWLEDGE BASE:
{formatted_nodes}

QUESTION: {user_question}
```

**Multi-turn:** Keep a conversation history in the frontend (array of {role, content}). On follow-up questions, send the full conversation history + the same source nodes (or re-search for the new question and merge source sets).

-----

## 2. LLM Edge Enrichment

**What:** When auto-linking creates an edge, also ask the LLM to describe the *specific* relationship between the two nodes. Replace “related: 0.47” with “contradicts longevity claim” or “supports embedding approach.”

**Why:** “Related” edges are noise. Labeled edges are signal. This is the difference between a pile of connected dots and a graph that tells a story. When you open Focus view, seeing “contradicts” vs “supports” vs “is part of” on each connection line transforms comprehension.

**Data model changes:**

Add to `edges` table:

```sql
ALTER TABLE edges ADD COLUMN relationship_type TEXT DEFAULT 'related';
-- Values: related, supports, contradicts, extends, is-part-of, inspired-by, 
--         similar-to, implements, questions, summarizes, example-of
ALTER TABLE edges ADD COLUMN relationship_description TEXT;
-- Free-text: "Both discuss transformer architecture but reach opposite conclusions"
```

**API changes:**

Modify the auto-linking flow in `create_node()`:

1. After computing top 5 similar nodes and creating edges…
1. For each new edge, fire an async LLM call (don’t block node creation)
1. LLM receives both nodes’ content and classifies the relationship
1. Update the edge with relationship_type and relationship_description

Background enrichment endpoint for existing edges:

```
POST /api/enrich-edges
Body: { "node_id": "abc123" }  // optional, enriches all edges for this node
// or no body = enrich all un-enriched edges

Response: { "enriched": 5, "skipped": 2 }
```

**LLM prompt template:**

```
Given two knowledge base entries, classify their relationship.

ENTRY A [{label_a}]: {content_a[:300]}
ENTRY B [{label_b}]: {content_b[:300]}

Respond in JSON:
{
  "relationship_type": one of: "supports", "contradicts", "extends", "is-part-of", "inspired-by", "similar-to", "implements", "questions", "summarizes", "example-of", "related",
  "description": "2-8 word description of the specific relationship",
  "confidence": 0.0-1.0
}

Choose "related" only if no more specific type fits.
The description should be specific to THESE entries, not generic.
```

**Frontend changes:**

- Focus view: render `relationship_description` on each connection line (or on hover)
- Focus view: color-code connection lines by relationship_type (green=supports, red=contradicts, blue=extends, etc.)
- Map/Semantic views: show relationship_description on hover over edges
- Detail panel: list connections with their relationship descriptions, not just labels
- Edge editing: let users manually set/override relationship_type via dropdown in Focus view

**Batch enrichment:** Add a “Enrich all connections” button in settings or as a `/cmd` command that processes all edges with `relationship_type = 'related'` through the LLM. Rate limit to ~2 requests/second to stay within API limits. Show progress bar.

-----

## 3. Resurface Forgotten Nodes

**What:** A daily prompt that surfaces nodes you’ve forgotten about. Low-temperature, unvisited nodes get a second chance. “You saved this 3 weeks ago and haven’t touched it. Still relevant?”

**Why:** The #1 failure mode of every PKM system is: stuff goes in and never comes back out. This closes the loop. Temperature decay already identifies forgotten nodes — now use that signal.

**Implementation — two surfaces:**

### 3a. Rediscovery Feed (passive)

Add a “Rediscovery” section to Stream view — 3 cards at the top, styled differently (subtle highlight or “From your archive” label). Selected by:

1. `temperature < 0.5` (cold)
1. `status = 'permanent'` (not inbox junk)
1. `last_visited` more than 7 days ago
1. Random selection weighted by *inverse* temperature (colder = more likely to surface)
1. Exclude nodes surfaced in the last 3 days (track in a `resurfaced_at` column or in-memory set)

No new API endpoint needed — the `/api/graph` response already includes temperature and last_visited. Frontend can compute this client-side, or add:

```
GET /api/resurface?count=3
Response: [{ node objects }]
```

### 3b. Daily Spark (active, via digest)

Extend the daily digest feature. When the user requests a digest (or on a scheduled cron), include a “Rediscovery” section:

```
You also saved these items recently but haven't revisited them:
• [Node Label 1] — saved 2 weeks ago. Connection to [Other Node] via "extends".
• [Node Label 2] — saved 3 weeks ago. No connections yet — consider linking it.
• [Node Label 3] — saved 10 days ago. Similar to [Recently Added Node].
```

The LLM can optionally generate a one-line “why this might still matter” for each resurfaced node based on its content and the user’s recent activity.

### 3c. Serendipity Pairs (advanced)

Pick 2 nodes with *moderate* cosine similarity (0.3-0.5 range — not obvious matches, not total misses) and prompt the user: “Do these connect? How?”

```
GET /api/serendipity
Response: {
  "node_a": { ... },
  "node_b": { ... },
  "similarity": 0.38,
  "llm_prompt": "These two entries share some thematic overlap around learning systems. Is there a connection worth making?"
}
```

Frontend: Show as a card in Stream or as a notification. User can: “Link them” (opens manual link dialog with pre-filled relationship), “Dismiss”, or “Not related.”

**Data model changes:**

```sql
ALTER TABLE nodes ADD COLUMN resurfaced_at TEXT;  -- ISO datetime, last time shown in rediscovery
```

-----

## 4. Multiple Named Canvases (Workspaces)

**What:** Instead of one global Map view, create multiple named spatial canvases. Each canvas contains a subset of nodes, arranged independently. A node can appear on multiple canvases. Deleting a canvas doesn’t delete the nodes.

**Why:** This is Heptabase’s core insight. One global canvas becomes unusable fast. Named canvases let you create focused thinking spaces: “Longevity Research”, “Open Mind Architecture”, “Q1 Planning.” Each is a context-specific spatial arrangement of relevant nodes.

**Data model changes:**

New table:

```sql
CREATE TABLE canvases (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  color TEXT,  -- accent color for tab/chip
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE canvas_nodes (
  canvas_id TEXT REFERENCES canvases(id) ON DELETE CASCADE,
  node_id TEXT REFERENCES nodes(id) ON DELETE CASCADE,
  x REAL,  -- position on THIS canvas (independent of other canvases)
  y REAL,
  pinned INTEGER DEFAULT 0,
  added_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (canvas_id, node_id)
);
```

The existing `nodes.x` and `nodes.y` columns become the “default” Map positions. Canvas-specific positions are in `canvas_nodes`.

**API changes:**

```
GET    /api/canvases                    -- list all canvases
POST   /api/canvases                    -- create canvas { name, description, color }
PATCH  /api/canvases/{id}               -- update canvas metadata
DELETE /api/canvases/{id}               -- delete canvas (nodes survive)
GET    /api/canvases/{id}/nodes         -- get nodes + canvas-specific positions
POST   /api/canvases/{id}/nodes         -- add nodes to canvas { node_ids: [] }
DELETE /api/canvases/{id}/nodes/{node_id} -- remove node from canvas
POST   /api/canvases/{id}/positions     -- batch save positions for this canvas
```

**Frontend:**

- Map view gets a canvas selector: dropdown or horizontal tabs showing canvas names
- “Default Map” is the existing global canvas (uses nodes.x, nodes.y)
- Named canvases load positions from canvas_nodes table
- “Add to canvas” action in Focus view and right-click context menu on nodes
- Drag a node from Stream/search results onto a canvas tab to add it
- “New Canvas” button creates an empty canvas, opens it, lets you name it
- Canvas nodes render the same way as Map nodes (force-directed for unpinned, fixed for pinned)
- Edges between nodes on the same canvas are shown; edges to nodes NOT on the canvas are shown as faded stubs with a “→ [label]” indicator (click to add that node to canvas)
- Canvas color shows as accent on the tab

**Smart canvas suggestions:**
When creating a new canvas, offer to auto-populate it:

- “Add all nodes tagged [type]”
- “Add all nodes connected to [selected node]”
- “Add all nodes from the last week”

-----

## 5. Chat With Selection

**What:** Select a set of nodes (or an entire canvas), open a chat panel, and have a conversation with the AI about just that subset. “Summarize what I’ve collected.” “What’s missing?” “Write a draft based on these.”

**Why:** This turns the graph from a storage system into a production tool. Capture → organize → synthesize → output. The chat-with-selection step is where fragments become essays, plans, or decisions.

**User experience:**

1. **From a canvas:** Multi-select nodes (shift+click, or lasso), then click “Chat about these” button that appears
1. **From Focus view:** “Chat about this + connections” button sends the center node + all neighbors to chat
1. **From Stream view:** Checkbox mode — select multiple cards, “Chat about selected” button appears in a floating action bar
1. **From a search result:** “Chat about these results” button

**API changes:**

```
POST /api/chat
Body: {
  "node_ids": ["abc123", "def456", "ghi789"],
  "message": "Summarize what I've collected here and identify gaps",
  "history": [  // optional, for multi-turn
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}

Response: {
  "reply": "Based on these 5 entries, you've been exploring...",
  "suggestions": [  // optional follow-up prompts
    "Want me to draft an outline from these notes?",
    "Should I identify contradictions between these entries?"
  ]
}
```

**Frontend:**

- Chat panel slides in from the right (similar to detail panel)
- Selected nodes shown as chips at the top of the panel (removable)
- Chat history below
- Input bar at bottom of panel
- Quick action buttons above input: “Summarize”, “Find gaps”, “Draft outline”, “Find contradictions”, “Suggest connections”
- “Save response” button on any assistant message — creates a new note node, auto-linked to all source nodes
- “Add to canvas” button — if chatting from a canvas context, add the saved note to that canvas

**LLM system prompt:**

```
You are helping a user think through a collection of their personal knowledge base entries.
The user has selected specific entries for you to work with.
Ground your responses in the provided entries. Cite entries as [Entry Label] when referencing them.
Be a thinking partner: synthesize, identify patterns, spot gaps, suggest connections.
When asked to draft or write, use the entries as source material.

SELECTED ENTRIES:
{formatted_nodes_with_full_content}
```

**Quick actions expand to pre-filled prompts:**

- “Summarize” → “Give me a concise summary of what these entries cover and how they relate.”
- “Find gaps” → “What topics or questions are implied by these entries but not directly addressed? What’s missing?”
- “Draft outline” → “Create a structured outline for a document/essay based on these entries.”
- “Find contradictions” → “Do any of these entries contradict each other? Where do they disagree?”
- “Suggest connections” → “What unexpected connections exist between these entries? What patterns emerge?”

-----

## 6. Structured Type Fields

**What:** Different node types have different metadata fields. A paper has authors, journal, year. A person has company, role, email. A task has priority, due date, status. These fields are queryable and displayable.

**Why:** Right now a “paper” and a “note” are the same data structure with a different color. Structured fields make nodes useful as *records*, not just text blobs. You can query “show me all papers from 2024” or “list tasks by priority.”

**Data model:**

Use the existing `metadata` JSON column. Define type schemas:

```python
TYPE_SCHEMAS = {
    "paper": {
        "authors": {"type": "text", "label": "Authors"},
        "journal": {"type": "text", "label": "Journal/Source"},
        "year": {"type": "number", "label": "Year"},
        "doi": {"type": "url", "label": "DOI"},
        "status": {"type": "select", "label": "Status", "options": ["to-read", "reading", "read", "cited"]},
        "rating": {"type": "number", "label": "Rating (1-5)"},
    },
    "task": {
        "priority": {"type": "select", "label": "Priority", "options": ["low", "medium", "high", "urgent"]},
        "due_date": {"type": "date", "label": "Due Date"},
        "status": {"type": "select", "label": "Status", "options": ["todo", "in-progress", "blocked", "done"]},
        "assigned_to": {"type": "text", "label": "Assigned To"},
    },
    "person": {
        "company": {"type": "text", "label": "Company"},
        "role": {"type": "text", "label": "Role"},
        "email": {"type": "email", "label": "Email"},
        "phone": {"type": "text", "label": "Phone"},
        "relationship": {"type": "select", "label": "Relationship", "options": ["friend", "colleague", "contact", "mentor", "family"]},
    },
    "project": {
        "status": {"type": "select", "label": "Status", "options": ["ideation", "active", "paused", "completed", "abandoned"]},
        "start_date": {"type": "date", "label": "Start Date"},
        "target_date": {"type": "date", "label": "Target Date"},
        "stakeholders": {"type": "text", "label": "Stakeholders"},
    },
    "url": {
        "site_name": {"type": "text", "label": "Site"},
        "author": {"type": "text", "label": "Author"},
        "published_date": {"type": "date", "label": "Published"},
        "read_status": {"type": "select", "label": "Status", "options": ["unread", "reading", "read", "archived"]},
    },
    "appointment": {
        "date": {"type": "datetime", "label": "Date & Time"},
        "location": {"type": "text", "label": "Location"},
        "attendees": {"type": "text", "label": "Attendees"},
        "recurring": {"type": "select", "label": "Recurring", "options": ["none", "daily", "weekly", "monthly"]},
    },
    "image": {
        "source": {"type": "text", "label": "Source"},
        "subjects": {"type": "text", "label": "Subjects"},
        "location": {"type": "text", "label": "Location"},
    },
    "idea": {
        "status": {"type": "select", "label": "Status", "options": ["raw", "developing", "validated", "implemented", "abandoned"]},
        "domain": {"type": "text", "label": "Domain"},
    },
    "note": {
        # Notes are intentionally unstructured — no required fields
    },
}
```

**API changes:**

```
GET /api/type-schema/{type}  -- returns the field schema for a node type
```

Modify `PATCH /api/node/{id}` to accept metadata field updates:

```json
{ "metadata": { "authors": "Smith et al.", "year": 2024, "status": "reading" } }
```

Modify `GET /api/search` to support field queries:

```
GET /api/search?type=paper&field_status=reading
GET /api/search?type=task&field_priority=high&field_status=todo
```

**Frontend:**

- **Detail panel / Focus view:** Show structured fields below the content. Editable inline. Select fields render as dropdowns. Date fields get a date picker.
- **Stream view:** Show key fields on cards. Tasks show priority badge and due date. Papers show authors and year. URLs show site name.
- **LLM auto-fill:** When creating a node, the LLM can attempt to populate structured fields from the content. If you paste a paper URL, the LLM extracts authors, journal, year into the metadata fields.
- **Table view (new, optional):** For any node type, offer a table/list view that shows nodes as rows with structured fields as columns. Filter and sort by any field. This is how you’d see “all papers sorted by year” or “all tasks by priority.” This is a stretch goal — implement it as a simple sortable table overlay.

**LLM auto-extraction prompt:**

```
Given this content being saved as a "{node_type}" entry, extract structured metadata.

Content: {content[:500]}
URL (if any): {url}

Return JSON with only the fields you can confidently extract:
{schema_fields_for_this_type}

Omit fields you're unsure about. Do not guess.
```

-----

## 7. Chat With Your Whiteboard (Canvas Chat)

**What:** Open a chat panel scoped to a specific canvas. The AI has context of all nodes on that canvas and their spatial arrangement. You can ask it to analyze, synthesize, or extend the canvas.

**Why:** This combines features #4 (canvases) and #5 (chat with selection) into something more powerful — a persistent, context-aware thinking partner for each workspace.

**Depends on:** Feature #4 (canvases) and Feature #5 (chat with selection).

**User experience:**

- When viewing a named canvas in Map view, a “Chat” button appears in the toolbar
- Opens the same chat panel as Feature #5, but pre-loaded with ALL nodes on the canvas
- Chat history is persisted per canvas (stored in canvas metadata or a new table)
- The AI knows the canvas name/description and all node contents
- Special capabilities:
  - “What should I add to this canvas?” → AI suggests nodes from the broader knowledge base that might be relevant (runs semantic search against canvas topic)
  - “Create a node about X” → AI creates a new node and adds it to the canvas
  - “Organize these into groups” → AI suggests spatial groupings and optionally creates section labels

**Data model:**

```sql
CREATE TABLE canvas_chats (
  id TEXT PRIMARY KEY,
  canvas_id TEXT REFERENCES canvases(id) ON DELETE CASCADE,
  role TEXT NOT NULL,  -- 'user' or 'assistant'
  content TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now'))
);
```

**API:**

```
GET  /api/canvases/{id}/chat              -- get chat history
POST /api/canvases/{id}/chat              -- send message
     Body: { "message": "What's missing from this canvas?" }
DELETE /api/canvases/{id}/chat            -- clear chat history
```

**LLM system prompt:**

```
You are a thinking partner helping the user work with their canvas "{canvas_name}".
Description: {canvas_description}

This canvas contains {node_count} entries:
{formatted_nodes}

Help the user think about this collection. You can:
- Summarize and synthesize
- Identify gaps and suggest additions
- Find contradictions or tensions
- Suggest how to organize or group the entries
- Draft outputs based on the canvas contents

When suggesting new entries to add, search the user's broader knowledge base.
Cite entries as [Entry Label] when referencing them.
```

**Frontend:**

- Chat panel identical to Feature #5 but with persistent history
- Additional quick actions: “Suggest additions”, “Identify themes”, “Create outline”
- When AI suggests adding a node from the broader KB, show it as a card with an “Add to canvas” button
- When AI suggests creating a new node, show a preview card with “Create & Add” button

-----

## Implementation Notes

### Priority Order

1. **Ask Your Knowledge Base** (#1) — standalone, no schema changes, highest immediate value
1. **LLM Edge Enrichment** (#2) — one ALTER TABLE, enriches all existing edges
1. **Resurface Forgotten Nodes** (#3) — minimal backend, mostly frontend cards
1. **Multiple Named Canvases** (#4) — new tables, significant frontend work
1. **Chat With Selection** (#5) — builds on #1, adds multi-node context
1. **Structured Type Fields** (#6) — uses existing metadata column, frontend forms
1. **Canvas Chat** (#7) — combines #4 and #5, build last

### Cost Management

Features #1, #2, #5, and #7 all make LLM API calls. Be mindful of costs:

- Edge enrichment (#2): batch process, don’t re-enrich already-enriched edges
- Ask/Chat (#1, #5, #7): only called on user action, not automatic
- Use `gpt-4o-mini` for all of these — fast, cheap, good enough for classification and synthesis
- Cache edge enrichment results — once labeled, an edge doesn’t need re-labeling unless nodes change
- For Ask (#1), limit context to 15 nodes max to keep token count reasonable

### Backward Compatibility

- All new columns use DEFAULT values — existing data works without migration pain
- Structured type fields (#6) use the existing `metadata` JSON column — no schema change needed
- New tables (canvases, canvas_nodes, canvas_chats) are additive
- The existing global Map view becomes “Default Canvas” — no breaking change

### Testing

- Edge enrichment: test with a few edges first, review quality, adjust prompt before batch
- Ask: test with questions you know the answer to from your own nodes
- Resurface: verify temperature decay is actually running and cold nodes exist
- Canvases: test that deleting a canvas doesn’t delete nodes, and that a node on multiple canvases has independent positions

### File Naming

Follow the `om-` prefix convention for any new files. If splitting server code into modules, use `om-enrichment.py`, `om-canvas.py`, etc. But prefer keeping everything in `om-server.py` unless it gets unwieldy.
