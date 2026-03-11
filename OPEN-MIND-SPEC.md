# Open Mind — Technical Specification

**Current version:** v19  
**File:** `om-viz.html` (~9000 lines, single-file frontend)  
**Server:** `om-server.py` (FastAPI + SQLite + embeddings)  
**Live URL:** `https://openmind.fahrenheitrequited.dev`  
**Host:** `jfischer@ceto.languageweaver.com` → SSH to `c-jfischer3`, port 4640  
**Server port:** 8250  
**Deploy:** `scp -P 4640 om-viz.html jfischer@ceto.languageweaver.com:/home/jfischer/claude/open-mind/`  
**Verify:** Top-left logo shows correct version (e.g. "v19")

---

## Architecture Overview

Single HTML file serves all UI — HTML, CSS, and JS together. Backend is a Python FastAPI server with SQLite storage, sentence-transformer embeddings, and Cloudflare tunnel for public access.

### Views

| View | Description |
|------|-------------|
| **Stream** | Chronological card feed of nodes |
| **Focus** | Poincaré disk hyperbolic graph (primary work area) |
| **Map** | Canvas-based 2D force-directed graph |
| **Semantic** | Force-directed graph with semantic/embedding layout |
| **Triage** | Inbox processing workflow |

### API Endpoints (server)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/graph` | GET | All nodes + edges + clusters |
| `/api/node/:id` | GET | Single node detail |
| `/api/node/:id` | PATCH | Update node |
| `/api/node/:id` | DELETE | Delete node |
| `/api/nodes/:id/similarities` | GET | Similarity scores for layout |
| `/api/add` | POST | Add text node |
| `/api/add-image` | POST | Add image node |
| `/api/add-pdf` | POST | Add PDF node |
| `/api/link` | POST | Create edge |
| `/api/edge/:id` | PATCH/DELETE | Modify/remove edge |
| `/api/enrich-edges` | POST | LLM-enrich edge labels |
| `/api/embedding-layout` | GET | Embedding positions |
| `/api/nl` | POST | Natural language command |
| `/api/transcribe` | POST | Audio transcription |
| `/api/positions` | POST | Save node positions |

**Note:** Frontend calls `/api/graph` (not `/api/nodes` + `/api/edges` separately).

---

## Focus View — Poincaré Disk

The primary visualization. Renders a hyperbolic Poincaré disk where the selected center node is large at center, neighbors arranged around it, and distant nodes shrink toward the boundary.

### Node Hierarchy

| Tier | Description | Visual | Hover behavior |
|------|-------------|--------|----------------|
| **Primary** | Center/focused node | Largest, accent border + glow ring | Shows detail on click |
| **Secondary** | Direct neighbors (up to 7 in graph mode, top-5 in similarity mode) | Medium size, have permanent rail cards + connectors | Hover highlights node + connector + rail card (all three) |
| **Tertiary** | All other visible nodes | Small, shrink toward boundary | Hover expands to primary size, shows ephemeral card + connector leading away from center |

### State Objects

```javascript
const pdState = {
  positions: Map,        // nodeId → [x,y] in Poincaré disk coords (-1..1)
  screenPositions: Map,  // nodeId → {sx, sy, r, vs} in screen pixels
  nodeEls: Map,          // nodeId → <g class="pd-node"> DOM element (cache)
  edgeEls: [],           // [{el, sourceId, targetId}] (cache)
  diskCx, diskCy, diskR, // disk center and radius in screen pixels
  layoutMode,            // 'graph' or 'similarity'
  animRAF,               // current animation frame id (null = idle)
  topSimilar: [],        // top-5 similar node ids (similarity mode)
};

const focusState = {
  active: boolean,
  centerId: string,
  centerData: object,
  selectedNodeId: string,
  history: [],           // navigation history stack
  isEditing: boolean,
  _detailOpen: boolean,
  _hoverTimer: number,
};
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `renderPoincareGraph()` | Full SVG rebuild (after navigation) |
| `_renderBlendedPositions(positions, cx, cy, R, svg, skipDetails)` | Hot render path |
| `_updatePdPositions(positions, cx, cy, R, skipDetails)` | Core position update |
| `updatePoincarePositions(a, cx, cy, R, svg)` | Called during fly-through frames |
| `animateToNode(targetId)` | Möbius fly-through (600ms) + settle (300ms) |
| `buildSummaryRail()` | Rebuilds right-side detail cards |
| `_updateConnectors()` | Redraws SVG Bézier connectors |
| `focusNavigate(id)` | Navigate to new center node |
| `_showFocusDetail()` / `_hideFocusDetail()` | Toggle full-screen detail panel |
| `_showEphemeral(id)` | Show ephemeral card + connector for tertiary node |
| `_removeEphemeral()` | Clean up ephemeral card + connector |
| `_highlightRailCard(id, on)` | Toggle accent border on rail card |
| `_highlightConnector(id, on)` | Toggle accent highlight on Bézier connector |
| `_highlightNode(id, on)` | Toggle accent stroke on SVG node + bring to front |

### Navigation

- **Single click** on node → 250ms debounce → `focusNavigate(id)` → fly-through to center
- **Double click** on node → navigate + open detail panel
- **Single click center** → open detail panel
- **Double click background** → go back in history
- **ESC from detail** → back to focus graph
- **L key** → toggle graph/similarity mode

### Node Sizing

Center node and hover target size: `Math.min(88, diskR * 0.35)` — scales with disk radius, caps at 88px.

This value is applied consistently across:
- Initial render (`renderPoincareGraph`)
- Position updates (`_updatePdPositions`)
- Wheel zoom handler
- Pinch zoom handler
- Hover animation system

Non-center nodes: `MIN_R=3` to `MAX_R=44`, scaled by `visualScale(pos)^0.6`.

### Hover System (Focus View)

**Implementation:** Position-based hit testing via `mousemove` on the SVG.

Instead of relying on DOM `mouseenter`/`mouseleave` (unreliable with overlapping SVG elements), the system:

1. On every `mousemove`, scans all node centers and finds the closest to the cursor
2. If closest node is different from current hover → switches hover
3. If no node is close enough → clears hover

This solves the "buried neighbor" problem: when a node expands to 88px and covers tiny nearby nodes, moving the mouse toward a hidden node's center still triggers its hover. The `mouseenter` DOM event is NOT used (it would fight with position-based testing since it fires on the topmost DOM element, not the closest center).

**Hover effects:**
- Node grows to center-node size (88px max) via exponential smoothing (factor 0.25)
- Only the currently hovered node is brought to front in SVG z-order (shrinking nodes stay where they are)
- For secondary (rail) nodes: rail card border, connector, and node all highlight in accent
- For tertiary nodes: ephemeral card + Bézier connector appear after 180ms delay

**Cleanup:** Hover clears when mouse is >2.5× the animated radius from the node center, or when mouse leaves the SVG entirely (safety net).

### Ephemeral Cards (Tertiary Hover)

Appear when hovering a node that doesn't have a permanent rail card.

- **Position:** Offset from node along radial direction (away from disk center), gap = `max(sp.r + 12, 24)px`
- **Connector:** Bézier curve from node edge to nearest card edge, following radial direction
- **Connector SVG:** Appended to `#focus-connector-svg`, classed `.focus-ephemeral-connector` so `_updateConnectors()` doesn't wipe it during rail rebuilds
- **Cleanup:** Removed on hover exit, node switch, or view switch

### Summary Rail (Right Side)

Detail cards for center node + top neighbors. Each has a Bézier connector from its node to the card.

- `_railCardState: Map<id, {currentY, targetY, height}>` — animated Y positions
- `_tickRailAnimation()` — exponential smoothing loop
- `_solveRailLayout()` — sorts cards by desired Y, enforces minimum spacing, clamps to rail bounds
- Cards auto-positioned to align vertically with their node's screen Y

**Rail card hover (reverse highlighting):** `mouseenter`/`mouseleave` on each card triggers `_highlightRailCard`, `_highlightConnector`, and `_highlightNode` — all three highlight in accent color simultaneously. Uses `window._focusHighlight*` globals since rail cards are created in `buildSummaryRail()` but highlight functions are scoped inside the hover setup block.

### Zoom

- **Wheel:** 3% per tick (reduced from 8% for smoother feel)
- **Pinch:** Proportional to finger distance
- **Limits:** min 80px, max 80% of available area
- Rail reserve is measured dynamically from actual rail element visibility

---

## Map & Semantic Views — Canvas Hover System

Both views render on `<canvas id="graph-canvas">` with a 2D context.

### Hover Behavior

When hovering a node in Map or Semantic view:

1. **Node expands** to 88px (matching Focus center node size)
2. **Detail card** appears to the right of the node (or left if near viewport edge)
3. **Bézier connector** links the node to the card
4. Card shows: type badge, label, content snippet (80 chars)

**Implementation:**
- `_updateCanvasHoverCard(node, mx, my)` — creates/positions card + connector SVG
- `_removeCanvasHoverCard()` — cleanup
- Card is a `.canvas-hover-card` div appended to `document.body` (fixed position)
- Connector is an SVG element with class `.canvas-hover-connector`

**Cleanup triggers:** Mouse leaves canvas, drag starts, pan starts, view switch.

### Canvas Node Rendering

```javascript
const baseR = n.radius * cam.zoom;
const isHovered = hoveredNode && hoveredNode.id === n.id;
const r = isHovered ? Math.max(baseR, 88) : baseR;
```

**Z-order:** The hovered node is drawn last by reordering the draw array — `nodes.filter(not hovered).concat([hovered])`. This ensures the expanded node always appears on top in the canvas (which has no DOM z-order, only paint order).

---

## CSS / Theming

### CSS Variables

```css
:root {
  --bg: #0a0a0f;  --bg2: #12121a;  --bg3: #1a1a26;
  --surface: #1e1e2e;  --border: #2a2a3e;
  --text: #e0dfe8;  --text-dim: #7a7a92;  --text-muted: #4a4a62;
  --accent: #6ee7b7;  --accent2: #34d399;
  --accent-dim: rgba(110, 231, 183, 0.15);
  --warm: #fbbf24;  --hot: #f97316;  --cool: #60a5fa;
  --font: 'DM Sans', sans-serif;  --mono: 'JetBrains Mono', monospace;
}
body.light-mode {
  --bg: #f4f4f8;  --surface: #ffffff;  --border: #d0d0e0;
  --text: #1a1a2e;  --accent: #059669;
}
```

### Node Type Colors (`TYPE_COLORS`)

| Type | Color |
|------|-------|
| paper | `#fb923c` |
| url | `#38bdf8` |
| image | `#e879f9` |
| daily | `#94a3b8` |
| note | `#6ee7b7` |
| appointment | `#fbbf24` |

### Theme Toggle

- **Dark mode (default):** Yellow filled sun icon on dark blue (#1a3a5c) background
- **Light mode:** White filled moon icon on near-black (#1a1a2e) background
- Icons: 18×18px, filled + stroked
- Hover: subtle scale(1.08)
- State persisted in `localStorage('om-theme')`

---

## Top Bar UI Elements

Left to right:
1. **Logo** — "Open Mind v19"
2. **View tabs** — Stream, Focus, Map, Semantic
3. **Inbox badge** — count of inbox-status nodes
4. **Theme toggle** — sun/moon button
5. **Pull slider** — controls similarity attraction tightness (0.1–1.0), always visible
6. **FPS counter** — updates every 500ms, color-coded (green ≥50, yellow ≥30, red <30)
7. **Stats bar** — "N nodes · M edges" + Enrich button

## Filter Bar (Bottom)

Fixed-position bar at `bottom: 96px` (above input area at `bottom: 24px`), `z-index: 201`, visible in **all views** (Stream, Focus, Map, Semantic). Two rows:

### Row 1: Temporal Filter
- **Label:** "Temporal" in accent color (70px fixed width)
- **Single track, two handles:** Left handle = start time, right handle = end time
- **Logarithmic mapping:** Recent time gets more slider space (power curve K=2.5). Rightmost 50% of slider covers ~18% of time range, making it easy to fine-tune recent dates. Formula: `ms = maxMs - span * (1 - pos)^K`
- **Date tick marks:** Auto-generated along track (weekly for <60 day spans, monthly otherwise), positioned using the same log mapping so they show where dates actually fall on the slider
- **Implementation:** Two overlapping `<input type="range">` (0–1000) on same track div. Track and fill are background divs. Inputs have `pointer-events:none` but thumbs have `pointer-events:all` via `::-webkit-slider-thumb` / `::-moz-range-thumb`
- **Fill bar:** Green bar between the two handles showing selected range (`left: lo%`, `right: (100-hi)%`)
- **Date labels:** Left shows start date, right shows end date, update live on drag
- **Presets:** all / month / week / today — set both handles programmatically
- **Range padding:** Min date extended to at least 90 days before earliest node for slider resolution
- **Filtering:** Nodes outside the time window dimmed to 12% opacity

### Row 2: Semantic Filter
- **Label:** "Semantic" in accent color (70px fixed width)
- **Input:** Full-width text field, filters nodes by content match (label, content, type, description, tags)
- **Behavior:** 200ms debounce, ESC to clear, case-insensitive substring match
- **Filtering:** Non-matching nodes dimmed to 12% opacity

### Implementation
- `_timeFilter` global: `{ startMs, endMs, minMs, maxMs, _hiddenIds: Set }`
- `_semanticQuery` string for text filter
- `_initFilters()` — called from `loadGraph()` and `enterFocusView()`, guarded by `_filterInited` to attach listeners only once
- `_applyFilters()` — single function combining temporal + semantic, builds unified `hiddenIds` Set, applies to:
  - **Focus:** SVG node `g.style.opacity`
  - **Map/Semantic:** Canvas `ctx.globalAlpha` via `_timeFilter._hiddenIds`
  - **Stream:** Card opacity + DOM reorder (matching cards on top, dimmed cards below)
- **Node date field:** API returns `n.created` (ISO string), NOT `created_at`
- Two plain native range inputs overlaid on one track — `pointer-events:none` on input, `pointer-events:all` on thumb only. No custom drag code needed.

---

## HTML Structure (Focus View)

```html
<div id="focus-layout" class="focus-layout">
  <div class="focus-graph-area" id="focus-graph-area">
    <svg id="pd-svg">
      <circle class="pd-boundary" />
      <circle class="pd-center-glow" />
      <g id="pd-edges">
        <path class="pd-edge" />...
      </g>
      <g id="pd-nodes">
        <g class="pd-node" data-id="X" transform="translate(sx,sy)">
          <defs><clipPath id="clip-X"><rect /></clipPath></defs>
          <rect />          <!-- background + border -->
          <image clip-path="url(#clip-X)" />  <!-- thumbnail -->
          <rect />          <!-- glow ring (center node only) -->
          <text />          <!-- label (text-only nodes) -->
        </g>
      </g>
    </svg>
    <svg id="focus-connector-svg" />   <!-- Bézier connectors (rail + ephemeral) -->
    <div id="focus-summary-rail" />    <!-- detail cards -->
  </div>
  <div id="focus-center" />            <!-- full-screen detail panel -->
</div>

<!-- Global elements (visible in all views) -->
<div id="filter-bar" />               <!-- Temporal + Semantic filters -->
<div id="input-area" />               <!-- Command input + hint chips -->
```

---

## Coordinate Systems

- **Poincaré coords:** `[x, y]` ∈ (-1, 1). Center = [0,0].
- **Screen coords:** `sx = diskCx + pos[0] * diskR`, `sy = diskCy + pos[1] * diskR`
- `sp.sx/sy` are **graphArea-relative**, NOT window coords
- Connector SVG: `viewBox="0 0 gaRect.width gaRect.height"`
- Rail Y offsets: `railRect.top - gaRect.top`
- Canvas hover cards: fixed positioning (window coords)

---

## Performance Architecture

- **DOM element cache** (`pdState.nodeEls`, `pdState.edgeEls`) built after every `svg.innerHTML` rebuild
- **During drag/momentum:** only `transform="translate(x,y)"` updated (GPU composited), sizes/opacities/edges skipped (`skipDetails=true`)
- **Sizes/edges:** only on settle (after animation stops)
- **Hover animation:** separate `requestAnimationFrame` loop (`_animateHoverSizes`), exponential smoothing factor 0.25

---

## Phone Detection

```javascript
const isPhone = matchMedia('(max-width:480px) and (pointer:coarse)').matches;
```

On phone:
- Only Focus view available (auto-enters on load)
- Summary rail hidden (`display: none !important`)
- Connector SVG hidden
- Touch handlers: tap = navigate, double-tap = navigate + detail, long-press ignored
- Drag/pan via touch with momentum

---

## SMS Input — Type Classification

Twilio webhook at `/api/sms`. Incoming SMS/MMS is processed and added to the knowledge graph.

### Flow
1. **Media (images/PDFs):** Downloaded, captioned/OCR'd, saved as `image` or `paper` nodes. SMS body text becomes the user description.
2. **Text-only:** Routed through `api_nl` → `parse_intent()` which uses LLM (gpt-4o-mini) to classify.

### LLM Classification
`parse_intent()` classifies into node types including the four special types:

| Type | Detection Signals | SMS Reply Style |
|------|------------------|-----------------|
| **chore** | Actionable: buy, fix, call, clean, errands | Brief: "Added. (N open chores)" |
| **diary** | Reflections, feelings, experiences, "today I..." | Warm minimal: "Noted." |
| **appointment** | Date/time references, meetings, scheduled events | Confirms parsed time: "Got it: Dentist, Tue @ 3pm" |
| **idea** | What-if, concepts, hypotheses, creative thoughts | Encouraging: "Captured! Relates to [topic]." |
| **note** | General capture, doesn't fit above | Simple: "Added to your mind map." |

### Type Schemas (metadata extraction)
After classification, `extract_type_fields()` pulls structured metadata:

| Type | Fields |
|------|--------|
| chore | `task_status` (todo/in-progress/done), `priority` (low/med/high), `depends_on` |
| diary | `mood` (great/good/okay/rough/bad), `location` |
| appointment | `appt_date`, `appt_status` (upcoming/completed/cancelled), `location`, `attendees`, `recurring` |
| idea | `idea_status` (raw/developing/validated/implemented/abandoned), `domain` |

### Response Enrichment
SMS responses include context: chore replies mention open chore count, appointment replies mention upcoming count.

---

## Known Issues / TODO

- [ ] Digest/daily nodes have no thumbnail — expand to large black square with small text. Need better rendering for text-only node expansion (formatted preview? colored card?)
- [ ] Ephemeral cards may still clip if node is very close to graph area edge
- [ ] Phone landscape with rail visible doesn't shift disk center (rail hidden on phone via CSS)

---

## Stream View — Type-Specific Rendering

When a type filter chip is active, Stream switches from the default card grid to a type-appropriate layout. Clicking any item selects it in the detail panel (right side) via `selectStreamCard()`.

### Filter Chips (top of Stream)
All | **Ideas** | **Chores** | **Appointments** | **Diary** | Inbox | URLs | Images | Papers | Today

### Renderers

| Filter | Layout | Sort Order | Special Features |
|--------|--------|------------|------------------|
| **Diary** | Journal — date headers, time + full text per entry | Chronological (oldest first) | Groups by day, includes `daily` type nodes. Inline edit (✎ → textarea), delete (✕). |
| **Appointments** | Agenda — large day number + month, label, time/detail | By due_date, upcoming first | Status: upcoming/completed/cancelled. ✓ marks done, ✕ cancels. "Hide past" toggle. Inline label edit. Shows location. |
| **Chores** | Todo list — checkbox + label + dependency + priority | Incomplete first, then by created | Checkbox toggles done (PATCH). "Hide done" toggle. Progress "N of M done". Inline label edit. Delete. |
| **Ideas** | Numbered list — number badge, label, snippet, tags | By created DESC | Shows connection count, domain tag. Inline label edit. Delete. |
| **All / other** | Default card grid | By created DESC | Existing Monopoly-style cards |

All four views have:
- **Inline edit** (✎ button) — replaces label/text with an input field, saves on blur or Enter, cancels on Escape
- **Delete** (✕ button) — confirms then DELETEs node from server and removes from local state
- **Click anywhere** opens in detail panel

### Appointment Status
| Status | Visual | Actions |
|--------|--------|---------|
| `upcoming` | Normal | ✓ complete, ✎ edit, ✕ cancel |
| `completed` | Faded + "done" badge (green) | ✎ edit |
| `cancelled` | Faded + "cancelled" badge (red, strikethrough) | ✎ edit |

### Chore Status
| Status | Visual | Actions |
|--------|--------|---------|
| `todo` | Normal, empty checkbox | Click checkbox → done, ✎ edit, ✕ delete |
| `in-progress` | Normal | Click checkbox → done |
| `done` | Faded, strikethrough, ✓ in checkbox | Click checkbox → todo |

### Node Types
| Type | Color (TYPE_COLORS) | Color (MONOPOLY) |
|------|-------------------|------------------|
| chore | `#f472b6` | `#E67E22` |
| diary | `#60a5fa` | `#3498DB` |
| idea | `#a78bfa` | `#F1C40F` |
| appointment | `#fbbf24` | `#3498DB` |

### Implementation
- `renderStreamCards()` dispatches to `renderDiaryList()`, `renderAppointmentList()`, `renderChoreList()`, or `renderIdeaList()` when the corresponding filter is active
- Each renderer creates a `.typed-list` container with type-specific item elements
- `filterStreamNodes()` applies type-specific sort orders (diary=chronological, appointments=by due date, chores=incomplete first)
- Chore checkbox toggle sends `PATCH /api/node/:id` with updated `meta.task_status`

---

## Common Patterns

**Adding a focus feature:** Init after `initPoincareLayout()` call in `enterFocusView()`

**CSS:** Always use `var(--accent)`, `var(--surface)`, `var(--text)` — never hardcoded colors

**Hot path (60fps):** Edit `_updatePdPositions()` — no DOM queries, no layout reads

**Version bump:** Find `Open Mind <span ...>vN</span>` and increment

**Typical session:**
1. `cp /mnt/user-data/outputs/om-viz.html /home/claude/om-viz.html`
2. Edit with `str_replace` tool
3. Bump version
4. `cp /home/claude/om-viz.html /mnt/user-data/outputs/om-viz.html`
5. Present file → user SCPs → hard reload
