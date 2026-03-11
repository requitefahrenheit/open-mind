#!/usr/bin/env python3
"""
Open Mind — Personal Knowledge Graph Server
=============================================
FastAPI + SQLite + Sentence-Transformers + OpenAI LLM
Serves a force-directed mind map interface.

Run:  python3 om-server.py
Port: 8250 (default)
"""

import os, json, re, uuid, logging, asyncio, threading, base64
import sqlite3, datetime, mimetypes
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from html import escape as html_escape

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Config ──────────────────────────────────────────
PORT = int(os.environ.get("OPENMIND_PORT", 8250))
DB_PATH = os.environ.get("OPENMIND_DB", os.path.expanduser("~/openmind/openmind.db"))
UPLOAD_DIR = os.environ.get("OPENMIND_UPLOADS", os.path.expanduser("~/openmind/uploads"))
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE = os.environ.get("TWILIO_PHONE_NUMBER", "")
EMBED_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.45  # auto-link above this cosine sim
MAX_AUTO_EDGES = 5           # max auto-links per add
LLM_MODEL = "gpt-4o-mini"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("openmind")

# ─── Globals ─────────────────────────────────────────
db: sqlite3.Connection = None
db_lock = threading.Lock()  # Protect SQLite from concurrent access
embedder = None  # SentenceTransformer
_openai_client = None

# ─── Database ────────────────────────────────────────
def init_db():
    global db
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")

    db.executescript("""
    CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        label TEXT,
        node_type TEXT DEFAULT 'note',
        status TEXT DEFAULT 'inbox',
        pinned INTEGER DEFAULT 0,
        starred INTEGER DEFAULT 0,
        x REAL,
        y REAL,
        image_path TEXT,
        url TEXT,
        due_date TEXT,
        temperature REAL DEFAULT 1.0,
        visit_count INTEGER DEFAULT 0,
        last_visited TEXT,
        embedding BLOB,
        metadata TEXT DEFAULT '{}',
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS edges (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        label TEXT DEFAULT 'related',
        weight REAL DEFAULT 1.0,
        auto_created INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
        FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE,
        UNIQUE(source_id, target_id)
    );

    CREATE TABLE IF NOT EXISTS daily_nodes (
        date TEXT PRIMARY KEY,
        node_id TEXT NOT NULL,
        FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
    CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
    CREATE INDEX IF NOT EXISTS idx_nodes_created ON nodes(created_at);
    CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
    CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
    """)

    # FTS for literal search
    try:
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(id, content, label, node_type, tokenize='porter')")
    except:
        pass  # already exists

    db.commit()
    log.info(f"Database ready: {DB_PATH}")


def rebuild_fts():
    """Rebuild the FTS index from nodes table."""
    with db_lock:
        db.execute("DELETE FROM nodes_fts")
        db.execute("INSERT INTO nodes_fts(id, content, label, node_type) SELECT id, content, label, node_type FROM nodes")
        db.commit()


# ─── Embeddings ──────────────────────────────────────
def init_embedder():
    global embedder
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBED_MODEL)
        log.info(f"Embedder loaded: {EMBED_MODEL}")
    except ImportError:
        log.warning("sentence-transformers not installed. Semantic search disabled.")
        embedder = None


def embed_text(text: str) -> Optional[bytes]:
    if embedder is None:
        return None
    vec = embedder.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32).tobytes()


def cosine_sim(a_bytes: bytes, b_bytes: bytes) -> float:
    a = np.frombuffer(a_bytes, dtype=np.float32)
    b = np.frombuffer(b_bytes, dtype=np.float32)
    return float(np.dot(a, b))


def semantic_search(query: str, limit: int = 10, threshold: float = 0.3):
    if embedder is None:
        return []
    q_vec = embedder.encode(query, normalize_embeddings=True).astype(np.float32).tobytes()
    with db_lock:
        rows = db.execute("SELECT id, content, label, node_type, status, embedding FROM nodes WHERE embedding IS NOT NULL").fetchall()
    results = []
    for r in rows:
        sim = cosine_sim(q_vec, r["embedding"])
        if sim >= threshold:
            results.append({
                "id": r["id"],
                "content": r["content"][:200],
                "label": r["label"],
                "node_type": r["node_type"],
                "status": r["status"],
                "score": round(sim, 4)
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


# ─── LLM ─────────────────────────────────────────────
def get_openai():
    global _openai_client
    if _openai_client is None and OPENAI_KEY:
        try:
            import httpx
            _openai_client = httpx.Client(
                base_url="https://api.openai.com/v1",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                timeout=30.0
            )
            log.info("OpenAI client ready")
        except ImportError:
            log.warning("httpx not installed. LLM features disabled.")
    return _openai_client


def llm_chat(system: str, user: str, model: str = None) -> str:
    client = get_openai()
    if not client:
        return ""
    try:
        r = client.post("/chat/completions", json={
            "model": model or LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        })
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error(f"LLM error: {e}")
        return ""


def parse_intent(text: str, recent_context: str = "") -> dict:
    """Use LLM to classify intent: add, search, link, or command."""
    system = """You are the intent parser for Open Mind, a personal knowledge graph.
Given user input, classify it and extract structured data. Respond ONLY with valid JSON.

{
  "action": "add" | "search" | "link" | "digest" | "unknown",
  "content": "the main text content",
  "label": "short label for the node (max 60 chars)",
  "node_type": "idea" | "url" | "paper" | "project" | "appointment" | "note" | "image" | "task",
  "url": "extracted URL if any, else null",
  "due_date": "ISO date if time-sensitive, else null",
  "search_query": "what to search for if action is search",
  "search_filters": {"type": null, "time": null},
  "response_text": "brief natural language response to send back to the user"
}

Rules:
- If the input contains a question or starts with "where", "show", "find", "what", "search" → action=search
- If the input contains a URL → action=add, node_type=url (or paper if arxiv/scholar)
- If the input mentions a date/time/deadline/meeting → add due_date, maybe node_type=appointment
- If the input starts with "link" or "connect" → action=link
- If the input asks for a summary/digest → action=digest
- Otherwise → action=add
- Always generate a good short label
- For search, extract the semantic query and any type/time filters
"""
    ctx = f"\nRecent context:\n{recent_context}" if recent_context else ""
    result = llm_chat(system, text + ctx)
    try:
        # Strip markdown fences if present
        result = re.sub(r'^```json\s*', '', result)
        result = re.sub(r'\s*```$', '', result)
        return json.loads(result)
    except:
        # Fallback: treat as simple add
        return {
            "action": "add",
            "content": text,
            "label": text[:60],
            "node_type": "note",
            "response_text": "Added to your mind map."
        }


# ─── URL Unfurling ───────────────────────────────────
def download_thumbnail(image_url: str) -> str | None:
    """Download an image from URL and save to uploads dir. Returns filename or None."""
    try:
        import httpx
        r = httpx.get(image_url, timeout=10, follow_redirects=True,
                      headers={"User-Agent": "OpenMind/1.0"})
        ct = r.headers.get("content-type", "")
        if not ct.startswith("image/"):
            return None
        if len(r.content) > 2 * 1024 * 1024:  # cap at 2MB
            return None
        ext = ".jpg"
        if "png" in ct: ext = ".png"
        elif "webp" in ct: ext = ".webp"
        elif "gif" in ct: ext = ".gif"
        fname = f"thumb_{gen_id()}{ext}"
        fpath = os.path.join(UPLOAD_DIR, fname)
        with open(fpath, "wb") as f:
            f.write(r.content)
        log.info(f"Downloaded thumbnail: {fname} ({len(r.content)} bytes)")
        return fname
    except Exception as e:
        log.warning(f"Thumbnail download failed for {image_url}: {e}")
        return None


def extract_page_text(html: str) -> str:
    """Extract readable text from HTML, stripping tags/scripts/styles."""
    # Remove script, style, nav, header, footer blocks
    cleaned = re.sub(r'<(script|style|nav|header|footer|aside)[^>]*>.*?</\1>', '', html, flags=re.S | re.I)
    # Remove all remaining tags
    text = re.sub(r'<[^>]+>', ' ', cleaned)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Cap at ~5000 chars for storage
    return text[:5000]


def unfurl_url(url: str) -> dict:
    """Fetch title, description, og:image, page text, and archive full HTML from a URL."""
    try:
        import httpx
        r = httpx.get(url, timeout=10, follow_redirects=True,
                      headers={"User-Agent": "OpenMind/1.0"})
        full_html = r.text
        html = full_html[:50000]  # grab more HTML for text extraction
        title = re.search(r'<title[^>]*>([^<]+)</title>', html, re.I)
        desc = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']', html, re.I)
        og_title = re.search(r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\'](.*?)["\']', html, re.I)
        og_desc = re.search(r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\'](.*?)["\']', html, re.I)
        og_img = re.search(r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\'](.*?)["\']', html, re.I)

        og_image_url = og_img.group(1) if og_img else None
        thumbnail = None
        if og_image_url:
            thumbnail = download_thumbnail(og_image_url)

        page_text = extract_page_text(html)

        # Archive full HTML (up to 1MB)
        html_archive = None
        if len(full_html.encode('utf-8', errors='replace')) <= 1_048_576:
            try:
                archive_fname = f"archive_{gen_id()}.html"
                archive_path = os.path.join(UPLOAD_DIR, archive_fname)
                with open(archive_path, "w", encoding="utf-8") as f:
                    f.write(full_html)
                html_archive = archive_fname
                log.info(f"Archived HTML: {archive_fname} ({len(full_html)} chars)")
            except Exception as e:
                log.warning(f"HTML archive failed for {url}: {e}")
        else:
            log.info(f"HTML too large to archive ({len(full_html)} chars): {url}")

        return {
            "title": (og_title or title).group(1).strip() if (og_title or title) else url,
            "description": (og_desc or desc).group(1).strip() if (og_desc or desc) else "",
            "image": og_image_url,
            "thumbnail": thumbnail,
            "page_text": page_text,
            "html_archive": html_archive
        }
    except Exception as e:
        log.warning(f"Unfurl failed for {url}: {e}")
        return {"title": url, "description": "", "image": None, "thumbnail": None, "page_text": "", "html_archive": None}


# ─── OCR ─────────────────────────────────────────────
def ocr_image(filepath: str) -> str:
    """Run tesseract OCR on an image, return extracted text."""
    try:
        import subprocess
        result = subprocess.run(
            ["tesseract", filepath, "stdout", "--psm", "6"],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip()
    except Exception as e:
        log.warning(f"OCR failed: {e}")
        return ""


def caption_image(image_path: str) -> str:
    """Use OpenAI Vision API to generate a description of an image."""
    client = get_openai()
    if not client:
        return ""
    try:
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".gif": "image/gif", ".webp": "image/webp"}
        mime = mime_map.get(ext, "image/jpeg")

        r = client.post("/chat/completions", json={
            "model": LLM_MODEL,  # gpt-4o-mini supports vision
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in 2-3 sentences. Focus on what's depicted, any visible text, and the overall context. Be concise."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}", "detail": "low"}}
                ]
            }],
            "max_tokens": 200,
            "temperature": 0.3
        })
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"Image captioning failed: {e}")
        return ""


def detailed_image_description(image_path: str) -> str:
    """Use OpenAI Vision API to generate a rich description for semantic matching."""
    client = get_openai()
    if not client:
        return ""
    try:
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                    ".gif": "image/gif", ".webp": "image/webp"}
        mime = mime_map.get(ext, "image/jpeg")

        r = client.post("/chat/completions", json={
            "model": LLM_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Provide a detailed description of this image for a knowledge graph. Include:\n"
                        "1. What is depicted (objects, people, scenes, text)\n"
                        "2. Visual style, colors, composition\n"
                        "3. Potential topics, themes, or categories\n"
                        "4. Any visible text, labels, or data\n"
                        "5. Emotional tone or context if applicable\n"
                        "Be thorough but structured. About 100-150 words."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}", "detail": "low"}}
                ]
            }],
            "max_tokens": 400,
            "temperature": 0.3
        })
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"Detailed image description failed: {e}")
        return ""


# ─── Edge Classification ─────────────────────────────
def classify_edges(source_text: str, targets: list) -> dict:
    """Use LLM to classify relationship types between a new node and its matches.

    Args:
        source_text: The new node's label + content text.
        targets: List of dicts with id, label, content snippet.

    Returns:
        Dict mapping target_id -> edge label string.
    """
    if not targets:
        return {}
    system = """You classify relationships between knowledge graph nodes.
Given a source node and target nodes, classify each relationship as EXACTLY one of:
discusses, extends, supports, contradicts, relates_to, inspired_by, depends_on, alternative_to

Respond ONLY with valid JSON: {"target_id": "label", ...}
Be precise. Use "discusses" for same-topic, "extends" for builds-upon, "supports" for evidence,
"contradicts" for opposing, "inspired_by" for creative influence, "depends_on" for prerequisites,
"alternative_to" for competing approaches, "relates_to" as fallback."""

    target_desc = "\n".join(
        f'- id="{t["id"]}": {t["label"]} — {t["content"][:120]}'
        for t in targets
    )
    user_msg = f'Source node: "{source_text}"\n\nTarget nodes:\n{target_desc}'

    result = llm_chat(system, user_msg)
    if not result:
        return {}
    try:
        # Strip markdown fences if present
        result = re.sub(r'^```json\s*', '', result)
        result = re.sub(r'\s*```$', '', result)
        parsed = json.loads(result)
        # Validate labels
        valid_labels = {"discusses", "extends", "supports", "contradicts",
                        "relates_to", "inspired_by", "depends_on", "alternative_to"}
        return {k: v for k, v in parsed.items() if v in valid_labels}
    except Exception as e:
        log.warning(f"classify_edges parse error: {e}")
        return {}


# ─── Node Operations ─────────────────────────────────
def sanitize_fts(query: str) -> str:
    """Sanitize a query for FTS5 MATCH. Wrap each word in quotes to avoid operator injection."""
    words = re.findall(r'[\w]+', query)
    if not words:
        return '""'
    return ' '.join(f'"{w}"' for w in words)


def gen_id() -> str:
    return uuid.uuid4().hex[:12]


def create_node(content: str, label: str = None, node_type: str = "note",
                status: str = "inbox", url: str = None, due_date: str = None,
                image_path: str = None, x: float = None, y: float = None,
                metadata: dict = None, pinned: bool = False) -> dict:
    """Create a node, compute embedding, auto-link, update FTS."""
    nid = gen_id()
    now = datetime.datetime.utcnow().isoformat()

    if not label:
        label = content[:60].replace('\n', ' ')

    # Embed
    embed_content = f"{label}. {content[:500]}"
    emb = embed_text(embed_content)

    # URL unfurling
    url_meta = {}
    if url:
        url_meta = unfurl_url(url)
        if not label or label == content[:60].replace('\n', ' '):
            label = url_meta.get("title", label)
        # Append scraped page text to content so it's searchable and visible
        page_text = url_meta.get("page_text", "")
        if page_text:
            desc = url_meta.get("description", "")
            scraped = f"{desc}\n\n{page_text}" if desc else page_text
            if content and content != url:
                content = f"{content}\n\n---\n\n{scraped}"
            else:
                content = scraped

    meta = metadata or {}
    meta.update(url_meta)
    # Don't store bulky page_text in metadata (it's already in content)
    meta.pop("page_text", None)

    # Phase 1: Insert node, compute similarity matches (inside lock)
    scored_matches = []
    target_info_for_classify = []
    with db_lock:
        db.execute("""
            INSERT INTO nodes (id, content, label, node_type, status, pinned, url, due_date,
                              image_path, x, y, embedding, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (nid, content, label, node_type, status, int(pinned), url, due_date,
              image_path, x, y, emb, json.dumps(meta), now, now))

        # FTS
        try:
            db.execute("INSERT INTO nodes_fts(id, content, label, node_type) VALUES (?, ?, ?, ?)",
                       (nid, content, label, node_type))
        except:
            pass

        # Find similar nodes for auto-linking
        if emb:
            rows = db.execute("SELECT id, content, label, embedding FROM nodes WHERE id != ? AND embedding IS NOT NULL", (nid,)).fetchall()
            scored = []
            for r in rows:
                sim = cosine_sim(emb, r["embedding"])
                if sim >= SIMILARITY_THRESHOLD:
                    scored.append((r["id"], sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            scored_matches = scored[:MAX_AUTO_EDGES]

            # Collect target info for LLM classification (while still in lock)
            if scored_matches and OPENAI_KEY:
                match_ids = {tid for tid, _ in scored_matches}
                for r in rows:
                    if r["id"] in match_ids:
                        target_info_for_classify.append({
                            "id": r["id"],
                            "label": r["label"] or "",
                            "content": (r["content"] or "")[:200]
                        })

        db.commit()

    # Phase 2: Classify edge labels via LLM (outside lock to avoid blocking)
    edge_labels = {}
    if target_info_for_classify and OPENAI_KEY:
        source_text = f"{label}. {content[:300]}"
        try:
            edge_labels = classify_edges(source_text, target_info_for_classify)
        except Exception as e:
            log.warning(f"classify_edges failed: {e}")

    # Phase 3: Insert edges (re-acquire lock)
    auto_edges = []
    with db_lock:
        for target_id, sim in scored_matches:
            eid = gen_id()
            edge_label = edge_labels.get(target_id, "related")
            try:
                db.execute("""INSERT INTO edges (id, source_id, target_id, label, weight, auto_created)
                             VALUES (?, ?, ?, ?, ?, 1)""",
                           (eid, nid, target_id, edge_label, round(sim, 3)))
                auto_edges.append({"target": target_id, "score": round(sim, 3), "label": edge_label})
            except sqlite3.IntegrityError:
                pass

        # Attach to today's daily node
        today = datetime.date.today().isoformat()
        daily = db.execute("SELECT node_id FROM daily_nodes WHERE date=?", (today,)).fetchone()
        if daily:
            eid = gen_id()
            try:
                db.execute("INSERT INTO edges (id, source_id, target_id, label, weight, auto_created) VALUES (?, ?, ?, 'daily', 0.5, 1)",
                           (eid, daily["node_id"], nid))
            except:
                pass

        db.commit()
    log.info(f"[ADD] {nid}: {label[:50]} ({node_type}) +{len(auto_edges)} auto-links")
    return {
        "id": nid, "label": label, "node_type": node_type, "status": status,
        "auto_edges": auto_edges, "url_meta": url_meta
    }


def ensure_daily_node():
    """Create today's daily node if it doesn't exist."""
    today = datetime.date.today().isoformat()
    with db_lock:
        existing = db.execute("SELECT node_id FROM daily_nodes WHERE date=?", (today,)).fetchone()
        if existing:
            return existing["node_id"]
        nid = gen_id()
        now = datetime.datetime.utcnow().isoformat()
        label = datetime.date.today().strftime("%A, %b %d")
        db.execute("""INSERT INTO nodes (id, content, label, node_type, status, pinned, created_at, updated_at)
                      VALUES (?, ?, ?, 'daily', 'active', 1, ?, ?)""",
                   (nid, f"Daily node for {today}", label, now, now))
        db.execute("INSERT INTO daily_nodes (date, node_id) VALUES (?, ?)", (today, nid))
        db.commit()
    log.info(f"[DAILY] Created: {label}")
    return nid


def get_full_graph():
    """Export entire graph for frontend rendering."""
    with db_lock:
        rows = db.execute("""
            SELECT id, content, label, node_type, status, pinned, starred, x, y,
                   image_path, url, due_date, temperature, visit_count, last_visited,
                   metadata, created_at, updated_at
            FROM nodes ORDER BY created_at DESC
        """).fetchall()

        erows = db.execute("SELECT id, source_id, target_id, label, weight, auto_created FROM edges").fetchall()

    nodes = []
    for r in rows:
        nodes.append({
            "id": r["id"],
            "content": r["content"][:300],
            "label": r["label"],
            "type": r["node_type"],
            "status": r["status"],
            "pinned": bool(r["pinned"]),
            "starred": bool(r["starred"]),
            "x": r["x"],
            "y": r["y"],
            "image": r["image_path"],
            "url": r["url"],
            "due_date": r["due_date"],
            "temperature": r["temperature"],
            "visits": r["visit_count"],
            "meta": json.loads(r["metadata"] or "{}"),
            "created": r["created_at"],
            "updated": r["updated_at"]
        })

    edges = []
    for r in erows:
        edges.append({
            "id": r["id"],
            "source": r["source_id"],
            "target": r["target_id"],
            "label": r["label"],
            "weight": r["weight"],
            "auto": bool(r["auto_created"])
        })

    clusters = compute_clusters()
    return {"nodes": nodes, "edges": edges, "clusters": clusters}


def compute_clusters():
    """Use union-find to detect clusters of tightly connected nodes.

    Queries edges with weight >= 0.55 (tighter than the 0.45 auto-link threshold).
    Returns clusters with >= 3 members.
    """
    with db_lock:
        erows = db.execute(
            "SELECT source_id, target_id FROM edges WHERE weight >= 0.55"
        ).fetchall()

    if not erows:
        return []

    # Union-Find
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for row in erows:
        union(row["source_id"], row["target_id"])

    # Group by root
    groups = {}
    for node_id in parent:
        root = find(node_id)
        groups.setdefault(root, []).append(node_id)

    # Filter to clusters with >= 3 members
    clusters = []
    for i, (root, members) in enumerate(groups.items()):
        if len(members) >= 3:
            clusters.append({
                "id": f"cluster_{i}",
                "members": members,
                "size": len(members)
            })

    return clusters


def update_temperature():
    """Decay temperature for unvisited nodes, boost for recently visited."""
    now = datetime.datetime.utcnow()
    with db_lock:
        rows = db.execute("SELECT id, temperature, last_visited, visit_count FROM nodes").fetchall()
        for r in rows:
            temp = r["temperature"]
            if r["last_visited"]:
                try:
                    last = datetime.datetime.fromisoformat(r["last_visited"])
                    hours_ago = (now - last).total_seconds() / 3600
                    decay = max(0.1, temp * (0.996 ** hours_ago))
                    temp = decay
                except:
                    temp = max(0.1, temp * 0.99)
            else:
                temp = max(0.1, temp * 0.99)
            db.execute("UPDATE nodes SET temperature=? WHERE id=?", (round(temp, 4), r["id"]))
        db.commit()


def visit_node(node_id: str):
    """Record a visit, boost temperature."""
    now = datetime.datetime.utcnow().isoformat()
    with db_lock:
        db.execute("""UPDATE nodes SET visit_count = visit_count + 1,
                      last_visited = ?, temperature = MIN(2.0, temperature + 0.3),
                      updated_at = ? WHERE id = ?""", (now, now, node_id))
        db.commit()


# ─── Daily Digest ────────────────────────────────────
def generate_daily_digest():
    """Generate a daily digest summarizing recent activity. Returns node dict or None."""
    today = datetime.date.today().isoformat()
    now = datetime.datetime.utcnow()
    yesterday = (now - datetime.timedelta(hours=24)).isoformat()

    with db_lock:
        # Check if digest already exists for today
        existing = db.execute(
            "SELECT id FROM nodes WHERE node_type='daily' AND label LIKE 'Digest:%' AND created_at >= ?",
            (today,)
        ).fetchone()
        if existing:
            return None

        # Nodes created in the last 24 hours
        new_nodes = db.execute(
            "SELECT id, label, node_type, status, due_date FROM nodes WHERE created_at >= ? ORDER BY created_at DESC",
            (yesterday,)
        ).fetchall()

        # Edges created in the last 24 hours (with node labels)
        new_edges = db.execute("""
            SELECT e.label as edge_label, e.weight,
                   s.label as source_label, t.label as target_label
            FROM edges e
            JOIN nodes s ON e.source_id = s.id
            JOIN nodes t ON e.target_id = t.id
            WHERE e.created_at >= ?
        """, (yesterday,)).fetchall()

        # Inbox count
        inbox_count = db.execute("SELECT COUNT(*) as c FROM nodes WHERE status='inbox'").fetchone()["c"]

        # Upcoming due dates (next 7 days)
        next_week = (now + datetime.timedelta(days=7)).isoformat()[:10]
        upcoming = db.execute(
            "SELECT label, due_date, node_type FROM nodes WHERE due_date IS NOT NULL AND due_date <= ? AND due_date >= ? ORDER BY due_date ASC",
            (next_week, today)
        ).fetchall()

    # No activity? Skip
    if not new_nodes and not new_edges:
        return None

    # Build context for LLM
    node_lines = [f"- {r['label']} ({r['node_type']}, {r['status']})" for r in new_nodes]
    edge_lines = [f"- {r['source_label']} <-> {r['target_label']} ({r['edge_label']}, weight {r['weight']:.2f})" for r in new_edges]
    due_lines = [f"- {r['label']} due {r['due_date']}" for r in upcoming]

    context = f"""Activity in the last 24 hours:

New nodes ({len(new_nodes)}):
{chr(10).join(node_lines) if node_lines else '(none)'}

New connections ({len(new_edges)}):
{chr(10).join(edge_lines) if edge_lines else '(none)'}

Inbox items: {inbox_count}

Upcoming due dates:
{chr(10).join(due_lines) if due_lines else '(none)'}
"""

    system = """You write concise daily digests for a personal knowledge graph called Open Mind.
Summarize the activity in 3-5 sentences. Mention key themes, notable new connections,
and any upcoming deadlines. Be brief and insightful. Do not use bullet points or headers."""

    summary = llm_chat(system, context)
    if not summary:
        # Fallback if LLM is unavailable
        summary = f"Today: {len(new_nodes)} new nodes, {len(new_edges)} new connections. {inbox_count} items in inbox."
        if upcoming:
            summary += f" {len(upcoming)} upcoming deadlines."

    # Create digest node
    label = f"Digest: {datetime.date.today().strftime('%b %d')}"
    result = create_node(
        content=summary,
        label=label,
        node_type="daily",
        status="active"
    )
    log.info(f"[DIGEST] Generated: {label}")
    return result


# ─── API Models ──────────────────────────────────────
class AddRequest(BaseModel):
    content: str
    label: Optional[str] = None
    node_type: Optional[str] = None
    url: Optional[str] = None
    due_date: Optional[str] = None
    pinned: Optional[bool] = False
    use_llm: Optional[bool] = True

class LinkRequest(BaseModel):
    source_id: str
    target_id: str
    label: Optional[str] = "related"

class UpdateRequest(BaseModel):
    label: Optional[str] = None
    content: Optional[str] = None
    status: Optional[str] = None
    node_type: Optional[str] = None
    pinned: Optional[bool] = None
    starred: Optional[bool] = None
    x: Optional[float] = None
    y: Optional[float] = None
    due_date: Optional[str] = None

class NLQueryRequest(BaseModel):
    text: str
    channel: Optional[str] = "web"  # web, sms, voice

class PositionUpdate(BaseModel):
    id: str
    x: float
    y: float

# ─── Lifespan ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_embedder()
    ensure_daily_node()
    # Rebuild FTS on startup
    try:
        rebuild_fts()
    except:
        pass
    # Start background tasks
    bg_task = asyncio.create_task(background_tasks())
    yield
    bg_task.cancel()
    if db:
        db.close()


# ─── App ─────────────────────────────────────────────
app = FastAPI(title="Open Mind", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Routes ──────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend."""
    html_path = Path(__file__).parent / "om-viz.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Open Mind</h1><p>Frontend not found. Place om-viz.html next to om-server.py.</p>")


@app.get("/api/graph")
async def api_graph():
    """Get full graph data for rendering."""
    data = await asyncio.to_thread(get_full_graph)
    return JSONResponse(data)


@app.post("/api/add")
async def api_add(req: AddRequest):
    """Add a node. Optionally use LLM to parse intent."""
    content = req.content.strip()
    if not content:
        raise HTTPException(400, "Content required")

    node_type = req.node_type
    label = req.label
    url = req.url

    # Extract URL from content if not provided
    if not url:
        url_match = re.search(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
        if url_match:
            url = url_match.group(0)

    # LLM parsing (blocking — run in thread)
    if req.use_llm and OPENAI_KEY:
        parsed = await asyncio.to_thread(parse_intent, content)
        if parsed.get("action") == "search":
            results = await asyncio.to_thread(semantic_search, parsed.get("search_query", content), 5)
            return JSONResponse({
                "action": "search",
                "results": results,
                "response": parsed.get("response_text", "")
            })
        node_type = node_type or parsed.get("node_type", "note")
        label = label or parsed.get("label")
        url = url or parsed.get("url")
        due_date = req.due_date or parsed.get("due_date")
    else:
        due_date = req.due_date

    result = await asyncio.to_thread(
        create_node, content=content, label=label, node_type=node_type,
        url=url, due_date=due_date, pinned=req.pinned
    )
    return JSONResponse(result)


@app.post("/api/add-image")
async def api_add_image(
    file: UploadFile = File(...),
    content: str = Form(""),
    label: str = Form("")
):
    """Add a node with an attached image."""
    # Save file
    ext = Path(file.filename).suffix or ".jpg"
    fname = f"{gen_id()}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    data = await file.read()
    with open(fpath, "wb") as f:
        f.write(data)

    # OCR + Vision captioning + detailed description (parallel)
    ocr_text = ocr_image(fpath)
    caption, detailed_desc = await asyncio.gather(
        asyncio.to_thread(caption_image, fpath),
        asyncio.to_thread(detailed_image_description, fpath)
    )

    full_content = content
    if caption:
        full_content = f"{content}\n\n[Caption]: {caption}" if content else f"[Caption]: {caption}"
    if detailed_desc:
        full_content = f"{full_content}\n\n[Description]: {detailed_desc}" if full_content else f"[Description]: {detailed_desc}"
    if ocr_text:
        full_content = f"{full_content}\n\n[OCR]: {ocr_text}" if full_content else f"[OCR]: {ocr_text}"

    if not label:
        label = caption[:60] if caption else (content[:60] if content else (ocr_text[:60] if ocr_text else file.filename))

    meta = {}
    if detailed_desc:
        meta["image_description"] = detailed_desc
    if content.strip():
        meta["user_notes"] = content.strip()

    result = create_node(
        content=full_content, label=label, node_type="image",
        image_path=fname, metadata=meta
    )
    return JSONResponse(result)


@app.post("/api/add-pdf")
async def api_add_pdf(
    file: UploadFile = File(...),
    content: str = Form(""),
    label: str = Form("")
):
    """Add a node from a PDF — extract text, snapshot first page as thumbnail."""
    ext = Path(file.filename).suffix or ".pdf"
    fname = f"{gen_id()}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    data = await file.read()
    with open(fpath, "wb") as f:
        f.write(data)

    pdf_text = ""
    thumb_fname = None
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(fpath)
        # Extract text from all pages (cap at 5000 chars)
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        pdf_text = "\n".join(pages_text)[:5000]
        # Snapshot first page as thumbnail
        if len(doc) > 0:
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 1.5x zoom
            thumb_fname = f"thumb_{gen_id()}.png"
            thumb_path = os.path.join(UPLOAD_DIR, thumb_fname)
            pix.save(thumb_path)
            log.info(f"PDF thumbnail: {thumb_fname}")
        doc.close()
    except ImportError:
        log.warning("PyMuPDF not installed — PDF text/thumbnail extraction skipped")
    except Exception as e:
        log.warning(f"PDF processing error: {e}")

    full_content = content
    if pdf_text:
        full_content = f"{content}\n\n{pdf_text}" if content else pdf_text

    if not label:
        label = Path(file.filename).stem[:60] or "PDF document"

    meta = {"pdf_file": fname}
    if thumb_fname:
        meta["thumbnail"] = thumb_fname
    if content.strip():
        meta["user_notes"] = content.strip()

    result = create_node(
        content=full_content, label=label, node_type="paper",
        image_path=thumb_fname, metadata=meta
    )
    return JSONResponse(result)


@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    """Transcribe audio via OpenAI Whisper API, then route through NL pipeline."""
    audio_bytes = await file.read()
    filename = file.filename or "audio.webm"
    content_type = file.content_type or "audio/webm"

    client = get_openai()
    if not client:
        raise HTTPException(503, "OpenAI client not configured")

    # Call Whisper API (synchronous httpx client, run in thread)
    def do_transcribe():
        r = client.post(
            "/audio/transcriptions",
            files={"file": (filename, audio_bytes, content_type)},
            data={"model": "whisper-1"},
            timeout=60.0
        )
        r.raise_for_status()
        return r.json().get("text", "")

    try:
        transcript = await asyncio.to_thread(do_transcribe)
    except Exception as e:
        log.error(f"Whisper transcription error: {e}")
        raise HTTPException(502, f"Transcription failed: {e}")

    if not transcript or not transcript.strip():
        return JSONResponse({"transcription": "", "action": "unknown",
                             "response": "Could not understand audio."})

    log.info(f"[VOICE] Transcribed: {transcript[:80]}")

    # Route through NL pipeline (same logic as api_nl)
    with db_lock:
        recent = db.execute("SELECT label, node_type FROM nodes ORDER BY created_at DESC LIMIT 10").fetchall()
    ctx = "\n".join([f"- {r['label']} ({r['node_type']})" for r in recent])

    parsed = await asyncio.to_thread(parse_intent, transcript, ctx)
    action = parsed.get("action", "add")

    if action == "search":
        query = parsed.get("search_query", transcript)
        sem_results = await asyncio.to_thread(semantic_search, query, 8, 0.25)

        fts_results = []
        try:
            with db_lock:
                fts_rows = db.execute(
                    "SELECT id, content, label, node_type FROM nodes_fts WHERE nodes_fts MATCH ? LIMIT 5",
                    (sanitize_fts(query),)
                ).fetchall()
            sem_ids = {r["id"] for r in sem_results}
            for r in fts_rows:
                if r["id"] not in sem_ids:
                    fts_results.append({
                        "id": r["id"], "content": r["content"][:200],
                        "label": r["label"], "node_type": r["node_type"],
                        "score": 0.5, "source": "fts"
                    })
        except:
            pass

        combined = sem_results + fts_results
        return JSONResponse({
            "transcription": transcript,
            "action": "search",
            "results": combined[:10],
            "response": parsed.get("response_text", f"Found {len(combined)} results.")
        })

    elif action == "add":
        url = parsed.get("url")
        result = await asyncio.to_thread(
            create_node,
            content=parsed.get("content", transcript),
            label=parsed.get("label"),
            node_type=parsed.get("node_type", "note"),
            url=url,
            due_date=parsed.get("due_date")
        )
        return JSONResponse({
            "transcription": transcript,
            "action": "add",
            "node": result,
            "response": parsed.get("response_text", f"Added: {result['label']}")
        })

    elif action == "digest":
        with db_lock:
            recent = db.execute("""
                SELECT label, node_type, status, due_date, created_at
                FROM nodes ORDER BY created_at DESC LIMIT 20
            """).fetchall()
        inbox = [r for r in recent if r["status"] == "inbox"]
        upcoming = [r for r in recent if r["due_date"]]
        summary = f"You have {len(inbox)} unprocessed inbox items."
        if upcoming:
            summary += f" {len(upcoming)} items with dates coming up."
        return JSONResponse({
            "transcription": transcript,
            "action": "digest",
            "inbox_count": len(inbox),
            "upcoming": [dict(r) for r in upcoming],
            "response": summary
        })

    else:
        return JSONResponse({
            "transcription": transcript,
            "action": "unknown",
            "response": parsed.get("response_text", "I'm not sure what to do with that. Try rephrasing?")
        })


@app.post("/api/nl")
async def api_nl(req: NLQueryRequest):
    """Natural language interface — the universal endpoint.
    Handles add, search, link, digest through LLM parsing."""
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Text required")

    # Get recent context for the LLM
    with db_lock:
        recent = db.execute("SELECT label, node_type FROM nodes ORDER BY created_at DESC LIMIT 10").fetchall()
    ctx = "\n".join([f"- {r['label']} ({r['node_type']})" for r in recent])

    parsed = await asyncio.to_thread(parse_intent, text, ctx)
    action = parsed.get("action", "add")

    if action == "search":
        query = parsed.get("search_query", text)
        sem_results = await asyncio.to_thread(semantic_search, query, 8, 0.25)

        # FTS search
        fts_results = []
        try:
            with db_lock:
                fts_rows = db.execute(
                    "SELECT id, content, label, node_type FROM nodes_fts WHERE nodes_fts MATCH ? LIMIT 5",
                    (sanitize_fts(query),)
                ).fetchall()
            sem_ids = {r["id"] for r in sem_results}
            for r in fts_rows:
                if r["id"] not in sem_ids:
                    fts_results.append({
                        "id": r["id"], "content": r["content"][:200],
                        "label": r["label"], "node_type": r["node_type"],
                        "score": 0.5, "source": "fts"
                    })
        except:
            pass

        combined = sem_results + fts_results
        return JSONResponse({
            "action": "search",
            "results": combined[:10],
            "response": parsed.get("response_text", f"Found {len(combined)} results.")
        })

    elif action == "add":
        url = parsed.get("url")
        # Extract user note if present (sent as [Note]: ... from frontend)
        user_note = ""
        raw_content = parsed.get("content", text)
        note_match = re.search(r'\[Note\]:\s*(.+)', text, re.S)
        if note_match:
            user_note = note_match.group(1).strip()
        meta = {}
        if user_note:
            meta["user_notes"] = user_note
        result = await asyncio.to_thread(
            create_node,
            content=raw_content,
            label=parsed.get("label"),
            node_type=parsed.get("node_type", "note"),
            url=url,
            due_date=parsed.get("due_date"),
            metadata=meta if meta else None
        )
        return JSONResponse({
            "action": "add",
            "node": result,
            "response": parsed.get("response_text", f"Added: {result['label']}")
        })

    elif action == "digest":
        with db_lock:
            recent = db.execute("""
                SELECT label, node_type, status, due_date, created_at
                FROM nodes ORDER BY created_at DESC LIMIT 20
            """).fetchall()
        inbox = [r for r in recent if r["status"] == "inbox"]
        upcoming = [r for r in recent if r["due_date"]]
        summary = f"You have {len(inbox)} unprocessed inbox items."
        if upcoming:
            summary += f" {len(upcoming)} items with dates coming up."
        return JSONResponse({
            "action": "digest",
            "inbox_count": len(inbox),
            "upcoming": [dict(r) for r in upcoming],
            "response": summary
        })

    else:
        return JSONResponse({
            "action": "unknown",
            "response": parsed.get("response_text", "I'm not sure what to do with that. Try rephrasing?")
        })


@app.post("/api/link")
async def api_link(req: LinkRequest):
    """Manually create an edge between two nodes."""
    eid = gen_id()
    try:
        with db_lock:
            db.execute("INSERT INTO edges (id, source_id, target_id, label, auto_created) VALUES (?, ?, ?, ?, 0)",
                       (eid, req.source_id, req.target_id, req.label))
            db.commit()
        return {"id": eid, "status": "linked"}
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Edge already exists")


@app.delete("/api/edge/{edge_id}")
async def api_delete_edge(edge_id: str):
    """Delete an edge."""
    with db_lock:
        db.execute("DELETE FROM edges WHERE id=?", (edge_id,))
        db.commit()
    return {"status": "deleted"}


@app.get("/api/node/{node_id}")
async def api_get_node(node_id: str):
    """Get full node details + connected nodes."""
    with db_lock:
        row = db.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Node not found")

    await asyncio.to_thread(visit_node, node_id)

    # Get connected nodes
    with db_lock:
        edge_rows = db.execute("""
            SELECT e.id as edge_id, e.label as edge_label, e.weight,
                   CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as other_id
            FROM edges e
            WHERE e.source_id = ? OR e.target_id = ?
        """, (node_id, node_id, node_id)).fetchall()

    connected = []
    for e in edge_rows:
        with db_lock:
            other = db.execute("SELECT id, label, node_type, status FROM nodes WHERE id=?",
                              (e["other_id"],)).fetchone()
        if other:
            connected.append({
                "edge_id": e["edge_id"],
                "edge_label": e["edge_label"],
                "weight": e["weight"],
                "id": other["id"],
                "label": other["label"],
                "type": other["node_type"],
                "status": other["status"]
            })

    return {
        "id": row["id"],
        "content": row["content"],
        "label": row["label"],
        "type": row["node_type"],
        "status": row["status"],
        "pinned": bool(row["pinned"]),
        "starred": bool(row["starred"]),
        "x": row["x"],
        "y": row["y"],
        "image": row["image_path"],
        "url": row["url"],
        "due_date": row["due_date"],
        "temperature": row["temperature"],
        "visits": row["visit_count"],
        "meta": json.loads(row["metadata"] or "{}"),
        "created": row["created_at"],
        "updated": row["updated_at"],
        "connected": connected
    }


@app.get("/api/similar-images/{node_id}")
async def api_similar_images(node_id: str, limit: int = 10):
    """Find images semantically similar to a given image node."""
    with db_lock:
        row = db.execute("SELECT embedding FROM nodes WHERE id=?", (node_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Node not found")
    if not row["embedding"]:
        return {"results": []}

    emb = row["embedding"]
    with db_lock:
        image_rows = db.execute(
            "SELECT id, label, image_path, metadata, embedding FROM nodes "
            "WHERE node_type='image' AND id != ? AND embedding IS NOT NULL",
            (node_id,)
        ).fetchall()

    scored = []
    for r in image_rows:
        sim = cosine_sim(emb, r["embedding"])
        if sim >= 0.3:
            meta = json.loads(r["metadata"] or "{}")
            scored.append({
                "id": r["id"],
                "label": r["label"],
                "image": r["image_path"],
                "score": round(sim, 3),
                "description": meta.get("image_description", "")
            })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"results": scored[:limit]}


@app.patch("/api/node/{node_id}")
async def api_update_node(node_id: str, req: UpdateRequest):
    """Update node properties."""
    updates = []
    params = []
    for field in ["label", "content", "status", "node_type", "due_date"]:
        val = getattr(req, field, None)
        if val is not None:
            updates.append(f"{field}=?")
            params.append(val)
    for field in ["x", "y"]:
        val = getattr(req, field, None)
        if val is not None:
            updates.append(f"{field}=?")
            params.append(val)
    for field in ["pinned", "starred"]:
        val = getattr(req, field, None)
        if val is not None:
            updates.append(f"{field}=?")
            params.append(int(val))

    if not updates:
        raise HTTPException(400, "No updates provided")

    updates.append("updated_at=?")
    params.append(datetime.datetime.utcnow().isoformat())
    params.append(node_id)

    with db_lock:
        db.execute(f"UPDATE nodes SET {', '.join(updates)} WHERE id=?", params)

        # Re-embed if content or label changed
        if req.content or req.label:
            row = db.execute("SELECT content, label FROM nodes WHERE id=?", (node_id,)).fetchone()
            if row:
                emb = embed_text(f"{row['label']}. {row['content'][:500]}")
                if emb:
                    db.execute("UPDATE nodes SET embedding=? WHERE id=?", (emb, node_id))

        db.commit()
    return {"status": "updated"}


@app.post("/api/positions")
async def api_save_positions(positions: List[PositionUpdate]):
    """Batch save node positions from frontend layout."""
    with db_lock:
        for p in positions:
            db.execute("UPDATE nodes SET x=?, y=?, pinned=1 WHERE id=?", (p.x, p.y, p.id))
        db.commit()
    return {"status": "saved", "count": len(positions)}


@app.delete("/api/node/{node_id}")
async def api_delete_node(node_id: str):
    """Delete a node and its edges."""
    with db_lock:
        db.execute("DELETE FROM nodes WHERE id=?", (node_id,))
        try:
            db.execute("DELETE FROM nodes_fts WHERE id=?", (node_id,))
        except:
            pass
        db.commit()
    return {"status": "deleted"}


@app.get("/api/search")
async def api_search(q: str = Query(...), limit: int = 10):
    """Combined semantic + FTS search."""
    results = await asyncio.to_thread(semantic_search, q, limit, 0.25)

    # Also FTS
    try:
        with db_lock:
            fts_rows = db.execute(
                "SELECT id, content, label, node_type FROM nodes_fts WHERE nodes_fts MATCH ? LIMIT ?",
                (sanitize_fts(q), limit)
            ).fetchall()
        existing_ids = {r["id"] for r in results}
        for r in fts_rows:
            if r["id"] not in existing_ids:
                results.append({
                    "id": r["id"], "content": r["content"][:200],
                    "label": r["label"], "node_type": r["node_type"],
                    "score": 0.5
                })
    except:
        pass

    return {"results": results[:limit]}


@app.get("/api/stats")
async def api_stats():
    """Graph statistics."""
    with db_lock:
        node_count = db.execute("SELECT COUNT(*) as c FROM nodes").fetchone()["c"]
        edge_count = db.execute("SELECT COUNT(*) as c FROM edges").fetchone()["c"]
        inbox_count = db.execute("SELECT COUNT(*) as c FROM nodes WHERE status='inbox'").fetchone()["c"]
        types = db.execute("SELECT node_type, COUNT(*) as c FROM nodes GROUP BY node_type").fetchall()
    return {
        "nodes": node_count,
        "edges": edge_count,
        "inbox": inbox_count,
        "types": {r["node_type"]: r["c"] for r in types}
    }


@app.post("/api/digest")
async def api_generate_digest():
    """Manually trigger daily digest generation."""
    result = await asyncio.to_thread(generate_daily_digest)
    if result:
        return JSONResponse({"status": "generated", "node": result})
    return JSONResponse({"status": "skipped", "reason": "Already generated or no activity"})


@app.get("/api/inbox")
async def api_inbox():
    """Get all inbox items for process mode triage."""
    with db_lock:
        rows = db.execute("""
            SELECT id, content, label, node_type, status, url, image_path,
                   temperature, due_date, created_at
            FROM nodes WHERE status='inbox'
            ORDER BY created_at ASC
        """).fetchall()
    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "content": r["content"][:300] if r["content"] else "",
            "label": r["label"],
            "type": r["node_type"],
            "url": r["url"],
            "image": r["image_path"],
            "temperature": r["temperature"],
            "due_date": r["due_date"],
            "created": r["created_at"]
        })
    return JSONResponse({"items": items, "total": len(items)})


# ─── Twilio SMS Webhook ─────────────────────────────
@app.post("/api/sms")
async def api_sms(request: Request):
    """Twilio webhook for incoming SMS/MMS."""
    form = await request.form()
    body = form.get("Body", "").strip()
    from_number = form.get("From", "")
    num_media = int(form.get("NumMedia", 0))

    log.info(f"[SMS] From {from_number}: {body[:80]} ({num_media} media)")

    response_text = ""

    # Handle images
    if num_media > 0:
        for i in range(num_media):
            media_url = form.get(f"MediaUrl{i}")
            media_type = form.get(f"MediaContentType{i}", "image/jpeg")
            if media_url and "image" in media_type:
                try:
                    import httpx
                    r = httpx.get(media_url, follow_redirects=True, timeout=30,
                                  auth=(TWILIO_SID, TWILIO_TOKEN))
                    ext = mimetypes.guess_extension(media_type) or ".jpg"
                    fname = f"{gen_id()}{ext}"
                    fpath = os.path.join(UPLOAD_DIR, fname)
                    with open(fpath, "wb") as f:
                        f.write(r.content)
                    ocr_text = ocr_image(fpath)
                    caption = caption_image(fpath)
                    detailed_desc = detailed_image_description(fpath)
                    if body:
                        img_content = body
                    elif caption:
                        img_content = f"[Caption]: {caption}"
                        if detailed_desc:
                            img_content += f"\n[Description]: {detailed_desc}"
                        if ocr_text:
                            img_content += f"\n[OCR]: {ocr_text}"
                    elif ocr_text:
                        img_content = f"[OCR]: {ocr_text}"
                    else:
                        img_content = "Image from SMS"
                    meta = {}
                    if detailed_desc:
                        meta["image_description"] = detailed_desc
                    create_node(content=img_content, node_type="image", image_path=fname,
                                label=caption[:60] if caption else None,
                                metadata=meta if meta else None)
                    response_text += f"Image saved. "
                    if caption:
                        response_text += f"Caption: {caption[:100]} "
                    if ocr_text:
                        response_text += f"OCR: {ocr_text[:100]}... "
                except Exception as e:
                    log.error(f"SMS media error: {e}")

    # Handle text
    if body:
        result = await api_nl(NLQueryRequest(text=body, channel="sms"))
        data = json.loads(result.body.decode())
        response_text += data.get("response", "Done.")

    # TwiML response — escape user content for XML safety
    safe_text = html_escape(response_text[:1500])
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response><Message>{safe_text}</Message></Response>"""
    return HTMLResponse(content=twiml, media_type="text/xml")


@app.get("/api/sms/status")
async def api_sms_status():
    configured = bool(TWILIO_SID and TWILIO_TOKEN and TWILIO_PHONE)
    return {
        "configured": configured,
        "phone": TWILIO_PHONE if configured else None,
        "webhook_url": "https://openmind.fahrenheitrequited.dev/api/sms"
    }


# ─── Reset ───────────────────────────────────────────
@app.delete("/api/reset")
async def api_reset(confirm: str = Query("")):
    """Wipe all nodes, edges, daily_nodes and FTS. Recreate today's daily node."""
    if confirm != "yes":
        raise HTTPException(400, "Pass ?confirm=yes to reset the database")
    with db_lock:
        db.execute("DELETE FROM edges")
        db.execute("DELETE FROM daily_nodes")
        db.execute("DELETE FROM nodes")
        try:
            db.execute("DELETE FROM nodes_fts")
        except:
            pass
        db.commit()
    # Recreate today's daily node
    ensure_daily_node()
    rebuild_fts()
    log.info("[RESET] Database wiped and daily node recreated")
    return {"status": "reset", "message": "Database cleared. Fresh daily node created."}


# ─── Image serving ───────────────────────────────────
@app.get("/uploads/{filename}")
async def serve_upload(filename: str):
    fpath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(fpath):
        return FileResponse(fpath)
    raise HTTPException(404, "File not found")


# ─── Background tasks ───────────────────────────────
async def background_tasks():
    """Periodic background work."""
    while True:
        await asyncio.sleep(3600)  # hourly
        try:
            ensure_daily_node()
            update_temperature()
            log.info("[BG] Temperature decay + daily node check")
        except Exception as e:
            log.error(f"[BG] Error: {e}")
        # Daily digest (once per day, idempotent)
        try:
            generate_daily_digest()
        except Exception as e:
            log.error(f"[BG] Digest error: {e}")


# ─── Run ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info(f"Starting Open Mind on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
