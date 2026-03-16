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
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Config ──────────────────────────────────────────
PORT = int(os.environ.get("OPENMIND_PORT", 8250))
DB_PATH = os.environ.get("OPENMIND_DB", os.path.expanduser("~/open-mind/openmind.db"))
UPLOAD_DIR = os.environ.get("OPENMIND_UPLOADS", os.path.expanduser("~/open-mind/uploads"))
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE = os.environ.get("TWILIO_PHONE_NUMBER", "")
EMBED_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.45  # auto-link above this cosine sim
MAX_AUTO_EDGES = 5           # max auto-links per add
LLM_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
ENRICH_SEMAPHORE_SIZE = 3

# ─── Structured Type Schemas (v3 feature 6) ─────────
TYPE_SCHEMAS = {
    "paper": {
        "authors": {"type": "text", "label": "Authors"},
        "journal": {"type": "text", "label": "Journal/Source"},
        "year": {"type": "number", "label": "Year"},
        "doi": {"type": "url", "label": "DOI"},
        "read_status": {"type": "select", "label": "Status", "options": ["to-read", "reading", "read", "cited"]},
        "rating": {"type": "number", "label": "Rating (1-5)"},
    },
    "task": {
        "priority": {"type": "select", "label": "Priority", "options": ["low", "medium", "high", "urgent"]},
        "task_due": {"type": "date", "label": "Due Date"},
        "task_status": {"type": "select", "label": "Status", "options": ["todo", "in-progress", "blocked", "done"]},
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
        "project_status": {"type": "select", "label": "Status", "options": ["ideation", "active", "paused", "completed", "abandoned"]},
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
        "appt_date": {"type": "datetime", "label": "Date & Time"},
        "location": {"type": "text", "label": "Location"},
        "attendees": {"type": "text", "label": "Attendees"},
        "recurring": {"type": "select", "label": "Recurring", "options": ["none", "daily", "weekly", "monthly"]},
    },
    "image": {
        "source": {"type": "text", "label": "Source"},
        "subjects": {"type": "text", "label": "Subjects"},
        "img_location": {"type": "text", "label": "Location"},
    },
    "idea": {
        "idea_status": {"type": "select", "label": "Status", "options": ["raw", "developing", "validated", "implemented", "abandoned"]},
        "domain": {"type": "text", "label": "Domain"},
    },
    "note": {},
    "daily": {},
}

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("openmind")

# ─── Globals ─────────────────────────────────────────
db: sqlite3.Connection = None
db_lock = threading.Lock()  # Protect SQLite from concurrent access
embedder = None  # SentenceTransformer
_openai_client = None
_enrich_semaphore = None

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
    except sqlite3.OperationalError:
        pass  # already exists

    # Add edge enrichment columns (v3 feature 2)
    try:
        db.execute("ALTER TABLE edges ADD COLUMN relationship_type TEXT DEFAULT 'related'")
    except sqlite3.OperationalError:
        pass  # column already exists
    try:
        db.execute("ALTER TABLE edges ADD COLUMN relationship_description TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists

    for _col in ["ALTER TABLE nodes ADD COLUMN enriched_at TEXT","ALTER TABLE nodes ADD COLUMN digest TEXT","ALTER TABLE nodes ADD COLUMN openai_embedding BLOB"]:
        try: db.execute(_col)
        except: pass
    # Add resurfaced_at column (v3 feature 3 — resurface forgotten nodes)
    try:
        db.execute("ALTER TABLE nodes ADD COLUMN resurfaced_at TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists

    # Canvas tables (v3 feature 4 — multiple named canvases)
    db.executescript("""
    CREATE TABLE IF NOT EXISTS canvases (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        color TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS canvas_nodes (
        canvas_id TEXT REFERENCES canvases(id) ON DELETE CASCADE,
        node_id TEXT REFERENCES nodes(id) ON DELETE CASCADE,
        x REAL,
        y REAL,
        pinned INTEGER DEFAULT 0,
        added_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (canvas_id, node_id)
    );

    CREATE INDEX IF NOT EXISTS idx_canvas_nodes_canvas ON canvas_nodes(canvas_id);
    CREATE INDEX IF NOT EXISTS idx_canvas_nodes_node ON canvas_nodes(node_id);

    CREATE TABLE IF NOT EXISTS canvas_chats (
        id TEXT PRIMARY KEY,
        canvas_id TEXT REFERENCES canvases(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_canvas_chats_canvas ON canvas_chats(canvas_id);
    """)

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
    if len(a) != len(b):
        return 0.0
    return float(np.dot(a, b))


def _temporal_boost(created_at_str, sim):
    try:
        import math
        from datetime import datetime
        age = (datetime.utcnow()-datetime.fromisoformat(created_at_str)).total_seconds()/86400
        return sim*(1+0.15*math.exp(-age*math.log(2)/30))
    except: return sim

def semantic_search(query: str, limit: int = 10, threshold: float = 0.3):
    if embedder is None:
        return []
    q_vec = embedder.encode(query, normalize_embeddings=True).astype(np.float32).tobytes()
    with db_lock:
        rows = db.execute("SELECT id, content, label, node_type, status, embedding, created_at FROM nodes WHERE embedding IS NOT NULL").fetchall()
    results = []
    for r in rows:
        sim = cosine_sim(q_vec, r["embedding"])
        if sim >= threshold:
            boosted = _temporal_boost(r["created_at"] or "", sim)
            results.append({
                "id": r["id"],
                "content": r["content"][:200],
                "label": r["label"],
                "node_type": r["node_type"],
                "status": r["status"],
                "score": round(boosted, 4),
                "raw_score": round(sim, 4)
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


def semantic_search_full(query: str, limit: int = 15, threshold: float = 0.3):
    """Like semantic_search but returns full content for Ask feature."""
    if embedder is None:
        return []
    q_vec = embedder.encode(query, normalize_embeddings=True).astype(np.float32).tobytes()
    with db_lock:
        rows = db.execute("SELECT id, content, label, node_type, status, embedding, created_at FROM nodes WHERE embedding IS NOT NULL").fetchall()
    results = []
    for r in rows:
        sim = cosine_sim(q_vec, r["embedding"])
        if sim >= threshold:
            boosted = _temporal_boost(r["created_at"] or "", sim)
            results.append({
                "id": r["id"],
                "content": r["content"],
                "label": r["label"],
                "node_type": r["node_type"],
                "status": r["status"],
                "score": round(boosted, 4),
                "raw_score": round(sim, 4)
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


def llm_chat_messages(messages: list, model: str = None, max_tokens: int = 1500) -> str:
    """LLM chat with full message history support."""
    client = get_openai()
    if not client:
        return ""
    try:
        r = client.post("/chat/completions", json={
            "model": model or LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
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
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback: treat as simple add
        return {
            "action": "add",
            "content": text,
            "label": text[:60],
            "node_type": "note",
            "response_text": "Added to your mind map."
        }


def extract_type_fields(content: str, node_type: str, url: str = None) -> dict:
    """Use LLM to extract structured metadata fields from node content."""
    schema = TYPE_SCHEMAS.get(node_type, {})
    if not schema:
        return {}
    client = get_openai()
    if not client:
        return {}

    fields_desc = json.dumps({k: v["label"] for k, v in schema.items()}, indent=2)
    url_line = f"\nURL: {url}" if url else ""

    system = "You extract structured metadata from content. Respond ONLY with valid JSON. Omit fields you cannot confidently extract. Do not guess."
    user_msg = f"""Given this content being saved as a "{node_type}" entry, extract structured metadata.

Content: {content[:500]}{url_line}

Available fields:
{fields_desc}

Return JSON with only the fields you can confidently extract. Use the field keys exactly as shown."""

    result = llm_chat(system, user_msg)
    if not result:
        return {}
    try:
        result = re.sub(r'^```json\s*', '', result)
        result = re.sub(r'\s*```$', '', result)
        parsed = json.loads(result)
        # Validate: only keep keys that are in the schema
        valid = {}
        for k, v in parsed.items():
            if k in schema and v is not None and v != "":
                valid[k] = v
        return valid
    except Exception as e:
        log.warning(f"extract_type_fields parse error: {e}")
        return {}


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
            "model": VISION_MODEL,  # gpt-4o for best caption quality
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in 3-5 sentences covering: what is depicted, any people or objects, visible text, setting/context, and notable visual details."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}", "detail": "high"}}
                ]
            }],
            "max_tokens": 600,
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
            "model": VISION_MODEL,
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
                        "Be thorough and specific. 150-200 words."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_data}", "detail": "high"}}
                ]
            }],
            "max_tokens": 800,
            "temperature": 0.3
        })
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning(f"Detailed image description failed: {e}")
        return ""



def get_openai_embedding(text):
    client = get_openai()
    if not client or not text.strip(): return None
    try:
        import numpy as _np
        r = client.post("/embeddings", json={"model": OPENAI_EMBED_MODEL, "input": text[:8000], "encoding_format": "float"})
        vec = _np.array(r.json()["data"][0]["embedding"], dtype=_np.float32)
        n = _np.linalg.norm(vec)
        if n > 0: vec /= n
        return vec.tobytes()
    except Exception as e:
        log.warning(f"OpenAI embedding failed: {e}")
        return None

def generate_node_digest(label, content, node_type):
    client = get_openai()
    if not client: return ""
    try:
        system = "You write concise 2-4 sentence narrative digests for personal knowledge graph nodes. Be direct and informative. No preamble."
        user = f"Type: {node_type}\nLabel: {label}\n\n{content[:600]}"
        return (llm_chat(system, user, model=LLM_MODEL) or "").strip()[:500]
    except Exception as e:
        log.warning(f"Digest generation failed: {e}")
        return ""

async def enrich_node_background(node_id, label, content, node_type):
    if not _enrich_semaphore or not OPENAI_KEY: return
    async with _enrich_semaphore:
        try:
            text = f"{label}. {content[:800]}"
            digest, oai_emb = await asyncio.gather(
                asyncio.to_thread(generate_node_digest, label, content, node_type),
                asyncio.to_thread(get_openai_embedding, text),
            )
            with db_lock:
                db.execute("UPDATE nodes SET digest=?, openai_embedding=?, enriched_at=datetime('now') WHERE id=?",
                           (digest or None, oai_emb, node_id))
                db.commit()
            log.info(f"[ENRICH] Node {node_id}: digest+oai_embedding stored")
        except Exception as e:
            log.error(f"Node enrichment error {node_id}: {e}")

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


def enrich_edge_relationship(edge_id, source_label, source_content, target_label, target_content):
    """Use LLM to classify the specific relationship between two connected nodes.

    Returns True on success, False on failure.
    """
    valid_types = {"supports", "contradicts", "extends", "is-part-of",
                   "inspired-by", "similar-to", "implements", "questions",
                   "summarizes", "example-of", "related", "discusses",
                   "relates_to", "inspired_by", "depends_on", "alternative_to"}

    system = """You classify relationships between knowledge graph entries.
Respond ONLY with valid JSON. No extra text."""

    user_msg = f"""Given two knowledge base entries, classify their relationship.

ENTRY A [{source_label}]: {(source_content or '')[:300]}
ENTRY B [{target_label}]: {(target_content or '')[:300]}

Respond in JSON:
{{
  "relationship_type": one of: "supports", "contradicts", "extends", "is-part-of", "inspired-by", "similar-to", "implements", "questions", "summarizes", "example-of", "related",
  "description": "2-8 word description of the specific relationship",
  "confidence": 0.0-1.0
}}

Choose "related" only if no more specific type fits.
The description should be specific to THESE entries, not generic."""

    result = llm_chat(system, user_msg)
    if not result:
        return False
    try:
        # Strip markdown fences if present
        result = re.sub(r'^```json\s*', '', result)
        result = re.sub(r'\s*```$', '', result)
        parsed = json.loads(result)

        rel_type = parsed.get("relationship_type", "related")
        if rel_type not in valid_types:
            rel_type = "related"
        description = parsed.get("description", "")
        confidence = parsed.get("confidence", 0.5)

        # Only update if confidence is reasonable
        if confidence < 0.2:
            return False

        with db_lock:
            db.execute(
                "UPDATE edges SET relationship_type = ?, relationship_description = ? WHERE id = ?",
                (rel_type, description[:100] if description else None, edge_id)
            )
            db.commit()
        log.info(f"[ENRICH] Edge {edge_id}: {rel_type} — {description}")
        return True
    except Exception as e:
        log.warning(f"enrich_edge_relationship parse error: {e}")
        return False


async def fire_edge_enrichment(node_id: str):
    """Background task: enrich all un-enriched edges for a node."""
    try:
        with db_lock:
            edges = db.execute("""
                SELECT e.id, e.source_id, e.target_id,
                       s.label as s_label, s.content as s_content,
                       t.label as t_label, t.content as t_content
                FROM edges e
                JOIN nodes s ON e.source_id = s.id
                JOIN nodes t ON e.target_id = t.id
                WHERE (e.source_id = ? OR e.target_id = ?)
                  AND (e.relationship_type IS NULL OR e.relationship_type = 'related')
                  AND e.label != 'daily'
            """, (node_id, node_id)).fetchall()

        if edges:
            async def _one(edge):
                if not _enrich_semaphore: return
                async with _enrich_semaphore:
                    await asyncio.to_thread(enrich_edge_relationship, edge["id"], edge["s_label"], edge["s_content"], edge["t_label"], edge["t_content"])
            await asyncio.gather(*[_one(e) for e in edges])
    except Exception as e:
        log.error(f"Edge enrichment error: {e}")


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

    # LLM auto-extraction of structured type fields
    if OPENAI_KEY and node_type in TYPE_SCHEMAS and TYPE_SCHEMAS[node_type]:
        try:
            extracted = extract_type_fields(content, node_type, url)
            if extracted:
                # Don't overwrite existing metadata keys
                for k, v in extracted.items():
                    if k not in meta:
                        meta[k] = v
                log.info(f"[EXTRACT] Auto-filled {len(extracted)} fields for {node_type}")
        except Exception as e:
            log.warning(f"Auto-extraction failed: {e}")

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
        except sqlite3.OperationalError:
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
            except sqlite3.IntegrityError:
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

        erows = db.execute("SELECT id, source_id, target_id, label, weight, auto_created, relationship_type, relationship_description FROM edges").fetchall()

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
            "digest": r["digest"] if "digest" in r.keys() else None,
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
            "auto": bool(r["auto_created"]),
            "relationship_type": r["relationship_type"] or "related",
            "relationship_description": r["relationship_description"],
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
                except (ValueError, TypeError):
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


# ─── Resurface Forgotten Nodes (v3 feature 3) ───────
import random as _random

def get_resurface_nodes(count: int = 3) -> list:
    """Get cold, permanent, unvisited nodes for rediscovery feed.
    Selection criteria:
    - temperature < 0.5 (cold)
    - status = 'permanent' (not inbox junk)
    - last_visited more than 7 days ago (or never visited)
    - Not resurfaced in the last 3 days
    Weighted by inverse temperature (colder = more likely).
    """
    now = datetime.datetime.utcnow()
    seven_days_ago = (now - datetime.timedelta(days=7)).isoformat()
    three_days_ago = (now - datetime.timedelta(days=3)).isoformat()

    with db_lock:
        rows = db.execute("""
            SELECT id, content, label, node_type, status, temperature, url,
                   image_path, created_at, last_visited, metadata, due_date,
                   starred, visit_count, resurfaced_at
            FROM nodes
            WHERE status = 'permanent'
              AND temperature < 0.5
              AND (last_visited IS NULL OR last_visited < ?)
              AND (resurfaced_at IS NULL OR resurfaced_at < ?)
              AND (node_type IS NULL OR node_type != 'daily')
        """, (seven_days_ago, three_days_ago)).fetchall()

    if not rows:
        return []

    # Weight by inverse temperature (colder = more likely to surface)
    candidates = []
    for r in rows:
        temp = r["temperature"] or 0.1
        weight = 1.0 / max(temp, 0.05)  # inverse temp, floor at 0.05
        candidates.append((dict(r), weight))

    # Weighted random sample
    selected = []
    remaining = list(candidates)
    pick_count = min(count, len(remaining))
    for _ in range(pick_count):
        if not remaining:
            break
        total_weight = sum(w for _, w in remaining)
        r_val = _random.random() * total_weight
        cumulative = 0
        for i, (node, w) in enumerate(remaining):
            cumulative += w
            if cumulative >= r_val:
                selected.append(node)
                remaining.pop(i)
                break

    # Mark as resurfaced
    now_iso = now.isoformat()
    with db_lock:
        for node in selected:
            db.execute("UPDATE nodes SET resurfaced_at = ? WHERE id = ?",
                       (now_iso, node["id"]))
        db.commit()

    # Format results
    results = []
    for node in selected:
        # Compute days since creation
        try:
            created = datetime.datetime.fromisoformat(node["created_at"])
            days_ago = (now - created).days
        except (ValueError, TypeError):
            days_ago = 0
        meta = {}
        try:
            meta = json.loads(node["metadata"]) if node["metadata"] else {}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        results.append({
            "id": node["id"],
            "label": node["label"],
            "content": (node["content"] or "")[:200],
            "node_type": node["node_type"],
            "temperature": node["temperature"],
            "created_at": node["created_at"],
            "days_ago": days_ago,
            "last_visited": node["last_visited"],
            "url": node["url"],
            "image_path": node["image_path"],
            "starred": node["starred"],
            "metadata": meta,
        })

    return results


def get_serendipity_pair() -> dict:
    """Pick 2 nodes with moderate cosine similarity (0.3-0.5) for serendipity prompt.
    Returns node pair + similarity + optional LLM-generated connection prompt.
    """
    with db_lock:
        rows = db.execute("""
            SELECT id, content, label, node_type, embedding, temperature,
                   image_path, url, created_at, metadata
            FROM nodes
            WHERE embedding IS NOT NULL
              AND (node_type IS NULL OR node_type != 'daily')
              AND status = 'permanent'
        """).fetchall()

    if len(rows) < 2:
        return {}

    # Find pairs with moderate similarity (0.3-0.5 range)
    best_pairs = []
    # Sample a subset if too many nodes to compare all pairs
    sample = rows if len(rows) <= 50 else _random.sample(list(rows), 50)

    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            sim = cosine_sim(sample[i]["embedding"], sample[j]["embedding"])
            if 0.3 <= sim <= 0.5:
                best_pairs.append((sample[i], sample[j], sim))

    if not best_pairs:
        # Relax range if nothing found
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                sim = cosine_sim(sample[i]["embedding"], sample[j]["embedding"])
                if 0.25 <= sim <= 0.55:
                    best_pairs.append((sample[i], sample[j], sim))

    if not best_pairs:
        return {}

    # Pick a random pair from candidates
    node_a, node_b, sim = _random.choice(best_pairs)

    def format_node(r):
        meta = {}
        try:
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return {
            "id": r["id"],
            "label": r["label"],
            "content": (r["content"] or "")[:200],
            "node_type": r["node_type"],
            "temperature": r["temperature"],
            "url": r["url"],
            "image_path": r["image_path"],
            "created_at": r["created_at"],
            "metadata": meta,
        }

    # Generate LLM prompt about the connection
    llm_prompt = ""
    if OPENAI_KEY:
        system = "Given two knowledge base entries, write a single curious question (15 words max) asking the user if they see a connection. Be specific to the content, not generic."
        user_msg = f"Entry A [{node_a['label']}]: {(node_a['content'] or '')[:200]}\n\nEntry B [{node_b['label']}]: {(node_b['content'] or '')[:200]}"
        llm_prompt = llm_chat(system, user_msg)

    if not llm_prompt:
        llm_prompt = f"These two entries share some thematic overlap. Is there a connection worth making?"

    return {
        "node_a": format_node(node_a),
        "node_b": format_node(node_b),
        "similarity": round(sim, 3),
        "llm_prompt": llm_prompt,
    }


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

    # Get resurfaced nodes for the digest (don't mark as resurfaced — digest is passive)
    seven_days_ago = (now - datetime.timedelta(days=7)).isoformat()
    three_days_ago = (now - datetime.timedelta(days=3)).isoformat()
    with db_lock:
        resurface_rows = db.execute("""
            SELECT n.id, n.label, n.node_type, n.temperature, n.created_at,
                   (SELECT GROUP_CONCAT(n2.label, ', ')
                    FROM edges e
                    JOIN nodes n2 ON (e.target_id = n2.id AND e.source_id = n.id)
                       OR (e.source_id = n2.id AND e.target_id = n.id)
                    WHERE n2.id != n.id LIMIT 3) as connections
            FROM nodes n
            WHERE n.status = 'permanent'
              AND n.temperature < 0.5
              AND (n.last_visited IS NULL OR n.last_visited < ?)
              AND (n.resurfaced_at IS NULL OR n.resurfaced_at < ?)
              AND (n.node_type IS NULL OR n.node_type != 'daily')
            ORDER BY RANDOM() LIMIT 3
        """, (seven_days_ago, three_days_ago)).fetchall()

    # Build context for LLM
    node_lines = [f"- {r['label']} ({r['node_type']}, {r['status']})" for r in new_nodes]
    edge_lines = [f"- {r['source_label']} <-> {r['target_label']} ({r['edge_label']}, weight {r['weight']:.2f})" for r in new_edges]
    due_lines = [f"- {r['label']} due {r['due_date']}" for r in upcoming]
    resurface_lines = []
    for r in resurface_rows:
        try:
            created = datetime.datetime.fromisoformat(r["created_at"])
            days = (now - created).days
        except (ValueError, TypeError):
            days = 0
        conn_str = f" Connected to: {r['connections']}." if r["connections"] else " No connections yet."
        resurface_lines.append(f"- [{r['label']}] — saved {days} days ago.{conn_str}")

    context = f"""Activity in the last 24 hours:

New nodes ({len(new_nodes)}):
{chr(10).join(node_lines) if node_lines else '(none)'}

New connections ({len(new_edges)}):
{chr(10).join(edge_lines) if edge_lines else '(none)'}

Inbox items: {inbox_count}

Upcoming due dates:
{chr(10).join(due_lines) if due_lines else '(none)'}

Forgotten nodes worth revisiting:
{chr(10).join(resurface_lines) if resurface_lines else '(none)'}
"""

    system = """You write concise daily digests for a personal knowledge graph called Open Mind.
Summarize the activity in 3-5 sentences. Mention key themes, notable new connections,
and any upcoming deadlines. If there are forgotten nodes worth revisiting, include a brief
"Rediscovery" note mentioning 1-2 of them and why they might still matter.
Be brief and insightful. Do not use bullet points or headers."""

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
    metadata: Optional[dict] = None  # Structured type fields

class NLQueryRequest(BaseModel):
    text: str
    channel: Optional[str] = "web"  # web, sms, voice

class PositionUpdate(BaseModel):
    id: str
    x: float
    y: float

class AskRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = None  # [{role, content}]

class EnrichEdgesRequest(BaseModel):
    node_id: Optional[str] = None

class UpdateEdgeRequest(BaseModel):
    relationship_type: Optional[str] = None
    relationship_description: Optional[str] = None
    label: Optional[str] = None

class CanvasCreate(BaseModel):
    name: str
    description: Optional[str] = None
    color: Optional[str] = None

class CanvasUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None

class CanvasAddNodes(BaseModel):
    node_ids: List[str]

class CanvasPositionUpdate(BaseModel):
    id: str  # node_id
    x: float
    y: float

class ChatRequest(BaseModel):
    node_ids: List[str]
    message: str
    history: Optional[List[dict]] = None  # [{role, content}]

class CanvasChatMessage(BaseModel):
    message: str

# ─── Lifespan ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _enrich_semaphore
    _enrich_semaphore = asyncio.Semaphore(ENRICH_SEMAPHORE_SIZE)
    init_db()
    init_embedder()
    ensure_daily_node()
    # Rebuild FTS on startup
    try:
        rebuild_fts()
    except Exception:
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

@app.get("/phonetest", response_class=HTMLResponse)
async def phonetest():
    """Minimal test page to verify connectivity."""
    import json
    with db_lock:
        row = db.execute("SELECT count(*) FROM nodes").fetchone()
    count = row[0]
    return HTMLResponse(f"""<!DOCTYPE html><html><body style="background:#000;color:#0f0;font:20px monospace;padding:40px">
<h1>PHONE TEST</h1><p>DB nodes: {count}</p><p>Time: {__import__('datetime').datetime.now().isoformat()}</p>
<p id="r">fetching api...</p>
<script>fetch('/api/stats').then(r=>r.json()).then(d=>document.getElementById('r').textContent='API: '+JSON.stringify(d)).catch(e=>document.getElementById('r').textContent='ERR: '+e)</script>
</body></html>""", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
async def index():
    """Serve the frontend."""
    html_path = Path(os.path.expanduser("~/open-mind/om-viz.html"))
    if html_path.exists():
        return HTMLResponse(html_path.read_text(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
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
    # Fire-and-forget edge enrichment + node digest/embedding
    if OPENAI_KEY:
        if result.get("auto_edges"):
            asyncio.create_task(fire_edge_enrichment(result["id"]))
        asyncio.create_task(enrich_node_background(result["id"], result["label"], content, node_type))
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
        except Exception:
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
        # Fire-and-forget edge enrichment + node enrichment
        if OPENAI_KEY:
            if result.get("auto_edges"):
                asyncio.create_task(fire_edge_enrichment(result["id"]))
            asyncio.create_task(enrich_node_background(result["id"], result["label"], result.get("content",""), result.get("node_type","note")))
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
        except Exception:
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
        # Fire-and-forget edge enrichment + node enrichment
        if OPENAI_KEY:
            if result.get("auto_edges"):
                asyncio.create_task(fire_edge_enrichment(result["id"]))
            asyncio.create_task(enrich_node_background(result["id"], result["label"], result.get("content",""), result.get("node_type","note")))
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


@app.post("/api/enrich-edges")
async def api_enrich_edges(req: EnrichEdgesRequest = EnrichEdgesRequest()):
    """Enrich un-enriched edges with LLM relationship classification."""
    if not OPENAI_KEY:
        raise HTTPException(400, "OpenAI API key required")

    with db_lock:
        if req.node_id:
            edges = db.execute("""
                SELECT e.id, e.source_id, e.target_id,
                       s.label as s_label, s.content as s_content,
                       t.label as t_label, t.content as t_content
                FROM edges e
                JOIN nodes s ON e.source_id = s.id
                JOIN nodes t ON e.target_id = t.id
                WHERE (e.source_id = ? OR e.target_id = ?)
                  AND (e.relationship_type IS NULL OR e.relationship_type = 'related')
                  AND e.label != 'daily'
            """, (req.node_id, req.node_id)).fetchall()
        else:
            edges = db.execute("""
                SELECT e.id, e.source_id, e.target_id,
                       s.label as s_label, s.content as s_content,
                       t.label as t_label, t.content as t_content
                FROM edges e
                JOIN nodes s ON e.source_id = s.id
                JOIN nodes t ON e.target_id = t.id
                WHERE (e.relationship_type IS NULL OR e.relationship_type = 'related')
                  AND e.label != 'daily'
            """).fetchall()

    enriched = 0
    skipped = 0
    for edge in edges:
        try:
            success = await asyncio.to_thread(
                enrich_edge_relationship,
                edge["id"],
                edge["s_label"], edge["s_content"],
                edge["t_label"], edge["t_content"]
            )
            if success:
                enriched += 1
            else:
                skipped += 1
            # Rate limit: ~2 requests/second
            await asyncio.sleep(0.5)
        except Exception as e:
            log.warning(f"Enrich edge {edge['id']} failed: {e}")
            skipped += 1

    return {"enriched": enriched, "skipped": skipped}


@app.patch("/api/edge/{edge_id}")
async def api_update_edge(edge_id: str, req: UpdateEdgeRequest):
    """Update an edge's relationship metadata."""
    valid_types = {"related", "supports", "contradicts", "extends", "is-part-of",
                   "inspired-by", "similar-to", "implements", "questions",
                   "summarizes", "example-of", "discusses", "extends",
                   "relates_to", "inspired_by", "depends_on", "alternative_to"}

    updates = []
    params = []
    if req.relationship_type is not None:
        if req.relationship_type not in valid_types:
            raise HTTPException(400, f"Invalid relationship_type: {req.relationship_type}")
        updates.append("relationship_type = ?")
        params.append(req.relationship_type)
    if req.relationship_description is not None:
        updates.append("relationship_description = ?")
        params.append(req.relationship_description)
    if req.label is not None:
        updates.append("label = ?")
        params.append(req.label)

    if not updates:
        raise HTTPException(400, "No fields to update")

    params.append(edge_id)
    with db_lock:
        db.execute(f"UPDATE edges SET {', '.join(updates)} WHERE id = ?", params)
        db.commit()
    return {"status": "updated"}


@app.get("/api/node/{node_id}")
async def api_get_node(node_id: str):
    """Get full node details + connected nodes."""
    with db_lock:
        row = db.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Node not found")

        # Record visit inside the same lock to avoid race with deletion
        now_ts = datetime.datetime.utcnow().isoformat()
        db.execute("""UPDATE nodes SET visit_count = visit_count + 1,
                      last_visited = ?, temperature = MIN(2.0, temperature + 0.3),
                      updated_at = ? WHERE id = ?""", (now_ts, now_ts, node_id))
        db.commit()

        # Get connected nodes in a single batch query
        edge_rows = db.execute("""
            SELECT e.id as edge_id, e.label as edge_label, e.weight,
                   e.relationship_type, e.relationship_description,
                   CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as other_id
            FROM edges e
            WHERE e.source_id = ? OR e.target_id = ?
        """, (node_id, node_id, node_id)).fetchall()

        other_ids = [e["other_id"] for e in edge_rows]
        neighbors = {}
        if other_ids:
            placeholders = ",".join("?" * len(other_ids))
            neighbor_rows = db.execute(
                f"SELECT id, label, node_type, status, image_path, metadata FROM nodes WHERE id IN ({placeholders})",
                other_ids
            ).fetchall()
            neighbors = {r["id"]: r for r in neighbor_rows}

    connected = []
    for e in edge_rows:
        other = neighbors.get(e["other_id"])
        if other:
            other_meta = {}
            try:
                other_meta = json.loads(other["metadata"] or "{}")
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
            connected.append({
                "edge_id": e["edge_id"],
                "edge_label": e["edge_label"],
                "weight": e["weight"],
                "relationship_type": e["relationship_type"] or "related",
                "relationship_description": e["relationship_description"],
                "id": other["id"],
                "label": other["label"],
                "type": other["node_type"],
                "status": other["status"],
                "image": other["image_path"],
                "thumbnail": other_meta.get("thumbnail")
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


@app.get("/api/nodes/{node_id}/similarities")
async def api_node_similarities(node_id: str):
    """Get cosine similarity scores between a node and all other nodes."""
    def _compute():
        with db_lock:
            row = db.execute("SELECT embedding FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not row:
            return None
        if not row["embedding"]:
            return {"similarities": []}
        emb = row["embedding"]
        with db_lock:
            rows = db.execute(
                "SELECT id, embedding FROM nodes WHERE id != ? AND embedding IS NOT NULL",
                (node_id,)
            ).fetchall()
        results = []
        for r in rows:
            sim = cosine_sim(emb, r["embedding"])
            results.append({"id": r["id"], "score": round(sim, 4)})
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"similarities": results}
    result = await asyncio.to_thread(_compute)
    if result is None:
        raise HTTPException(404, "Node not found")
    return result


@app.get("/api/embedding-layout")
async def api_embedding_layout():
    """Return PCA 2D projection of all node embeddings for spatial layout."""
    def _compute():
        with db_lock:
            rows = db.execute(
                "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL"
            ).fetchall()
        if len(rows) < 2:
            return {"positions": {r["id"]: [0.0, 0.0] for r in rows}}
        ids = [r["id"] for r in rows]
        vecs = np.array([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        # PCA to 2D via SVD (no sklearn needed)
        mean = vecs.mean(axis=0)
        centered = vecs - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ Vt[:2].T  # (N, 2)
        # Normalize to [-1, 1]
        max_abs = np.abs(proj).max()
        if max_abs > 0:
            proj = proj / max_abs
        positions = {}
        for i, nid in enumerate(ids):
            positions[nid] = [round(float(proj[i, 0]), 4), round(float(proj[i, 1]), 4)]
        return {"positions": positions}
    result = await asyncio.to_thread(_compute)
    return result


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

    has_field_updates = bool(updates)
    has_meta_update = req.metadata is not None

    if not has_field_updates and not has_meta_update:
        raise HTTPException(400, "No updates provided")

    with db_lock:
        if has_field_updates:
            updates.append("updated_at=?")
            params.append(datetime.datetime.utcnow().isoformat())
            params.append(node_id)
            db.execute(f"UPDATE nodes SET {', '.join(updates)} WHERE id=?", params)

        # Merge metadata
        if has_meta_update:
            row = db.execute("SELECT metadata FROM nodes WHERE id=?", (node_id,)).fetchone()
            if row:
                existing_meta = json.loads(row["metadata"] or "{}")
                existing_meta.update(req.metadata)
                db.execute("UPDATE nodes SET metadata=?, updated_at=? WHERE id=?",
                           (json.dumps(existing_meta), datetime.datetime.utcnow().isoformat(), node_id))

        db.commit()

        # Re-embed if content or label changed (after UPDATE so we get new values)
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
        except sqlite3.OperationalError:
            pass
        db.commit()
    return {"status": "deleted"}


@app.get("/api/search")
async def api_search(request: Request, q: str = Query(""), limit: int = 10,
                     type: str = Query(None)):
    """Combined semantic + FTS search with optional type and field filters."""
    # Extract field_* query parameters
    field_filters = {}
    for key, val in request.query_params.items():
        if key.startswith("field_"):
            field_name = key[6:]  # strip "field_" prefix
            field_filters[field_name] = val

    results = []
    if q:
        results = await asyncio.to_thread(semantic_search, q, limit * 2, 0.25)

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
        except Exception:
            pass

    # If no query text but we have filters, load all nodes
    if not q and (type or field_filters):
        with db_lock:
            rows = db.execute(
                "SELECT id, content, label, node_type, metadata FROM nodes ORDER BY created_at DESC LIMIT ?",
                (min(limit * 5, 500),)
            ).fetchall()
        results = [{
            "id": r["id"], "content": (r["content"] or "")[:200],
            "label": r["label"], "node_type": r["node_type"],
            "score": 1.0, "_metadata": r["metadata"]
        } for r in rows]

    # Apply type filter
    if type:
        results = [r for r in results if r.get("node_type") == type]

    # Apply field filters (match against metadata JSON)
    if field_filters:
        filtered = []
        for r in results:
            # Fetch metadata if not already loaded
            meta_str = r.get("_metadata")
            if meta_str is None:
                with db_lock:
                    row = db.execute("SELECT metadata FROM nodes WHERE id=?", (r["id"],)).fetchone()
                meta_str = row["metadata"] if row else "{}"
            try:
                meta = json.loads(meta_str or "{}")
            except (json.JSONDecodeError, ValueError, TypeError):
                meta = {}
            match = True
            for fname, fval in field_filters.items():
                if str(meta.get(fname, "")).lower() != fval.lower():
                    match = False
                    break
            if match:
                filtered.append(r)
        results = filtered

    # Clean up internal fields
    for r in results:
        r.pop("_metadata", None)

    return {"results": results[:limit]}


@app.get("/api/semantic-filter")
async def api_semantic_filter(q: str, threshold: float = 0.3, limit: int = 200):
    """Return node IDs ranked by embedding similarity to a text query."""
    if not q.strip():
        return {"results": []}

    query_emb = await asyncio.to_thread(embed_text, q.strip())
    if query_emb is None:
        raise HTTPException(500, "Embedding failed")

    with db_lock:
        rows = db.execute(
            "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL"
        ).fetchall()

    scored = []
    for r in rows:
        sim = cosine_sim(query_emb, r["embedding"])
        if sim >= threshold:
            scored.append({"id": r["id"], "score": round(float(sim), 4)})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"results": scored[:limit], "query": q.strip()}


@app.post("/api/ask")
async def api_ask(req: AskRequest):
    """Ask your knowledge base - synthesized Q&A over nodes."""
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question required")

    # 1. Semantic search (top 15)
    sem_results = await asyncio.to_thread(semantic_search_full, question, 15, 0.25)

    # 2. FTS search and merge
    try:
        with db_lock:
            fts_rows = db.execute(
                """SELECT id, content, label, node_type FROM nodes_fts
                   WHERE nodes_fts MATCH ? LIMIT 10""",
                (sanitize_fts(question),)
            ).fetchall()
        existing_ids = {r["id"] for r in sem_results}
        for r in fts_rows:
            if r["id"] not in existing_ids:
                sem_results.append({
                    "id": r["id"], "content": r["content"],
                    "label": r["label"], "node_type": r["node_type"],
                    "score": 0.5
                })
    except Exception:
        pass

    # Limit to top 15 sources
    sources = sem_results[:15]

    if not sources:
        return JSONResponse({
            "answer": "I couldn't find any relevant entries in your knowledge base for this question.",
            "sources": []
        })

    # 3. Build context string
    formatted_nodes = "\n\n".join([
        f"[{s['label']}] (id:{s['id']}): {s['content'][:500]}"
        for s in sources
    ])

    # 4. Build messages array
    system_prompt = f"""You are an assistant helping a user query their personal knowledge base.
Answer the question using ONLY the knowledge base entries provided below.
When referencing specific entries, cite them as [Entry Label].
If the knowledge base doesn't contain enough information, say so.
Be concise and direct. Synthesize across entries when relevant.
Do not make up information not present in the entries.

KNOWLEDGE BASE:
{formatted_nodes}"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history for multi-turn
    if req.history:
        for msg in req.history:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current question
    messages.append({"role": "user", "content": question})

    # 5. Call LLM
    answer = await asyncio.to_thread(llm_chat_messages, messages, None, 1500)

    if not answer:
        return JSONResponse({
            "answer": "Sorry, I couldn't generate an answer. The LLM service may be unavailable.",
            "sources": [{"id": s["id"], "label": s["label"], "relevance": s["score"]} for s in sources]
        })

    # 6. Extract citations - find [Label] patterns and match to sources
    cited_labels = re.findall(r'\[([^\]]+)\]', answer)
    source_map = {s["label"].lower(): s for s in sources}

    # Build sources array with citation matching
    source_list = []
    cited_ids = set()
    for label in cited_labels:
        label_lower = label.lower()
        if label_lower in source_map:
            s = source_map[label_lower]
            if s["id"] not in cited_ids:
                cited_ids.add(s["id"])
                source_list.append({"id": s["id"], "label": s["label"], "relevance": s["score"]})

    # Add uncited sources at the end
    for s in sources:
        if s["id"] not in cited_ids:
            source_list.append({"id": s["id"], "label": s["label"], "relevance": s.get("score", 0.5)})

    log.info(f"[ASK] Q: {question[:80]} → {len(source_list)} sources, {len(answer)} chars")

    return JSONResponse({
        "answer": answer,
        "sources": source_list
    })


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Chat with a selection of nodes — synthesize, analyze, and explore."""
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "Message required")
    if not req.node_ids:
        raise HTTPException(400, "At least one node_id required")

    # 1. Fetch full content for each selected node
    nodes_data = []
    with db_lock:
        for nid in req.node_ids:
            row = db.execute(
                "SELECT id, label, content, node_type FROM nodes WHERE id = ?",
                (nid,)
            ).fetchone()
            if row:
                nodes_data.append({
                    "id": row["id"],
                    "label": row["label"],
                    "content": row["content"] or "",
                    "type": row["node_type"]
                })

    if not nodes_data:
        raise HTTPException(404, "No valid nodes found for the given IDs")

    # 2. Build context string from selected nodes
    formatted_nodes = "\n\n".join([
        f"[{n['label']}] (id:{n['id']}, type:{n['type']}): {n['content'][:500]}"
        for n in nodes_data
    ])

    # 3. Build messages array with system prompt
    system_prompt = f"""You are helping a user think through a collection of their personal knowledge base entries.
The user has selected specific entries for you to work with.
Ground your responses in the provided entries. Cite entries as [Entry Label] when referencing them.
Be a thinking partner: synthesize, identify patterns, spot gaps, suggest connections.
When asked to draft or write, use the entries as source material.

SELECTED ENTRIES:
{formatted_nodes}"""

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history for multi-turn
    if req.history:
        for msg in req.history:
            if msg.get("role") in ("user", "assistant") and msg.get("content"):
                messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current message
    messages.append({"role": "user", "content": message})

    # 4. Call LLM
    reply = await asyncio.to_thread(llm_chat_messages, messages, None, 2000)

    if not reply:
        return JSONResponse({
            "reply": "Sorry, I couldn't generate a response. The LLM service may be unavailable.",
            "sources": [{"id": n["id"], "label": n["label"], "type": n["type"]} for n in nodes_data],
            "suggestions": []
        })

    # 5. Build sources list
    sources = [{"id": n["id"], "label": n["label"], "type": n["type"]} for n in nodes_data]

    # 6. Generate contextual follow-up suggestions via heuristics
    suggestions = []
    node_count = len(nodes_data)
    types_present = {n["type"] for n in nodes_data}

    if node_count >= 3:
        suggestions.append("Want me to draft an outline from these notes?")
    if node_count >= 2:
        suggestions.append("Should I identify contradictions between these entries?")
    if "idea" in types_present:
        suggestions.append("Want me to explore how these ideas connect?")
    if "task" in types_present or "project" in types_present:
        suggestions.append("Should I create an action plan from these entries?")
    if "url" in types_present or "paper" in types_present:
        suggestions.append("Want me to summarize the key takeaways from these sources?")
    if node_count == 1:
        suggestions.append("Want me to expand on this entry with related questions?")
        suggestions.append("Should I suggest connections to other entries?")

    # Limit to 2-3 suggestions
    suggestions = suggestions[:3]

    log.info(f"[CHAT] msg: {message[:80]} → {node_count} nodes, {len(reply)} chars")

    return JSONResponse({
        "reply": reply,
        "sources": sources,
        "suggestions": suggestions
    })


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


@app.get("/api/type-schema/{node_type}")
async def api_type_schema(node_type: str):
    """Return the structured field schema for a node type."""
    schema = TYPE_SCHEMAS.get(node_type, {})
    return JSONResponse({"type": node_type, "fields": schema})


@app.get("/api/type-schemas")
async def api_all_type_schemas():
    """Return all type schemas."""
    return JSONResponse(TYPE_SCHEMAS)


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
                   temperature, due_date, created_at, metadata
            FROM nodes WHERE status='inbox'
            ORDER BY created_at ASC
        """).fetchall()
    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "content": r["content"] or "",
            "label": r["label"],
            "type": r["node_type"],
            "url": r["url"],
            "image": r["image_path"],
            "temperature": r["temperature"],
            "due_date": r["due_date"],
            "created": r["created_at"],
            "meta": json.loads(r["metadata"] or "{}") if r["metadata"] else {}
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

    # Handle media (images and PDFs)
    # Strategy: download files quickly, then process LLM enrichment in background
    # This avoids Twilio's 15-second webhook timeout on multi-image MMS
    media_downloaded = 0
    pending_images = []  # (fpath, fname) tuples for background processing
    if num_media > 0:
        import httpx
        for i in range(num_media):
            media_url = form.get(f"MediaUrl{i}")
            media_type = form.get(f"MediaContentType{i}", "")
            if not media_url:
                continue
            is_image = "image" in media_type
            is_pdf = "pdf" in media_type
            if not is_image and not is_pdf:
                log.info(f"[SMS] Skipping unsupported media type: {media_type}")
                continue
            try:
                r = httpx.get(media_url, follow_redirects=True, timeout=30,
                              auth=(TWILIO_SID, TWILIO_TOKEN))
                ext = mimetypes.guess_extension(media_type) or (".jpg" if is_image else ".pdf")
                fname = f"{gen_id()}{ext}"
                fpath = os.path.join(UPLOAD_DIR, fname)
                with open(fpath, "wb") as f:
                    f.write(r.content)
                media_downloaded += 1

                if is_image:
                    # Create placeholder node immediately with user text
                    placeholder_content = body or "Image from SMS (processing...)"
                    nid = create_node(
                        content=placeholder_content, node_type="image", image_path=fname,
                        label=body[:60] if body else f"SMS image {i+1}",
                        metadata={"sms_processing": True}
                    )
                    pending_images.append((fpath, fname, nid.get("id") if isinstance(nid, dict) else None))

                elif is_pdf:
                    pdf_content = body or "PDF from SMS"
                    meta = {"pdf_file": fname}
                    create_node(content=pdf_content, node_type="paper",
                                label=body[:60] if body else f"PDF from SMS",
                                metadata=meta)

            except Exception as e:
                log.error(f"SMS media download error: {e}")

        if media_downloaded > 0:
            response_text = f"{media_downloaded} item{'s' if media_downloaded > 1 else ''} received. Processing descriptions in background."

        # Fire background enrichment for all downloaded images
        if pending_images:
            async def _enrich_sms_images(images, user_body):
                """Background task: enrich SMS images with OCR + Vision after webhook returns."""
                for fpath, fname, node_id in images:
                    if not node_id:
                        continue
                    try:
                        ocr_text, caption, detailed_desc = await asyncio.gather(
                            asyncio.to_thread(ocr_image, fpath),
                            asyncio.to_thread(caption_image, fpath),
                            asyncio.to_thread(detailed_image_description, fpath)
                        )
                        generated_parts = []
                        if caption:
                            generated_parts.append(f"[Caption]: {caption}")
                        if detailed_desc:
                            generated_parts.append(f"[Description]: {detailed_desc}")
                        if ocr_text:
                            generated_parts.append(f"[OCR]: {ocr_text}")
                        generated = "\n".join(generated_parts)
                        if user_body and generated:
                            img_content = f"{user_body}\n\n{generated}"
                        elif user_body:
                            img_content = user_body
                        elif generated:
                            img_content = generated
                        else:
                            img_content = "Image from SMS"
                        meta = {}
                        if detailed_desc:
                            meta["image_description"] = detailed_desc
                        # Remove processing flag
                        meta.pop("sms_processing", None)
                        label = caption[:60] if caption else (user_body[:60] if user_body else fname)
                        # Update the placeholder node
                        emb = embed_text(f"{label}. {img_content[:500]}")
                        with db_lock:
                            db.execute("""
                                UPDATE nodes SET content=?, label=?, metadata=?, embedding=?, updated_at=?
                                WHERE id=?
                            """, (img_content, label, json.dumps(meta), emb,
                                  datetime.datetime.utcnow().isoformat(), node_id))
                            db.commit()
                            try:
                                db.execute("UPDATE nodes_fts SET content=?, label=? WHERE id=?",
                                          (img_content, label, node_id))
                            except Exception:
                                pass
                        log.info(f"[SMS] Enriched image node {node_id}: {label[:40]}")
                        # Fire edge enrichment
                        try:
                            asyncio.create_task(fire_edge_enrichment(node_id))
                        except Exception:
                            pass
                    except Exception as e:
                        log.error(f"[SMS] Background enrichment failed for {fname}: {e}")

            asyncio.create_task(_enrich_sms_images(pending_images, body))

    # Handle text (skip if already used as description for media above)
    if body and num_media == 0:
        # If SMS has a URL + extra text, format so api_nl saves text as user_notes
        import re as _re
        url_match = _re.search(r'(https?://\S+)', body)
        if url_match:
            url_part = url_match.group(1)
            text_part = body.replace(url_part, '').strip()
            if text_part:
                nl_text = f"{url_part}\n\n[Note]: {text_part}"
            else:
                nl_text = body
        else:
            nl_text = body
        result = await api_nl(NLQueryRequest(text=nl_text, channel="sms"))
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


# ─── Canvas Endpoints (v3 feature 4) ─────────────────

@app.get("/api/canvases")
async def api_list_canvases():
    """List all canvases."""
    with db_lock:
        rows = db.execute("SELECT * FROM canvases ORDER BY created_at DESC").fetchall()
    return JSONResponse([{
        "id": r["id"], "name": r["name"], "description": r["description"],
        "color": r["color"], "created_at": r["created_at"], "updated_at": r["updated_at"],
    } for r in rows])


@app.post("/api/canvases")
async def api_create_canvas(req: CanvasCreate):
    """Create a new canvas."""
    cid = gen_id()
    now = datetime.datetime.utcnow().isoformat()
    with db_lock:
        db.execute(
            "INSERT INTO canvases (id, name, description, color, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (cid, req.name, req.description, req.color, now, now)
        )
        db.commit()
    log.info(f"[CANVAS] Created: {req.name} ({cid})")
    return {"id": cid, "name": req.name, "description": req.description, "color": req.color,
            "created_at": now, "updated_at": now}


@app.patch("/api/canvases/{canvas_id}")
async def api_update_canvas(canvas_id: str, req: CanvasUpdate):
    """Update canvas metadata."""
    now = datetime.datetime.utcnow().isoformat()
    with db_lock:
        existing = db.execute("SELECT * FROM canvases WHERE id=?", (canvas_id,)).fetchone()
        if not existing:
            raise HTTPException(404, "Canvas not found")
        updates = []
        params = []
        if req.name is not None:
            updates.append("name=?")
            params.append(req.name)
        if req.description is not None:
            updates.append("description=?")
            params.append(req.description)
        if req.color is not None:
            updates.append("color=?")
            params.append(req.color)
        if updates:
            updates.append("updated_at=?")
            params.append(now)
            params.append(canvas_id)
            db.execute(f"UPDATE canvases SET {', '.join(updates)} WHERE id=?", params)
            db.commit()
    return {"status": "updated"}


@app.delete("/api/canvases/{canvas_id}")
async def api_delete_canvas(canvas_id: str):
    """Delete a canvas (nodes survive)."""
    with db_lock:
        db.execute("DELETE FROM canvases WHERE id=?", (canvas_id,))
        db.commit()
    log.info(f"[CANVAS] Deleted: {canvas_id}")
    return {"status": "deleted"}


@app.get("/api/canvases/{canvas_id}/nodes")
async def api_canvas_nodes(canvas_id: str):
    """Get nodes on a canvas with canvas-specific positions."""
    with db_lock:
        # Verify canvas exists
        canvas = db.execute("SELECT * FROM canvases WHERE id=?", (canvas_id,)).fetchone()
        if not canvas:
            raise HTTPException(404, "Canvas not found")

        rows = db.execute("""
            SELECT n.id, n.content, n.label, n.node_type, n.status, n.pinned as global_pinned,
                   n.starred, n.image_path, n.url, n.due_date, n.temperature,
                   n.visit_count, n.metadata, n.created_at, n.updated_at,
                   cn.x, cn.y, cn.pinned as canvas_pinned
            FROM canvas_nodes cn
            JOIN nodes n ON cn.node_id = n.id
            WHERE cn.canvas_id = ?
        """, (canvas_id,)).fetchall()

        # Get edges between nodes on this canvas
        node_ids = [r["id"] for r in rows]
        edge_rows = []
        if node_ids:
            placeholders = ','.join('?' for _ in node_ids)
            edge_rows = db.execute(f"""
                SELECT id, source_id, target_id, label, weight, auto_created,
                       relationship_type, relationship_description
                FROM edges
                WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
            """, node_ids + node_ids).fetchall()

    node_id_set = set(r["id"] for r in rows)

    nodes_out = []
    for r in rows:
        meta = {}
        try:
            meta = json.loads(r["metadata"] or "{}")
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        nodes_out.append({
            "id": r["id"],
            "content": (r["content"] or "")[:300],
            "label": r["label"],
            "type": r["node_type"],
            "status": r["status"],
            "pinned": bool(r["canvas_pinned"]),  # Use canvas-specific pinned
            "starred": bool(r["starred"]),
            "x": r["x"],  # Canvas-specific position
            "y": r["y"],
            "image": r["image_path"],
            "url": r["url"],
            "due_date": r["due_date"],
            "temperature": r["temperature"],
            "visits": r["visit_count"],
            "meta": meta,
            "created": r["created_at"],
            "updated": r["updated_at"]
        })

    edges_out = []
    for r in edge_rows:
        on_canvas = r["source_id"] in node_id_set and r["target_id"] in node_id_set
        edges_out.append({
            "id": r["id"],
            "source": r["source_id"],
            "target": r["target_id"],
            "label": r["label"],
            "weight": r["weight"],
            "auto": bool(r["auto_created"]),
            "relationship_type": r["relationship_type"] or "related",
            "relationship_description": r["relationship_description"],
            "on_canvas": on_canvas,  # True if both endpoints on canvas
        })

    return {
        "canvas": {
            "id": canvas["id"], "name": canvas["name"],
            "description": canvas["description"], "color": canvas["color"]
        },
        "nodes": nodes_out,
        "edges": edges_out
    }


@app.post("/api/canvases/{canvas_id}/nodes")
async def api_canvas_add_nodes(canvas_id: str, req: CanvasAddNodes):
    """Add nodes to a canvas."""
    added = 0
    now = datetime.datetime.utcnow().isoformat()
    with db_lock:
        canvas = db.execute("SELECT id FROM canvases WHERE id=?", (canvas_id,)).fetchone()
        if not canvas:
            raise HTTPException(404, "Canvas not found")
        for nid in req.node_ids:
            try:
                db.execute(
                    "INSERT OR IGNORE INTO canvas_nodes (canvas_id, node_id, added_at) VALUES (?, ?, ?)",
                    (canvas_id, nid, now)
                )
                added += 1
            except sqlite3.IntegrityError:
                pass
        db.commit()
    log.info(f"[CANVAS] Added {added} nodes to {canvas_id}")
    return {"status": "added", "count": added}


@app.delete("/api/canvases/{canvas_id}/nodes/{node_id}")
async def api_canvas_remove_node(canvas_id: str, node_id: str):
    """Remove a node from a canvas (node still exists globally)."""
    with db_lock:
        db.execute("DELETE FROM canvas_nodes WHERE canvas_id=? AND node_id=?", (canvas_id, node_id))
        db.commit()
    return {"status": "removed"}


@app.post("/api/canvases/{canvas_id}/positions")
async def api_canvas_positions(canvas_id: str, positions: List[CanvasPositionUpdate]):
    """Batch save canvas-specific positions."""
    with db_lock:
        for p in positions:
            db.execute(
                "UPDATE canvas_nodes SET x=?, y=?, pinned=1 WHERE canvas_id=? AND node_id=?",
                (p.x, p.y, canvas_id, p.id)
            )
        db.commit()
    return {"status": "saved", "count": len(positions)}


# ─── Canvas Chat ──────────────────────────────────────
@app.get("/api/canvases/{canvas_id}/chat")
async def api_canvas_chat_history(canvas_id: str):
    """Fetch all chat messages for a canvas."""
    with db_lock:
        rows = db.execute(
            "SELECT id, role, content, created_at FROM canvas_chats WHERE canvas_id=? ORDER BY created_at ASC",
            (canvas_id,)
        ).fetchall()
    messages = [{"id": r["id"], "role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows]
    return {"messages": messages}


@app.post("/api/canvases/{canvas_id}/chat")
async def api_canvas_chat(canvas_id: str, req: CanvasChatMessage):
    """Chat with a canvas — AI has full context of all canvas entries."""
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "Message required")

    # 1. Verify canvas exists, fetch canvas info
    with db_lock:
        canvas_row = db.execute(
            "SELECT id, name, description FROM canvases WHERE id=?", (canvas_id,)
        ).fetchone()
    if not canvas_row:
        raise HTTPException(404, "Canvas not found")

    canvas_name = canvas_row["name"]
    canvas_description = canvas_row["description"] or ""

    # 2. Fetch all nodes on this canvas
    nodes_data = []
    with db_lock:
        rows = db.execute(
            """SELECT n.id, n.label, n.content, n.node_type
               FROM canvas_nodes cn JOIN nodes n ON cn.node_id = n.id
               WHERE cn.canvas_id = ?""",
            (canvas_id,)
        ).fetchall()
        for row in rows:
            nodes_data.append({
                "id": row["id"],
                "label": row["label"],
                "content": row["content"] or "",
                "type": row["node_type"]
            })

    # 3. Load existing chat history (last 20 messages)
    with db_lock:
        history_rows = db.execute(
            "SELECT role, content FROM canvas_chats WHERE canvas_id=? ORDER BY created_at ASC LIMIT 20",
            (canvas_id,)
        ).fetchall()
    prior_history = [{"role": r["role"], "content": r["content"]} for r in history_rows]

    # 4. Build canvas-aware system prompt
    formatted_nodes = "\n\n".join([
        f"[{n['label']}] (id:{n['id']}, type:{n['type']}): {n['content'][:500]}"
        for n in nodes_data
    ])

    desc_line = f"\nDescription: {canvas_description}" if canvas_description else ""
    system_prompt = f"""You are a thinking partner helping the user work with their canvas "{canvas_name}".{desc_line}

This canvas contains {len(nodes_data)} entries:
{formatted_nodes}

Help the user think about this collection. You can:
- Summarize and synthesize
- Identify gaps and suggest additions
- Find contradictions or tensions
- Suggest how to organize or group the entries
- Draft outputs based on the canvas contents

When suggesting new entries to add, describe them clearly.
Cite entries as [Entry Label] when referencing them."""

    # 5. Build messages array
    messages = [{"role": "system", "content": system_prompt}]
    for msg in prior_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    # 6. Call LLM
    reply = await asyncio.to_thread(llm_chat_messages, messages, None, 2000)

    if not reply:
        return JSONResponse({
            "reply": "Sorry, I couldn't generate a response. The LLM service may be unavailable.",
            "sources": [{"id": n["id"], "label": n["label"], "type": n["type"]} for n in nodes_data],
            "suggestions": []
        })

    # 7. Save both user message and assistant reply to canvas_chats
    now = datetime.datetime.utcnow().isoformat()
    user_msg_id = gen_id()
    assistant_msg_id = gen_id()
    with db_lock:
        db.execute(
            "INSERT INTO canvas_chats (id, canvas_id, role, content, created_at) VALUES (?, ?, 'user', ?, ?)",
            (user_msg_id, canvas_id, message, now)
        )
        db.execute(
            "INSERT INTO canvas_chats (id, canvas_id, role, content, created_at) VALUES (?, ?, 'assistant', ?, ?)",
            (assistant_msg_id, canvas_id, reply, now)
        )
        db.commit()

    # 8. Build sources and suggestions
    sources = [{"id": n["id"], "label": n["label"], "type": n["type"]} for n in nodes_data]

    # Generate contextual follow-up suggestions via heuristics
    suggestions = []
    node_count = len(nodes_data)
    types_present = {n["type"] for n in nodes_data}

    if node_count >= 3:
        suggestions.append("Want me to draft an outline from these entries?")
    if node_count >= 2:
        suggestions.append("Should I identify contradictions between these entries?")
    if "idea" in types_present:
        suggestions.append("Want me to explore how these ideas connect?")
    if "task" in types_present or "project" in types_present:
        suggestions.append("Should I create an action plan from these entries?")
    if "url" in types_present or "paper" in types_present:
        suggestions.append("Want me to summarize the key takeaways from these sources?")
    if node_count == 1:
        suggestions.append("Want me to expand on this entry with related questions?")
        suggestions.append("Should I suggest connections to other entries?")
    if node_count == 0:
        suggestions.append("This canvas is empty — want to add some entries first?")

    suggestions = suggestions[:3]

    log.info(f"[CANVAS-CHAT] canvas={canvas_id} msg: {message[:80]} → {node_count} nodes, {len(reply)} chars")

    return JSONResponse({
        "reply": reply,
        "sources": sources,
        "suggestions": suggestions
    })


@app.delete("/api/canvases/{canvas_id}/chat")
async def api_canvas_chat_clear(canvas_id: str):
    """Clear all chat messages for a canvas."""
    with db_lock:
        db.execute("DELETE FROM canvas_chats WHERE canvas_id=?", (canvas_id,))
        db.commit()
    log.info(f"[CANVAS-CHAT] Cleared chat history for canvas={canvas_id}")
    return {"status": "cleared"}


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
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("DELETE FROM canvas_nodes")
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("DELETE FROM canvases")
        except sqlite3.OperationalError:
            pass
        try:
            db.execute("DELETE FROM canvas_chats")
        except sqlite3.OperationalError:
            pass
        db.commit()
    # Recreate today's daily node
    ensure_daily_node()
    rebuild_fts()
    log.info("[RESET] Database wiped and daily node recreated")
    return {"status": "reset", "message": "Database cleared. Fresh daily node created."}


# ─── Resurface Endpoints (v3 feature 3) ─────────────
@app.get("/api/resurface")
async def api_resurface(count: int = Query(3, ge=1, le=10)):
    """Get cold, forgotten nodes for rediscovery feed."""
    results = await asyncio.to_thread(get_resurface_nodes, count)
    return JSONResponse(results)


@app.get("/api/serendipity")
async def api_serendipity():
    """Get a serendipity pair — two moderately similar nodes that might connect."""
    result = await asyncio.to_thread(get_serendipity_pair)
    if not result:
        return JSONResponse({"error": "Not enough permanent nodes for serendipity"}, status_code=404)
    return JSONResponse(result)


# ─── Image serving ───────────────────────────────────
@app.get("/uploads/{filename}")
async def serve_upload(filename: str):
    fpath = os.path.join(UPLOAD_DIR, filename)
    # Prevent path traversal attacks
    if not os.path.realpath(fpath).startswith(os.path.realpath(UPLOAD_DIR) + os.sep):
        raise HTTPException(403, "Forbidden")
    if os.path.exists(fpath):
        return FileResponse(fpath)
    raise HTTPException(404, "File not found")


# ─── Remote Control (SSE push to browser) ──────────────────
_sse_clients: set = set()  # set of asyncio.Queue

async def _broadcast_sse(event: str, data: dict):
    """Push an event to all connected SSE clients."""
    global _sse_clients
    payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    dead = set()
    for q in _sse_clients:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.add(q)
    if dead:
        log.info(f"[SSE] Removed {len(dead)} dead clients")
    _sse_clients -= dead
    log.info(f"[SSE] Broadcast '{event}' to {len(_sse_clients)} live clients")

@app.get("/api/events")
async def api_events():
    """SSE stream for browser remote control."""
    global _sse_clients
    q = asyncio.Queue(maxsize=50)
    _sse_clients.add(q)
    async def stream():
        try:
            yield "event: connected\ndata: {}\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15)
                    yield payload
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass
        finally:
            _sse_clients.discard(q)
            log.info(f"[SSE] Client disconnected, {len(_sse_clients)} remaining")
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

class RemoteCommand(BaseModel):
    action: str  # "navigate", "focus_search", "switch_view"
    query: Optional[str] = None
    node_id: Optional[str] = None
    view: Optional[str] = None

@app.post("/api/remote-command")
async def api_remote_command(cmd: RemoteCommand):
    """Push a command to connected browsers via SSE."""
    data = {"action": cmd.action}
    
    if cmd.action == "navigate" and cmd.query:
        # Search for best matching node
        results = await asyncio.to_thread(semantic_search, cmd.query, 1, 0.15)
        if results:
            data["node_id"] = results[0]["id"]
            data["label"] = results[0].get("label", "")
        else:
            return JSONResponse({"ok": False, "error": f"No node matching '{cmd.query}'"})
    elif cmd.action == "navigate" and cmd.node_id:
        data["node_id"] = cmd.node_id
    elif cmd.action == "switch_view" and cmd.view:
        data["view"] = cmd.view
    
    client_count = len(_sse_clients)
    log.info(f"[REMOTE] Broadcasting to {client_count} SSE clients: {data}")
    await _broadcast_sse("command", data)
    return JSONResponse({"ok": True, "sse_clients": client_count, **data})

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
    uvicorn.run(app, host="127.0.0.1", port=PORT)
