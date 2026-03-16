#!/usr/bin/env python3
"""
Backfill enrichment for all existing nodes that lack digest or openai_embedding.
Calls the live server's internal functions via direct import.
Runs in batches with rate-limiting to avoid hammering the OpenAI API.
"""
import os, sys, time, sqlite3, asyncio, logging
os.chdir('/home/jfischer/claude/_open-mind')
sys.path.insert(0, '.')

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('backfill')

DB_PATH = '/home/jfischer/open-mind/openmind.db'
OPENAI_KEY = os.environ.get('OPENAI_API_KEY', '')
BATCH = 5          # nodes per batch
DELAY = 1.5        # seconds between batches
SKIP_TYPES = {'daily', 'digest'}  # structural nodes, not worth enriching

if not OPENAI_KEY:
    log.error('OPENAI_API_KEY not set')
    sys.exit(1)

# ── bootstrap server internals ──────────────────────────────────
import numpy as np
import httpx, json, re, threading

db_lock = threading.Lock()
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row

_client = httpx.Client(
    base_url='https://api.openai.com/v1',
    headers={'Authorization': f'Bearer {OPENAI_KEY}'},
    timeout=60.0
)

LLM_MODEL = 'gpt-4o-mini'
OPENAI_EMBED_MODEL = 'text-embedding-3-small'

def get_openai_embedding(text):
    if not text.strip(): return None
    try:
        r = _client.post('/embeddings', json={
            'model': OPENAI_EMBED_MODEL,
            'input': text[:8000],
            'encoding_format': 'float'
        })
        r.raise_for_status()
        vec = np.array(r.json()['data'][0]['embedding'], dtype=np.float32)
        n = np.linalg.norm(vec)
        if n > 0: vec /= n
        return vec.tobytes()
    except Exception as e:
        log.warning(f'Embedding failed: {e}')
        return None

def generate_digest(label, content, node_type):
    try:
        system = 'You write concise 2-4 sentence narrative digests for personal knowledge graph nodes. Be direct and informative. No preamble.'
        user = f'Type: {node_type}\nLabel: {label}\n\n{content[:600]}'
        r = _client.post('/chat/completions', json={
            'model': LLM_MODEL,
            'messages': [{'role':'system','content':system},{'role':'user','content':user}],
            'max_tokens': 300,
            'temperature': 0.3
        })
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()[:500]
    except Exception as e:
        log.warning(f'Digest failed: {e}')
        return ''

def enrich_node(row):
    nid, label, content, node_type = row['id'], row['label'] or '', row['content'] or '', row['node_type'] or 'note'
    text = f'{label}. {content[:800]}'
    digest = generate_digest(label, content, node_type)
    oai_emb = get_openai_embedding(text)
    conn.execute(
        "UPDATE nodes SET digest=?, openai_embedding=?, enriched_at=datetime('now') WHERE id=?",
        (digest or None, oai_emb, nid)
    )
    conn.commit()
    log.info(f'  [{node_type}] {label[:50]} — digest:{len(digest)}chars emb:{"yes" if oai_emb else "no"}')

# ── fetch nodes needing enrichment ──────────────────────────────
rows = conn.execute("""
    SELECT id, label, content, node_type
    FROM nodes
    WHERE (enriched_at IS NULL OR openai_embedding IS NULL)
      AND node_type NOT IN ('daily','digest')
      AND length(content) > 10
    ORDER BY temperature DESC, created_at DESC
""").fetchall()

total = len(rows)
log.info(f'Nodes to enrich: {total}')

if total == 0:
    log.info('Nothing to do.')
    sys.exit(0)

ok = err = 0
for i, row in enumerate(rows):
    try:
        enrich_node(row)
        ok += 1
    except Exception as e:
        log.error(f'  FAIL {row["id"]}: {e}')
        err += 1
    if (i + 1) % BATCH == 0:
        pct = (i+1)/total*100
        log.info(f'Progress: {i+1}/{total} ({pct:.0f}%) ok={ok} err={err}')
        time.sleep(DELAY)

log.info(f'DONE. ok={ok} err={err} total={total}')
