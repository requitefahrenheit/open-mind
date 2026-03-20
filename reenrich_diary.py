#!/usr/bin/env python3
"""
Re-enrich diary nodes with a better digest prompt.
Diary entries are Claude conversation logs — the digest should read as
a natural memory entry: what was explored, what was learned, the texture
of the exchange. First person plural ('we'), conversational, not clinical.
"""
import os, sys, time, sqlite3, httpx, numpy as np

DB = '/home/jfischer/open-mind/openmind.db'
OPENAI_KEY = os.environ.get('OPENAI_API_KEY', '')
if not OPENAI_KEY:
    print('OPENAI_API_KEY not set'); sys.exit(1)

LLM_MODEL = 'gpt-4o-mini'
EMBED_MODEL = 'text-embedding-3-small'
BATCH_DELAY = 1.0

client = httpx.Client(
    base_url='https://api.openai.com/v1',
    headers={'Authorization': f'Bearer {OPENAI_KEY}'},
    timeout=60.0
)

SYSTEM = """You write memory digests for a personal knowledge graph. 
These are diary entries from conversations between Jeremy (J) and Claude.
Write 2-4 sentences in a natural, first-person-plural voice (\"we explored\", \"J wanted to know\", \"turned out that\").
Capture: what was actually explored or discovered, any surprising or interesting turn, the emotional or intellectual texture.
Be specific — name the actual thing discussed. Avoid generic academic language.
No preamble. No \"This entry...\". Just the memory."""

def make_digest(label, content):
    try:
        r = client.post('/chat/completions', json={
            'model': LLM_MODEL,
            'messages': [
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': f'Label: {label}\n\n{content[:800]}'}
            ],
            'max_tokens': 200,
            'temperature': 0.4
        })
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f'  digest error: {e}'); return None

def make_embedding(label, content):
    try:
        text = f'{label}. {content[:800]}'
        r = client.post('/embeddings', json={
            'model': EMBED_MODEL,
            'input': text[:8000],
            'encoding_format': 'float'
        })
        r.raise_for_status()
        vec = np.array(r.json()['data'][0]['embedding'], dtype=np.float32)
        n = np.linalg.norm(vec)
        if n > 0: vec /= n
        return vec.tobytes()
    except Exception as e:
        print(f'  embedding error: {e}'); return None

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT id, label, content FROM nodes WHERE node_type='diary' AND content IS NOT NULL"
).fetchall()

print(f'Re-enriching {len(rows)} diary nodes...')
ok = err = 0

for i, row in enumerate(rows):
    nid, label, content = row['id'], row['label'] or '', row['content'] or ''
    digest = make_digest(label, content)
    emb = make_embedding(label, content)
    if digest:
        conn.execute(
            "UPDATE nodes SET digest=?, openai_embedding=?, enriched_at=datetime('now') WHERE id=?",
            (digest, emb, nid)
        )
        conn.commit()
        print(f'  [{i+1}/{len(rows)}] {label[:45]}')
        print(f'    {digest[:140]}')
        ok += 1
    else:
        err += 1
    time.sleep(BATCH_DELAY)

print(f'\nDone. ok={ok} err={err}')
