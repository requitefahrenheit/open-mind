#!/usr/bin/env python3
"""
Auto-canvas detector for OpenMind.

Detects cohesive clusters using three signals:
  1. Recency burst   — many nodes in same topic touched recently
  2. Semantic density — tight OAI-embedding cluster (DBSCAN)
  3. Graph community  — densely connected subgraph (Louvain-style greedy)

Combines signals, names each candidate with GPT-4o-mini,
then either prints proposals (dry run) or creates the canvases.

Usage:
  python3 auto_canvas.py             # dry run: show proposals
  python3 auto_canvas.py --write     # create canvases + add nodes
  python3 auto_canvas.py --min 4     # min nodes to form a canvas (default 4)
  python3 auto_canvas.py --days 14   # recency window in days (default 14)
"""
import os, sys, sqlite3, uuid, json, datetime, math, argparse
import numpy as np
import httpx

DB     = '/home/jfischer/open-mind/openmind.db'
OPENAI = os.environ.get('OPENAI_API_KEY', '')
MODEL  = 'gpt-4o-mini'

# Type-to-colour mapping (matches frontend)
TYPE_COLORS = {
    'idea': '#a78bfa', 'url': '#38bdf8', 'paper': '#fb923c',
    'note': '#6ee7b7', 'diary': '#60a5fa', 'image': '#e879f9',
    'chore': '#f472b6', 'task': '#f472b6', 'appointment': '#fbbf24',
}

def gen_id(): return uuid.uuid4().hex[:12]

# ───────────────────────────────────────────────────────────────────────────
# Signal 1: Recency burst
# Nodes updated in the last N days, grouped by semantic proximity.
# A "burst" is 4+ nodes that are also closely related (sim > 0.55).
# ───────────────────────────────────────────────────────────────────────────
def recency_clusters(nodes, vecs, id2idx, days=14, sim_thresh=0.55, min_size=4):
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    recent_ids = [
        n['id'] for n in nodes
        if (n['updated_at'] or n['created_at'] or '') > cutoff.isoformat()
        and n['node_type'] not in ('daily', 'digest')
        and n['id'] in id2idx
    ]
    if len(recent_ids) < min_size:
        return []

    # Pairwise sim among recent nodes
    idxs = [id2idx[i] for i in recent_ids]
    rv = np.array([vecs[i] for i in idxs])
    S = rv @ rv.T

    # Simple single-linkage grouping
    groups = {i: {i} for i in range(len(recent_ids))}
    parent = list(range(len(recent_ids)))

    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
            groups[a] = groups[a] | groups.get(b, {b})
            groups.pop(b, None)

    for i in range(len(recent_ids)):
        for j in range(i+1, len(recent_ids)):
            if float(S[i, j]) >= sim_thresh:
                union(i, j)

    results = []
    seen = set()
    for root, members in groups.items():
        if find(root) != root: continue
        if len(members) < min_size: continue
        member_ids = [recent_ids[m] for m in members]
        # Score = avg internal similarity * recency bonus
        midxs = [id2idx[i] for i in member_ids]
        mv = np.array([vecs[i] for i in midxs])
        ms = mv @ mv.T
        n = len(midxs)
        avg_sim = (ms.sum() - n) / max(n * (n-1), 1)
        key = frozenset(member_ids)
        if key not in seen:
            seen.add(key)
            results.append({
                'ids': member_ids,
                'score': float(avg_sim) * 1.5,  # recency bonus
                'signal': 'recency',
            })
    return results


# ───────────────────────────────────────────────────────────────────────────
# Signal 2: Semantic density (DBSCAN-style)
# Find groups where avg pairwise similarity is high regardless of recency.
# ───────────────────────────────────────────────────────────────────────────
def semantic_clusters(nodes, vecs, id2idx, sim_thresh=0.62, min_size=4):
    valid = [n for n in nodes
             if n['id'] in id2idx
             and n['node_type'] not in ('daily', 'digest')]
    if len(valid) < min_size:
        return []

    idxs = [id2idx[n['id']] for n in valid]
    ids  = [n['id'] for n in valid]
    mv   = np.array([vecs[i] for i in idxs])
    S    = mv @ mv.T

    # For each node find its neighbours above threshold
    neighbours = {i: set() for i in range(len(ids))}
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            if float(S[i, j]) >= sim_thresh:
                neighbours[i].add(j)
                neighbours[j].add(i)

    # Core points: >= min_size-1 neighbours
    visited = set()
    clusters = []
    for i in range(len(ids)):
        if i in visited: continue
        if len(neighbours[i]) < min_size - 1: continue
        # BFS expand
        cluster = {i}
        queue   = list(neighbours[i])
        while queue:
            nb = queue.pop()
            if nb in cluster: continue
            cluster.add(nb)
            visited.add(nb)
            if len(neighbours[nb]) >= min_size - 1:
                queue.extend(neighbours[nb] - cluster)
        visited.add(i)
        if len(cluster) >= min_size:
            member_ids = [ids[m] for m in cluster]
            midxs = [id2idx[ids[m]] for m in cluster]
            mv2 = np.array([vecs[k] for k in midxs])
            ms  = mv2 @ mv2.T
            n   = len(midxs)
            avg = (ms.sum() - n) / max(n*(n-1), 1)
            clusters.append({
                'ids': member_ids,
                'score': float(avg),
                'signal': 'semantic',
            })
    return clusters


# ───────────────────────────────────────────────────────────────────────────
# Signal 3: Graph community (greedy modularity)
# Find nodes densely inter-connected via existing edges.
# ───────────────────────────────────────────────────────────────────────────
def graph_communities(nodes, edges, min_size=4, weight_thresh=0.5):
    node_set = {n['id'] for n in nodes if n['node_type'] not in ('daily','digest')}
    # Build adjacency: only significant-weight edges
    adj = {nid: {} for nid in node_set}
    for e in edges:
        s, t, w = e['source'], e['target'], float(e.get('weight') or 0.5)
        if s in node_set and t in node_set and w >= weight_thresh:
            adj[s][t] = w
            adj[t][s] = w

    # Simple label propagation (3 iterations)
    labels = {nid: nid for nid in node_set}
    for _ in range(5):
        order = list(node_set)
        np.random.shuffle(order)
        for nid in order:
            if not adj[nid]: continue
            # Pick label of highest-weight neighbour
            best = max(adj[nid].items(), key=lambda x: x[1])
            labels[nid] = labels[best[0]]

    # Group by label
    groups = {}
    for nid, lbl in labels.items():
        groups.setdefault(lbl, []).append(nid)

    results = []
    for lbl, members in groups.items():
        if len(members) < min_size: continue
        # Compute internal density
        internal = sum(
            1 for m in members for nb in adj[m]
            if nb in set(members)
        ) / 2  # undirected
        possible = len(members) * (len(members)-1) / 2
        density  = internal / max(possible, 1)
        if density >= 0.3:  # at least 30% of possible edges exist
            results.append({
                'ids': members,
                'score': density,
                'signal': 'graph',
            })
    return results


# ───────────────────────────────────────────────────────────────────────────
# Merge overlapping proposals
# ───────────────────────────────────────────────────────────────────────────
def merge_proposals(all_proposals, overlap_thresh=0.75):
    """
    If two proposals share >= overlap_thresh fraction of nodes, merge them.
    Keep the one with highest score as primary; union the node sets.
    """
    merged = []
    used   = set()
    by_score = sorted(all_proposals, key=lambda x: x['score'], reverse=True)
    for i, p in enumerate(by_score):
        if i in used: continue
        group = dict(p)
        group['ids'] = set(p['ids'])
        group['signals'] = {p['signal']}
        for j, q in enumerate(by_score):
            if j <= i or j in used: continue
            ps, qs = set(p['ids']), set(q['ids'])
            overlap = len(ps & qs) / max(len(ps | qs), 1)
            if overlap >= overlap_thresh:
                group['ids'] |= qs
                group['signals'].add(q['signal'])
                group['score'] = max(group['score'], q['score'])
                used.add(j)
        used.add(i)
        group['ids'] = list(group['ids'])
        merged.append(group)
    return merged


# ───────────────────────────────────────────────────────────────────────────
# Name a proposal with GPT-4o-mini
# ───────────────────────────────────────────────────────────────────────────
def name_cluster(node_labels, node_types, client):
    sample = [f'[{t}] {l}' for t, l in zip(node_types[:12], node_labels[:12])]
    prompt = (
        'These nodes form a cohesive cluster in a personal knowledge graph.\n'
        'Nodes:\n' + '\n'.join(sample) + '\n\n'
        'Give this cluster a short, specific name (2-5 words, title case). '
        'Name the actual topic, not a generic category. '
        'Respond with ONLY the name, nothing else.'
    )
    try:
        r = client.post('/chat/completions', json={
            'model': MODEL,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 20,
            'temperature': 0.3,
        })
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content'].strip().strip('"').strip("'")
    except Exception as e:
        print(f'  naming error: {e}')
        return 'Auto Cluster'


# ───────────────────────────────────────────────────────────────────────────
# Pick a colour based on dominant node type
# ───────────────────────────────────────────────────────────────────────────
def dominant_color(node_types):
    from collections import Counter
    most_common = Counter(node_types).most_common(1)[0][0]
    return TYPE_COLORS.get(most_common, '#6ee7b7')


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--write',  action='store_true')
parser.add_argument('--min',    type=int,   default=4)
parser.add_argument('--days',   type=int,   default=14)
parser.add_argument('--skip-existing', action='store_true', default=True,
                    help='Skip canvases whose name already exists (default: on)')
args = parser.parse_args()

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# Load nodes
node_rows = conn.execute('''
    SELECT id, label, node_type, temperature, visit_count,
           created_at, updated_at, openai_embedding, embedding
    FROM nodes
    WHERE node_type NOT IN ('daily','digest')
''').fetchall()
nodes = [dict(r) for r in node_rows]

# Load edges
edge_rows = conn.execute(
    'SELECT source_id as source, target_id as target, weight FROM edges'
).fetchall()
edges = [dict(e) for e in edge_rows]

# Load existing canvas names to avoid duplication
existing_canvas_names = {
    r[0].lower() for r in conn.execute('SELECT name FROM canvases').fetchall()
}

# Build embedding matrix (prefer OAI)
meta = []; vecs = []
for n in nodes:
    if n['openai_embedding']:
        v = np.frombuffer(n['openai_embedding'], dtype=np.float32).copy()
    elif n['embedding']:
        v = np.frombuffer(n['embedding'], dtype=np.float32).copy()
    else:
        continue
    nv = np.linalg.norm(v)
    if nv > 0: v /= nv
    meta.append(n)
    vecs.append(v)

id2idx = {m['id']: i for i, m in enumerate(meta)}
print(f'Nodes loaded: {len(meta)}  Edges: {len(edges)}')

# Run all three detectors
print('Running detectors...')
all_proposals = []
all_proposals += recency_clusters(meta, vecs, id2idx,
                                  days=args.days, min_size=args.min)
all_proposals += semantic_clusters(meta, vecs, id2idx,
                                   min_size=args.min)
all_proposals += graph_communities(meta, edges, min_size=args.min)

print(f'Raw proposals: {len(all_proposals)}')

# Merge overlapping
proposals = merge_proposals(all_proposals)
proposals.sort(key=lambda x: x['score'], reverse=True)
print(f'After merge: {len(proposals)}')

if not proposals:
    print('No clusters found. Try --min 3 or --days 30.')
    sys.exit(0)

# Name each with GPT if key available
client = None
if OPENAI:
    client = httpx.Client(
        base_url='https://api.openai.com/v1',
        headers={'Authorization': f'Bearer {OPENAI}'},
        timeout=30.0
    )

id2node = {n['id']: n for n in meta}

def _name_sim(a, b):
    wa=set(a.lower().split()); wb=set(b.lower().split())
    return len(wa&wb)/max(len(wa|wb),1)

final_proposals = []
proposed_names  = []
for p in proposals:
    member_nodes = [id2node[i] for i in p['ids'] if i in id2node]
    labels = [n['label'] or '' for n in member_nodes]
    types  = [n['node_type'] for n in member_nodes]

    name = name_cluster(labels, types, client) if client else 'Auto Cluster'

    # Skip if exact name already in DB
    if args.skip_existing and name.lower() in existing_canvas_names:
        print(f'  SKIP (exists in DB): {name}')
        continue
    # Skip if too similar to another name proposed this run
    if any(_name_sim(name, pn) >= 0.6 for pn in proposed_names):
        print(f'  SKIP (duplicate): {name}')
        continue
    proposed_names.append(name)

    color  = dominant_color(types)
    type_breakdown = {}
    for t in types: type_breakdown[t] = type_breakdown.get(t, 0) + 1

    final_proposals.append({
        'name': name,
        'color': color,
        'score': round(p['score'], 3),
        'signals': list(p.get('signals', {p.get('signal','?')})),
        'ids': p['ids'],
        'labels': labels,
        'types': type_breakdown,
    })

print()
print('=' * 60)
print(f'CANVAS PROPOSALS ({len(final_proposals)})')
print('=' * 60)
for i, p in enumerate(final_proposals):
    print(f'\n{i+1}. "{p["name"]}"  score={p["score"]}  signals={p["signals"]}')
    print(f'   {len(p["ids"])} nodes | types: {p["types"]}')
    for lbl in p['labels'][:8]:
        print(f'   • {lbl[:60]}')
    if len(p['labels']) > 8:
        print(f'   • ... +{len(p["labels"])-8} more')

if args.write:
    print()
    print('Creating canvases...')
    now = datetime.datetime.utcnow().isoformat()
    created = 0
    for p in final_proposals:
        cid = gen_id()
        conn.execute(
            'INSERT INTO canvases (id, name, description, color, created_at, updated_at) VALUES (?,?,?,?,?,?)',
            (cid, p['name'],
             f'Auto-generated ({"|".join(p["signals"])}). Score: {p["score"]}',
             p['color'], now, now)
        )
        for nid in p['ids']:
            try:
                conn.execute(
                    'INSERT OR IGNORE INTO canvas_nodes (canvas_id, node_id, added_at) VALUES (?,?,?)',
                    (cid, nid, now)
                )
            except Exception:
                pass
        conn.commit()
        print(f'  Created: "{p["name"]}" ({len(p["ids"])} nodes)')
        created += 1
    print(f'\nDone. {created} canvases created.')
else:
    print()
    print('DRY RUN -- pass --write to create these canvases')
