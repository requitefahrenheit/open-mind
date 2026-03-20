#!/usr/bin/env python3
import sqlite3, numpy as np, uuid, sys

DB = '/home/jfischer/open-mind/openmind.db'
THRESH_OAI = 0.60
THRESH_ML = 0.45
MAX_PER_NODE = 8
DRY = '--write' not in sys.argv

def gen_id(): return uuid.uuid4().hex[:12]

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

rows = conn.execute("SELECT id,label,node_type,embedding,openai_embedding FROM nodes WHERE (embedding IS NOT NULL OR openai_embedding IS NOT NULL) AND node_type NOT IN ('daily','digest')").fetchall()
print(f'Nodes: {len(rows)}, OAI: {sum(1 for r in rows if r["openai_embedding"])}')

meta=[]; vecs=[]
for r in rows:
    if r['openai_embedding']:
        v=np.frombuffer(r['openai_embedding'],dtype=np.float32).copy(); use_oai=True
    else:
        v=np.frombuffer(r['embedding'],dtype=np.float32).copy(); use_oai=False
    n=np.linalg.norm(v)
    if n>0: v/=n
    meta.append((r['id'],r['label'] or '',r['node_type'],use_oai))
    vecs.append(v)

existing=set()
for e in conn.execute('SELECT source_id,target_id FROM edges').fetchall():
    existing.add((e[0],e[1])); existing.add((e[1],e[0]))
print(f'Existing edges: {len(existing)//2}')

oai_idx=[i for i,m in enumerate(meta) if m[3]]
ml_idx=[i for i,m in enumerate(meta) if not m[3]]

new_edges=[]
if len(oai_idx)>1:
    ov=np.array([vecs[i] for i in oai_idx])
    S=ov@ov.T
    for ii,i in enumerate(oai_idx):
        for jj,j in enumerate(oai_idx):
            if jj<=ii: continue
            s=float(S[ii,jj])
            if s>=THRESH_OAI and (meta[i][0],meta[j][0]) not in existing:
                new_edges.append((meta[i][0],meta[j][0],s,True))

if len(ml_idx)>1:
    mv=np.array([vecs[i] for i in ml_idx])
    S2=mv@mv.T
    for ii,i in enumerate(ml_idx):
        for jj,j in enumerate(ml_idx):
            if jj<=ii: continue
            s=float(S2[ii,jj])
            if s>=THRESH_ML and (meta[i][0],meta[j][0]) not in existing:
                new_edges.append((meta[i][0],meta[j][0],s,False))

new_edges.sort(key=lambda x:x[2],reverse=True)
id2m={m[0]:m for m in meta}
nc={}
final=[]
for src,tgt,sim,oai in new_edges:
    if nc.get(src,0)<MAX_PER_NODE and nc.get(tgt,0)<MAX_PER_NODE:
        final.append((src,tgt,sim,oai))
        nc[src]=nc.get(src,0)+1; nc[tgt]=nc.get(tgt,0)+1

print(f'New edges (before cap): {len(new_edges)}, after cap: {len(final)}')
print(f'Cross-type: {sum(1 for s,t,_,__ in final if id2m[s][2]!=id2m[t][2])}')
print('\nTop 25:')
for src,tgt,sim,oai in final[:25]:
    sm,tm=id2m[src],id2m[tgt]
    print(f'  {sim:.3f} [{sm[2]}] {sm[1][:36]} <-> [{tm[2]}] {tm[1][:36]}')

if DRY:
    print('\nDRY RUN -- pass --write'); sys.exit(0)

written=0
for src,tgt,sim,oai in final:
    try:
        conn.execute('INSERT INTO edges(id,source_id,target_id,label,weight,auto_created) VALUES(?,?,?,?,?,1)',(gen_id(),src,tgt,'related',round(sim,4)))
        written+=1
    except sqlite3.IntegrityError:
        pass
conn.commit()
print(f'Done. Written: {written}, total edges: {conn.execute("SELECT count(*) FROM edges").fetchone()[0]}')
