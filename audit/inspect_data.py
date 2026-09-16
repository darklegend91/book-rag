import json,os
from collections import Counter
from pathlib import Path
import numpy as np
from bookrag.config import load_config
from bookrag.index.store import Store
cfg=load_config();s=Store(cfg.index_dir).load()
print('CACHE SETTINGS', {k:os.environ.get(k) for k in ['HF_HOME','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','TIKTOKEN_CACHE_DIR']})
print('INDEX',s.vectors.shape,'finite',bool(np.isfinite(s.vectors).all()),'unique ids',len(set(c.id for c in s.chunks)))
print('NORMS',float(np.linalg.norm(s.vectors,axis=1).min()),float(np.linalg.norm(s.vectors,axis=1).max()))
print('CHUNKS',json.dumps({'total':len(s),'over_700':sum(c.token_count>700 for c in s.chunks),'over_4500':sum(c.token_count>4500 for c in s.chunks),'max_tokens':max(c.token_count for c in s.chunks),'max_page_span':max(c.page_end-c.page_start for c in s.chunks)},indent=2))
for b in s.books():print('BOOK',b['book_id'],b['title'],b['n_chunks'],b['n_pages'])
print('SKIPPED',json.dumps(s.manifest.get('skipped',[])))
for f in sorted(cfg.index_dir.glob('eval_*.json')):
 data=json.loads(f.read_text());print('EXISTING EVAL',f.name,json.dumps(data)[:1400])
print('LARGEST',[(c.id,c.token_count,c.page_start,c.page_end) for c in sorted(s.chunks,key=lambda c:-c.token_count)[:5]])
