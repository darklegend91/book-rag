import hashlib,json
from pathlib import Path
from collections import Counter
from bookrag.index.store import Store
from bookrag.ingest.loaders import discover_books,load_document,text_quality
from bookrag.config import load_config
cfg=load_config();s=Store(cfg.index_dir).load()
for p in discover_books(cfg.books_dir):
 digest=hashlib.sha256(p.read_bytes()).hexdigest()
 pages,meta=load_document(p)
 print(json.dumps(dict(file=p.name,sha256=digest,pages=len(pages),quality=text_quality(pages))),flush=True)
print('CHUNK PAGE SPANS',dict(Counter(c.page_end-c.page_start for c in s.chunks)))
print('EXACT UNIQUE CHUNK TEXT',len(set(c.text for c in s.chunks)),'of',len(s.chunks))
