import json,time,traceback
from bookrag.config import load_config
from bookrag.retrieve.pipeline import Retriever
cfg=load_config()
r=Retriever(cfg)
started=time.time()
try:
 for q in ['What is the Carnot cycle?','How is entropy defined statistically?','How do I prepare chocolate cake?','What is ikigai?','What is a phased array antenna?']:
  t=time.time();result=r.retrieve(q)
  print(json.dumps(dict(question=q,seconds=time.time()-t,grounded=result.grounded,top_score=result.top_score,citations=result.citations,context_chars=len(result.context),ranked_ids=result.ranked_ids)),flush=True)
except Exception:
 traceback.print_exc()
 print('elapsed',time.time()-started,flush=True)
 raise
