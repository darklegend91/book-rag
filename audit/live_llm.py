"""Real Ollama + real indexed excerpts; retrieval intentionally stubbed because BGE is not cached."""
import json,time
from pathlib import Path
from types import SimpleNamespace
from bookrag.config import load_config
from bookrag.index.store import Store
from bookrag.schemas import ScoredChunk,Question
from bookrag.retrieve.pipeline import Retriever,RetrievalResult
from bookrag.chat.engine import ChatEngine
from bookrag.llm.client import client_from_config
from bookrag.paper.verifier import verify_question
cfg=load_config();llm=client_from_config(cfg);llm.timeout_s=90;llm._client.timeout=90
s=Store(cfg.index_dir).load();cs=[c for c in s.chunks if c.book_id.startswith('thermal-physics')]
r=object.__new__(Retriever);r.cfg=cfg
scored=[ScoredChunk(c,1,rerank_score=1) for c in cs]
context,citations=r.build_context(scored)
rr=RetrievalResult('',[],scored,[c.id for c in cs],context,citations,1,True)
ret=SimpleNamespace(reranker=True,retrieve=lambda *a,**k:rr,arbiter=SimpleNamespace(before=lambda *a:None))
engine=ChatEngine(cfg,retriever=ret,llm=llm)
for q in ['What is the Carnot cycle? Answer in at most 80 words.','Give a chocolate cake recipe. Answer in at most 80 words.']:
 engine.reset();t=time.time();answer=engine.ask(q)
 print(json.dumps(dict(test='live_answer_fixed_context',question=q,seconds=round(time.time()-t,2),grounded=answer.grounded,text=answer.text,used_citations=answer.used_citations)),flush=True)
q=Question(number=1,section='A',text='State the Carnot efficiency formula.',marks=2,qtype='short',bloom='remember',topic='Carnot cycle',answer='Efficiency = 1 - Tc/Th, using absolute temperatures.')
t=time.time();v=verify_question(llm,q,context)
print(json.dumps(dict(test='live_verifier',seconds=round(time.time()-t,2),verdict=v.__dict__)),flush=True)
