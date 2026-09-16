"""Live read-only audit: BGE + Qwen, exports only to audit/live-paper/."""
import json,time
from pathlib import Path
from bookrag.config import load_config
from bookrag.chat.engine import ChatEngine
from bookrag.retrieve.pipeline import Retriever
from bookrag.paper.generator import PaperGenerator
from bookrag.paper.blueprint import Blueprint,SectionSpec
from bookrag.paper.export import export_paper
from bookrag.ingest.chunker import count_tokens
cfg=load_config();r=Retriever(cfg);engine=ChatEngine(cfg,retriever=r)
thermal=next(b['book_id'] for b in r.store.books() if b['book_id'].startswith('thermal-physics'))
ikigai=next(b['book_id'] for b in r.store.books() if 'ikigai' in b['book_id'])
cases=[
 ('Carnot efficiency','What is the Carnot cycle efficiency?', [thermal],True),
 ('Statistical entropy','How is entropy defined statistically?', [thermal],True),
 ('Gas law','What is the ideal gas law?', [thermal],True),
 ('Gibbs equilibrium','How is Gibbs free energy related to the equilibrium constant?', [thermal],True),
 ('Ikigai','What does ikigai mean?', [ikigai],True),
 ('Antenna','What is a phased array antenna?',None,True),
 ('Cake','How do I bake a chocolate cake?', [thermal],False),
 ('Football','Who won the 2022 FIFA World Cup?', [thermal],False),
 ('Restricted book','What does ikigai mean?', [thermal],False),
 ('Unknown id','What is entropy?', ['does-not-exist'],False),
]
results=[]
for name,q,ids,expected in cases:
 t=time.perf_counter();rr=r.retrieve(q,book_ids=ids)
 # This is the chat gate, stricter than the retrieval floor.
 accepts=rr.grounded and rr.top_score>=float(cfg.get('grounding.answer_threshold'))
 record=dict(test='retrieval',name=name,query=q,book_ids=ids,expected=expected,accepted=accepts,matched=accepts==expected,top_score=rr.top_score,context_tokens=count_tokens(rr.context),context_citations=len(rr.citations),returned_chunks=len(rr.results),ranked_ids=rr.ranked_ids,seconds=round(time.perf_counter()-t,2))
 results.append(record);print(json.dumps(record),flush=True)
print(json.dumps(dict(test='retrieval_summary',matched=sum(x['matched'] for x in results),n=len(results))),flush=True)
for q in ['What is the Carnot cycle? Keep the answer concise.','And its efficiency formula?','Give me a chocolate cake recipe.']:
 t=time.perf_counter();parts=list(engine.ask(q,book_ids=[thermal],stream=True));ans=engine.last_answer
 print(json.dumps(dict(test='end_to_end_stream_chat',question=q,search_query=ans.search_query,grounded=ans.grounded,text=ans.text,used_citations=ans.used_citations,seconds=round(time.perf_counter()-t,2),stream_chars=sum(map(len,parts)))),flush=True)
bp=Blueprint(title='Audit Thermal Physics Paper',sections=[SectionSpec('A',type='short',count=1,marks_each=2),SectionSpec('B',type='mcq',count=1,marks_each=1)],topics=['Carnot cycle','Entropy'])
gen=PaperGenerator(cfg,retriever=r,llm=engine.llm)
t=time.perf_counter();paper=gen.generate(bp,book_ids=[thermal],progress=lambda msg:print(json.dumps(dict(test='paper_progress',message=msg)),flush=True))
files=export_paper(paper,Path('audit/live-paper'),'thermal-audit')
print(json.dumps(dict(test='end_to_end_paper',seconds=round(time.perf_counter()-t,2),requested=paper.report.requested,accepted=paper.report.accepted,retried=paper.report.retried,total_marks=paper.total_marks,verified=[q.verified for q in paper.questions],exports=[str(p) for p in files],questions=[q.to_dict() for q in paper.questions])),flush=True)
