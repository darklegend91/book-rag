import copy,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
import streamlit as st
from streamlit.testing.v1 import AppTest
from bookrag.config import load_config,Config
from bookrag.paper.generator import Paper,GenerationReport
from bookrag.paper.blueprint import Blueprint,SectionSpec
from bookrag.schemas import Question
original=load_config()
source=Path('app.py').read_text()+'\nst.session_state.audit_filter = selected_books\n'
with tempfile.TemporaryDirectory() as d:
 raw=copy.deepcopy(original.raw);raw['paths']=dict(index_dir=str(original.index_dir),books_dir=d+'/books',pyq_dir=d+'/pyqs',export_dir=d+'/exports')
 cfg=Config(raw=raw)
 paper=Paper(Blueprint(sections=[SectionSpec('A',count=1,marks_each=2)]),questions=[Question(1,'A','What is heat?',2,'short','remember','Heat',answer='Energy transfer.',verified=True)],report=GenerationReport(requested=1,accepted=1))
 fake_llm=Mock();fake_llm.health_check.return_value={'primary_ok':True};fake_llm.model_catalog.return_value=[]
 fake_engine=Mock();fake_engine.active_model.return_value=cfg.get('llm.primary');fake_engine.ask.side_effect=RuntimeError('simulated backend disconnect')
 with patch('bookrag.config.load_config',return_value=cfg),patch('bookrag.llm.client.client_from_config',return_value=fake_llm),patch('bookrag.chat.engine.ChatEngine',return_value=fake_engine),patch('bookrag.paper.generator.PaperGenerator') as gen,patch('bookrag.memory.ollama_loaded',return_value=[]),patch('bookrag.memory.system_memory_gb',return_value=(16,8)):
  st.cache_resource.clear();gen.return_value.generate.return_value=paper
  a=AppTest.from_string(source).run(timeout=30)
  print('initial exceptions',[e.message for e in a.exception])
  a.multiselect[0].set_value([]).run(timeout=30)
  print('empty selection resolves to',a.session_state.audit_filter,'-> None passed to engine')
  next(b for b in a.button if b.label=='Generate paper').click().run(timeout=30)
  print('generate exceptions',[e.message for e in a.exception],'downloads',len(a.get('download_button')))
  a.text_input[0].set_value('New title').run(timeout=30)
  print('after unrelated rerun downloads',len(a.get('download_button')),'saved preview',bool(a.session_state.paper_md))
  a.chat_input[0].set_value('What is heat?').run(timeout=30)
  print('backend error exceptions',[e.message for e in a.exception])
  print('chat model',fake_engine.active_model.return_value,'paper client is sidebar client',gen.call_args.kwargs['llm'] is fake_llm)
