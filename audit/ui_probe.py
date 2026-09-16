from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from streamlit.testing.v1 import AppTest
source=Path('app.py').read_text()+'''\nst.session_state.audit_engine = get_engine()\n'''
with patch('bookrag.llm.client.OllamaClient.health_check',return_value={'primary_ok':True}), patch('bookrag.llm.client.OllamaClient.model_catalog',return_value=[]),patch('bookrag.memory.ollama_loaded',return_value=[]),patch('bookrag.memory.system_memory_gb',return_value=(16,8)),patch('bookrag.chat.engine.Retriever',return_value=SimpleNamespace()):
 a=AppTest.from_string(source).run(timeout=30)
 print('SESSION A errors', [e.message for e in a.exception])
 engine_a=a.session_state.audit_engine
 engine_a.history.append({'role':'user','content':'Private session A question'})
 b=AppTest.from_string(source).run(timeout=30)
 print('SESSION B errors', [e.message for e in b.exception])
 engine_b=b.session_state.audit_engine
 print('SHARED ENGINE',engine_a is engine_b,'SECOND SESSION HISTORY',engine_b.history)
 a.multiselect[0].set_value([]).run(timeout=30)
 print('ZERO BOOKS SELECTED',a.multiselect[0].value,'selected_books',a.session_state.filtered_state.get('selected_books','local variable'))
 print('TABS',[t.label for t in a.tabs])
