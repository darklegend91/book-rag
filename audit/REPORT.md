# Book RAG project audit

Date: 15 September 2026

## Assessment

This is a useful local prototype: the modules separate document loading, retrieval, chat, paper generation, and exports clearly. The earlier audit defects have largely been fixed, but the current implementation still has the shortcomings listed below.

Application source and the existing index/books/exports were not changed. Audit scripts and evidence are under `audit/`. Model downloads were explicitly authorized for the live tests. This workspace has no `.git` directory, so a commit-history review was unavailable.

## Test coverage and interpretation

- Syntax compilation of `app.py` and `bookrag`: passed.
- Installed dependency consistency (`pip check`): passed. This does not prove compatibility across clean installations or audit dependency vulnerabilities.
- 20 offline regression tests: passed after the fixes.
- 36 project regression tests in `tests/test_fixes.py`: passed.
- 11 additional deterministic paper/evaluation probes reproduced malformed-output, export, marks, deduplication, and scoring issues.
- Streamlit AppTest: app startup and all four tabs rendered with mocked service dependencies. Two sessions shared an actual `ChatEngine`. Additional flows checked empty book selection, generation/download persistence, and backend errors.
- Real input parsing: inspected all seven PDFs. Four pass the current text-quality gate; three are skipped. Two accepted PDFs are byte-identical.
- Existing index: 1,389 unique chunk IDs; vector matrix 1,389 × 1,024, finite values, approximately unit-length vectors. Only **737 unique chunk texts**.
- Existing saved evaluations: three retrieval cases and sixteen faithfulness claims. These are historical results, not fresh proof of system accuracy.
- Real BGE retrieval tests: five queries completed; supported questions were accepted and an off-topic cake request was refused.

### Evidence and rerun commands

Run from the project root:

```sh
.venv/bin/python -m unittest discover -s audit -p 'test_audit.py' -v
.venv/bin/python audit/paper_probe.py
PYTHONPATH=. .venv/bin/python audit/ui_probe.py
PYTHONPATH=. .venv/bin/python audit/ui_flow_probe.py
PYTHONPATH=. .venv/bin/python audit/inspect_data.py
PYTHONPATH=. .venv/bin/python audit/corpus_probe.py
```

The regression suite is intentionally red until the application defects are fixed. Test fixtures use temporary directories. The UI probes simulate service behavior and do not constitute browser visual/accessibility validation.

## Highest-priority defects

### 1. P1 — Separate users share chat history and model state

**Location:** `app.py:19–22`, `bookrag/chat/engine.py:75`.

`get_engine()` caches a mutable `ChatEngine` with `st.cache_resource`. Two independent AppTest sessions returned the same engine object; a private message inserted in session A appeared in session B's engine history. That history is used for query condensation and generation. Clearing one user's chat resets the shared engine as well.


### 2. P1 — The best retrieval hit can be absent from the actual prompt

**Location:** `bookrag/retrieve/pipeline.py:156–163, 174–205`.

Neighbor expansion sorts passages by book/ordinal before token budgeting. `build_context()` then stops at the first passage that does not fit. A long earlier neighbor can exclude the best hit—or leave the entire context empty. The result still has `grounded=True` and a high `top_score`. Chat subsequently calls the LLM even with empty context.

**Reproduced:** a small high-confidence answer after a long neighbor resulted in an empty context. A high-score/empty-context retrieval result still triggered generation.


### 3. P1 — Negative verifier responses can be treated as approval

**Location:** `bookrag/paper/verifier.py:46–63`.

`bool(data.get("answerable"))` converts the string `"false"` to `True`. A response with `answerable="false"` and confidence 0.99 was accepted. A syntactically valid JSON list crashes at `.get()` instead of producing a failed verdict.


### 4. P1 — “No answer key” still writes answers to JSON

**Location:** `bookrag/paper/export.py:158–170`.

`include_key=False` affects Markdown/DOCX, while JSON calls `to_json(paper)` unconditionally. The resulting JSON still contains `answer` and `correct_option`. This matters when the generated output bundle is distributed to students.

**Reproduced:** a no-key export still contained `"answer": "Same temperature."`.


### 5. P2 — Citation presence is mistaken for grounding

**Location:** `bookrag/chat/engine.py:194–213`.

The finalizer only requires at least one in-range marker. It neither rejects other invalid markers nor checks claim support. `Heat is energy [1]. Invented statement [99].` was marked grounded when only citation [1] existed. Uncited answers are correctly refused, but that is only a basic formatting check.


### 6. P2 — MCQs can pass without complete options or complete checking

**Location:** `bookrag/paper/verifier.py:63, 82–85`.

An MCQ with no options bypasses the option checker and can pass. A checker response evaluating only one of four options can also pass. Separately, malformed generator items such as `{"questions":["bad item"]}` crash generation at `generator.py:194` instead of retrying.


## Ingestion and retrieval shortcomings

| Priority | Finding and evidence |
|---|---|
| P2 | **Chunk size is not a hard limit.** A sentence without punctuation became a 3,999-token chunk with a 100-token target in the offline fallback-tokenizer probe. Existing data has 82 chunks above 700 tokens, largest 4,326. `ingest/chunker.py:206–219`. |
| P2 | **Chapter fragments acquire incorrect metadata.** A short Chapter 1 introduction was prepended to Chapter 2 and cited as page 2. `chunker.py:125–155`. |
| P2 | **Page ranges become broad and misleading.** `_group_units()` does not break on page changes; subsequent sentence chunks inherit the whole unit's page range. A first-page-only chunk cited pages 1–2; existing chunks span up to 23 inclusive pages. `chunker.py:62–89, 178–183`. |
| P2 | **DOCX tables are silently omitted.** A table cell containing a unique fact disappeared on load. `loaders.py:176–190`. |
| P2 | **Markdown heading levels are lost.** An H2 heading became a new chapter because synthetic font size was treated as level 1. `loaders.py:198–204`, `structure.py:94–103`. |
| P2 | **Header cleanup deletes matching body text.** Once text is classified as furniture, every matching occurrence is removed, even mid-page. `loaders.py:104–107`. |
| P2 | **Book IDs collide.** Different files with the same basename and byte size get the same ID. `loaders.py:23–26`. |
| P2 | **Book filter can be bypassed for unknown IDs.** A one-book store searched with `['missing']` returned its book because filtering depends on the number of requested IDs. `index/store.py:107`. |
| P2 | **Nearly half the index is duplicate material.** `air.pdf` and the phased-array textbook have the same SHA-256; 652 redundant chunks in 1,389. |
| P2 | **Index rebuilds are not atomic.** Vectors, chunks, BM25, and manifest are overwritten sequentially. Readers or crashes can observe mismatched files. `index/store.py:45–60`. Static finding; destructive interruption of the real index was not attempted. |
| P3 | **Reusing a Store after save leaves old BM25 cached.** Saving two chunks, searching, then saving one chunk left a two-element lexical score array. `index/store.py:45–63`. |

The three rejected PDFs are visibly poor text inputs according to fresh extraction metrics.  The thermodynamics “book” currently contains only three pages, so it is not broad textbook coverage.

## Paper generation, evaluation, and UI shortcomings

- **Incorrect maximum marks for optional sections:** offering three 2-mark questions with “Answer ANY ONE” exported maximum marks 6 instead of 2 (`generator.py:52–53`, `export.py:36`). Static inspection: `cli paper --marks` sets a field generation does not use (`cli.py:359–360`). 
- **Verification disabled still produces `verified=True`, confidence 1.0:** `generator.py:147–149`. This is a static, direct code finding. 
- **Identical short questions escape deduplication:** `State Ohm's law.` compared with itself returns false (`generator.py:239–247`) because meaningful short words are removed. 
- **PYQ structure inference adds together multiple papers:** without parsed sections, two papers worth four marks each produced an eight-mark section list with declared total four (`paper/pyq.py:88–92`). 
- **Faithfulness can overstate quality:** two extracted claims with only one supported returned judgment yielded 100%, counting only one claim (`evaluate/harness.py:121–125`).  The current 12-claim cap also limits coverage of long answers.
- **Retrieval evaluation misreports empty runs:** if every synthetic question generation fails, the probe returned `n=1` with zero evaluated cases and ordinary numeric metrics. 
- **Empty UI book selection searches everything:** `app.py:62`, `175`, `263`. 
- **Paper downloads disappear on rerun:** a generated paper initially had two download controls; changing an unrelated title field removed both while the preview remained. `app.py:286–300`. 
- **Backend failures interrupt the UI:** a simulated disconnect escaped the chat handler and became an AppTest exception (`app.py:169–178`). 
- **Model switching is not consistently applied:** the chat engine and sidebar/paper/PYQ client are separately cached. Chat changes its engine model, whereas paper generation uses `get_llm()` (`app.py:24–27, 170–172, 260–261`). 
- **Health checks can give false reassurance:** Ollama considers any variant of a model family sufficient (`llm/client.py:83–88`), and the fallback router ignores returned `primary_ok=False` when its HTTP call succeeds (`llm/router.py:116–131`). `doctor` checks Python imports but does not verify that embedding/reranker weights are cached; that omission was encountered during this audit.

## Remaining shortcomings

- Short factual answers can bypass claim checking. The default sentence-citation mode is disabled, and `_statements()` ignores very short statements; for example, a two-word cited claim is not independently checked.
- Duplicate claim-check verdict IDs are silently overwritten. Conflicting verdicts can therefore resolve to whichever duplicate appears last.
- A malformed but type-valid `source_ids` value such as an integer still raises `TypeError` in `_resolve_sources` and can abort a paper slot.
- Duplicate section names are not rejected. Marks are grouped by name and resolved against the first matching section, so two sections both named `A` can have incorrect totals.
- DOCX output does not include the optional-attempt marks line that Markdown shows. A paper can display three questions while its header awards only one attempt, without the DOCX explaining that rule.
- A title containing `/` creates an invalid nested export path and makes result storage fail. The job is cleared before this export error is shown, so the completed paper state is lost from the UI.
- The fallback router changes the primary model only on its primary client. After failover, changing models leaves the fallback client using its old model and reports the wrong active model.
- Retrieval citations can still be noisy: one query returned eight citations from duplicate books and broad multi-page neighbor ranges even though one chunk was the relevance hit. The answer may be correct while the source list is harder to audit.
- The current on-disk index has not been rebuilt with the new ingestion code. It still contains 1,389 chunks, 82 above the configured 700-token target, and two byte-identical books, so the running app retains those old data-quality problems.
- Full paper generation against the configured remote model was not completed because the execution environment's usage limit blocked the authorized run. Live retrieval used BGE locally; live Qwen answer checks from the prior audit used fixed excerpts.
- Query expansion failures are swallowed and silently degrade retrieval, making service/model failures hard to distinguish from ordinary no-expansion configuration.
- The app has no authentication or authorization boundary if exposed beyond a trusted local machine; uploaded books and generated papers are accessible to every connected user.

## Other engineering gaps

- No original automated regression suite or CI configuration was found.
- Dependency requirements have lower bounds but no lockfile, so fresh installations are not reproducible.
- No model/index compatibility check protects against changing embedding models while retaining an old index.
- Index writes are not coordinated across sessions, and there is no background-job cancellation/resume flow.
- `ingest --path` replaces the searchable index with the selected subset rather than adding to the existing library.
- The README's blanket local-only/privacy wording does not cover the configurable remote OpenAI-compatible backend; actual data handling depends on the chosen endpoint.

## Live tests

Real Ollama with configured Qwen 3 8B and the three existing thermodynamics excerpts:

| Check | Observed result | Elapsed |
|---|---|---:|
| Carnot-cycle answer | Grounded answer with citation [2] | 25.28 s |
| Chocolate cake request | Explicitly refused as outside source | 1.01 s |
| Supported Carnot question verification | Approved, confidence 1.0 | 475.96 s |

These three checks used fixed real excerpts, not live BGE retrieval. The slow verifier call is one observation during this audit, not a stable performance benchmark or proof of a particular root cause.  HTTP read timeouts are not necessarily total job deadlines.

The current BGE retrieval run completed five cases: Carnot cycle (grounded, 30.8 s), statistical entropy (grounded, 9.4 s), chocolate cake (refused, 8.6 s), ikigai (grounded, 7.9 s), and phased-array antenna (grounded, 12.2 s). Full paper generation against the remote model was blocked by the execution environment's usage limit.
