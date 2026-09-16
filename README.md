# Book RAG — Chat With Your Books & Build Exam Papers

A fully local retrieval-augmented generation pipeline over a set of books. It does three things:

1. **Chat** with one or many books, with every claim cited to book, chapter and page — and an explicit refusal when the books don't cover the question.
2. **Generate question papers** *from the book and nothing else*, where every question is independently verified as answerable from the retrieved source before it ships.
3. **Learn from previous-year question papers (PYQs)** — extract an examiner's section layout, mark bands, command verbs, phrasing and recurring topics, then write *new* questions from the book in that same pattern.

By default everything runs on your machine: no API keys, no data leaving the box. Generation uses **Qwen 3 via Ollama**; retrieval uses **BGE-M3** embeddings and a **BGE cross-encoder reranker**. If you set `llm.provider: openai` (or enable it as the fallback), retrieved book excerpts, your questions and generated papers are sent to that endpoint, and its own data-handling terms apply.

---

## Features

- **Private, local AI** — with the default Ollama backend the language model runs locally, so books, questions and answers are not sent to an external API. The optional OpenAI-compatible backend sends them to whichever server you configure.
- **Multi-format book library** — ingest PDF, EPUB, DOCX, Markdown and plain-text books, including multiple books in one searchable index.
- **Structure-aware indexing** — preserves chapters, sections and page metadata; removes repeated PDF headers and footers; and avoids splitting chunks across chapter boundaries.
- **Grounded book chat** — ask one-off questions or hold a multi-turn conversation, restrict retrieval to a selected book, inspect the source excerpts and receive chapter/page citations.
- **Hallucination control** — hybrid retrieval, cross-encoder reranking and relevance thresholds make the assistant explicitly refuse questions that the indexed books do not cover.
- **Verified question-paper generation** — create short-answer, long-answer, MCQ, numerical and true/false papers from a configurable section blueprint. Each draft is checked against the source before acceptance.
- **Previous-year paper (PYQ) analysis** — learn section structure, mark bands, command verbs, topic frequency and writing style from past papers while keeping the new questions grounded only in the books.
- **Multiple export formats** — save generated papers and answer keys as Markdown, DOCX and JSON, including generation and verification details.
- **CLI and web UI** — use the full workflow from the terminal or through a four-tab Streamlit interface for Chat, Question Papers, PYQs and Library management.
- **Built-in evaluation and memory controls** — measure retrieval/answer faithfulness, inspect system readiness and unload local models when memory is tight.

---

## Table of contents

- [Features](#features)
- [Why this design](#why-this-design)
- [How to run](#how-to-run)
- [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
- [How accuracy is enforced](#how-accuracy-is-enforced)
- [Memory: fitting this in 16 GB](#memory-fitting-this-in-16-gb)
- [CLI reference](#cli-reference)
- [The web UI](#the-web-ui)
- [Configuration reference](#configuration-reference)
- [Code map](#code-map)
- [Evaluation](#evaluation)
- [Access and security](#access-and-security)
- [Tuning guide](#tuning-guide)
- [Deploying on a server](#deploying-on-a-server)
- [Troubleshooting](#troubleshooting)

---

## Why this design

The hard requirement is **questions from the book, never outside it**. A naive RAG loop (embed → top-k cosine → stuff into prompt) fails that requirement in three specific ways, and each stage below exists to close one of them:

| Failure mode | What goes wrong | The countermeasure here |
|---|---|---|
| **Retrieval misses** | The model gets weak context and fills the gap from pretraining. The answer looks plausible and is wrong. | Hybrid dense+lexical retrieval over multiple query rewrites, fused with RRF. Recall is made generous *before* precision is imposed. |
| **Retrieval hits something merely related** | Cosine similarity rewards topical overlap, not answerability. | A cross-encoder reranks the survivors and a hard score floor discards weak ones. Below the floor, the LLM is never called at all. |
| **The model embellishes** | Even with good context, a 4–8B model adds a detail the source doesn't contain. | Every generated question gets an **independent verification pass** — a separate call, temperature 0, moderator persona, no memory of how the question was written. Failures are regenerated with the reason fed back; repeat failures are dropped. |

The ordering matters: **recall first, precision once, late**. Chasing precision early (a high similarity threshold on the bi-encoder, say) throws away passages the cross-encoder would have recognised.

---

## How to run

Run all commands below from the project root.

### 1. Install the prerequisites

You need:

- macOS or Linux;
- Python 3.11;
- [Ollama](https://ollama.com/download);
- at least 11 GB of free disk space for the downloaded models;
- 16 GB of RAM recommended; and
- an internet connection for the initial model downloads.

Create a virtual environment and install the Python dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Ollama and download Qwen 3

Start the Ollama server in one terminal and leave it running:

```bash
ollama serve
```

In a second terminal, download the configured language model:

```bash
ollama pull qwen3:8b
```

If the Ollama desktop application is already running, you do not need to run `ollama serve` separately.

### 3. Check the setup

Verify Ollama, the model, Python dependencies, the compute device, the index and available memory:

```bash
python -m bookrag.cli doctor
```

An absent index is expected until the next step.

### 4. Add and index your books

Place one or more supported files (`.pdf`, `.epub`, `.docx`, `.md` or `.txt`) in `data/books/`, then run:

```bash
python -m bookrag.cli ingest
```

The first indexing run downloads BGE-M3 and the BGE reranker (about 2.3 GB each); later runs reuse the cached models. Re-run `ingest` whenever you add, remove or replace books. You can inspect the resulting library with:

```bash
python -m bookrag.cli books
```

### 5. Launch the web application

```bash
streamlit run app.py
```

Streamlit opens the app in your browser, normally at `http://localhost:8501`. The web UI lets you upload and index books, chat with them, analyse previous-year papers and generate downloadable question papers.

### 6. Or use the CLI

Ask a single question:

```bash
python -m bookrag.cli ask "Explain the Carnot cycle and derive its efficiency."
```

Start a multi-turn chat:

```bash
python -m bookrag.cli chat
```

Generate a question paper from the default blueprint in `config.yaml`:

```bash
python -m bookrag.cli paper --title "Midterm Examination"
```

Generated Markdown, DOCX and JSON files are written to `data/exports/`. See the [CLI reference](#cli-reference) for book filters, topic restrictions, PYQ styling, custom blueprints and evaluation commands.

---

## The pipeline, stage by stage

```
                          ┌────────────────────────────────┐
  data/books/*.pdf  ────► │ 1. LOAD    loaders.py          │  layout-aware extraction
                          │            font sizes, spans   │  + header/footer stripping
                          └───────────────┬────────────────┘
                                          ▼
                          ┌────────────────────────────────┐
                          │ 2. STRUCTURE  structure.py     │  TOC + typography →
                          │               chapter/section  │  every line tagged
                          └───────────────┬────────────────┘
                                          ▼
                          ┌────────────────────────────────┐
                          │ 3. CHUNK      chunker.py       │  token-sized, overlapped,
                          │               ~700 tok         │  never across a chapter
                          └───────────────┬────────────────┘
                                          ▼
                          ┌────────────────────────────────┐
                          │ 4. EMBED+INDEX  embedder.py    │  BGE-M3 fp16 → vectors.npy
                          │                 store.py       │  + BM25 → bm25.pkl
                          └───────────────┬────────────────┘
                                          │
   ═══════════════════════ query time ════╪══════════════════════════════
                                          ▼
   question ──► 5. EXPAND ──► 6. HYBRID ──► 7. RRF ──► 8. RERANK ──► 9. FLOOR
                multi-query    dense +      fuse       cross-        discard
                + HyDE         BM25                    encoder       weak hits
                                                                       │
                          ┌────────────────────────────────────────────┤
                          ▼                                            ▼
              ┌───────────────────────┐                   ┌────────────────────────┐
              │ 10a. CHAT engine.py   │                   │ 10b. PAPER generator.py│
              │  grounded + cited     │                   │  write → VERIFY → retry│
              │  or explicit refusal  │                   │  → dedupe → export     │
              └───────────────────────┘                   └────────────────────────┘
```

### 1. Load — `bookrag/ingest/loaders.py`

PDFs are read through PyMuPDF's **span tree**, not `page.get_text()`. That preserves font size and boldness per line, which stage 2 needs. EPUB, DOCX and Markdown map their heading levels onto synthetic font sizes so one structure detector serves every format.

`strip_page_furniture()` removes running headers and footers: a line that repeats near the top or bottom of ≥30% of pages is furniture. Left in, "Principles of Thermal Physics · Page 47" contaminates every chunk and drags down retrieval precision.

### 2. Structure — `bookrag/ingest/structure.py`

Rebuilds the chapter/section hierarchy from two sources, in priority order:

1. The PDF's **embedded table of contents**, when present — authoritative.
2. **Font-size outliers**: a line whose size exceeds the modal body size by `heading_size_ratio`, short enough and shaped like a title (numbered `4.2 …`, or `Chapter/Unit/Part …`, or Title Case).

Every line ends up tagged with the chapter and section it lives under. This is what makes `Ch. 2 > 2.3 The Carnot Cycle, pp. 88–89` a real citation rather than a guess, and what lets the paper generator target a syllabus topic.

### 3. Chunk — `bookrag/ingest/chunker.py`

Exam questions hinge on complete definitions and derivations, so chunking is semantic, not arithmetic:

- sized in **real tokens** (tiktoken), not characters;
- **never merges across a chapter boundary**;
- prefers to break at section → paragraph → sentence, in that order;
- **overlaps** consecutive chunks by `chunk_overlap_tokens`, so a concept split across a boundary is recoverable from either side;
- folds sub-minimum fragments into a neighbour (backward normally, forward for a leading fragment) rather than emitting low-signal noise;
- cites a chunk by the section it **starts** in — a chunk spanning 1.1–1.3 cites 1.1, not 1.3.

**Contextual headers**: the text that gets *embedded* is prefixed with `[Book > Chapter > Section]`, while the text the LLM quotes back is not. This measurably lifts retrieval on topic-style queries without polluting quoted answers.

### 4. Embed & index — `bookrag/index/`

BGE-M3 (1024-dim, multilingual, long-context) in fp16, plus a BM25 index over the same texts.

The store is **one `.npy` matrix and one `.jsonl`** — no vector database. A dedicated ANN index only pays off above roughly a million vectors; a few books is 10k–100k chunks, where an exact float32 matrix multiply returns in milliseconds and is *exactly* correct. Given the brief is maximum accuracy, taking ANN's recall loss would be a pure downside.

### 5–7. Query expansion, hybrid retrieval, fusion

**Multi-query** (`retrieval.multi_query`): the LLM rewrites your question three ways — formal textbook terminology, broader conceptual phrasing, narrow keywords. A student rarely phrases a question the way the book does.

**HyDE** (`retrieval.hyde`): the LLM writes a short hypothetical *answer* and that gets embedded too. A hypothetical answer lives much closer in embedding space to the real passage than the question does. Its factual accuracy is irrelevant — it's a retrieval probe, never shown to anyone.

**Hybrid**: every variant runs through *both* dense and BM25. Dense finds paraphrases; BM25 finds the exact tokens dense models blur — symbols, formula names, rare proper nouns, statute numbers.

**RRF fusion**: `score(d) = Σ 1/(k + rank(d))` over all lists. Rank-based, so it needs no score normalisation between two retrievers whose scores aren't comparable. The original query's lists are weighted above its rewrites.

### 8–9. Rerank and floor — `bookrag/retrieve/rerank.py`

A bi-encoder scores query and passage *independently*, so it captures topical similarity but not whether the passage **answers** the query. The cross-encoder reads both together and is far more discriminative.

It also makes the refusal threshold meaningful: its scores are **probabilities in [0, 1]** from an "is this relevant" head (sentence-transformers applies the sigmoid for you), so a low top score is genuine evidence the books don't cover the question. Measured on a thermodynamics index, on-book questions score 0.29–0.96 and off-book ones 0.00002–0.0004 — three orders of magnitude of separation. Chunks below `rerank.min_score` are discarded; if nothing survives, **the LLM is never called**.

Surviving chunks get **neighbour expansion** (`retrieval.neighbor_window`) — adjacent chunks are pulled in at a discounted score and the set is re-sorted into reading order, so a derivation truncated by a chunk boundary is still complete and still in order.

### 10a. Chat — `bookrag/chat/engine.py`

- **History condensation**: "and its derivation?" is rewritten into a standalone query before retrieval, so pronouns don't destroy recall.
- **Citations**: `[n]` markers are resolved against real chunk metadata. The model cannot invent a page number, because it never writes one.
- **Refusal**: below threshold, or on a `NOT_IN_SOURCE:` reply, you get an explicit "not covered" rather than a confident paraphrase of a weak hit.

### 10b. Paper generation — `bookrag/paper/`

Per question slot, not one giant call:

1. **Pick a topic** — PYQ-weighted if a profile is loaded, else round-robin over detected chapters, with used topics penalised so the paper spans the syllabus.
2. **Retrieve evidence for that topic** — each question is grounded in evidence chosen *for it*.
3. **Generate** against that evidence, in the PYQ house style if supplied.
4. **Verify** independently — see below.
5. **Deduplicate** by token overlap against questions already accepted, catching the near-restatements a model produces when two chapters cover the same concept. Command verbs are stripped before comparison, so "State X" and "Define X" collide. The threshold is 0.45, set from measurement: real output produced *"Define the term 'heat capacity' and state its two forms"* vs *"Define the term 'heat capacity' for a substance"* at 0.571, and two Zeroth-Law questions at 0.50 — both obvious duplicates that a 0.6 cutoff let through.

6. **Sanitise** deterministically. A 4–8B model leaks source references and taxonomy labels no matter how firmly the prompt forbids them, so `_strip_source_references()` removes them by regex: *"as described in Chapter 1"*, *"from the given relation"*, *"according to the text"*, and taxonomy verbs (*"Create an expression"* → *"Derive an expression"*). "Analyse" and "Evaluate" are left alone — those are legitimate exam verbs.

Slot-by-slot means one bad question can be retried without regenerating the paper.

Output goes to `data/exports/` as Markdown, DOCX and JSON. The JSON carries the full generation report — acceptance rate, retries, dropped topics and every rejection with the moderator's reason.

---

## How accuracy is enforced

Five independent mechanisms. Each one alone is defeatable; together they're what makes "from the book, not outside it" hold.

**1. Grounding rules in the prompt.** One shared `GROUNDING_RULES` block, centralised in `prompts.py`: only the excerpts are admissible, every factual sentence carries a citation marker, and the model must emit `NOT_IN_SOURCE:` rather than guess.

**2. The reranker floor.** Structural, not persuasive. If nothing clears `rerank.min_score`, no LLM call happens, so there is nothing to hallucinate from.

**3. Independent verification** — `bookrag/paper/verifier.py`. The generator wrote the question while looking at the excerpts, so asking it "is this grounded?" is self-marking. Verification is a **separate call** with a moderator persona, temperature 0, given only the excerpts and the finished question — no memory of how it was produced. It rejects when:

- answering needs knowledge not in the excerpts,
- the excerpts are too thin for the marks allotted,
- the question is ambiguous,
- the model answer states anything unsupported,
- an MCQ's correct option isn't supported, or a distractor is also correct.

A rejection feeds its reason back into a retry (`grounding.verification_retries`). Questions that fail repeatedly are **dropped, not shipped**. A verifier *error* counts as a failure, never a pass.

**MCQs get a second, narrower pass.** The general moderator reliably catches ungrounded content but not two distractors that are logically *equivalent*. Measured: it passed an option set containing both *"the total entropy of an isolated system never decreases"* and *"…increases or remains constant"* — the same claim twice — at confidence 1.00. So every MCQ additionally goes through `_check_mcq_options()`, which judges each option **independently** and requires exactly one to be true, explicitly instructed to compare meaning rather than wording. If the checker's answer disagrees with the generator's, the answer key is realigned; if two options survive, the question is rejected and regenerated.

**4. Deterministic post-processing.** The prompt forbids "according to the text" and similar tells, but a 4–8B model leaks one occasionally, so `_strip_source_references()` removes them by regex. Belt and braces.

**5. Measurement.** `bookrag/evaluate/harness.py` turns "feels better" into a number — see [Evaluation](#evaluation).

### Where PYQs fit

PYQs drive **pattern transfer, not content**. They contribute section layout, mark bands, command verbs, phrasing conventions and topic weighting. They are *never* used as source material for answers — that would break the grounding guarantee. New questions are always written from book excerpts, in the PYQ's style.

---

## Memory: fitting this in 16 GB

This pipeline can want four models resident at once — an LLM for generation, a second smaller LLM for cheap calls, a bi-encoder, and a cross-encoder. Naively that is **~15 GB** and the machine swaps.

### Measured per stage

On an M4 / 16 GB, using the Metal allocator (`torch.mps.driver_allocated_memory`) rather than process RSS — on Apple Silicon model weights live in unified memory that RSS does not track, so RSS *understates* usage badly here:

| Stage | Python (RSS) | Encoders (MPS) | Ollama | **Total** |
|---|---|---|---|---|
| Baseline (torch imported) | 0.21 GB | — | — | **0.21 GB** |
| **Ingest** (BGE-M3 only) | 2.08 GB | 1.10 GB | — | **3.18 GB** |
| ↳ after `unload_after_ingest` | 2.01 GB | 0.00 GB | — | **2.01 GB** |
| **Retrieval** (BGE-M3 + reranker) | 2.70 GB | 2.13 GB | — | **4.83 GB** |
| **Chat** (+ qwen3:8b) | 0.81 GB | 2.13 GB | 6.08 GB | **9.02 GB** |
| **Paper generation** (generate + verify) | 0.83 GB | 2.13 GB | 6.08 GB | **9.04 GB** |

Peak transient process RSS during a full run: **4.36 GB**.

Two things worth reading off that table:

- **Ingest never loads the LLM or the reranker.** It is the cheapest stage at ~3.2 GB, so you can index a large library on a busy machine.
- **Chat and paper generation cost the same.** Paper generation makes far *more* LLM calls, but they are sequential against one resident model, so peak memory is flat. A long paper costs time, not RAM.

### What totals look like before and after tuning

| | Naive | Tuned |
|---|---|---|
| Ollama (`qwen3:8b`) | 7.4 GB @ 16k ctx | 6.1 GB @ 8k ctx |
| Second LLM (`qwen3:4b-instruct`) | ~3.0 GB | 0 GB |
| Encoders (fp32 → fp16) | ~4.6 GB | 2.13 GB |
| **Total at steady state** | **~15 GB** | **~9.0 GB** |

### The four levers

All live in `config.yaml` under `memory:`, implemented in `bookrag/memory.py`.

**1. One LLM, not two.** `llm.fast` deliberately points at the *same* model as `llm.primary`. Two distinct models means Ollama holds both. Point `fast` at `qwen3:4b-instruct` only if you have ≥32 GB. *Saves ~3.0 GB.*

**2. Halve the context.** `num_ctx: 8192` keeps the KV cache near 0.6 GB; 16384 costs an extra ~1.3 GB. `retrieval.max_context_tokens` is set to 4500 to leave room inside it for the system prompt and the answer — **raise them together or not at all.** *Saves ~1.3 GB.*

**3. fp16 encoders.** `memory.fp16_encoders: true` loads BGE-M3 and the reranker in half precision: ~2.3 GB → ~1.1 GB each, at no measurable accuracy cost (these encoders are published in fp16 — a spot-check scored 0.99605 in fp16 vs 0.99607 in fp32). *Saves ~2.3 GB.*

> The dtype must be set at **load time** via `model_kwargs={"torch_dtype": torch.float16}`. Calling `.half()` and reassigning the inner module breaks sentence-transformers 6.x at inference, with a misleading traceback that points at the tokenizer.

**4. Release the allocator cache after reranking.** `memory.empty_cache_after_rerank: true`. Scoring 40 passages at 1024 tokens leaves ~1.05 GB of activation buffers cached by Metal, on top of only ~1.06 GB of live weights — the cache is as large as the model. Returning it after each rerank nearly halves the stage. Batch size barely affects this; the high-water mark is set by the largest single batch and then retained. *Saves ~1.05 GB.*

Ingest also releases the embedder when it finishes (`memory.unload_after_ingest`), which is why the table shows 3.18 GB → 2.01 GB.

Reclaim memory at any time:

```bash
.venv/bin/python -m bookrag.cli free --all
```

`doctor` reports free memory and warns if more than one LLM is resident. As a hard guard you can also cap Ollama globally (takes effect on its next restart):

```bash
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
```

**If you're still tight**, in the order I'd try them:

1. `llm.num_ctx: 6144` with `retrieval.max_context_tokens: 3000` — saves ~0.5 GB, costs some answer depth on long questions.
2. `rerank.candidates: 20` — saves little memory (the cache is dominated by the largest batch, not the count) but roughly halves rerank latency.
3. `llm.primary: qwen3:4b-instruct` — drops Ollama from 6.1 GB to ~3.0 GB, the single biggest remaining win, but costs noticeable quality on question verification and multi-step reasoning. Try 1 and 2 first.

Lowering `embedding.batch_size` helps only during ingest, which is already the cheapest stage.

---

## CLI reference

```bash
.venv/bin/python -m bookrag.cli <command>
```

| Command | What it does |
|---|---|
| `doctor` | Check Ollama, models, index, dependencies, device, free memory |
| `ingest` | Parse, chunk, embed and index everything in `data/books/` |
| `ingest --path FILE` | Index a single file |
| `books` | List indexed books and their detected chapters |
| `ask "question"` | One grounded, cited answer. `--show-context` to see the excerpts |
| `chat` | Interactive multi-turn session (`/reset`, `/sources`, `/quit`) |
| `pyq <file-or-dir>` | Analyse previous-year papers into a reusable profile |
| `paper` | Generate a verified question paper |
| `free [--all]` | Evict LLMs from Ollama to reclaim memory |
| `eval retrieval -n 30` | Self-supervised Hit@k and MRR |
| `eval faithfulness <file>` | Fraction of answer claims actually supported |

Useful flags:

- `--book <book_id>` restricts chat or generation to one book (ids come from `books`).
- `--blueprint <file.json>` supplies a full paper spec.
- `--pyq <profile.json>` applies PYQ style and topic weighting; add `--pyq-structure` to also copy the section layout.
- `--topics "Entropy,Phase Transitions"` restricts the syllabus.
- `--no-key` omits the answer key.

### Worked example: a paper in an examiner's style

Analyse three past papers, then generate a new one that mirrors their structure:

```bash
.venv/bin/python -m bookrag.cli pyq data/pyqs
```

```bash
.venv/bin/python -m bookrag.cli paper --pyq data/index/pyq_profile.json --pyq-structure --title "End-Semester Examination"
```

---

## The web UI

```bash
.venv/bin/streamlit run app.py
```

Four tabs:

- **💬 Chat** — streaming answers with expandable sources, and a warning banner when a question isn't covered.
- **📝 Question Paper** — build the section blueprint interactively, toggle PYQ style, watch the generate-and-verify log live, then download Markdown or DOCX. Rejected drafts are shown with the moderator's reasoning.
- **🗂 Previous-Year Papers** — upload PYQs, analyse them, and inspect the extracted questions, Bloom mix, recurring topics and derived style guide.
- **📖 Library** — upload books, build the index, browse each book's detected outline.

The sidebar shows Ollama health and lets you restrict every operation to a subset of books.

---

## Configuration reference

All tuning lives in `config.yaml`. The knobs that matter most:

| Key | Default | Effect |
|---|---|---|
| `ingest.chunk_tokens` | 700 | Larger = more context per hit, fewer, blunter hits |
| `ingest.chunk_overlap_tokens` | 120 | Insurance against concepts split at a boundary |
| `ingest.heading_size_ratio` | 1.15 | Lower if chapters aren't being detected |
| `retrieval.multi_query` / `hyde` | false | Recall boosters; each costs a full LLM call on the critical path before any searching starts |
| `retrieval.neighbor_window` | 1 | Adjacent chunks pulled in for continuity |
| `retrieval.max_context_tokens` | 4500 | **Must fit inside `llm.num_ctx`** alongside the answer |
| `rerank.candidates` | 40 | Chunks entering the cross-encoder. Main speed/RAM lever |
| `rerank.top_k` | 8 | Chunks reaching the LLM |
| `llm.provider` | ollama | `openai` for any OpenAI-compatible server, `ollama` for local Ollama. Set via `BOOKRAG_LLM_PROVIDER` in `.env` |
| `llm.openai.base_url` | — | Your server, including the `/v1` suffix (`BOOKRAG_LLM_BASE_URL`) |
| `llm.fallback.enabled` | true | Retry on the fallback backend when the primary is unreachable, per call |
| `llm.primary` | qwen3:8b | Startup default only — switch at runtime from the sidebar, `ask --model`, or `/model`. Run `cli models` for what each installed model is good for |
| `memory.profile` | balanced | `balanced` keeps all three models resident and only frees under pressure; `conservative` hands the machine to one stage at a time |
| `rerank.min_score` | 0.02 | **The main anti-hallucination lever.** A probability, not a logit. Raise to refuse more |
| `rerank.max_length` | 512 | Truncation per (query, passage) pair; rerank cost scales with it |
| `rerank.candidates` | 20 | Rerank cost is linear in this — the biggest retrieval lever |
| `grounding.answer_threshold` | 0.05 | Chat refuses below this, without calling the LLM at all |
| `grounding.verify_questions` | true | Turning this off removes the strongest accuracy guarantee |
| `grounding.min_verify_confidence` | 0.7 | Raise for a stricter paper and a lower acceptance rate |
| `llm.temperature` | 0.1 | Low = faithful. Don't raise it for answering |
| `llm.num_ctx` | 8192 | See [Memory](#memory-fitting-this-in-16-gb) |
| `memory.fp16_encoders` | true | Halves encoder RAM (~2.3 GB saved) |
| `memory.empty_cache_after_rerank` | true | Returns ~1 GB of allocator cache after each rerank |
| `memory.unload_after_ingest` | true | Frees the embedder when indexing finishes |

---

## Code map

```
config.yaml                  every tuning knob
app.py                       Streamlit UI (4 tabs)
bookrag/
  config.py                  typed dotted-path access to config.yaml
  schemas.py                 Chunk, Page, ScoredChunk, Question
  memory.py                  fp16, unloading, Ollama eviction, RAM reporting
  cli.py                     Typer CLI
  ingest/
    loaders.py               PDF/EPUB/DOCX/TXT → pages + font metadata
    structure.py             TOC + typography → chapter/section hierarchy
    chunker.py               structure-aware, token-accurate chunking
  index/
    embedder.py              BGE-M3, fp16, lazy + cached
    store.py                 vectors.npy + chunks.jsonl + bm25.pkl
    builder.py               ingest orchestration
  retrieve/
    hybrid.py                dense + BM25, RRF fusion
    rerank.py                BGE cross-encoder
    pipeline.py              expansion → retrieval → rerank → floor → context
  llm/
    client.py                Ollama chat/stream/JSON-with-repair
    prompts.py               all prompts, centralised
  chat/
    engine.py                grounded chat, citations, refusal, condensation
  paper/
    blueprint.py             sections, marks, Bloom allocation, topic discovery
    pyq.py                   PYQ parsing → style/pattern profile
    generator.py             slot-by-slot generate → verify → dedupe
    verifier.py              independent groundedness moderation
    export.py                Markdown / DOCX / JSON
  evaluate/
    harness.py               retrieval and faithfulness metrics
data/
  books/    pyqs/    index/    exports/
```

---

## Evaluation

Both evaluations are **self-supervised** — no hand-labelled dataset needed. Run them after any change to chunking, embeddings or thresholds.

**Retrieval.** Samples chunks, has the model write a question that only that chunk answers, then checks whether retrieval puts that chunk back on top.

> Measured against `RetrievalResult.relevance_rank()`, not the order of `results`. `results` is deliberately sorted into *reading order* after neighbour expansion so derivations stay coherent for the LLM — scoring against it measures document position rather than retrieval quality, which understates Hit@1 badly.

```bash
.venv/bin/python -m bookrag.cli eval retrieval -n 30
```

Reports Hit@1, Hit@3, Hit@k, MRR and refusal rate. To measure what query expansion is actually buying you, A/B it:

```bash
.venv/bin/python -m bookrag.cli eval retrieval -n 30 --no-expand
```

**Faithfulness.** Splits real answers into atomic claims and checks each against the retrieved context. Takes a file of one question per line.

```bash
.venv/bin/python -m bookrag.cli eval faithfulness my_questions.txt
```

Reports are written to `data/index/eval_*.json`.

---

### Gold-set evaluation (the number to trust)

The evaluations above grade the system with questions it wrote itself. A gold set is checked by a person and includes questions the books do **not** answer, so it measures what users see.

```bash
# 1. Draft ~30 questions from your indexed books, plus off-topic controls
.venv/bin/python -m bookrag.cli eval draft-gold -n 30
# 2. Review data/eval/gold_draft.jsonl: fix questions, pages and keywords,
#    delete bad lines, add your own, set "reviewed": true. Save as gold.jsonl.
# 3. Score the real pipeline against it
.venv/bin/python -m bookrag.cli eval gold data/eval/gold.jsonl
# 4. Compare a setting without editing config.yaml
.venv/bin/python -m bookrag.cli eval gold data/eval/gold.jsonl --set retrieval.max_context_tokens=12000
```

It reports, for answerable questions, how often the system answered, cited the right page, retrieved the right page and used the expected keywords; for unanswerable ones, how often it correctly refused; and latency. Each run is saved to `data/eval/gold-results-<time>.json`. The line format is documented in `bookrag/evaluate/gold.py`.

---

## Access and security

The web app has no per-user separation: anyone who reaches it can upload books, read generated papers and use the LLM server. `.streamlit/config.toml` therefore binds it to `127.0.0.1`. To serve other machines, change `server.address` **and** set `BOOKRAG_APP_PASSWORD` in `.env`; the app refuses to start on a network address without a password.

---

## Tuning guide

**Answers are too vague / miss detail.** Raise `rerank.top_k` and `retrieval.max_context_tokens` together — and raise `llm.num_ctx` to match, or the context will be silently truncated.

**It refuses things that are in the book.** Lower `rerank.min_score` and `grounding.answer_threshold` (more negative). First confirm the chapter was actually ingested with `books`.

**It answers things that aren't in the book.** Raise both thresholds toward 0. Verify `grounding.verify_questions: true`.

**Chapters aren't detected** (`books` shows blank chapters). The PDF likely has no TOC and uniform font sizes. Lower `ingest.heading_size_ratio` to ~1.08. Structure detection degrades gracefully — retrieval still works, citations just lose chapter names.

**Too many questions are dropped.** Lower `grounding.min_verify_confidence` to ~0.55, or raise `grounding.verification_retries`. A high drop rate usually means the topic genuinely lacks substance in the book — check `ungrounded_topics` in the exported JSON.

**Generation is slow.** `retrieval.hyde` and `retrieval.multi_query` now ship off (one fewer LLM call each per query); lower `rerank.candidates` further, lower `retrieval.max_context_tokens` (it sets the prefill the 8B model must chew through before the first token), or set `grounding.verify_questions: false` — the last one trades away the strongest accuracy guarantee, so prefer the others.

---

## Deploying on a server

Measured on this project's 16 GB Apple Silicon box, 780-chunk index, remote vLLM backend.

**Before it is reachable by anyone else**

1. Set `BOOKRAG_APP_PASSWORD` in `.env` and change `server.address` in `.streamlit/config.toml`. The app refuses to serve a network address without a password.
2. Put it behind a reverse proxy with TLS. Streamlit speaks plain HTTP and the password is sent as form data.
3. In `.streamlit/config.toml`, set `fileWatcherType = "none"` and raise `maxUploadSize` if your books exceed 200 MB.

**Warm up at boot.** A cold first query costs ~14 s of model loading, paid by whoever asks first:

```bash
.venv/bin/python -m bookrag.cli warmup
```

Run it right after starting the app (systemd `ExecStartPost=`, a Docker healthcheck, or by hand).

**Logs.** Set `BOOKRAG_LOG_LEVEL=INFO` to get retrieval degradations and backend failovers on stderr, where journald or Docker collects them.

**Offline model loading.** `embedding.local_files_only` and `rerank.local_files_only` default to `auto`: once the weights are cached, startup never contacts Hugging Face, so it cannot be delayed or broken by an update check.

**Where the time goes, and what to tune**

| | Measured |
|---|---|
| Page render, after the first | 0.03 s (the sidebar's status and model list refresh in the background) |
| Retrieval, warm | 2.74 s, of which reranking is 2.67 s |
| Answer generation | 1–10 s, on the LLM server |
| Cold start | 13.9 s (both encoders) |
| Memory | ~1.0–1.4 GB for the two encoders, shared by all sessions |

Reranking dominates, and it runs one query at a time: four simultaneous users measured 10.8 s each. Options, cheapest first:

- **Tune it.** `rerank.candidates: 12` with `rerank.max_length: 384` measured 1.28 s (2.1x faster) and returned the same top passage on all four test queries. Confirm on your own questions with `eval gold` before keeping it.
- **Move the encoders to the GPU box.** vLLM (and similar) can serve BGE embeddings and reranking; that removes ~1 GB of local memory, the cold start, and the one-at-a-time limit.
- **Run more replicas** behind the proxy with sticky sessions, if the encoders stay local. Each replica loads its own copy of the models.

Do **not** disable the reranker to gain speed: the refusal threshold (`grounding.answer_threshold`) is a reranker score, so without it nothing is ever refused and off-book questions get answered.

**Scaling the library.** Dense search is an exact matmul and stays fast well past this size; `rank_bm25` scores every chunk in Python on every query, so replace it (e.g. `bm25s`, or a sparse matrix) if the library grows past a few tens of thousands of chunks.

---

## Troubleshooting

**`Cannot reach Ollama`** — run `ollama serve`, then `.venv/bin/python -m bookrag.cli doctor`.

**`No index at data/index`** — run `ingest`. If it reports 0 files, check your PDFs are in `data/books/`.

**Ingest produces very few chunks / a book is skipped as unreadable** — the PDF is scanned images or a glyph-shredded export. With `ingest.ocr: auto` (the default) ingest OCRs such files automatically once `ocrmypdf` is installed (`brew install ocrmypdf`), caching the result under the index dir.

**Machine swaps / everything crawls** — see [Memory](#memory-fitting-this-in-16-gb). Start with `cli free --all` and confirm `llm.fast` equals `llm.primary`.

**`AttributeError` from the reranker** — you're on sentence-transformers 6.x with a `.half()` patch applied. Set dtype via `model_kwargs` at load time instead; the shipped code already does.

**Answers truncate mid-sentence** — `retrieval.max_context_tokens` is too close to `llm.num_ctx`. Leave ~3000 tokens of headroom for generation.
