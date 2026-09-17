"""Streamlit UI: ingest, chat, PYQ analysis and paper generation.

    streamlit run app.py
"""
from __future__ import annotations

import hmac
import os
import re
import threading
import time
from pathlib import Path

import streamlit as st

from bookrag.config import load_config
from bookrag.index.store import Store
from bookrag.jobs import Job
from bookrag.logs import setup_logging

setup_logging()

st.set_page_config(page_title="Book RAG — Chat & Exam Builder", page_icon="📚", layout="wide")
cfg = load_config()

_LOCAL_ADDRESSES = {"127.0.0.1", "localhost", "::1"}
# Set by nginx, Cloudflare Tunnel and similar. A proxy on this machine makes
# every visitor look local, so these decide whether the app is really local-only.
_PROXY_HEADERS = ("cf-connecting-ip", "x-forwarded-for", "x-real-ip", "forwarded")
LOGIN_MAX_FAILURES = 5
LOGIN_LOCKOUT_S = 15 * 60


@st.cache_resource
def login_failures() -> dict:
    """Recent failed sign-ins per client, shared by every session. Per-session
    counting would not slow an attacker down: a new connection is a new session."""
    return {"lock": threading.Lock(), "by_client": {}}


def request_headers() -> dict[str, str]:
    try:
        return {k.lower(): v for k, v in st.context.headers.items()}
    except Exception:
        return {}


def client_id(headers: dict[str, str]) -> str:
    """The visitor's address. Behind a proxy the socket peer is the proxy itself,
    so the header it adds is used instead."""
    forwarded = (headers.get("cf-connecting-ip")
                 or headers.get("x-forwarded-for", "").split(",")[0].strip()
                 or headers.get("x-real-ip"))
    if forwarded:
        return forwarded
    try:
        return st.context.ip_address or "unknown"
    except Exception:
        return "unknown"


def require_access() -> None:
    """Gate the whole app before anything else runs.

    There is no per-user separation inside: anyone who reaches the app can
    upload books, read generated papers and spend the LLM server's time. So it
    serves this machine only (see .streamlit/config.toml), unless a password is
    set -- and it refuses to serve the network, or a proxy/tunnel, without one.
    """
    password = os.environ.get("BOOKRAG_APP_PASSWORD") or ""
    if not password:
        configured = cfg.get("app.password")
        password = "" if configured is None else str(configured)
    address = str(st.get_option("server.address") or "").strip()
    headers = request_headers()
    proxied = any(h in headers for h in _PROXY_HEADERS)

    if not password:
        if address in _LOCAL_ADDRESSES and not proxied:
            return
        if proxied:
            st.error("This app is being reached through a proxy or tunnel, so it is not "
                     "local-only, but no password is set. Set `BOOKRAG_APP_PASSWORD` in .env "
                     "and restart.")
        else:
            st.error("This app is reachable from other machines "
                     f"(server.address = {address or 'all interfaces'}) but no password is set. "
                     "Either keep it on this machine with `server.address = \"127.0.0.1\"` in "
                     ".streamlit/config.toml, or set `BOOKRAG_APP_PASSWORD` in .env and restart.")
        st.stop()
    if st.session_state.get("authenticated"):
        return
    st.title("📚 Book RAG")
    with st.form("login"):
        attempt = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        client, failures, now = client_id(headers), login_failures(), time.monotonic()
        with failures["lock"]:
            by_client = failures["by_client"]
            for key in [k for k, times in by_client.items() if now - times[-1] >= LOGIN_LOCKOUT_S]:
                del by_client[key]
            recent = [t for t in by_client.get(client, []) if now - t < LOGIN_LOCKOUT_S]
            locked = len(recent) >= LOGIN_MAX_FAILURES
            ok = not locked and hmac.compare_digest(attempt.encode("utf-8"), password.encode("utf-8"))
            if ok:
                by_client.pop(client, None)
            elif not locked:
                by_client[client] = recent + [now]
        if ok:
            st.session_state.authenticated = True
            st.rerun()
        if locked:
            st.error(f"Too many wrong passwords. Try again in {LOGIN_LOCKOUT_S // 60} minutes.")
        else:
            st.error("Wrong password.")
    st.stop()


require_access()

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>#|~$])")


def plain(text: str) -> str:
    """Escape text for markdown-rendering elements, so a title such as
    "_OceanofPDF.com_" or a marker like "[2]" shows literally."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


# ---------------------------------------------------------------- resources
def index_exists() -> bool:
    return Store.exists(cfg.index_dir)


def index_version() -> str:
    """Changes whenever the index is rebuilt, so cached readers are replaced."""
    ptr = cfg.index_dir / "CURRENT"
    if ptr.exists():
        return ptr.read_text(encoding="utf-8").strip()
    legacy = cfg.index_dir / "chunks.jsonl"
    return str(legacy.stat().st_mtime_ns) if legacy.exists() else ""


@st.cache_resource(show_spinner="Loading models and index...", max_entries=1)
def get_retriever(version: str):
    """Shared by every session: the encoders and index are read-only here and
    far too heavy to load once per browser tab."""
    from bookrag.retrieve.pipeline import Retriever
    retriever = Retriever(cfg)
    retriever.shared = True
    return retriever


@st.cache_resource(show_spinner=False, max_entries=1)
def get_store(version: str) -> Store:
    """The index, loaded once per build and shared by every session.

    The sidebar and Library tab called Store.load() on every rerun, re-reading
    chunks.jsonl, vectors.npy and the BM25 pickle each time -- work that grows
    with the library and buys nothing, since a build publishes a new version.
    """
    return Store(cfg.index_dir).load()


@st.cache_resource
def status_slot() -> dict:
    """Backend status shared by every session: the health of the configured
    servers is the same for all of them."""
    return {"status": None, "refreshing": False, "lock": threading.Lock()}


def _refresh_status(slot: dict) -> None:
    from bookrag.llm.client import client_from_config
    status: dict = {"at": time.monotonic(), "health": None, "error": "", "catalog": []}
    try:
        llm = client_from_config(cfg)
        try:
            status["health"] = llm.health_check()
        except Exception as exc:
            status["error"] = str(exc)
        try:
            status["catalog"] = llm.model_catalog()
        except Exception:
            pass
    finally:
        with slot["lock"]:
            slot["status"], slot["refreshing"] = status, False


def backend_status(force: bool = False) -> dict:
    """Model-server health and catalogue, refreshed in the background.

    Both are network calls, and a rerun happens on every click and every
    progress tick; an unreachable backend costs a full connect timeout each
    time, which added ~10 s to every interaction. Now only the first load of
    the process waits for them: later refreshes run in a thread, so a click
    renders from the last known status instead of blocking on the network.
    """
    slot = status_slot()
    ttl = float(cfg.get("ui.status_ttl_s", 20))
    status = slot["status"]
    if force or status is None:
        with slot["lock"]:
            slot["refreshing"] = True
        _refresh_status(slot)
        return slot["status"]
    if time.monotonic() - status["at"] >= ttl and not slot["refreshing"]:
        with slot["lock"]:
            if not slot["refreshing"]:
                slot["refreshing"] = True
                threading.Thread(target=_refresh_status, args=(slot,),
                                 daemon=True, name="status-refresh").start()
    return status


@st.cache_data(ttl=5, show_spinner=False)
def memory_snapshot(host: str, ask_ollama: bool) -> tuple[float, float, list[dict]]:
    """Free memory and resident models, at most once every 5 s across sessions."""
    from bookrag.memory import ollama_loaded, system_memory_gb
    total, available = system_memory_gb()
    return total, available, (ollama_loaded(host) if ask_ollama else [])


@st.cache_resource
def index_job_slot() -> dict:
    """Process-wide, so every session sees the one index build in progress
    and none can start a second."""
    return {"job": None, "lock": threading.Lock()}


def get_llm():
    """One client per session, so one user's model switch doesn't change
    which model answers everyone else."""
    if "llm" not in st.session_state:
        from bookrag.llm.client import client_from_config
        st.session_state.llm = client_from_config(cfg)
    return st.session_state.llm


def get_engine():
    """One ChatEngine per session. It holds conversation history, which feeds
    query condensing and generation; a process-wide cached engine leaked one
    user's questions into another's answers."""
    retriever = get_retriever(index_version())
    engine = st.session_state.get("engine")
    if engine is None or engine.retriever is not retriever:
        from bookrag.chat.engine import ChatEngine
        fresh = ChatEngine(cfg, retriever=retriever, llm=get_llm())
        if engine is not None:           # index rebuilt: keep the conversation
            fresh.history = engine.history
        st.session_state.engine = fresh
    return st.session_state.engine


def apply_model_choice() -> None:
    """Point this session's client at the sidebar's model. Chat, paper
    generation and PYQ analysis all go through this one client."""
    want = st.session_state.get("active_model")
    llm = get_llm()
    applied = st.session_state.get("applied_model") or llm.primary
    if not want or want == applied:
        return
    llm.primary = want
    st.session_state.applied_model = want
    # Only a local Ollama holds models we can evict; a remote server manages its own.
    if applied and str(cfg.get("llm.provider", "ollama")) == "ollama":
        from bookrag.memory import ollama_unload
        ollama_unload(applied, llm.host)


@st.fragment(run_every=1.0)
def job_panel(job: Job, label: str, cancel_key: str) -> None:
    """Live progress for a background job, refreshed without rerunning the page."""
    if not job.running:
        st.rerun()                        # finished: let the full page show the outcome
    elapsed = int(job.elapsed)
    st.info(f"{label} running · {elapsed // 60}m {elapsed % 60:02d}s")
    st.code("\n".join(job.tail(14)) or "starting...")
    if job.cancel_requested:
        st.caption("Cancelling — stops after the current step finishes.")
    elif st.button("Cancel", key=cancel_key):
        job.cancel()


def store_paper_result(job: Job) -> None:
    """Move a finished paper into session state, saving the export files.

    The paper took minutes to generate, so a failed save (a full disk, a
    read-only folder) is reported without losing it: the preview and the
    Markdown download still work from memory.
    """
    from bookrag.paper.export import export_paper, safe_stem, to_markdown
    result, meta = job.result, job.meta
    r = result.report
    md = to_markdown(result, include_key=meta["include_key"])
    saved: list[str] = []
    docx_file = None
    export_error = ""
    try:
        files = export_paper(result, cfg.export_dir,
                             f"paper_{safe_stem(meta['title'].lower(), max_len=40)}",
                             include_key=meta["include_key"])
        saved = [f.name for f in files]
        docx_file = next(((f.name, f.read_bytes()) for f in files if f.suffix == ".docx"), None)
    except Exception as exc:
        export_error = f"{type(exc).__name__}: {exc}"
    st.session_state["paper_result"] = {
        "metrics": (f"{r.accepted}/{meta['planned']}", result.total_marks,
                    f"{r.acceptance_rate:.0%}", r.retried),
        "cancelled": r.cancelled,
        "saved": saved,
        "export_error": export_error,
        "md": md,
        "docx": docx_file,
        "rejections": list(r.rejections),
    }
    st.session_state["paper_md"] = md


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📚 Book RAG")
    _provider = str(cfg.get("llm.provider", "ollama"))
    _status = backend_status()
    _, _refresh_col = st.columns([3, 1])
    if _refresh_col.button("Refresh", help="Re-check the model server now"):
        backend_status(force=True)
        st.rerun()
    health = _status["health"]
    if health is None:
        st.error(f"Model server unreachable ({_provider})\n\n{plain(_status['error'])}")
    else:
        # A server that lists no models can't confirm the configured one.
        _unconfirmed = health.get("models_listed") is False or any(
            isinstance(v, dict) and v.get("models_listed") is False for v in health.values())
        if health.get("router"):
            for _label in ("openai", "ollama"):
                _e = health.get(_label)
                if not _e:
                    continue
                if not _e.get("ok"):
                    st.warning(f"{_label} down — {plain(str(_e.get('error', ''))[:70])}")
                elif _e.get("models_listed") is False:
                    st.warning(f"{_label} up · lists no models, so the configured one "
                               "can't be confirmed")
                elif not _e.get("primary_ok", True):
                    st.warning(f"{_label} up · configured model not installed")
                else:
                    st.success(f"{_label} up · {plain(_e.get('host', ''))}")
            st.caption(f"answering via **{health['last_used']}**"
                       + (f" · {health['failures']} failover(s)" if health["failures"] else ""))
        elif health["primary_ok"]:
            st.success(f"{_provider} up · {plain(health.get('host', ''))}")
        elif _unconfirmed:
            st.warning(f"{_provider} up · the server lists no models, so the configured "
                       "one can't be confirmed")
        else:
            st.warning(f"{_provider} up · configured model not available")
        if not health["primary_ok"] and not _unconfirmed and _provider == "ollama":
            st.warning(f"Pull it: `ollama pull {cfg.get('llm.primary')}`")

    # None searches every book; a list restricts to those books.
    selected_books: list[str] | None = None
    no_books_selected = False
    if index_exists():
        store = get_store(index_version())
        st.caption(f"{len(store)} chunks · {len(store.books())} book(s)")
        _title_counts: dict[str, int] = {}
        for b in store.books():
            _title_counts[b["title"]] = _title_counts.get(b["title"], 0) + 1
        # Two books can share a title; the label must still tell them apart.
        titles = {(b["title"] if _title_counts[b["title"]] == 1
                   else f"{b['title']} ({b['book_id'][-6:]})"): b["book_id"]
                  for b in store.books()}
        picked = st.multiselect("Restrict to books", list(titles), default=list(titles))
        if titles and not picked:
            no_books_selected = True
            st.warning("No books selected. Pick at least one to chat or generate a paper.")
        elif len(picked) < len(titles):
            selected_books = [titles[t] for t in picked]
    else:
        st.warning("No index yet — build one in the Library tab.")

    st.divider()
    # Model picker. Deliberately driven off the lightweight client rather than
    # get_engine(), so opening the app does not load the index and encoders --
    # the choice is applied when a question or paper is actually requested.
    from bookrag.memory import ollama_loaded, ollama_unload, system_memory_gb
    from bookrag.llm.catalog import summary_line
    _catalog = list(_status["catalog"])
    if not _catalog:
        _catalog = [{"name": get_llm().primary, "role": "configured default",
                     "size_gb": 0.0, "tok_s": 0.0, "tok_s_measured": False,
                     "use_for": "", "avoid_for": "", "chat_capable": True}]
    _by_name = {d["name"]: d for d in _catalog}
    _models = list(_by_name)
    _default = st.session_state.get("active_model", get_llm().primary)
    if _default not in _models:
        _default = _models[0]
    st.session_state.active_model = st.selectbox(
        "Answering model", _models,
        index=_models.index(_default),
        format_func=lambda n: summary_line(_by_name[n]),
        help="Loaded on the next question. Switching evicts the previous model.",
    )
    _pick = _by_name[st.session_state.active_model]
    if not _pick.get("chat_capable", True):
        st.error(_pick["use_for"])
    elif _pick.get("use_for"):
        st.caption(f"**Use for:** {_pick['use_for']}")
        if _pick.get("avoid_for"):
            st.caption(f"**Avoid for:** {_pick['avoid_for']}")
        if _pick.get("examples"):
            with st.expander("Questions this model suits"):
                for _ex in _pick["examples"]:
                    st.caption(f"• {_ex}")
    if st.button("Release model", use_container_width=True,
                 help="Evict it from Ollama now and give the RAM back. "
                      "The next question reloads it."):
        _host = cfg.get("llm.host", "http://localhost:11434")
        _before = system_memory_gb()[1]
        with st.spinner("Evicting from Ollama..."):
            # ollama_unload blocks until the pages are actually back, so the
            # before/after figures below describe reality rather than a
            # snapshot taken mid-eviction.
            _freed = [m["name"] for m in ollama_loaded(_host)
                      if ollama_unload(m["name"], _host)]
        _after = system_memory_gb()[1]
        if _freed:
            st.success(f"Released {', '.join(_freed)} — "
                       f"{_before:.1f} → {_after:.1f} GB free")
        else:
            st.info("Nothing was resident.")

    _fallback_provider = (str(cfg.get("llm.fallback.provider", ""))
                          if cfg.get("llm.fallback.enabled", True) else "")
    _uses_ollama = "ollama" in {_provider, _fallback_provider}
    _total, _avail, _resident = memory_snapshot(
        str(cfg.get("llm.host", "http://localhost:11434")), _uses_ollama)
    if _total:
        _low = _avail < float(cfg.get("memory.min_free_gb", 2.0))
        st.caption(f"{'⚠️' if _low else '🟢'} {_avail:.1f} / {_total:.1f} GB free "
                   f"· profile `{cfg.get('memory.profile', 'balanced')}`")
    for _m in _resident:
        st.caption(f"resident: {plain(_m['name'])} ({_m['size_gb']} GB)")
    st.caption(f"Books dir: `{cfg.books_dir}`")
    st.caption(f"Embeddings: `{cfg.get('embedding.model')}`")
    st.caption(f"Reranker: `{cfg.get('rerank.model')}`")


tab_chat, tab_paper, tab_pyq, tab_lib = st.tabs(
    ["💬 Chat", "📝 Question Paper", "🗂 Previous-Year Papers", "📖 Library"]
)

# ---------------------------------------------------------------- chat tab
with tab_chat:
    if not index_exists():
        st.info("Build the index first (Library tab).")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        c1, c2 = st.columns([6, 1])
        with c2:
            if st.button("Clear", use_container_width=True):
                st.session_state.messages = []
                if "engine" in st.session_state:
                    st.session_state.engine.reset()
                st.rerun()

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("citations"):
                    with st.expander("Sources"):
                        for line in m["citations"]:
                            st.caption(plain(line))

        if prompt := st.chat_input("Ask something from the book...",
                                   disabled=no_books_selected):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                sources: list[str] = []
                # Held in an st.empty so an answer that fails the citation or
                # claim check after streaming can be replaced, not merely warned about.
                slot = st.empty()
                try:
                    engine = get_engine()
                    apply_model_choice()
                    with st.spinner("Searching and checking against the books..."):
                        stream = engine.ask(prompt, book_ids=selected_books, stream=True)
                        with slot.container():
                            text = st.write_stream(stream)
                    ans = engine.last_answer
                    if ans and ans.grounded:
                        # Only the passages the answer cites, identical labels merged.
                        sources = ans.sources()
                    if ans and not ans.grounded:
                        slot.empty()
                        text = ans.text
                        slot.warning(text)
                    if ans and ans.retrieval and ans.retrieval.warnings:
                        st.caption("⚠️ " + plain(" · ".join(ans.retrieval.warnings)))
                    if ans and ans.retrieval and ans.retrieval.timings:
                        t = ans.retrieval.timings
                        st.caption(" · ".join(f"{k} {v:.2f}s" for k, v in t.items()))
                    if sources:
                        with st.expander("Sources"):
                            for line in sources:
                                st.caption(plain(line))
                except Exception as exc:
                    slot.empty()
                    text = f"Couldn't answer: {exc}"
                    hint = ("The GPU ran out of memory for the search models. Set "
                            "`embedding.device: cpu` in config.yaml, or give them room by "
                            "lowering vLLM's --gpu-memory-utilization."
                            if "out of memory" in str(exc).lower()
                            else "Check the model server (sidebar) and try again.")
                    slot.error(plain(text) + "\n\n" + hint)
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "citations": sources}
            )

# ---------------------------------------------------------------- paper tab
with tab_paper:
    if not index_exists():
        st.info("Build the index first (Library tab).")
    else:
        from bookrag.paper import blueprint as bp_mod

        st.subheader("Paper setup")
        col1, col2, col3 = st.columns(3)
        title = col1.text_input("Title", cfg.get("paper.title", "End-Semester Examination"))
        duration = col2.text_input("Duration", cfg.get("paper.duration", "3 Hours"))
        include_key = col3.checkbox("Include answer key", value=True)

        profile_path = cfg.index_dir / "pyq_profile.json"
        use_pyq = False
        pyq_structure = False
        if profile_path.exists():
            use_pyq = st.checkbox("Use previous-year paper style & topic weighting", value=True)
            if use_pyq:
                pyq_structure = st.checkbox("Also copy the PYQ section structure", value=False)
        else:
            st.caption("No PYQ profile yet — analyse papers in the PYQ tab to unlock style matching.")

        st.markdown("**Sections**")
        default_sections = cfg.get("paper.sections", []) or []
        n_sections = st.number_input("Number of sections", 1, 6, len(default_sections) or 3)
        section_specs = []
        for i in range(int(n_sections)):
            d = default_sections[i] if i < len(default_sections) else {}
            c = st.columns([3, 2, 1, 1])
            name = c[0].text_input("Name", d.get("name", f"Section {chr(65+i)}"), key=f"sn{i}")
            qtype = c[1].selectbox("Type", ["short", "long", "mcq", "numerical", "truefalse"],
                                   index=["short", "long", "mcq", "numerical", "truefalse"].index(
                                       d.get("type", "short")), key=f"st{i}")
            count = c[2].number_input("Count", 1, 30, int(d.get("count", 5)), key=f"sc{i}")
            marks = c[3].number_input("Marks", 1, 25, int(d.get("marks_each", 2)), key=f"sm{i}")
            section_specs.append(bp_mod.SectionSpec(name=name, type=qtype,
                                                    count=int(count), marks_each=int(marks),
                                                    instructions=d.get("instructions", ""),
                                                    attempt=int(d.get("attempt", 0) or 0)))

        total = sum(s.total_marks for s in section_specs)
        st.caption(f"Total: **{total} marks**, {sum(s.count for s in section_specs)} questions")

        topics_raw = st.text_area("Restrict to topics (one per line, blank = whole book)", "")

        paper_job: Job | None = st.session_state.get("paper_job")
        paper_running = paper_job is not None and paper_job.running

        if st.button("Generate paper", type="primary",
                     disabled=no_books_selected or paper_running):
            from bookrag.paper.generator import PaperGenerator
            from bookrag.paper.pyq import PYQProfile

            try:
                profile = PYQProfile.load(profile_path) if (use_pyq and profile_path.exists()) else None
                bp = (profile.to_blueprint(title=title) if (profile and pyq_structure)
                      else bp_mod.Blueprint(title=title, duration=duration,
                                            sections=section_specs,
                                            bloom_mix=dict(cfg.get("paper.bloom_mix", {}) or {})))
                bp.title, bp.duration = title, duration
                if topics_raw.strip():
                    bp.topics = [t.strip() for t in topics_raw.splitlines() if t.strip()]
                if profile:
                    bp.style_guide = profile.style_guide

                problems = bp.problems()
                if problems:
                    st.error("Fix the paper setup first: " + plain("; ".join(problems)))
                else:
                    engine = get_engine()
                    apply_model_choice()
                    gen = PaperGenerator(cfg, retriever=engine.retriever, llm=get_llm(), pyq=profile)
                    # Generation runs for minutes (every question is verified), so
                    # it runs in the background and can be cancelled.
                    job = Job("paper", gen.generate, bp, book_ids=selected_books)
                    job.meta = {"include_key": include_key, "title": title,
                                "planned": sum(s.count for s in bp.sections)}
                    st.session_state.paper_job = job.start()
                    st.session_state.pop("paper_result", None)
                    st.rerun()
            except Exception as exc:
                st.error(f"Paper generation failed to start: {plain(exc)}\n\n"
                         "Check the model server (sidebar) and try again.")

        if paper_running:
            job_panel(paper_job, "Paper generation", "cancel_paper")
        elif paper_job is not None:
            if paper_job.status == "failed":
                st.error(f"Paper generation failed: {plain(paper_job.error)}\n\n"
                         "Check the model server (sidebar) and try again.")
            elif paper_job.result is not None:
                store_paper_result(paper_job)
            st.session_state.paper_job = None

        res = st.session_state.get("paper_result")
        if res and not paper_running:
            if res.get("cancelled"):
                st.warning("Generation was cancelled. The questions accepted before that "
                           "are kept below.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Questions", res["metrics"][0])
            m2.metric("Total marks", res["metrics"][1])
            m3.metric("Verified", res["metrics"][2])
            m4.metric("Retries", res["metrics"][3])
            if res.get("export_error"):
                st.error(f"Saving the paper to {plain(cfg.export_dir)} failed "
                         f"({plain(res['export_error'])}). It is still shown below and can be "
                         "downloaded as Markdown.")
            else:
                st.success("Saved: " + plain(", ".join(res["saved"])))
            st.download_button("Download Markdown", res["md"], file_name="question_paper.md",
                               key="dl_md")
            if res["docx"]:
                st.download_button("Download DOCX", res["docx"][1], file_name=res["docx"][0],
                                   key="dl_docx")
            if res["rejections"]:
                with st.expander(f"{len(res['rejections'])} rejected drafts (moderator feedback)"):
                    for rej in res["rejections"]:
                        st.caption(f"**{plain(rej['topic'][:60])}** — {plain(rej['reason'])}")
            st.markdown("---")
            st.markdown(res["md"])

# ---------------------------------------------------------------- pyq tab
with tab_pyq:
    st.subheader("Analyse previous-year question papers")
    st.caption("PYQs are used for *pattern transfer* — section layout, mark bands, "
               "command verbs, phrasing and recurring topics. Answers still come "
               "only from the books.")

    pyq_dir = cfg.path("paths.pyq_dir")
    pyq_dir.mkdir(parents=True, exist_ok=True)
    uploads = st.file_uploader("Upload PYQ papers", type=["pdf", "txt", "docx"],
                               accept_multiple_files=True)
    if uploads:
        for up in uploads:
            (pyq_dir / Path(up.name).name).write_bytes(up.getbuffer())
        st.success(f"Saved {len(uploads)} file(s) to {plain(pyq_dir)}")

    existing = sorted(p for p in pyq_dir.glob("*") if p.suffix.lower() in {".pdf", ".txt", ".docx"})
    st.write(f"**{len(existing)} paper(s) available:** " +
             (plain(", ".join(p.name for p in existing)) or "none"))

    if existing and st.button("Analyse papers", type="primary"):
        from bookrag.paper.pyq import analyze_pyqs

        log = st.empty()
        lines: list[str] = []

        def progress(msg):
            lines.append(str(msg))
            log.code("\n".join(lines[-12:]))

        try:
            apply_model_choice()
            with st.spinner("Parsing papers..."):
                profile = analyze_pyqs(existing, get_llm(), progress=progress)
            profile.save(cfg.index_dir / "pyq_profile.json")
            st.success(f"Extracted {len(profile.questions)} questions from {len(profile.papers)} paper(s)")
        except Exception as exc:
            st.error(f"PYQ analysis failed: {plain(exc)}")

    profile_path = cfg.index_dir / "pyq_profile.json"
    if profile_path.exists():
        from bookrag.paper.pyq import PYQProfile
        profile = PYQProfile.load(profile_path)
        c1, c2, c3 = st.columns(3)
        c1.metric("Questions", len(profile.questions))
        c2.metric("Papers", len(profile.papers))
        c3.metric("Total marks", profile.total_marks or "—")
        if profile.bloom_mix:
            st.bar_chart(profile.bloom_mix)
        if profile.priority_topics():
            st.write("**Recurring topics:** " + plain(", ".join(profile.priority_topics(15))))
        if profile.style_guide:
            with st.expander("Derived style guide"):
                st.markdown(profile.style_guide)
        with st.expander("Extracted questions"):
            for q in profile.questions[:80]:
                st.caption(f"[{q.marks}m · {plain(q.bloom)}] {plain(q.text)}")

# ---------------------------------------------------------------- library
with tab_lib:
    st.subheader("Library")
    books_dir = cfg.books_dir
    books_dir.mkdir(parents=True, exist_ok=True)

    uploads = st.file_uploader("Upload books", type=["pdf", "epub", "txt", "md", "docx"],
                               accept_multiple_files=True, key="bookup")
    if uploads:
        for up in uploads:
            (books_dir / Path(up.name).name).write_bytes(up.getbuffer())
        st.success(f"Saved {len(uploads)} file(s). Rebuild the index below.")

    from bookrag.index.builder import build_index, cached_book_count
    from bookrag.ingest.loaders import discover_books
    found = discover_books(books_dir)
    st.write(f"**{len(found)} file(s) in `{books_dir}`:** " +
             (plain(", ".join(p.name for p in found)) or "none"))

    job_slot = index_job_slot()
    index_job: Job | None = job_slot["job"]
    index_running = index_job is not None and index_job.running

    if found and st.button("Build / rebuild index", type="primary", disabled=index_running):
        with job_slot["lock"]:
            current = job_slot["job"]
            if current is None or not current.running:
                # Runs in the background: the tab stays usable, any session can
                # watch or cancel it, and an interrupted build resumes next time.
                # Sessions pick up the new index via index_version().
                job_slot["job"] = Job("index", build_index, cfg).start()
        st.rerun()

    if index_running:
        job_panel(index_job, "Index build", "cancel_index")
    elif index_job is not None:
        if index_job.status == "done" and index_job.result:
            st.success(f"Indexed {index_job.result['n_chunks']} chunks in "
                       f"{index_job.result['build_seconds']}s")
        elif index_job.status == "cancelled":
            st.warning("Index build cancelled. The existing index is unchanged, and books "
                       "embedded before the cancel are cached, so the next build resumes "
                       "from there.")
        elif index_job.status == "failed":
            st.error(f"Index build failed: {plain(index_job.error)}")
    if not index_running:
        n_cached = cached_book_count(cfg)
        if n_cached:
            st.caption(f"{n_cached} book(s) embedded by an interrupted build will be reused "
                       "by the next build.")

    if index_exists():
        store = get_store(index_version())
        for b in store.books():
            with st.expander(f"{b['title']} — {b['n_pages']} pages, {b['n_chunks']} chunks"):
                for h in b.get("outline", [])[:60]:
                    st.caption("  " * (h["level"] - 1) + f"{plain(h['title'])}  (p.{h['page']})")
