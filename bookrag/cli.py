"""Command-line interface.

    python -m bookrag.cli doctor
    python -m bookrag.cli ingest
    python -m bookrag.cli chat
    python -m bookrag.cli ask "what is entropy?"
    python -m bookrag.cli pyq data/pyqs
    python -m bookrag.cli paper --marks 70 --pyq data/index/pyq_profile.json
    python -m bookrag.cli eval retrieval -n 30
    python -m bookrag.cli eval draft-gold -n 30
    python -m bookrag.cli eval gold data/eval/gold.jsonl --set retrieval.max_context_tokens=12000
"""
from __future__ import annotations

import json
import signal
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from bookrag.config import load_config

app = typer.Typer(add_completion=False, help="Chat with your books and build exam papers.")
eval_app = typer.Typer(help="Measure retrieval and faithfulness.")
app.add_typer(eval_app, name="eval")
console = Console()


def _cfg(path: Optional[str]):
    from bookrag.logs import setup_logging
    setup_logging()
    return load_config(path)


def _with_overrides(cfg, pairs: Optional[List[str]]):
    """Apply repeatable --set key=value options; returns (config, overrides)."""
    from bookrag.config import parse_overrides
    try:
        overrides = parse_overrides(pairs)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--set")
    return (cfg.with_overrides(overrides) if overrides else cfg), overrides


def _project_path(cfg, value: str) -> Path:
    """A path as typed, if it exists or is absolute; otherwise relative to the project."""
    p = Path(value)
    return p if p.is_absolute() or p.exists() else cfg.root / p


@contextmanager
def _graceful_stop(what: str):
    """First Ctrl+C asks the work to stop at its next checkpoint (keeping what
    is done); a second Ctrl+C aborts immediately. Yields a should_cancel."""
    stop = threading.Event()

    def handler(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        stop.set()
        console.print(f"\n[yellow]Stopping {what} after the current step — "
                      "press Ctrl+C again to abort now.[/]")

    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield stop.is_set
    finally:
        signal.signal(signal.SIGINT, previous)


def _model_status(entry: dict, key: str) -> str:
    if entry.get("models_listed") is False:
        return "[yellow]unconfirmed[/]"          # server lists no models to check against
    return "[green]ok[/]" if entry.get(key) else "[red]missing[/]"


def _weights_cached(repo: str) -> Optional[bool]:
    """True if a model's weights are in the local cache; None if undeterminable."""
    from bookrag.weights import weights_cached
    return weights_cached(repo)


# ---------------------------------------------------------------- doctor
@app.command()
def doctor(config: Optional[str] = typer.Option(None, "--config")):
    """Check Ollama, models, index and dependencies."""
    cfg = _cfg(config)
    from bookrag.llm.client import client_from_config, LLMError

    table = Table("check", "status", "detail")

    try:
        health = client_from_config(cfg).health_check()
        if health.get("router"):
            # Report each backend separately: "one of them is up" is exactly the
            # thing you do not want summarised into a single green tick.
            seen: set[str] = set()
            for label in (health.get("last_used"), "openai", "ollama"):
                if not label or label in seen or label not in health:
                    continue
                seen.add(label)
                entry = health[label]
                table.add_row(f"llm: {label}",
                              "[green]up[/]" if entry.get("ok") else "[red]down[/]",
                              entry.get("host", "") if entry.get("ok")
                              else str(entry.get("error", ""))[:60])
                if entry.get("ok"):
                    table.add_row(f"  {label} model", _model_status(entry, "primary_ok"), "")
            table.add_row("llm: answering",
                          "[green]ok[/]" if health["primary_ok"] else "[red]no usable backend[/]",
                          health["last_used"])
        else:
            llm_provider = str(cfg.get("llm.provider", "ollama"))
            names = ((cfg.get("llm.primary"), cfg.get("llm.fast")) if llm_provider == "ollama"
                     else (cfg.get("llm.openai.model"),
                           cfg.get("llm.openai.fast_model") or cfg.get("llm.openai.model")))
            table.add_row(llm_provider, "[green]up[/]", health["host"])
            table.add_row("primary model", _model_status(health, "primary_ok"), str(names[0]))
            table.add_row("fast model", _model_status(health, "fast_ok"), str(names[1]))
    except LLMError as exc:
        table.add_row("ollama", "[red]down[/]", str(exc)[:90])

    try:
        from bookrag.index.store import Store
        store = Store(cfg.index_dir).load()
        table.add_row("index", "[green]ok[/]",
                      f"{len(store)} chunks from {len(store.books())} book(s)")
    except Exception as exc:
        table.add_row("index", "[yellow]absent[/]", str(exc)[:90])

    for mod, label in [("torch", "torch"), ("sentence_transformers", "sentence-transformers"),
                       ("pymupdf", "PyMuPDF"), ("rank_bm25", "rank-bm25")]:
        try:
            __import__(mod)
            table.add_row(label, "[green]ok[/]", "")
        except ImportError as exc:
            table.add_row(label, "[red]missing[/]", str(exc)[:60])

    # Importable is not the same as usable: the encoders' weights are a separate
    # multi-GB download that otherwise only happens (or fails) on first use.
    weights = [("embedding weights", cfg.get("embedding.model", "BAAI/bge-m3"))]
    if bool(cfg.get("rerank.enabled", True)):
        weights.append(("reranker weights", cfg.get("rerank.model", "BAAI/bge-reranker-v2-m3")))
    for label, repo in weights:
        state = _weights_cached(str(repo))
        table.add_row(label,
                      {True: "[green]ok[/]", False: "[yellow]not downloaded[/]"}.get(state, "[yellow]unknown[/]"),
                      str(repo) if state else f"{repo} — downloads on first use")

    try:
        from bookrag.index.embedder import resolve_device
        table.add_row("device", "[green]" + resolve_device(cfg.get("embedding.device", "auto")) + "[/]", "")
    except Exception:
        pass

    provider = str(cfg.get("llm.provider", "ollama"))
    table.add_row("llm provider", "[green]ok[/]", provider)
    if provider == "openai" or str(cfg.get("llm.fallback.provider", "")) == "openai":
        table.add_row("  server", "", f"{cfg.get('llm.openai.base_url')} "
                                      f"(model: {cfg.get('llm.openai.model') or '[red]unset[/]'})")
    if bool(cfg.get("llm.fallback.enabled", True)):
        table.add_row("  fallback", "[green]on[/]", str(cfg.get("llm.fallback.provider", "ollama")))

    from bookrag.memory import system_memory_gb, ollama_loaded, PROFILES
    total, available = system_memory_gb()
    if total:
        low = available < float(cfg.get("memory.min_free_gb", 2.0))
        table.add_row("memory", "[red]tight[/]" if low else "[green]ok[/]",
                      f"{available:.1f} GB free of {total:.1f} GB")
    profile = str(cfg.get("memory.profile", "balanced"))
    known = profile in PROFILES
    table.add_row("memory profile", "[green]ok[/]" if known else "[yellow]warn[/]",
                  profile if known else f"{profile} — unknown, falling back to balanced")
    if total and low and profile == "balanced":
        table.add_row("", "[yellow]hint[/]",
                      "Memory is tight. Try memory.profile: conservative.")
    loaded = ollama_loaded(cfg.get("llm.host", "http://localhost:11434"))
    if loaded:
        table.add_row("ollama resident", "", ", ".join(
            f"{m['name']} ({m['size_gb']} GB)" for m in loaded))
        if len(loaded) > 1:
            table.add_row("", "[yellow]warn[/]",
                          "2+ LLMs resident. Set OLLAMA_MAX_LOADED_MODELS=1 "
                          "or run `cli free`.")

    console.print(table)


@app.command()
def warmup(config: Optional[str] = typer.Option(None, "--config"),
           query: str = typer.Option("warm up the encoders", "--query")):
    """Load the encoders and run one query, so the first real user doesn't wait.

    Run it at boot (systemd ExecStartPost, a Docker healthcheck, or just after
    `streamlit run`): a cold first query costs ~14 s of model loading.
    """
    cfg = _cfg(config)
    import time
    from bookrag.retrieve.pipeline import Retriever

    started = time.perf_counter()
    retriever = Retriever(cfg)
    loaded = time.perf_counter()
    rr = retriever.retrieve(query)
    console.print(f"index: {len(retriever.store)} chunks · load {loaded - started:.1f}s · "
                  f"first query {time.perf_counter() - loaded:.1f}s · "
                  f"stages { {k: round(v, 2) for k, v in rr.timings.items()} }")
    try:
        from bookrag.llm.client import client_from_config
        health = client_from_config(cfg).health_check()
        console.print(f"llm: {'ok' if health.get('primary_ok') else 'no usable backend'}")
    except Exception as exc:
        console.print(f"llm: unreachable ({str(exc)[:80]})", style="yellow", markup=False)


@app.command()
def models(config: Optional[str] = typer.Option(None, "--config")):
    """List installed models with what each is good for in this pipeline."""
    cfg = _cfg(config)
    from bookrag.llm.client import client_from_config
    from bookrag.llm.catalog import summary_line

    catalog = client_from_config(cfg).model_catalog()
    if not catalog:
        console.print("[red]Cannot reach Ollama.[/] Start it with `ollama serve`.")
        raise typer.Exit(1)
    active = cfg.get("llm.primary")
    for d in catalog:
        mark = "[green]*[/]" if d["name"] == active else " "
        console.print(f"{mark} [bold]{summary_line(d)}[/]")
        console.print(f"    [dim]use:[/]   {d['use_for']}")
        if d.get("avoid_for"):
            console.print(f"    [dim]avoid:[/] {d['avoid_for']}")
        for ex in d.get("examples", []):
            console.print(f"    [dim]e.g.[/]   \"{ex}\"")
        console.print()
    console.print("[dim]* = llm.primary in config (startup default). "
                  "Switch per-run with `ask --model` or `/model` in chat.[/]")


@app.command()
def free(config: Optional[str] = typer.Option(None, "--config"),
         all_models: bool = typer.Option(False, "--all", help="Evict every model, not just ours")):
    """Evict LLMs from Ollama to reclaim memory now."""
    cfg = _cfg(config)
    from bookrag.memory import ollama_loaded, ollama_unload_all, ollama_unload, system_memory_gb

    host = cfg.get("llm.host", "http://localhost:11434")
    before = ollama_loaded(host)
    if not before:
        console.print("[dim]No models resident.[/]")
    elif all_models:
        for name in ollama_unload_all(host):
            console.print(f"  evicted {name}")
    else:
        for name in {cfg.get("llm.primary"), cfg.get("llm.fast")}:
            if any(m["name"] == name for m in before):
                ollama_unload(name, host)
                console.print(f"  evicted {name}")

    total, available = system_memory_gb()
    if total:
        console.print(f"[green]{available:.1f} GB free of {total:.1f} GB[/]")


# ---------------------------------------------------------------- ingest
@app.command()
def ingest(config: Optional[str] = typer.Option(None, "--config"),
           path: Optional[str] = typer.Option(None, "--path",
                                              help="Add (or refresh) one file in the existing index"),
           replace: bool = typer.Option(False, "--replace",
                                        help="With --path: make the index contain only this file")):
    """Parse, chunk, embed and index the books.

    Without --path, rebuilds the index from the whole books dir.
    """
    cfg = _cfg(config)
    from bookrag.index.builder import build_index
    from bookrag.jobs import Cancelled

    paths = [Path(path)] if path else None
    try:
        with _graceful_stop("the build") as should_cancel:
            manifest = build_index(cfg, paths, progress=lambda m: console.print(m),
                                   replace=replace if path else True,
                                   should_cancel=should_cancel)
    except (Cancelled, KeyboardInterrupt):
        console.print("\n[yellow]Build stopped.[/] The existing index is unchanged. Books "
                      "embedded so far are cached — run the same command again to resume.")
        raise typer.Exit(130)
    table = Table("book", "pages", "chunks")
    for b in manifest["books"]:
        table.add_row(b["title"][:50], str(b["n_pages"]), str(b["n_chunks"]))
    console.print(table)


@app.command()
def books(config: Optional[str] = typer.Option(None, "--config")):
    """List indexed books and their detected chapters."""
    cfg = _cfg(config)
    from bookrag.index.store import Store

    store = Store(cfg.index_dir).load()
    for b in store.books():
        console.print(f"\n[bold]{b['title']}[/]  [dim]({b['book_id']})[/]")
        console.print(f"  {b['n_pages']} pages, {b['n_chunks']} chunks")
        for ch in store.chapters(b["book_id"])[:25]:
            console.print(f"    - {ch[:80]}")


# ---------------------------------------------------------------- ask/chat
@app.command()
def ask(question: str,
        config: Optional[str] = typer.Option(None, "--config"),
        book: Optional[str] = typer.Option(None, "--book", help="Restrict to one book_id"),
        model: Optional[str] = typer.Option(None, "--model", help="Answering model (evicts the configured one)"),
        release: bool = typer.Option(False, "--release", help="Evict the model when done"),
        show_context: bool = typer.Option(False, "--show-context")):
    """Ask a single grounded question."""
    cfg = _cfg(config)
    from bookrag.chat.engine import ChatEngine

    engine = ChatEngine(cfg)
    if model:
        engine.use_model(model)
    with console.status("retrieving + answering..."):
        answer = engine.ask(question, book_ids=[book] if book else None)
    console.print(Markdown(answer.formatted()))
    for warning in (answer.retrieval.warnings if answer.retrieval else []):
        console.print(f"warning: {warning}", style="yellow", markup=False)
    if show_context and answer.retrieval:
        console.print("\n[dim]--- context ---[/]")
        console.print(answer.retrieval.context[:4000])
    if release:
        info = engine.release_model()
        console.print(f"[dim]released {info['model']} — "
                      f"{info['available_gb']:.1f} GB free[/]")


@app.command()
def chat(config: Optional[str] = typer.Option(None, "--config"),
         book: Optional[str] = typer.Option(None, "--book"),
         model: Optional[str] = typer.Option(None, "--model", help="Answering model")):
    """Interactive multi-turn chat. Commands: /reset /sources /model /release /quit"""
    cfg = _cfg(config)
    from bookrag.chat.engine import ChatEngine

    engine = ChatEngine(cfg)
    if model:
        engine.use_model(model)
    console.print(f"[bold]Book chat[/] — /reset, /sources, /model, /release, /quit "
                  f"[dim](model: {engine.active_model()})[/]\n")
    last = None
    while True:
        try:
            q = console.input("[bold cyan]you >[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in {"/quit", "/exit"}:
            break
        if q == "/reset":
            engine.reset()
            console.print("[dim]history cleared[/]")
            continue
        if q.startswith("/model"):
            parts = q.split(maxsplit=1)
            if len(parts) == 1:
                from bookrag.llm.catalog import summary_line
                for d in engine.llm.model_catalog():
                    mark = "*" if d["name"] == engine.active_model() else " "
                    console.print(f"  {mark} {summary_line(d)}")
                    console.print(f"      [dim]use: {d['use_for']}[/]")
                    if d.get("avoid_for"):
                        console.print(f"      [dim]avoid: {d['avoid_for']}[/]")
                    for ex in d.get("examples", []):
                        console.print(f"      [dim]e.g.   \"{ex}\"[/]")
                console.print("[dim]  /model <name> to switch[/]")
            else:
                info = engine.use_model(parts[1].strip())
                console.print(f"[dim]{info['previous']} -> {info['active']}"
                              f"{' (previous released)' if info['released_previous'] else ''}"
                              f" — {info['available_gb']:.1f} GB free[/]")
            continue
        if q == "/release":
            info = engine.release_model()
            console.print(f"[dim]released {info['model']} — "
                          f"{info['available_gb']:.1f} GB free[/]")
            continue
        if q == "/sources":
            if last and last.retrieval:
                for i, c in enumerate(last.retrieval.citations, 1):
                    console.print(f"  [{i}] {c}")
            continue

        console.print("[bold green]book >[/] ", end="")
        stream = engine.ask(q, book_ids=[book] if book else None, stream=True)
        printed: list[str] = []
        for piece in stream:
            printed.append(piece)
            console.print(piece, end="")
        console.print()
        last = engine.last_answer
        if last and not last.grounded and last.text.strip() != "".join(printed).strip():
            # The streamed text failed a citation or claim check after it was shown.
            console.print(f"⚠ {last.text}", style="yellow", markup=False)
        if last and last.grounded:
            for line in last.sources():
                console.print(f"  {line}", style="dim", markup=False)
        for warning in (last.retrieval.warnings if last and last.retrieval else []):
            console.print(f"warning: {warning}", style="yellow", markup=False)


# ---------------------------------------------------------------- pyq
@app.command()
def pyq(source: str = typer.Argument(..., help="PYQ file or directory"),
        config: Optional[str] = typer.Option(None, "--config"),
        out: Optional[str] = typer.Option(None, "--out")):
    """Analyse previous-year papers into a reusable style/pattern profile."""
    cfg = _cfg(config)
    from bookrag.ingest.loaders import SUPPORTED
    from bookrag.llm.client import client_from_config
    from bookrag.paper.pyq import analyze_pyqs

    src = Path(source)
    paths = ([p for p in sorted(src.rglob("*")) if p.suffix.lower() in SUPPORTED]
             if src.is_dir() else [src])
    if not paths:
        raise typer.BadParameter(f"No supported files under {src}")

    profile = analyze_pyqs(paths, client_from_config(cfg), progress=lambda m: console.print(m))
    out_path = Path(out) if out else cfg.index_dir / "pyq_profile.json"
    profile.save(out_path)

    console.print(f"\n[green]Saved[/] {out_path}")
    console.print(f"  papers: {len(profile.papers)}  questions: {len(profile.questions)}")
    console.print(f"  mark bands: {profile.marks_distribution}")
    console.print(f"  bloom mix: {profile.bloom_mix}")
    console.print(f"  top topics: {', '.join(profile.priority_topics(10))}")


# ---------------------------------------------------------------- paper
@app.command()
def paper(config: Optional[str] = typer.Option(None, "--config"),
          blueprint: Optional[str] = typer.Option(None, "--blueprint", help="JSON/YAML blueprint"),
          pyq_profile: Optional[str] = typer.Option(None, "--pyq", help="pyq_profile.json"),
          use_pyq_structure: bool = typer.Option(False, "--pyq-structure",
                                                 help="Mirror the PYQ paper's section layout"),
          title: Optional[str] = typer.Option(None, "--title"),
          marks: Optional[int] = typer.Option(None, "--marks"),
          topics: Optional[str] = typer.Option(None, "--topics", help="Comma-separated"),
          book: Optional[str] = typer.Option(None, "--book"),
          out: Optional[str] = typer.Option(None, "--out", help="Output filename stem"),
          no_key: bool = typer.Option(False, "--no-key", help="Omit the answer key")):
    """Generate a verified question paper from the indexed books."""
    cfg = _cfg(config)
    from bookrag.paper import blueprint as bp_mod
    from bookrag.paper.export import export_paper, safe_stem
    from bookrag.paper.generator import PaperGenerator
    from bookrag.paper.pyq import PYQProfile

    profile = PYQProfile.load(Path(pyq_profile)) if pyq_profile else None

    if blueprint:
        bp = bp_mod.load(blueprint)
    elif profile and use_pyq_structure:
        bp = profile.to_blueprint(title=title)
        console.print("[dim]blueprint taken from PYQ structure[/]")
    else:
        bp = bp_mod.from_config(cfg)

    if title:
        bp.title = title
    if marks:
        # Marks come from the sections; --marks is a check, not a setting the
        # generator could honour by itself.
        if marks != bp.computed_marks:
            raise typer.BadParameter(
                f"the blueprint's sections add up to {bp.computed_marks} marks, not {marks}. "
                "Adjust the sections (paper.sections in config, or --blueprint) to match.",
                param_hint="--marks")
        bp.total_marks = marks
    if topics:
        bp.topics = [t.strip() for t in topics.split(",") if t.strip()]
    if profile and not bp.style_guide:
        bp.style_guide = profile.style_guide
    problems = bp.problems()
    if problems:
        raise typer.BadParameter("; ".join(problems), param_hint="paper sections")

    console.print(f"[bold]{bp.title}[/] — {bp.computed_marks} marks across "
                  f"{len(bp.sections)} section(s)")

    gen = PaperGenerator(cfg, pyq=profile)
    with _graceful_stop("generation") as should_cancel:
        result = gen.generate(bp, book_ids=[book] if book else None,
                              progress=lambda m: console.print(m),
                              should_cancel=should_cancel)
    if result.report.cancelled:
        console.print(f"[yellow]Generation stopped early — exporting the "
                      f"{len(result.questions)} question(s) accepted so far.[/]")

    stem = out or f"paper_{safe_stem(bp.title.lower(), max_len=40)}"
    written = export_paper(result, cfg.export_dir, stem, include_key=not no_key)

    r = result.report
    console.print(f"\n[bold]Generated {r.accepted}/{r.requested} questions[/] "
                  f"({result.total_marks} marks) — "
                  f"verified {r.acceptance_rate:.0%}, {r.retried} retries, "
                  f"{r.rejected} dropped, {r.duplicates} duplicates")
    for p in written:
        console.print(f"  [green]->[/] {p}")


# ---------------------------------------------------------------- eval
@eval_app.command("retrieval")
def eval_retrieval(config: Optional[str] = typer.Option(None, "--config"),
                   n: int = typer.Option(30, "-n", help="Number of sampled chunks"),
                   no_expand: bool = typer.Option(False, "--no-expand",
                                                  help="Disable multi-query/HyDE to A/B it"),
                   overrides: Optional[List[str]] = typer.Option(
                       None, "--set", help="Override a setting for this run, e.g. "
                                           "--set rerank.top_k=6 (repeatable)")):
    """Self-supervised retrieval quality: Hit@k and MRR."""
    cfg, _ = _with_overrides(_cfg(config), overrides)
    from bookrag.evaluate.harness import retrieval_eval, save_report

    m = retrieval_eval(cfg, n_samples=n, expand=not no_expand,
                       progress=lambda s: console.print(f"[dim]{s}[/]"))
    console.print(f"\n[bold]{m.summary()}[/]")
    path = save_report(m, cfg.index_dir / "eval_retrieval.json")
    console.print(f"[dim]report -> {path}[/]")


@eval_app.command("faithfulness")
def eval_faithfulness(questions_file: str = typer.Argument(..., help="One question per line"),
                      config: Optional[str] = typer.Option(None, "--config"),
                      overrides: Optional[List[str]] = typer.Option(
                          None, "--set", help="Override a setting for this run (repeatable)")):
    """Fraction of answer claims actually supported by retrieved context."""
    cfg, _ = _with_overrides(_cfg(config), overrides)
    from bookrag.evaluate.harness import faithfulness_eval, save_report

    qs = [l.strip() for l in Path(questions_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    res = faithfulness_eval(cfg, qs, progress=lambda s: console.print(f"[dim]{s}[/]"))
    if res["faithfulness"] is None:
        console.print("\n[bold yellow]Faithfulness: not measured[/] — no claims were scored.")
    else:
        console.print(f"\n[bold]Faithfulness: {res['faithfulness']:.1%}[/] "
                      f"({res['supported_claims']}/{res['total_claims']} claims)")
    if res.get("truncated_claims"):
        console.print(f"[yellow]{res['truncated_claims']} claims beyond evaluate.max_claims "
                      "were not scored.[/]")
    save_report(res, cfg.index_dir / "eval_faithfulness.json")


@eval_app.command("draft-gold")
def eval_draft_gold(config: Optional[str] = typer.Option(None, "--config"),
                    n: int = typer.Option(30, "-n", help="Answerable questions to draft"),
                    out: str = typer.Option("data/eval/gold_draft.jsonl", "--out",
                                            help="Where to write the draft (JSON Lines)"),
                    seed: int = typer.Option(7, "--seed")):
    """Draft a question set from the indexed books, for a person to review."""
    cfg = _cfg(config)
    from bookrag.evaluate.gold import draft_gold, write_jsonl

    if n < 1:
        raise typer.BadParameter("must be at least 1", param_hint="-n")
    cases = draft_gold(cfg, n=n, seed=seed,
                       progress=lambda s: console.print(s, style="dim", markup=False))
    path = write_jsonl(cases, _project_path(cfg, out) if Path(out).is_absolute()
                       else cfg.root / out)
    console.print(f"\nWrote {len(cases)} draft cases -> {path}", style="green", markup=False)
    console.print("Review every line before trusting the scores: fix questions, pages and "
                  "keywords, delete bad cases, add your own (including questions the books do "
                  "NOT answer), then set \"reviewed\": true.\n"
                  f"Run it with: python -m bookrag.cli eval gold {path}", markup=False)


@eval_app.command("gold")
def eval_gold(questions_file: str = typer.Argument(..., help="Gold set, JSON Lines "
                                                             "(format: bookrag/evaluate/gold.py)"),
              config: Optional[str] = typer.Option(None, "--config"),
              overrides: Optional[List[str]] = typer.Option(
                  None, "--set", help="Override a setting for this run, e.g. "
                                      "--set retrieval.max_context_tokens=12000 (repeatable)"),
              limit: int = typer.Option(0, "--limit", help="Only run the first N cases")):
    """Score answers against a reviewed question set: pages, keywords, refusals, latency."""
    from bookrag.evaluate.gold import load_gold, run_gold_eval, summary_text
    from bookrag.evaluate.harness import save_report

    cfg, applied = _with_overrides(_cfg(config), overrides)
    try:
        cases = load_gold(_project_path(cfg, questions_file))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="QUESTIONS_FILE")
    if limit > 0:
        cases = cases[:limit]
    res = run_gold_eval(cfg, cases, progress=lambda s: console.print(s, style="dim", markup=False))
    res["overrides"] = applied
    console.print("\n" + summary_text(res), markup=False)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = save_report(res, cfg.root / "data" / "eval" / f"gold-results-{stamp}.json")
    console.print(f"report -> {path}", style="dim", markup=False)


if __name__ == "__main__":
    app()
