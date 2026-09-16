"""The retrieval pipeline, end to end.

    query
      -> multi-query rewrite + HyDE probe        (recall)
      -> dense + BM25 over every variant         (recall)
      -> RRF fusion                              (consolidation)
      -> cross-encoder rerank                    (precision)
      -> score floor                             (refusal / no-hallucination)
      -> neighbour expansion + token budgeting   (answerability)
      -> numbered context block                  (citability)

Recall stages come first and stay generous; precision is imposed once, late, by
the cross-encoder. That ordering is what lets the score floor be strict without
losing genuinely relevant passages.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from bookrag.config import Config
from bookrag.index.embedder import Embedder, embedder_from_config
from bookrag.index.store import Store
from bookrag.ingest.chunker import count_tokens, truncate_tokens
from bookrag.llm.client import OllamaClient, client_from_config
from bookrag.memory import Arbiter, arbiter_from_config
from bookrag.llm.prompts import HYDE_SYSTEM, MULTIQUERY_SYSTEM, MULTIQUERY_USER
from bookrag.retrieve.hybrid import hybrid_search
from bookrag.retrieve.rerank import Reranker, reranker_from_config
from bookrag.schemas import ScoredChunk

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    query: str
    variants: list[str]
    results: list[ScoredChunk]   # reading order (for coherent context)
    ranked_ids: list[str]        # relevance order, pre-neighbour-expansion
    context: str                 # numbered, token-budgeted context block
    citations: list[str]         # citation string per [n]
    top_score: float
    grounded: bool               # did anything clear the score floor?
    timings: dict[str, float] = field(default_factory=dict)   # seconds per stage
    passages: list[str] = field(default_factory=list)          # text of each [n], as the LLM saw it
    warnings: list[str] = field(default_factory=list)          # degraded stages, e.g. failed expansion

    def chunk_ids(self) -> list[str]:
        """Ids in reading order — what the LLM sees."""
        return [sc.chunk.id for sc in self.results]

    def relevance_rank(self, chunk_id: str) -> int:
        """1-based rank by relevance, or 0 if absent.

        `results` is deliberately in reading order so derivations stay coherent,
        which makes it the wrong list to measure retrieval quality against.
        Evaluation must use this instead.
        """
        return self.ranked_ids.index(chunk_id) + 1 if chunk_id in self.ranked_ids else 0


class Retriever:
    def __init__(self, cfg: Config, store: Store | None = None,
                 embedder: Embedder | None = None,
                 reranker: Reranker | None = None,
                 llm: OllamaClient | None = None,
                 arbiter: Arbiter | None = None):
        self.cfg = cfg
        self.store = store or Store(cfg.index_dir).load()
        # Query vectors from one model searched against passage vectors from
        # another return confident nonsense, not an error. Refuse up front.
        built_with = self.store.manifest.get("embedding_model")
        configured = cfg.get("embedding.model")
        if embedder is None and built_with and configured and built_with != configured:
            raise RuntimeError(
                f"The index was built with {built_with}, but embedding.model is "
                f"{configured}. Rebuild the index (`python -m bookrag.cli ingest`) "
                f"or set embedding.model back to {built_with}."
            )
        self.embedder = embedder or embedder_from_config(cfg)
        self.reranker = reranker if reranker is not None else reranker_from_config(cfg)
        self._llm = llm
        self._llm_configured = llm is not None
        self.arbiter = arbiter or arbiter_from_config(cfg)
        self.arbiter.register(embedder=self.embedder, reranker=self.reranker)

    @property
    def llm(self) -> OllamaClient:
        if self._llm is None:
            self._llm = client_from_config(self.cfg)
        return self._llm

    # ------------------------------------------------------------ expansion
    def expand_query(self, query: str) -> list[str]:
        """Original query first; rewrites and a HyDE probe after it."""
        return self._expand(query)[0]

    def _expand(self, query: str) -> tuple[list[str], list[str]]:
        """(variants, warnings). Expansion is an optimisation, never a hard
        failure -- but a silent one hides a broken model server behind quietly
        worse recall, so every failure is logged and reported to the caller."""
        variants = [query]
        warnings: list[str] = []
        if self.cfg.get("retrieval.multi_query", True):
            n = int(self.cfg.get("retrieval.multi_query_n", 3))
            try:
                data = self.llm.complete_json(
                    MULTIQUERY_SYSTEM, MULTIQUERY_USER.format(question=query, n=n),
                    fast=True, temperature=0.3,
                )
                queries = data.get("queries") if isinstance(data, dict) else None
                for q in (queries if isinstance(queries, list) else [])[:n]:
                    if isinstance(q, str) and q.strip() and q.strip() != query:
                        variants.append(q.strip())
            except Exception as exc:
                log.warning("multi-query expansion failed: %s", exc)
                warnings.append(f"query rewriting failed ({str(exc)[:120]}); "
                                "searched without the rewritten queries")
        if self.cfg.get("retrieval.hyde", True):
            try:
                probe = self.llm.complete(HYDE_SYSTEM, query, fast=True,
                                          temperature=0.4, max_tokens=200)
                if probe.strip():
                    variants.append(probe.strip())
            except Exception as exc:
                log.warning("HyDE probe failed: %s", exc)
                warnings.append(f"HyDE probe failed ({str(exc)[:120]}); "
                                "searched without it")
        return variants, warnings

    # ------------------------------------------------------------ retrieval
    def retrieve(self, query: str, top_k: int | None = None,
                 book_ids: list[str] | None = None,
                 expand: bool = True,
                 min_score: float | None = None) -> RetrievalResult:
        cfg = self.cfg
        top_k = top_k or int(cfg.get("rerank.top_k", 8))
        floor = float(cfg.get("rerank.min_score", 0.02)) if min_score is None else min_score

        timings: dict[str, float] = {}
        t = time.perf_counter()

        variants, warnings = self._expand(query) if expand else ([query], [])
        timings["expand"] = time.perf_counter() - t; t = time.perf_counter()

        self.arbiter.before("embed")
        query_vecs = self.embedder.embed_queries(variants)
        timings["embed"] = time.perf_counter() - t; t = time.perf_counter()

        fused = hybrid_search(
            self.store, variants, query_vecs,
            dense_top_k=int(cfg.get("retrieval.dense_top_k", 40)),
            bm25_top_k=int(cfg.get("retrieval.bm25_top_k", 40)),
            rrf_k=int(cfg.get("retrieval.rrf_k", 60)),
            book_ids=book_ids,
            limit=int(cfg.get("rerank.candidates", 40)),
        )
        timings["search"] = time.perf_counter() - t; t = time.perf_counter()
        if not fused:
            return RetrievalResult(query, variants, [], [], "", [], float("-inf"),
                                   False, timings, [], warnings)

        scored = [ScoredChunk(chunk=self.store.chunks[idx], score=score,
                              dense_rank=extra["dense_rank"], bm25_rank=extra["bm25_rank"])
                  for idx, score, extra in fused]

        # Precision pass. Reranking against the *original* query only — the
        # rewrites served their purpose during recall and would blur the target.
        if self.reranker:
            self.arbiter.before("rerank")
            rr = self.reranker.score(query, [sc.chunk.text for sc in scored])
            for sc, s in zip(scored, rr):
                sc.rerank_score = s
            scored.sort(key=lambda sc: -(sc.rerank_score or float("-inf")))
            top_score = scored[0].rerank_score or float("-inf")
            kept = [sc for sc in scored if (sc.rerank_score or float("-inf")) >= floor][:top_k]
        else:
            top_score = scored[0].score
            kept = scored[:top_k]
        timings["rerank"] = time.perf_counter() - t; t = time.perf_counter()

        # Capture relevance order before neighbour expansion reorders for reading.
        ranked_ids = [sc.chunk.id for sc in kept]
        selected = self._select_context(self._expand_neighbors(kept))
        context, citations = self._render_context(selected)
        results = [sc for sc, _ in selected]
        # Only what the LLM will actually see counts: a hit that clears the
        # floor but did not fit the budget grounds nothing.
        in_context = {sc.chunk.id for sc in results}
        ranked_ids = [cid for cid in ranked_ids if cid in in_context]
        grounded = bool(ranked_ids) and bool(context.strip())
        timings["context"] = time.perf_counter() - t
        timings["total"] = sum(timings.values())
        return RetrievalResult(query, variants, results, ranked_ids, context,
                               citations, top_score, grounded, timings,
                               [body for _, body in selected], warnings)

    def _expand_neighbors(self, kept: list[ScoredChunk]) -> list[ScoredChunk]:
        """Pull in adjacent chunks so a truncated definition is still complete."""
        window = int(self.cfg.get("retrieval.neighbor_window", 1))
        if window <= 0 or not kept:
            return kept
        have = {sc.chunk.id for sc in kept}
        extra: list[ScoredChunk] = []
        for sc in kept:
            for n in self.store.neighbors(sc.chunk, window):
                if n.id not in have:
                    have.add(n.id)
                    # Inherit a discounted score: useful context, not a hit.
                    extra.append(ScoredChunk(chunk=n, score=sc.score * 0.5,
                                             # Reranker scores are probabilities,
                                             # so discount multiplicatively; a
                                             # flat -5.0 was logit-scale thinking.
                                             rerank_score=(sc.rerank_score or 0) * 0.01,
                                             is_neighbor=True))
        merged = kept + extra
        # Reading order keeps derivations and worked examples coherent.
        merged.sort(key=lambda sc: (sc.chunk.book_id, sc.chunk.ordinal))
        return merged

    # ------------------------------------------------------------- context
    def build_context(self, scored: list[ScoredChunk],
                      max_tokens: int | None = None) -> tuple[str, list[str]]:
        return self._render_context(self._select_context(scored, max_tokens))

    def _select_context(self, scored: list[ScoredChunk],
                        max_tokens: int | None = None) -> list[tuple[ScoredChunk, str]]:
        """Fill the token budget by relevance, then restore the input order.

        Budgeting in reading order let a long neighbour ahead of the best hit
        use up the budget and push that hit out of the prompt entirely.
        """
        budget = max_tokens or int(self.cfg.get("retrieval.max_context_tokens", 6000))

        def priority(item: tuple[int, ScoredChunk]):
            pos, sc = item
            s = sc.rerank_score if sc.rerank_score is not None else sc.score
            return (sc.is_neighbor, -s, pos)

        chosen: dict[int, str] = {}
        used = 0
        for pos, sc in sorted(enumerate(scored), key=priority):
            header_cost = count_tokens(f"[{len(scored)}] {sc.chunk.citation()}") + 4
            body = sc.chunk.text.strip()
            cost = header_cost + count_tokens(body)
            if used + cost > budget:
                room = budget - used - header_cost
                # The best hit is never dropped for being long: cut it to fit.
                if chosen or sc.is_neighbor or room < 20:
                    continue
                body = truncate_tokens(body, room)
                cost = header_cost + count_tokens(body)
            chosen[pos] = body
            used += cost
        return [(scored[pos], chosen[pos]) for pos in sorted(chosen)]

    @staticmethod
    def _render_context(selected: list[tuple[ScoredChunk, str]]) -> tuple[str, list[str]]:
        parts = [f"[{i}] {sc.chunk.citation()}\n{body}"
                 for i, (sc, body) in enumerate(selected, start=1)]
        return "\n\n".join(parts), [sc.chunk.citation() for sc, _ in selected]


def retriever_from_config(cfg: Config) -> Retriever:
    return Retriever(cfg)
