"""Measure the pipeline instead of trusting it.

Two evaluations, both self-supervised — no hand-labelled dataset required:

* **Retrieval (`retrieval_eval`)**: sample chunks, have the fast model write a
  question that only that chunk answers, then check whether retrieval puts that
  chunk back on top. Reports Hit@k, MRR and the reranker's contribution.
* **Faithfulness (`faithfulness_eval`)**: take real answers, split into claims,
  and ask the model whether each claim is supported by the retrieved context.

Run these after any change to chunking, embeddings or thresholds — they turn
"feels better" into a number.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from bookrag.config import Config
from bookrag.llm.client import client_from_config
from bookrag.retrieve.pipeline import Retriever

QGEN_SYSTEM = """Write ONE specific question that is answered by the passage below and
essentially nowhere else — use its distinctive terms, names or numbers. Do not
write a generic question. Do not mention "the passage". Output the question only."""

CLAIM_SYSTEM = """Split the answer into atomic factual claims. Return JSON only:
{"claims": ["...", "..."]}"""

SUPPORT_SYSTEM = """For each claim, decide whether the excerpts support it. Return JSON only:
{"verdicts": [{"claim": "...", "supported": true|false}]}"""


@dataclass
class RetrievalMetrics:
    n: int = 0                     # cases actually evaluated
    attempted: int = 0             # chunks sampled
    failed_generation: int = 0     # samples lost because no question was written
    # None, not 0.0, when nothing was evaluated: a zero would read as "retrieval
    # is useless" when the truth is "nothing was measured".
    hit_at_1: float | None = None
    hit_at_3: float | None = None
    hit_at_k: float | None = None
    mrr: float | None = None
    refusal_rate: float | None = None
    per_case: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        if not self.n:
            return (f"n=0 — nothing evaluated ({self.failed_generation} of "
                    f"{self.attempted} question generations failed)")
        failed = f"  ({self.failed_generation} q-gen failures)" if self.failed_generation else ""
        return (f"n={self.n}  Hit@1={self.hit_at_1:.1%}  Hit@3={self.hit_at_3:.1%}  "
                f"Hit@k={self.hit_at_k:.1%}  MRR={self.mrr:.3f}  "
                f"refusals={self.refusal_rate:.1%}{failed}")


def retrieval_eval(cfg: Config, n_samples: int = 30, seed: int = 7,
                   expand: bool = True, progress=print) -> RetrievalMetrics:
    retriever = Retriever(cfg)
    llm = client_from_config(cfg)
    rng = random.Random(seed)

    # Only sample chunks with enough substance to support a specific question.
    pool = [c for c in retriever.store.chunks if c.token_count >= 150]
    if not pool:
        pool = retriever.store.chunks
    sample = rng.sample(pool, min(n_samples, len(pool)))

    m = RetrievalMetrics(attempted=len(sample))
    ranks: list[float] = []
    refusals = 0

    for i, chunk in enumerate(sample, 1):
        try:
            question = llm.complete(QGEN_SYSTEM, chunk.text[:2500], fast=True,
                                    temperature=0.3, max_tokens=100).strip()
        except Exception as exc:
            m.failed_generation += 1
            progress(f"  ! q-gen failed: {exc}")
            continue
        if not question:
            m.failed_generation += 1
            continue
        rr = retriever.retrieve(question, expand=expand)
        # Relevance rank, NOT position in the reading-ordered context list.
        rank = rr.relevance_rank(chunk.id)
        if not rr.grounded:
            refusals += 1
        ranks.append(1.0 / rank if rank else 0.0)
        m.per_case.append({"question": question, "gold": chunk.id,
                           "rank": rank, "top_score": rr.top_score})
        progress(f"  [{i}/{len(sample)}] rank={rank or 'miss'}  {question[:70]}")

    m.n = len(m.per_case)
    if m.n:
        m.hit_at_1 = sum(1 for c in m.per_case if c["rank"] == 1) / m.n
        m.hit_at_3 = sum(1 for c in m.per_case if 0 < c["rank"] <= 3) / m.n
        m.hit_at_k = sum(1 for c in m.per_case if c["rank"] > 0) / m.n
        m.mrr = sum(ranks) / len(ranks)
        m.refusal_rate = refusals / m.n
    return m


def _norm_claim(text: str) -> str:
    return " ".join(str(text).lower().split())


def _count_supported(claims: list[str], verdicts) -> int:
    """Supported claims among those extracted.

    Verdicts are matched to claims by text, falling back to position only when
    the judge returned exactly one verdict per claim. A claim with no verdict is
    unsupported: dividing by the verdicts returned let a judge that skipped the
    hard claims report 100%.
    """
    from bookrag.paper.verifier import _as_bool

    verdicts = [v for v in verdicts if isinstance(v, dict)] if isinstance(verdicts, list) else []
    by_text = {_norm_claim(v["claim"]): _as_bool(v.get("supported"))
               for v in verdicts if v.get("claim")}
    ok = 0
    for i, claim in enumerate(claims):
        key = _norm_claim(claim)
        if key in by_text:
            ok += by_text[key]
        elif len(verdicts) == len(claims):
            ok += _as_bool(verdicts[i].get("supported"))
    return ok


def faithfulness_eval(cfg: Config, questions: list[str], progress=print) -> dict:
    from bookrag.chat.engine import ChatEngine

    engine = ChatEngine(cfg)
    llm = engine.llm
    max_claims = int(cfg.get("evaluate.max_claims", 40))
    total_claims = supported_claims = truncated = 0
    rows = []

    for q in questions:
        engine.reset()
        ans = engine.ask(q)
        if not ans.grounded:
            rows.append({"question": q, "refused": True, "claims": 0, "supported": 0})
            progress(f"  refused: {q[:60]}")
            continue
        try:
            data = llm.complete_json(CLAIM_SYSTEM, ans.text, fast=True, temperature=0.0)
            raw = data.get("claims") if isinstance(data, dict) else None
            claims = [str(c).strip() for c in (raw if isinstance(raw, list) else [])
                      if str(c).strip()]
            if len(claims) > max_claims:
                truncated += len(claims) - max_claims
                progress(f"  ! {len(claims)} claims, scoring the first {max_claims}")
                claims = claims[:max_claims]
            if not claims:
                progress(f"  ! no claims extracted for '{q[:40]}', not scored")
                rows.append({"question": q, "refused": False, "claims": 0,
                             "supported": 0, "unscored": True})
                continue
            data = llm.complete_json(
                SUPPORT_SYSTEM,
                f"EXCERPTS\n{ans.retrieval.context}\n\nCLAIMS\n" +
                "\n".join(f"- {c}" for c in claims),
                fast=False, temperature=0.0,
            )
            verdicts = data.get("verdicts") if isinstance(data, dict) else None
        except Exception as exc:
            progress(f"  ! eval failed for '{q[:40]}': {exc}")
            continue
        ok = _count_supported(claims, verdicts)
        total_claims += len(claims)
        supported_claims += ok
        rows.append({"question": q, "refused": False,
                     "claims": len(claims), "supported": ok})
        progress(f"  {ok}/{len(claims)} claims supported — {q[:60]}")

    return {
        "faithfulness": supported_claims / total_claims if total_claims else None,
        "total_claims": total_claims,
        "supported_claims": supported_claims,
        "truncated_claims": truncated,
        "rows": rows,
    }


def save_report(data, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data.__dict__ if hasattr(data, "__dict__") else data
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
