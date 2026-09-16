"""Hybrid dense + lexical retrieval fused with Reciprocal Rank Fusion.

Dense search finds paraphrases ("what causes entropy to rise" -> a passage that
never uses the word "cause"). BM25 finds exact tokens dense models blur:
symbols, statute numbers, rare proper nouns, formula names. Textbooks need both.

RRF fuses the two ranked lists without needing their scores to be comparable:
    score(d) = sum over lists of 1 / (k + rank(d))
"""
from __future__ import annotations

from bookrag.index.store import Store


def rrf_fuse(ranked_lists: list[list[int]], k: int = 60,
             weights: list[float] | None = None) -> list[tuple[int, float]]:
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[int, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, doc_idx in enumerate(lst, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def hybrid_search(store: Store, queries: list[str], query_vecs,
                  dense_top_k: int = 40, bm25_top_k: int = 40, rrf_k: int = 60,
                  book_ids: list[str] | None = None,
                  limit: int = 40) -> list[tuple[int, float, dict]]:
    """Run every query variant through both retrievers and fuse all lists at once."""
    ranked_lists: list[list[int]] = []
    dense_ranks: dict[int, int] = {}
    bm25_ranks: dict[int, int] = {}

    for i, q in enumerate(queries):
        dense = store.dense_search(query_vecs[i], dense_top_k, book_ids)
        lex = store.bm25_search(q, bm25_top_k, book_ids)
        d_idx = [idx for idx, _ in dense]
        l_idx = [idx for idx, _ in lex]
        ranked_lists.append(d_idx)
        ranked_lists.append(l_idx)
        for r, idx in enumerate(d_idx, 1):
            dense_ranks[idx] = min(dense_ranks.get(idx, 10**9), r)
        for r, idx in enumerate(l_idx, 1):
            bm25_ranks[idx] = min(bm25_ranks.get(idx, 10**9), r)

    # The original query's lists are weighted above its rewrites.
    weights: list[float] = []
    for i in range(len(queries)):
        w = 1.0 if i == 0 else 0.7
        weights.extend([w, w * 0.8])   # dense list, then lexical list

    fused = rrf_fuse(ranked_lists, k=rrf_k, weights=weights)[:limit]
    return [(idx, score, {"dense_rank": dense_ranks.get(idx),
                          "bm25_rank": bm25_ranks.get(idx)}) for idx, score in fused]
