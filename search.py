"""Retrieval over the two indexes.

    from search import Index
    idx = Index()
    hits  = idx.search_single(query, k=5)   # hybrid, across all single-source fatwas
    panel = idx.search_multi(query, k=1)    # the four-school comparison record

WHY HYBRID
----------
Pure vector search is weak on the rare domain terms this corpus is full of -
riba, masah, mudarabah, istihada, khul', talaq. BM25 nails exactly those and is
weak where embeddings are strong. Reciprocal rank fusion combines the two
without needing score calibration, which is the whole reason to use RRF rather
than a weighted sum of a cosine and a BM25 score that live on different scales.

WHY TWO INDEXES
---------------
The 121 multi_school records would never win top-k against ~3,900 single-source
fatwas. Searching them separately costs nothing and guarantees the four-school
panel fires whenever there is a real match. app.py turns that into a visible
coverage badge rather than a silent absence.

DIVERSITY
---------
`search_single` caps how many hits any one source may occupy (MAX_PER_SOURCE).
Without it a single prolific darul-ifta can fill all five slots, which defeats
the point of having assembled a multi-orientation corpus.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

from data.raw.config import (
    ABSTAIN_THRESHOLD,
    CORPUS,
    EMBED_DIMS,
    EMBED_MODEL,
    FIQH_THRESHOLD,
    LOCAL_EMBED_MODEL,
    RRF_K,
    TOP_K_ISLAMQA,
    VECTORS_MULTI,
    VECTORS_SINGLE,
)
from data.raw.schema import Doc, load_corpus

MAX_PER_SOURCE = 2      # of TOP_K_ISLAMQA, so no single site owns the results
TOKEN = re.compile(r"[a-z0-9']+")


def tokenise(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


class Index:
    """Loads the corpus and both vector files once, then answers queries."""

    def __init__(self) -> None:
        docs = load_corpus(CORPUS)
        self.single = [d for d in docs if not d.is_multi_school]
        self.multi = [d for d in docs if d.is_multi_school]

        self.v_single = np.load(VECTORS_SINGLE) if VECTORS_SINGLE.exists() else None
        self.v_multi = np.load(VECTORS_MULTI) if VECTORS_MULTI.exists() else None
        if self.v_single is not None and len(self.v_single) != len(self.single):
            raise RuntimeError(
                f"vectors_single has {len(self.v_single)} rows but the corpus has "
                f"{len(self.single)} single_source docs - re-run embed.py"
            )

        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenise(d.bm25_text) for d in self.single])
        self._embedder = None

    # -- query embedding ---------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        if self._embedder is None:
            self._embedder = self._make_embedder()
        v = self._embedder(query).astype(np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def _make_embedder(self):
        dims = self.v_single.shape[1] if self.v_single is not None else EMBED_DIMS
        if os.getenv("OPENAI_API_KEY") and dims == EMBED_DIMS:
            from openai import OpenAI

            client = OpenAI()

            def openai_embed(q: str) -> np.ndarray:
                r = client.embeddings.create(
                    model=EMBED_MODEL, input=[q], dimensions=EMBED_DIMS
                )
                return np.array(r.data[0].embedding)

            return openai_embed

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(LOCAL_EMBED_MODEL)
        return lambda q: model.encode([q], convert_to_numpy=True)[0]

    # -- retrieval ---------------------------------------------------------
    def search_single(self, query: str, k: int = TOP_K_ISLAMQA) -> list[Doc]:
        """Hybrid cosine + BM25, fused by RRF, then capped per source."""
        if not self.single:
            return []

        bm_rank = np.argsort(-self.bm25.get_scores(tokenise(query)))
        if self.v_single is not None:
            cos = self.v_single @ self.embed_query(query)
            vec_rank = np.argsort(-cos)
        else:
            cos, vec_rank = np.zeros(len(self.single)), bm_rank

        # RRF: 1/(K + rank). Rank-based, so the two score scales never meet.
        fused = np.zeros(len(self.single))
        for rank, i in enumerate(vec_rank[:200]):
            fused[i] += 1.0 / (RRF_K + rank)
        for rank, i in enumerate(bm_rank[:200]):
            fused[i] += 1.0 / (RRF_K + rank)

        out: list[Doc] = []
        per_source: dict[str, int] = {}
        for i in np.argsort(-fused):
            if fused[i] <= 0:
                break
            doc = self.single[i]
            if per_source.get(doc.source, 0) >= MAX_PER_SOURCE:
                continue
            per_source[doc.source] = per_source.get(doc.source, 0) + 1
            doc.score = float(cos[i])       # cosine is the interpretable one
            out.append(doc)
            if len(out) == k:
                break
        return out

    def search_multi(self, query: str, k: int = 1) -> list[Doc]:
        """Plain cosine over the small multi-school index."""
        if not self.multi or self.v_multi is None:
            return []
        cos = self.v_multi @ self.embed_query(query)
        out = []
        for i in np.argsort(-cos)[:k]:
            doc = self.multi[i]
            doc.score = float(cos[i])
            out.append(doc)
        return out

    # -- the decision app.py renders --------------------------------------
    def retrieve(self, query: str) -> dict:
        """Everything the UI needs, including the coverage and abstain states."""
        hits = self.search_single(query)
        panel = self.search_multi(query, k=1)
        best_panel = panel[0] if panel else None

        show_schools = bool(best_panel and best_panel.score >= FIQH_THRESHOLD)
        top = max([h.score for h in hits] + [best_panel.score if best_panel else 0.0])
        return {
            "query": query,
            "hits": hits,
            "panel": best_panel if show_schools else None,
            "show_schools": show_schools,
            "abstain": top < ABSTAIN_THRESHOLD,
            "top_score": top,
        }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from data.raw.demo_queries import QUERIES

    idx = Index()
    for q in sys.argv[1:] or QUERIES:
        r = idx.retrieve(q)
        state = ("ABSTAIN" if r["abstain"]
                 else "four-school" if r["show_schools"] else "single-source")
        print(f"\n=== {q}\n    [{state}] top={r['top_score']:.3f}")
        if r["panel"]:
            print(f"    PANEL {r['panel'].score:.3f}  {r['panel'].title[:60]}")
        for h in r["hits"]:
            print(f"      {h.score:.3f}  {h.orientation:<9} {h.title[:58]}")
