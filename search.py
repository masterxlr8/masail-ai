"""Retrieval over the fatwa index.

    from search import Index
    idx = Index()
    hits = idx.search(query, k=5)   # hybrid, across the whole corpus

WHY HYBRID
----------
Pure vector search is weak on the rare domain terms this corpus is full of -
riba, masah, mudarabah, istihada, khul', talaq. BM25 nails exactly those and is
weak where embeddings are strong. Reciprocal rank fusion combines the two
without needing score calibration, which is the whole reason to use RRF rather
than a weighted sum of a cosine and a BM25 score that live on different scales.

DIVERSITY
---------
The corpus has six distinct voices, and they do not line up with sites:

    islamqa      Salafi                     askimam      Hanafi
    islamqaorg   Hanafi / Maliki / Shafi'i / Hanbali

Capping per SITE therefore did not do what it looked like it did. islamqaorg is
organised by school, so its two slots could both land in the same section, and
"where do you place your hands in prayer" - the canonical four-way split, and the
whole reason this corpus exists - came back as two Hanbali fatwas that barely
disagreed. The site cap was satisfied; the point of it was not.

So the cap keys on (source, orientation), and `search` fills in two passes: one
hit per voice first, in fused order, then the remaining slots by rank up to
MAX_PER_VOICE each. Breadth before depth - without the second pass a query
covered by only one voice would return a single card.

The first pass is bounded by DIVERSITY_MARGIN, and that bound is the load-bearing
part. Breadth cannot be manufactured: ask where the hands go in prayer and the
corpus has four Hanbali answers and nothing else on qabd, so an unbounded first
pass spends four of five slots on a Salafi answer about raising the hands, a
Maliki one about joining prayers, and so on - a page that looks like four schools
disagreeing when three of them were never asked. A new voice is only worth a slot
if it is within DIVERSITY_MARGIN of the best hit; otherwise the slot goes back to
depth, and a single-school answer is reported honestly as a single-school answer.

RELEVANCE
---------
top-k always returns k things, however bad the k-th is. ABSTAIN_THRESHOLD only
guarded the BEST hit, so one strong match licensed four weak ones and each of
those still got a card and a source badge. `search` now drops any hit that fails
RELEVANCE_THRESHOLD *and* is not strong lexically (BM25_STRONG_RANK) - both
signals have to dislike it. That is the cheap half of the filter; the accurate
half is generate.CardOut.answers_question, where the model has read the document.

HITS ARE COPIES
---------------
`score` is a per-query value living on a Doc that the corpus owns and reuses for
every query, in every session (app.py caches one Index for the whole server). So
the searchers hand back `dataclasses.replace` copies rather than the corpus
objects: a second question must not rewrite the scores a first answer is still
displaying, and two concurrent users must not overwrite each other's.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import replace

import numpy as np

from data.raw.config import (
    ABSTAIN_THRESHOLD,
    BM25_STRONG_RANK,
    CORPUS,
    EMBED_DIMS,
    EMBED_MODEL,
    LOCAL_EMBED_MODEL,
    RELEVANCE_THRESHOLD,
    RRF_K,
    TOP_K_ISLAMQA,
    VECTORS_SINGLE,
)
from data.raw.schema import Doc, load_corpus

MAX_PER_VOICE = 2       # of TOP_K_ISLAMQA, so no one site+school owns the results
DIVERSITY_MARGIN = 0.08  # cosine a new voice may trail the best hit by, and still
                         # be worth a slot. See DIVERSITY.
TOKEN = re.compile(r"[a-z0-9']+")


def tokenise(s: str) -> list[str]:
    return TOKEN.findall(s.lower())


def voice(doc: Doc) -> tuple[str, str]:
    """The unit diversity is measured in: who published it, in which school."""
    return doc.source, doc.orientation


class Index:
    """Loads the corpus and the vector file once, then answers queries."""

    def __init__(self) -> None:
        self.docs = load_corpus(CORPUS)

        # The schema still permits multi_school records, but the four-school
        # rendering path was removed with FiqhQA - one would be indexed here and
        # then render as a card with an empty body.
        multi = [d.id for d in self.docs if d.is_multi_school]
        if multi:
            raise RuntimeError(
                f"{len(multi)} multi_school doc(s) in the corpus ({multi[:3]}...) but "
                "the four-school path was removed - see config.SOURCE_META"
            )

        # Not optional. ABSTAIN_THRESHOLD and RELEVANCE_THRESHOLD are cosine
        # values, so without vectors every score is 0 and the app abstains on
        # everything - a silent, total failure that looks like a thin corpus.
        if not VECTORS_SINGLE.exists():
            raise RuntimeError(f"missing {VECTORS_SINGLE.name} - run embed.py")
        self.vectors = np.load(VECTORS_SINGLE)
        if len(self.vectors) != len(self.docs):
            raise RuntimeError(
                f"{VECTORS_SINGLE.name} has {len(self.vectors)} rows but the corpus "
                f"has {len(self.docs)} docs - re-run embed.py"
            )

        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([tokenise(d.bm25_text) for d in self.docs])
        self._embedder = None

    # -- query embedding ---------------------------------------------------
    def embed_query(self, query: str) -> np.ndarray:
        if self._embedder is None:
            self._embedder = self._make_embedder()
        v = self._embedder(query).astype(np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def _make_embedder(self):
        # The stored vectors decide the model, not the environment: an
        # OPENAI_API_KEY appearing next to a corpus embedded with bge-small must
        # NOT switch the query side to a different vector space.
        dims = self.vectors.shape[1]
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
    def search(
        self, query: str, k: int = TOP_K_ISLAMQA, qv: np.ndarray | None = None
    ) -> list[Doc]:
        """Hybrid cosine + BM25, fused by RRF, then spread across the voices.

        Results are in FUSED rank order; `score` is the cosine, which is the
        interpretable number but not the one that decided the ordering, so it is
        not monotonic down the list.
        """
        if not self.docs:
            return []

        if qv is None:
            qv = self.embed_query(query)
        cos = self.vectors @ qv
        vec_rank = np.argsort(-cos)
        bm_rank = np.argsort(-self.bm25.get_scores(tokenise(query)))

        # RRF: 1/(K + rank). Rank-based, so the two score scales never meet.
        fused = np.zeros(len(self.docs))
        for rank, i in enumerate(vec_rank[:200]):
            fused[i] += 1.0 / (RRF_K + rank)
        for rank, i in enumerate(bm_rank[:200]):
            fused[i] += 1.0 / (RRF_K + rank)

        # Strong lexical evidence, kept as a set so a rare-term match can survive
        # a failing cosine. See config.BM25_STRONG_RANK.
        lexical = set(int(i) for i in bm_rank[:BM25_STRONG_RANK])

        ranked: list[int] = []
        for i in np.argsort(-fused):
            if fused[i] <= 0:
                break
            # Neither signal likes it: not semantically close, not a keyword
            # match either. Carding this would attribute a real scholar's real
            # fatwa to a question it never addressed.
            if cos[i] < RELEVANCE_THRESHOLD and int(i) not in lexical:
                continue
            ranked.append(int(i))

        # Pass 1 takes the best hit from each voice that is close enough to the
        # best hit overall to be worth the slot; pass 2 backfills by rank - see
        # DIVERSITY. Both walk the same fused order, so the result stays ranked
        # within each pass; it is only the passes that reorder anything.
        if not ranked:
            return []
        # Measured against the top-ranked hit, not the best cosine anywhere in
        # the list: fusion can leave a very high cosine far down the ranking, and
        # letting that set the bar quietly excludes voices from the first pass.
        floor = cos[ranked[0]] - DIVERSITY_MARGIN

        taken: dict[tuple[str, str], int] = {}
        chosen: list[int] = []
        for limit, near in ((1, True), (MAX_PER_VOICE, False)):
            for i in ranked:
                if len(chosen) == k:
                    break
                if near and cos[i] < floor:
                    continue
                v = voice(self.docs[i])
                if taken.get(v, 0) >= limit or i in chosen:
                    continue
                taken[v] = taken.get(v, 0) + 1
                chosen.append(i)

        return [replace(self.docs[i], score=float(cos[i])) for i in chosen]

    # -- the decision app.py renders --------------------------------------
    def retrieve(self, query: str) -> dict:
        """Everything the UI needs, including the abstain state."""
        hits = self.search(query)
        top = max([h.score for h in hits] + [0.0])
        return {
            "query": query,
            "hits": hits,
            "abstain": top < ABSTAIN_THRESHOLD,
            "top_score": top,
        }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from data.raw.demo_queries import QUERIES

    idx = Index()
    for q in sys.argv[1:] or QUERIES:
        r = idx.retrieve(q)
        state = "ABSTAIN" if r["abstain"] else "answered"
        print(f"\n=== {q}\n    [{state}] top={r['top_score']:.3f}  {len(r['hits'])} hit(s)")
        for h in r["hits"]:
            print(f"      {h.score:.3f}  {h.orientation:<9} {h.title[:58]}")
