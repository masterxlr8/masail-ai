"""Embed the corpus into two vector indexes.

    python embed.py            # OpenAI if OPENAI_API_KEY is set, else local
    python embed.py --local    # force the free local model

TWO INDEXES, ON PURPOSE
-----------------------
multi_school records are 121 of ~4,000. In one shared index they would almost
never win top-k, and the four-school comparison - the headline feature - would
silently never fire. So they get their own index. Searching 121 vectors costs
nothing, which means the panel fires whenever there is a decent match regardless
of what the single-source index returns.

WHAT GETS EMBEDDED
------------------
Doc.embed_text = title + question, deliberately NOT the answer. Queries are
questions, and question-to-question similarity is much stronger than
question-to-answer. The answer is still fully searchable through BM25 in
search.py, which is the half of retrieval that handles rare terms well.

Vectors are L2-normalised on write, so cosine similarity at query time is a
single matmul.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from data.raw.config import (
    CORPUS,
    EMBED_BATCH,
    EMBED_DIMS,
    EMBED_MODEL,
    LOCAL_EMBED_MODEL,
    VECTORS_MULTI,
    VECTORS_SINGLE,
)
from data.raw.schema import Doc, load_corpus


def normalise(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


def embed_openai(texts: list[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        chunk = texts[i: i + EMBED_BATCH]
        resp = client.embeddings.create(
            model=EMBED_MODEL, input=chunk, dimensions=EMBED_DIMS
        )
        out += [d.embedding for d in resp.data]
        print(f"    {min(i + EMBED_BATCH, len(texts))}/{len(texts)}", end="\r")
    print()
    return normalise(np.array(out))


def embed_local(texts: list[str]) -> np.ndarray:
    """Free CPU fallback. Slower, 384 dims, no API key needed."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(LOCAL_EMBED_MODEL)
    v = model.encode(texts, batch_size=64, show_progress_bar=True,
                     convert_to_numpy=True)
    return normalise(v)


def main(argv: list[str]) -> int:
    use_local = "--local" in argv or not os.getenv("OPENAI_API_KEY")
    backend = embed_local if use_local else embed_openai
    print(f"Embedding with {'local ' + LOCAL_EMBED_MODEL if use_local else EMBED_MODEL}")

    docs = load_corpus(CORPUS)
    single = [d for d in docs if not d.is_multi_school]
    multi = [d for d in docs if d.is_multi_school]

    for name, group, path in (("single_source", single, VECTORS_SINGLE),
                              ("multi_school", multi, VECTORS_MULTI)):
        if not group:
            print(f"  {name}: none, skipped")
            continue
        print(f"  {name}: {len(group)} docs")
        vecs = backend([d.embed_text for d in group])
        np.save(path, vecs)
        print(f"    -> {path.name}  {vecs.shape}  {vecs.nbytes / 1e6:.1f} MB")

    # Row order == corpus order within each group. search.py rebuilds the same
    # split from corpus.json, so nothing needs to be stored alongside.
    print("\nRow order matches corpus.json filtered by record_type.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
