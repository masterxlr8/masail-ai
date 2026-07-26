"""Embed the corpus into one vector index.

    python embed.py            # OpenAI if OPENAI_API_KEY is set, else local
    python embed.py --local    # force the free local model

There used to be a second index for the 121 FiqhQA four-school records, which
needed their own so they could not be crowded out of top-k by ~3,900
single-source fatwas. Those records were removed (see config.SOURCE_META), so
one index over the whole corpus is all that is left to build.

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
    print(f"  {len(docs)} docs")
    vecs = backend([d.embed_text for d in docs])
    np.save(VECTORS_SINGLE, vecs)
    print(f"    -> {VECTORS_SINGLE.name}  {vecs.shape}  {vecs.nbytes / 1e6:.1f} MB")

    # Row order == corpus order. search.py loads the same file in the same
    # order, so nothing needs to be stored alongside to align them.
    print("\nRow order matches corpus.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
