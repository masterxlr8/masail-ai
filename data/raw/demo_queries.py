"""The 7 demo queries. Every one is verified against the built corpus.

`python demo_queries.py` re-checks that each expected doc is still present -
run it after any change to sampling, cleaning or SAMPLE_PINS.

Chosen so the demo shows all three states: the four-school panel firing on a
genuine classical split, single-source contemporary answers with an orientation
badge, and a clean abstain.
"""

from __future__ import annotations

FOUR_SCHOOL = [
    {
        "query": "Does touching a woman break wudu?",
        "expect_doc": "fiqhqa:21",
        "why": "CENTREPIECE. Agreement='Disagreement'. The canonical split - "
               "Hanafi no, Shafi'i yes, Maliki/Hanbali with desire. Easy to explain.",
    },
    {
        "query": "Where do you place your hands in prayer?",
        "expect_doc": "fiqhqa:1",
        "why": "Qabd. Disagreement, and visually obvious to a non-specialist judge.",
    },
    {
        "query": "Does a divorce pronounced during menstruation count?",
        "expect_doc": "fiqhqa:48",
        "why": "Disagreement, Marriage category - shows the panel is not just Purity.",
    },
]

SINGLE_SOURCE = [
    {
        "query": "Is a conventional mortgage permissible?",
        "why": "32 fatwas in corpus. Contemporary - FiqhQA cannot cover it, so the "
               "coverage badge must read 'single-source (Salafi orientation)'.",
    },
    {
        "query": "What is the ruling on buying and selling bitcoin?",
        "expect_doc": "islamqa:360668",
        "why": "The ONLY bitcoin fatwa in 15,296. Survives sampling via SAMPLE_PINS - "
               "if this query fails, the pin broke.",
    },
    {
        "query": "Is life insurance allowed in Islam?",
        "why": "17 fatwas. Second contemporary case.",
    },
]

ABSTAIN = [
    {
        "query": "What are the rules for prayer on Mars?",
        "why": "Must abstain, not invent. Demo this deliberately - a RAG that knows "
               "when to stop beats one that always answers.",
    },
]

ALL = FOUR_SCHOOL + SINGLE_SOURCE + ABSTAIN
QUERIES = [q["query"] for q in ALL]


def verify() -> int:
    """Assert every expected doc is still in the corpus."""
    from data.raw.config import CORPUS
    from data.raw.schema import load_corpus

    by_id = {d.id: d for d in load_corpus(CORPUS)}
    missing = 0
    for q in ALL:
        want = q.get("expect_doc")
        if not want:
            print(f"  --  {q['query']}")
        elif want in by_id:
            print(f"  ok  {q['query']}\n        -> {want}  {by_id[want].title[:60]}")
        else:
            print(f"  MISSING {want} for: {q['query']}")
            missing += 1
    print(f"\n{len(ALL)} queries, {missing} missing expected doc(s)")
    return 1 if missing else 0


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(verify())
