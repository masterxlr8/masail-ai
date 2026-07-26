"""The 7 demo queries. Every one is verified against the built corpus.

`python demo_queries.py` re-checks that each expected doc is still present -
run it after any change to sampling, cleaning or SAMPLE_PINS.

Chosen so the demo shows both states: attributed answers that genuinely disagree
across orientations, and a clean abstain.

These used to be split into FOUR_SCHOOL and SINGLE_SOURCE, back when 121 FiqhQA
records supplied all four madhhabs' positions from one row. Those records were
removed (see config.SOURCE_META), so the classical splits are now demonstrated
the same way as everything else - several sites of different leanings answering
the same question, each under its own name.
"""

from __future__ import annotations

CLASSICAL_SPLITS = [
    {
        "query": "Does touching a woman break wudu?",
        "why": "CENTREPIECE. The canonical split - Hanafi no, Shafi'i yes, "
               "Maliki/Hanbali with desire. Should surface sites of several "
               "leanings disagreeing, each attributed to its own mufti.",
    },
    {
        "query": "Is it permissible to eat seafood other than fish?",
        "why": "The best cross-madhhab demo in the corpus. Hanafi (AskImam) restricts "
               "halal sea life to fish on a hadith about carrion; Shafi'i and "
               "IslamQA.info permit all of it. Three schools, three cards, one real "
               "disagreement - and everyday enough that a non-specialist sees it.",
    },
    {
        "query": "Is it allowed to combine two prayers when travelling?",
        "why": "Retrieves all five voices - Hanafi, Shafi'i, Hanbali, Maliki, "
               "IslamQA.info - which is the widest spread any query gets.",
    },
    {
        "query": "Where do you place your hands in prayer?",
        "why": "Qabd. Kept deliberately as the counter-example: the corpus has four "
               "Hanbali answers on qabd and nothing else, so this one SHOULD come "
               "back single-school. If it ever renders four madhhabs, the diversity "
               "pass has started manufacturing breadth - see search.DIVERSITY.",
    },
    {
        "query": "Does a divorce pronounced during menstruation count?",
        "why": "Disagreement in the Marriage category - shows the split is not just Purity.",
    },
]

CONTEMPORARY = [
    {
        "query": "Is a conventional mortgage permissible?",
        "why": "32 fatwas in corpus. Contemporary - no classical text covers it, so "
               "every card here is a named modern mufti.",
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

ALL = CLASSICAL_SPLITS + CONTEMPORARY + ABSTAIN
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
