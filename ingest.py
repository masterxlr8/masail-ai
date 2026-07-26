"""Collate every source into ONE common corpus: data/corpus.json

    python ingest.py                  # all registered sources
    python ingest.py islamqa askimam  # only these
    python ingest.py --full           # ignore ISLAMQA_SAMPLE_FRAC, take all 15,296

ADDING A SCRAPED SOURCE
-----------------------
1. Add an entry to SOURCE_META in config.py (label, url, orientation, license).
   `orientation` must already exist in config.ORIENTATIONS.
2. Write an adapter here that returns list[Doc], building every record through
   `make_doc` - see `adapt_askimam_stub` for the exact call.
3. Register it in ADAPTERS.
Nothing downstream changes: embed.py, search.py and app.py are source-agnostic.

`make_doc` fills id, native_id, content_hash, provenance and the retrieval
fields itself, and `validate_docs` fails the run loudly if anything is missing,
so a scraper only has to supply title / question / answer / categories / url.

FINDINGS THIS HANDLES (measured, not assumed)
---------------------------------------------
IslamQA  19,052 rows -> 15,296 unique fatwas. 3,256 URLs are cross-listed under
         several topics with byte-identical question+answer; left in, one fatwa
         can occupy several top-5 slots and crowd out other sources, so we dedupe
         by URL and MERGE the topic labels. 100% of questions carry a 'Question'
         prefix, 100% of answers a 'Praise be to Allah.' prefix (3 doubled),
         15,035 rows contain non-breaking spaces, 7 rows are stubs.
         Then sampled to ISLAMQA_SAMPLE_FRAC, stratified by topic.

FiqhQA (121 four-school rows) was removed - see the note above SOURCE_META in
config.py. Every source here now carries a per-fatwa URL a reader can open.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date

import pandas as pd

from data.raw.config import (
    CATEGORY_FIXES,
    CORPUS,
    DATA,
    HF_PARQUET,
    ISLAMQA_SAMPLE_FRAC,
    MIN_ANSWER_CHARS,
    RAW,
    SAMPLE_PINS,
    SAMPLE_SEED,
    SCRAPED,
    SOURCE_META,
)
from data.raw.schema import (
    SINGLE_SOURCE,
    Doc,
    Position,
    content_duplicates,
    content_hash,
    save_corpus,
    validate_docs,
)

TODAY = date.today().isoformat()


# --------------------------------------------------------------------------- #
# shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _load_parquet(name: str) -> pd.DataFrame:
    """Read the cached parquet, downloading and caching on first run."""
    cached = RAW / f"{name}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    print(f"  downloading {name} ...")
    df = pd.read_parquet(HF_PARQUET[name])
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cached)
    return df


def clean_text(s: str) -> str:
    """Normalise whitespace and strip boilerplate prefixes.

    Prefixes are stripped repeatedly: 3 IslamQA answers carry a doubled
    'Praise be to Allah.'. A genuine sentence beginning with the word
    'Question' survives, because only the prefix form is matched.
    """
    if not isinstance(s, str):
        return ""
    for _ in range(3):
        new = re.sub(r"^Question\s*\n", "", s)
        new = re.sub(r"^Praise be to All?aa?h\.\s*", "", new)
        if new == s:
            break
        s = new
    s = s.replace("\xa0", " ").replace("​", "").replace("‎", "")
    s = s.replace("\t", " ")
    # A single lost byte from the upstream scrape, always an apostrophe.
    s = re.sub(r"(\w)�(\w)", r"\1'\2", s)
    s = re.sub(r"[ ]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def stratified_sample(df: pd.DataFrame, strat_col: str, frac: float,
                      seed: int) -> pd.DataFrame:
    """Take `frac` of each stratum, never fewer than one row.

    Plain random sampling could leave a thin topic empty; proportional-per-topic
    keeps the corpus mix identical to the full set and guarantees every topic
    still has at least one fatwa. Seeded, so the corpus is reproducible.
    """
    if frac >= 1.0:
        return df
    keep: list = []
    for _, group in df.groupby(strat_col, sort=True):
        n = max(1, round(len(group) * frac))
        keep.extend(group.sample(n=n, random_state=seed).index)
    return df.loc[sorted(set(keep))]


def make_doc(source: str, native_id, *, title, question, answer,
             categories, url, record_type=SINGLE_SOURCE, **extra) -> Doc:
    """Build a Doc with identity, provenance and derived fields filled in.

    Every adapter goes through here so no source can forget or misspell a field.
    Computed for you: id, content_hash, source_label, source_url, orientation,
    license, scholar default, retrieved_at, and (as Doc properties) embed_text
    and bm25_text.
    """
    meta = SOURCE_META[source]
    title, question, answer = clean_text(title), clean_text(question), clean_text(answer)
    positions: list[Position] = extra.get("positions") or []

    return Doc(
        id=f"{source}:{native_id}",
        source=source,
        native_id=str(native_id),
        content_hash=content_hash(title, question, answer),
        source_label=meta["source_label"],
        source_url=meta["source_url"],
        record_type=record_type,
        # Aggregators (islamqa.org spans all four madhhabs) set this per record;
        # single-orientation sites fall back to their SOURCE_META default.
        orientation=extra.get("orientation") or meta["orientation"],
        url=url,
        license=meta["license"],
        language=extra.get("language", "en"),
        retrieved_at=TODAY,
        date_published=extra.get("date_published"),
        title=title,
        question=question,
        answer=answer,
        categories=[CATEGORY_FIXES.get(c, c) for c in categories if c],
        scholar=extra.get("scholar", meta.get("scholar")),
        madhhab=extra.get("madhhab"),
        question_arabic=extra.get("question_arabic"),
        answer_arabic=extra.get("answer_arabic"),
        positions=positions or None,
    )


# --------------------------------------------------------------------------- #
# adapters                                                                    #
# --------------------------------------------------------------------------- #
def adapt_islamqa(frac: float = ISLAMQA_SAMPLE_FRAC) -> list[Doc]:
    df = _load_parquet("islamqa").copy()
    before = len(df)

    for col in ("title", "question", "answer"):
        df[col] = df[col].fillna("").map(clean_text)

    # Drop stubs before deduping so a stub can't win its group.
    df = df[df["answer"].str.len() >= MIN_ANSWER_CHARS]
    dropped = before - len(df)

    # One fatwa cross-listed under N topics arrives as N identical rows:
    # keep one, merge the topic labels.
    topics = (
        df.groupby("article_url")["topic"]
        .apply(lambda s: sorted({t for t in s if isinstance(t, str) and t.strip()}))
        .to_dict()
    )
    df = df.drop_duplicates("article_url", keep="first")
    merged = before - dropped - len(df)
    unique = len(df)

    # Stratify on the fatwa's primary (alphabetically first) merged topic.
    df["_primary"] = df["article_url"].map(lambda u: (topics.get(u) or ["Uncategorised"])[0])
    sampled = stratified_sample(df, "_primary", frac, SAMPLE_SEED)

    # Rescue demo-critical fatwas the sample would otherwise drop.
    pinned = pd.Index([])
    if frac < 1.0 and SAMPLE_PINS:
        blob = (df.title + " " + df.question + " " + df.answer).str.lower()
        mask = blob.str.contains("|".join(SAMPLE_PINS), regex=True, na=False)
        pinned = df.index[mask].difference(sampled.index)
        df = df.loc[sampled.index.union(pinned).sort_values()]
    else:
        df = sampled

    docs = [
        make_doc(
            "islamqa", r.original_id,
            title=r.title, question=r.question, answer=r.answer,
            categories=topics.get(r.article_url, []), url=r.article_url,
        )
        for r in df.itertuples(index=False)
    ]
    pct = 100 * len(docs) / max(unique, 1)
    print(f"  islamqa   {before:>6} rows -> {unique:>6} unique "
          f"(dropped {dropped} stubs, merged {merged} cross-listed duplicates)")
    print(f"            {unique:>6} unique -> {len(docs):>6} docs "
          f"({pct:.1f}% stratified over {df['_primary'].nunique()} topics, seed {SAMPLE_SEED}"
          f"{f', +{len(pinned)} pinned' if len(pinned) else ''})")
    return docs


def adapt_askimam_stub() -> list[Doc]:
    """TEMPLATE for a scraped source - not registered yet.

    Scrape into rows of {id, title, question, answer, categories, url} then:

        return [
            make_doc(
                "askimam", row["id"],
                title=row["title"], question=row["question"], answer=row["answer"],
                categories=row["categories"], url=row["url"],
                madhhab="hanafi",                 # single_source WITH a known school
                date_published=row.get("date"),   # ISO date, or omit
            )
            for row in rows
        ]

    Label, site url, orientation, licence and scholar come from SOURCE_META
    automatically; id, content_hash and retrieved_at are computed. Run
    `python ingest.py askimam` and validate_docs will name any field you missed.
    """
    return []


def adapt_scraped(source: str):
    """Load a scrape.py output file. It is already in the common schema.

    This is the whole point of the design: scrape.py emits Doc records directly,
    so ingest just reads and revalidates them. No per-source parsing here.
    """
    def _adapt() -> list[Doc]:
        path = SCRAPED[source]
        if not path.exists():
            print(f"  {source:<10} no file at {path} - run `python scrape.py {source}`")
            return []
        with open(path, encoding="utf-8") as fh:
            docs = [Doc.from_dict(d) for d in json.load(fh)]
        by_orient = Counter(d.orientation for d in docs)
        print(f"  {source:<10} {len(docs):>6} docs from {path.name}  "
              + " ".join(f"{k}:{v}" for k, v in sorted(by_orient.items())))
        return docs

    return _adapt


ADAPTERS = {
    "islamqa": adapt_islamqa,
    "islamqaorg": adapt_scraped("islamqaorg"),
    "askimam": adapt_scraped("askimam"),
}


# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    full = "--full" in argv
    wanted = [a for a in argv if not a.startswith("--")] or list(ADAPTERS)
    unknown = [s for s in wanted if s not in ADAPTERS]
    if unknown:
        print(f"unknown source(s): {unknown}. available: {list(ADAPTERS)}")
        return 1

    DATA.mkdir(parents=True, exist_ok=True)
    print("Collating corpus ...")

    docs: list[Doc] = []
    for name in wanted:
        fn = ADAPTERS[name]
        docs.extend(fn(frac=1.0) if full and name == "islamqa" else fn())

    errors = validate_docs(docs)
    if errors:
        print(f"\n{len(errors)} validation error(s):")
        for e in errors[:20]:
            print(f"  {e}")
        return 1

    dupes = content_duplicates(docs)
    if dupes:
        print(f"  note: {len(dupes)} content_hash group(s) share identical text "
              f"across ids - same fatwa republished, review before scaling up")

    # search.py indexes the corpus as one flat list, so a multi_school record
    # would sit in it unrenderable. The schema still allows the type; nothing
    # currently emits it, and this is where that would stop being true quietly.
    n_multi = sum(d.is_multi_school for d in docs)
    if n_multi:
        print(f"\n{n_multi} multi_school doc(s) present, but the four-school "
              "rendering path was removed - see config.SOURCE_META. Either drop "
              "them or restore that path before shipping.")
        return 1

    save_corpus(docs, CORPUS)
    print(f"\nWrote {CORPUS} - {len(docs)} docs")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
