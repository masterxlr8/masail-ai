"""The common structure every source collapses into.

ONE record type (`Doc`) for every source, present and future. Adding a scraped
source (askimam, daruliftaa, ...) means writing an adapter that emits `Doc`s -
nothing downstream changes.

Design rules, so we stay in sync:

1. `record_type` is the only real branch in the whole system:
     'single_source'  one scholar/site answering  -> use `answer`
     'multi_school'   several schools compared    -> use `positions`
   Everything else (retrieval, badging, storage) is source-agnostic.

2. The stored JSON holds CONTENT + PROVENANCE only. Retrieval text
   (`embed_text`, `bm25_text`) is DERIVED on the Doc, never persisted - it is a
   pure function of the content, and storing it doubled the corpus (bm25_text
   alone was 46% of the file). A scraped adapter therefore cannot get these
   wrong, because it never supplies them.

3. `categories` is always a list, never a string. IslamQA cross-lists one fatwa
   under several topics; deduping by URL merges those topics rather than
   discarding them. Single-topic sources get a one-element list.

4. Every Doc carries `orientation`, drawn from the fixed ORIENTATIONS vocabulary
   in config.py. The UI badges it on every card - we never present a
   single-madhhab source as neutral, and a free-text field would let a scraper
   silently split one badge into three spellings.

5. Provenance fields (`scholar`, `license`, `retrieved_at`, `source_url`,
   `date_published`) are on every record because a scraped corpus needs them and
   retrofitting is painful. `scholar` holds the answering mufti OR the issuing
   institution - aggregators like islamqa.org give us the latter.

5b. NO SOURCE-SPECIFIC FIELDS. Every field here means the same thing for every
   source, or it does not belong. FiqhQA's `Agreement` column was dropped for
   exactly this reason: it was meaningful for 121 records and null for the rest,
   which is a per-source annotation wearing a common-schema costume. Anything
   like that lives in the source's own raw file and is joined at eval time.

6. `content_hash` is computed, not supplied. It is what makes re-scraping safe:
   same hash = unchanged page (skip), changed hash = the fatwa was edited, and
   the same hash under two different `id`s = the same fatwa republished across
   two sites, which happens constantly between Hanafi darul-iftas.

7. `validate_docs()` runs at the end of every ingest. A scraper that omits a
   field or invents an orientation fails loudly at ingest rather than producing
   a card with a blank badge three hours later.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from data.raw.config import MAX_ANSWER_CHARS_FOR_LLM, MIN_ANSWER_CHARS, ORIENTATIONS

SINGLE_SOURCE = "single_source"
MULTI_SCHOOL = "multi_school"
RECORD_TYPES = [SINGLE_SOURCE, MULTI_SCHOOL]

SCHOOL_KEYS = ["hanafi", "shafii", "maliki", "hanbali"]

VERDICTS = [
    "permissible",
    "impermissible",
    "disliked",
    "recommended",
    "obligatory",
    "depends",
]

_WS = re.compile(r"\s+")


def content_hash(title: str, question: str, answer: str) -> str:
    """Stable fingerprint of a fatwa's substance.

    Whitespace and case are normalised away so a re-scrape that only rewraps
    lines does not read as an edit. 16 hex chars is ample for ~10^5 records.
    """
    blob = _WS.sub(" ", f"{title} {question} {answer}").strip().lower()
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class Position:
    """One school's stance within a `multi_school` record."""

    school: str        # 'hanafi' | 'shafii' | 'maliki' | 'hanbali'
    school_label: str  # 'Hanafi' | "Shafi'i" | 'Maliki' | 'Hanbali'
    text: str
    has_opinion: bool  # False where the source states no recorded view


@dataclass
class Doc:
    """A single retrievable unit.

    For a single-source fatwa that is one Q&A. For a multi-school record it is
    one question carrying every school's position - deliberately NOT one record
    per school, so a single retrieval hit yields the complete comparison.
    """

    # --- identity & provenance -------------------------------------------
    id: str                 # '<source>:<native_id>' - globally unique
    source: str             # 'islamqa' | 'fiqhqa' | 'askimam' | ...
    native_id: str          # the source's own id, kept separately so a scraper
                            # can resume without string-splitting `id`
    content_hash: str       # computed; see module docstring rule 6
    source_label: str       # human-readable, shown on the card badge
    source_url: str         # site root, for attribution
    record_type: str        # SINGLE_SOURCE | MULTI_SCHOOL
    orientation: str        # one of config.ORIENTATIONS
    url: str                # canonical link to THIS fatwa
    license: str
    language: str = "en"
    retrieved_at: str = ""       # ISO date WE snapshotted the source
    date_published: str | None = None  # ISO date the SOURCE published it

    # --- content (every record) ------------------------------------------
    title: str = ""
    question: str = ""
    answer: str = ""        # single_source: the fatwa body. multi_school: summary statement.
    categories: list[str] = field(default_factory=list)

    # --- optional, source-dependent --------------------------------------
    scholar: str | None = None          # e.g. 'Mufti Ebrahim Desai'
    madhhab: str | None = None          # SCHOOL_KEYS; set when a single_source has a known school
    question_arabic: str | None = None  # original-language question
    answer_arabic: str | None = None    # original-language anchor text
    positions: list[Position] | None = None   # multi_school only

    # --- runtime only ------------------------------------------------------
    score: float = 0.0

    # ---------------------------------------------------------------------
    @property
    def is_multi_school(self) -> bool:
        return self.record_type == MULTI_SCHOOL

    @property
    def stated_positions(self) -> list[Position]:
        """Positions excluding schools with no recorded opinion."""
        return [p for p in (self.positions or []) if p.has_opinion]

    @property
    def embed_text(self) -> str:
        """What gets embedded: title + question, deliberately NOT the answer.

        User queries are questions, and question-to-question similarity is far
        stronger than question-to-answer.
        """
        return f"{self.title}\n{self.question}".strip()

    @property
    def bm25_text(self) -> str:
        """What BM25 indexes: the full substance.

        Keyword search is what catches rare domain terms (riba, masah,
        mudarabah, istihada) that embeddings handle poorly, so the body and
        every stated school position belong here.
        """
        parts = [self.title, self.question, self.answer]
        parts += [p.text for p in self.stated_positions]
        return "\n".join(p for p in parts if p).strip()

    def llm_context(self) -> str:
        """Answer text truncated for use inside an LLM prompt."""
        body = self.answer
        if len(body) > MAX_ANSWER_CHARS_FOR_LLM:
            body = body[:MAX_ANSWER_CHARS_FOR_LLM].rsplit(" ", 1)[0] + " ..."
        return body

    def problems(self) -> list[str]:
        """Everything wrong with this record. Empty list = valid."""
        bad: list[str] = []

        for f in ("id", "source", "native_id", "content_hash", "source_label",
                  "source_url", "record_type", "orientation", "url", "license",
                  "language", "retrieved_at"):
            if not getattr(self, f):
                bad.append(f"missing {f}")

        # `question` may legitimately be empty - some aggregated pages are
        # articles rather than Q&A. What must never be empty is embed_text,
        # or the record is unreachable by vector search.
        if not self.embed_text:
            bad.append("empty embed_text (both title and question are blank)")

        if self.id != f"{self.source}:{self.native_id}":
            bad.append("id is not '<source>:<native_id>'")
        if self.record_type not in RECORD_TYPES:
            bad.append(f"record_type {self.record_type!r} not in {RECORD_TYPES}")
        if self.orientation not in ORIENTATIONS:
            bad.append(f"orientation {self.orientation!r} not in config.ORIENTATIONS")
        if self.madhhab is not None and self.madhhab not in SCHOOL_KEYS:
            bad.append(f"madhhab {self.madhhab!r} not in {SCHOOL_KEYS}")
        if not isinstance(self.categories, list) or not all(
            isinstance(c, str) and c for c in self.categories
        ):
            bad.append("categories must be a list of non-empty strings")

        if self.is_multi_school:
            got = [p.school for p in (self.positions or [])]
            if got != SCHOOL_KEYS:
                bad.append(f"multi_school positions must be exactly {SCHOOL_KEYS}, got {got}")
        else:
            if self.positions:
                bad.append("single_source must not carry positions")
            if len(self.answer) < MIN_ANSWER_CHARS:
                bad.append(f"answer shorter than MIN_ANSWER_CHARS ({len(self.answer)})")

        return bad

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("score", None)  # runtime only, never persisted
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Doc:
        d = dict(d)
        if d.get("positions"):
            d["positions"] = [Position(**p) for p in d["positions"]]
        d.pop("score", None)
        return cls(**d)


@dataclass
class RulingCard:
    """One source's or one school's ruling. Never a merged verdict."""

    doc_id: str
    attribution: str   # 'Hanafi school' | 'IslamQA.info (Salafi)'
    verdict: str       # one of VERDICTS
    one_line: str
    reasoning: str
    evidences: list[dict] = field(default_factory=list)  # {type, ref, quote}
    conditions: list[str] = field(default_factory=list)
    unverified_quotes: list[str] = field(default_factory=list)


@dataclass
class Comparison:
    """Where sources agree and diverge. Deliberately has NO overall verdict."""

    agreement: list[str]
    divergence: list[dict]  # {point, positions: [{who, stance}]}
    turns_on: str


# ---------------------------------------------------------------------------
def validate_docs(docs: list[Doc]) -> list[str]:
    """Hard corpus-wide checks. Returns human-readable errors, empty if clean."""
    errors: list[str] = []
    seen: set[str] = set()
    for d in docs:
        errors += [f"{d.id}: {p}" for p in d.problems()]
        if d.id in seen:
            errors.append(f"{d.id}: duplicate id")
        seen.add(d.id)
    return errors


def content_duplicates(docs: list[Doc]) -> dict[str, list[str]]:
    """Groups of doc ids sharing identical content.

    Not an error: darul-iftas republish each other's fatwas, and the same page
    can live at two URLs. Reported at ingest so we can decide per case whether
    to drop one or keep both as independent attestations.
    """
    by_hash: dict[str, list[str]] = {}
    for d in docs:
        by_hash.setdefault(d.content_hash, []).append(d.id)
    return {h: ids for h, ids in by_hash.items() if len(ids) > 1}


def save_corpus(docs: list[Doc], path) -> None:
    """Write the merged corpus as a JSON array, one object per record."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([d.to_dict() for d in docs], fh, ensure_ascii=False, indent=1)


def load_corpus(path) -> list[Doc]:
    with open(path, encoding="utf-8") as fh:
        return [Doc.from_dict(d) for d in json.load(fh)]
