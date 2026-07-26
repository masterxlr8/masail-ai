"""The generation half of the RAG loop: retrieved Docs -> RulingCards -> Comparison.

    from search import Index
    from generate import answer

    result = answer("Does touching a woman break wudu?", Index())
    result.state        # 'abstain' | 'answered'
    result.cards        # list[RulingCard] - one per retrieved document
    result.comparison   # Comparison | None

WHAT THIS LAYER IS ALLOWED TO DO
--------------------------------
Extract and attribute. Nothing else. The product rule from plan.md is that we
never merge sources into one synthesised ruling, because a merged verdict is a
position no scholar actually holds. So:

  * every card is scoped to EXACTLY ONE document, and the prompt says so;
  * the comparison call is given the finished cards and forbidden to adjudicate -
    `Comparison` has no overall-verdict field to put an adjudication in even if
    the model tried;
  * every evidence quote is checked back against the source text before it
    renders (`verify_quotes`), and anything that fails moves to
    `unverified_quotes` rather than being silently dropped or silently shown.

WHO A CARD SPEAKS FOR
---------------------
The answering mufti or site, and only them. A card once read "Hanafi school -
qabd is a Sunnah act", built from a single Hanafi fatwa; that is a claim about
several centuries of scholarship resting on one page, and madhhabs disagree
internally all the time. `attribution` is therefore always a name a reader can
follow to a document - `attribution_for` builds it - and both prompts below
forbid generalising from one fatwa to a school. The site's leaning is still
shown, labelled as the site's.

THE TWO GENERATION PATHS
------------------------
  source cards  one structured call per Doc, scoped to that Doc. <=5 calls.
                That call also decides `answers_question`, which is the second
                half of the relevance filter: search.py drops what is cheaply
                provable noise, and this drops what only looks relevant until you
                read it. A false there discards the card.
  comparison    one call over the finished cards. 1 call.

TEMPERATURE
-----------
1.0 - the API default. See `sampling_args` below for why it is sometimes passed
and sometimes omitted; the behaviour is the same either way.
"""

from __future__ import annotations

import concurrent.futures as cf
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from data.raw.config import (
    ABSTAIN_THRESHOLD,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_NO_TEMPERATURE_MODELS,
    LLM_TEMPERATURE,
)
from data.raw.schema import VERDICTS, Comparison, Doc, RulingCard

load_dotenv()   # ANTHROPIC_API_KEY, from .env. Never hardcode it here.


# ---------------------------------------------------------------------------
# Client and sampling
# ---------------------------------------------------------------------------
_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    """The shared client, built on first use.

    No api_key= argument: the SDK resolves ANTHROPIC_API_KEY from the
    environment (populated by load_dotenv above, or by Streamlit's secrets on
    deploy). Built lazily so importing this module never needs a key - the
    notebook can walk through the retrieval half with no credentials at all.
    """
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def sampling_args(model: str = LLM_MODEL) -> dict:
    """`{"temperature": 1.0}` or `{}`, whichever the model accepts.

    Temperature stopped being a free parameter on the current generation:

      claude-sonnet-5              only the DEFAULT value is accepted; any other
                                   value returns a 400.
      claude-opus-5, fable-5,      the parameter is removed entirely - sending it
      opus-4.8, opus-4.7           at all returns a 400.

    The default IS 1.0, which is the value we want, so passing it on Sonnet 5 and
    omitting it on Opus 5 produce identical sampling. This function exists so
    that switching LLM_MODEL in config.py cannot break the request.
    """
    if model in LLM_NO_TEMPERATURE_MODELS:
        return {}
    return {"temperature": LLM_TEMPERATURE}


# ---------------------------------------------------------------------------
# Structured-output schemas
#
# These are the LLM's response contract, deliberately separate from schema.py's
# RulingCard: the model supplies only the fields it is qualified to supply.
# `doc_id` and `attribution` are ours - filled in from the Doc, so a card can
# never be mis-attributed by a hallucination.
# ---------------------------------------------------------------------------
Verdict = Literal[
    "permissible", "impermissible", "disliked", "recommended", "obligatory", "depends"
]
assert set(Verdict.__args__) == set(VERDICTS), "Verdict literal drifted from schema.VERDICTS"


class Evidence(BaseModel):
    type: Literal["quran", "hadith", "scholarly"]
    ref: str = Field(description="Surah:ayah, hadith collection and number, or the scholar/work cited")
    quote: str = Field(description="Verbatim span copied from the document. Do not paraphrase.")


class CardOut(BaseModel):
    """One document's ruling, as the model reports it.

    `answers_question` comes FIRST on purpose: it is the relevance gate, and the
    model should decide whether the document is on topic before it has spent a
    paragraph arguing that it is. Retrieval's cosine floor only ever saw a title
    and a question; this field is judged against the document body, which is why
    it is the one that catches "Playing with and selling Pokemon cards" surfacing
    on a question about bitcoin.
    """

    answers_question: bool = Field(
        description="True only if this document actually addresses the user's question. "
        "Same broad topic is NOT enough - a fatwa on combining prayers while "
        "travelling does not answer where to place your hands in prayer. False if "
        "the document is merely adjacent, or answers a different question about the "
        "same subject. When genuinely unsure, answer false."
    )
    verdict: Verdict
    one_line: str = Field(description="The ruling in one sentence, under 25 words.")
    summary: str = Field(
        description="A summary of this fatwa for someone who will not open the "
        "original: what it was asked, what it rules, the reasoning it gives, and any "
        "caveat or exception it attaches. 3-5 sentences, plain English, in the "
        "source's own terms. Write 'this answer' or the source's name - never 'the "
        "Hanafi school' or 'the madhhab', which one fatwa cannot speak for."
    )
    evidences: list[Evidence] = Field(default_factory=list, max_length=4)
    conditions: list[str] = Field(
        default_factory=list,
        description="Qualifications the ruling depends on. Empty if the ruling is unconditional.",
    )


class DivergencePosition(BaseModel):
    who: str = Field(description="The attribution string of the card, copied exactly.")
    stance: str = Field(description="That source's stance on this specific point, under 20 words.")


class DivergencePoint(BaseModel):
    point: str = Field(description="The single question the sources answer differently.")
    positions: list[DivergencePosition]


class StandaloneQuery(BaseModel):
    """A follow-up turn rewritten so retrieval can use it on its own."""

    query: str = Field(description="The self-contained question. Under 20 words.")
    is_followup: bool = Field(
        description="True if the rewrite actually needed the earlier turns to make sense."
    )


class ComparisonOut(BaseModel):
    """Where the cards agree and diverge. No overall verdict field, by design."""

    agreement: list[str] = Field(description="Points every source states. Empty list if none.")
    divergence: list[DivergencePoint]
    turns_on: str = Field(
        description="The one factual or interpretive question the disagreement reduces "
        "to - a definition, a hadith's authenticity, the scope of a condition. Written "
        "as a plain explanation of why these answers differ, one sentence. If the "
        "sources agree, say what they agree on and stop."
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
NO_ADJUDICATION = (
    "You are an extraction layer over a corpus of published fatwas. You report what "
    "a source says; you never rule, and you never decide which source is right.\n"
    "Hard rules:\n"
    "1. Use ONLY the document text supplied in this message. If it does not say "
    "something, that something does not go in your answer.\n"
    "2. Every `quote` must be a verbatim span copied character-for-character from "
    "the supplied text. Never reconstruct a verse or hadith from memory - if the "
    "document does not quote it, omit the evidence entirely.\n"
    "3. Never write 'the correct view is', 'the stronger opinion', 'the majority is "
    "right', or any equivalent. You are not adjudicating.\n"
    "4. Preserve conditions. A permissible-if-X ruling reported as plain "
    "'permissible' is a misquotation of a scholar.\n"
    "5. Attribute to the ANSWERING PARTY, never to a school of law. One fatwa tells "
    "you what its author holds; it does not tell you what the Hanafi, Shafi'i, Maliki "
    "or Hanbali school holds, and every madhhab contains internal disagreement. Write "
    "'this answer holds', 'the mufti holds', or the site's name. Never 'the Hanafi "
    "school holds', 'the madhhab's position is', or 'according to the Hanbalis'. If "
    "the document ITSELF reports what a school holds, you may say the document says "
    "so - that is quoting it, not generalising from it."
)

CARD_PROMPT = """Document to extract from - this is the ONLY source you may use.

<answered_by>{who}</answered_by>
<site_leaning>{orientation}</site_leaning>
<title>{title}</title>
<question>{question}</question>
<answer>
{answer}
</answer>

Extract this one document's ruling on the user's question: "{query}"

FIRST decide `answers_question`. This document was returned by a keyword and
vector search, which retrieves things that merely share vocabulary with the
question - being about the same broad subject is not the same as answering it.
If it does not address the question, set answers_question to false; the card is
then discarded and nothing else you write here is shown, so do not stretch the
document to fit.

Then pick the verdict enum that this document's own conclusion matches. Use
"depends" when the document makes the ruling conditional rather than stating one
outcome.

`site_leaning` is the madhhab this SITE generally follows. It is context for you,
not a claim you may make: this is one fatwa, so write about what this answer says
and do not attribute its ruling to the school as a whole.

Then write `summary`. The reader will most likely not open the original document,
so this is what they will actually know about this fatwa - give them its substance
(what was asked, what it concludes, why, and what it excludes), not a restatement
of `one_line`.
"""

CONDENSE_PROMPT = """Here is a conversation. Rewrite the LAST user message as a question that
stands on its own, so it can be used for search without the earlier turns.

{history}

Last user message: "{question}"

Resolve pronouns and elisions ("what about the Hanafi view?" after a question about mortgages
becomes "What is the Hanafi ruling on conventional mortgages?"). If the last message is already
self-contained, return it unchanged and set is_followup to false. Do not answer it, do not broaden
it, and do not add topics the user did not raise.
"""

COMPARISON_PROMPT = """Below are finished ruling cards on the question: "{query}"

{cards}

Each `who` is ONE mufti or ONE site that answered - not a school of law, even
where the name signals a madhhab. Copy each `who` exactly as given, and never
rewrite it into "the Hanafi school" or similar; these are individual published
answers, and a school's full position is not what we retrieved.

Report where they agree and where they diverge.

You must NOT produce an overall verdict, and there is no field for one. Do not
say which position is stronger, safer, more authoritative, or more widely held.
A reader must finish this able to state each position accurately - not able to
say who won.

`turns_on` explains why these answers differ: name the single underlying question
the disagreement reduces to (a definition, a hadith's authenticity, a condition's
scope) in one plain sentence. If the sources do not actually disagree, say what
they share and stop.
"""


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------
_NON_WORD = re.compile(r"[^a-z0-9]+")


def _norm(s: str) -> str:
    """Lowercase, strip punctuation and whitespace differences.

    A quote and its source differ constantly by a curly apostrophe, a
    transliteration bracket, or a line wrap. Those are not fabrications; a
    strict substring test would flag every one of them and train us to ignore
    the flag.
    """
    return _NON_WORD.sub(" ", s.lower()).strip()


def verify_quotes(card: RulingCard, source_text: str) -> RulingCard:
    """Move every evidence quote that is not in `source_text` to `unverified_quotes`.

    Mutates and returns the card. This is the guardrail that separates "the
    document cites Bukhari 5063" from "the model remembers a hadith". The UI
    renders unverified quotes in a warning block rather than hiding them - a
    silent drop would look like the model behaved.
    """
    haystack = _norm(source_text)
    kept, flagged = [], []
    for ev in card.evidences:
        quote = (ev.get("quote") or "").strip()
        if quote and _norm(quote) in haystack:
            kept.append(ev)
        else:
            flagged.append(quote or "(empty quote)")
    card.evidences = kept
    card.unverified_quotes = flagged
    return card


# ---------------------------------------------------------------------------
# The API call
# ---------------------------------------------------------------------------
def _parse(
    prompt: str,
    schema: type[BaseModel],
    model: str = LLM_MODEL,
    system: str = NO_ADJUDICATION,
) -> BaseModel:
    """One structured call. Returns a validated instance of `schema`.

    `messages.parse` constrains decoding to the JSON schema and validates the
    result, so there is no brittle regex-the-JSON-out-of-prose step and no
    retry loop for malformed output.
    """
    try:
        resp = client().messages.parse(
            model=model,
            max_tokens=LLM_MAX_TOKENS,
            **sampling_args(model),
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        return resp.parsed_output
    except ValidationError as e:
        # Almost always truncation: the model thinks first and emits JSON second,
        # so hitting LLM_MAX_TOKENS cuts the JSON mid-string. Say that, rather
        # than letting a raw "EOF while parsing" surface.
        raise RuntimeError(
            f"{schema.__name__} did not parse - most likely the response hit "
            f"LLM_MAX_TOKENS ({LLM_MAX_TOKENS}); raise it in config.py. {e}"
        ) from e
    except anthropic.NotFoundError as e:
        raise RuntimeError(f"model {model!r} not available to this key: {e}") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"rate limited - back off and retry: {e}") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"API returned {e.status_code}: {e}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"could not reach the API: {e}") from e


# ---------------------------------------------------------------------------
# Path 1 - one structured call per retrieved document.
# ---------------------------------------------------------------------------
def attribution_for(doc: Doc) -> str:
    """Who actually answered: a named mufti or issuing body, plus where it is published.

    NOT the madhhab. `doc.orientation` says which school a site generally follows,
    and putting that on the card header - "Hanbali school - ..." - turns one page
    into the settled position of a legal tradition that argues with itself. The
    leaning still renders, next to the link, as the site's.

    `scholar` is a person on askimam and islamqa.info and the issuing darul-ifta
    on islamqa.org, which is the right granularity either way: it is the party a
    reader would go back to. Falls back to the site when a record has none.
    """
    who = (doc.scholar or "").strip()
    if not who or who.lower() in doc.source_label.lower():
        return doc.source_label
    return f"{who}, {doc.source_label}"


def source_card(doc: Doc, query: str) -> RulingCard | None:
    """One card, scoped to one document, with every quote verified against it.

    Returns None when the model reports the document does not answer the
    question - the second half of the relevance filter that starts with
    RELEVANCE_THRESHOLD in search.py. Free, because the call already had to
    happen; accurate, because unlike the cosine it has read the whole document.
    """
    out: CardOut = _parse(
        CARD_PROMPT.format(
            who=attribution_for(doc),
            orientation=doc.orientation,
            title=doc.title,
            question=doc.question,
            answer=doc.llm_context(),      # truncated to MAX_ANSWER_CHARS_FOR_LLM
            query=query,
        ),
        CardOut,
    )
    if not out.answers_question:
        return None
    card = RulingCard(
        doc_id=doc.id,
        attribution=attribution_for(doc),
        verdict=out.verdict,
        one_line=out.one_line,
        summary=out.summary,
        evidences=[e.model_dump() for e in out.evidences],
        conditions=out.conditions,
    )
    # Verify against the FULL answer, not llm_context() - a quote from the tail
    # of a truncated document is real, and flagging it would be a false alarm.
    return verify_quotes(card, f"{doc.title}\n{doc.question}\n{doc.answer}")


# ---------------------------------------------------------------------------
# Path 2 - the comparison layer.
# ---------------------------------------------------------------------------
def compare(cards: list[RulingCard], query: str) -> Comparison | None:
    """Agreement and divergence across the finished cards. No overall verdict."""
    if len(cards) < 2:
        return None      # nothing to compare; one card is already the answer

    rendered = "\n\n".join(
        f"<card who=\"{c.attribution}\">\n"
        f"verdict: {c.verdict}\n"
        f"{c.one_line}\n{c.summary}"
        + ("\nconditions: " + "; ".join(c.conditions) if c.conditions else "")
        + "\n</card>"
        for c in cards
    )
    out: ComparisonOut = _parse(
        COMPARISON_PROMPT.format(query=query, cards=rendered), ComparisonOut
    )
    return Comparison(
        agreement=out.agreement,
        divergence=[d.model_dump() for d in out.divergence],
        turns_on=out.turns_on,
    )


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Conversation support
# ---------------------------------------------------------------------------
CONDENSE_SYSTEM = (
    "You rewrite the last message of a conversation into a standalone search query. "
    "You never answer the question and you never add information that is not in the "
    "conversation - you are resolving references, nothing more."
)


def condense_query(history: list[tuple[str, str]], question: str) -> tuple[str, bool]:
    """Rewrite a follow-up into a self-contained query. Returns (query, was_followup).

    "What about the Hanafi view?" is meaningless to a vector index - it embeds to
    nothing useful and BM25 sees three stopwords. Chat UIs live on exactly that
    kind of turn, so the alternative to rewriting is a conversation that silently
    stops retrieving after the first message.

    `history` is [(user_text, assistant_summary), ...], oldest first. Only the
    last few turns are used: a chat about wudu that drifts to inheritance should
    not have inheritance queries contaminated by wudu.
    """
    if not history:
        return question, False

    rendered = "\n".join(
        f"User: {u}\nAssistant: {a}" for u, a in history[-3:]
    )
    out: StandaloneQuery = _parse(
        CONDENSE_PROMPT.format(history=rendered, question=question),
        StandaloneQuery,
        system=CONDENSE_SYSTEM,
    )
    return out.query.strip() or question, out.is_followup


def disambiguate(cards: list[RulingCard]) -> list[RulingCard]:
    """Make every `attribution` unique.

    MAX_PER_SOURCE allows two hits from one site, and two fatwas from the same
    site can genuinely differ (one on interest-bearing mortgages, one on an
    interest-free variant). Left alone, both cards read "Mufti Ebrahim Desai,
    AskImam" and the comparison then lists that name twice with contradictory
    stances, which looks like the mufti contradicting himself. Suffixing the fatwa
    number makes them two documents again.
    """
    counts: dict[str, int] = {}
    for c in cards:
        counts[c.attribution] = counts.get(c.attribution, 0) + 1
    for c in cards:
        if counts[c.attribution] > 1:
            c.attribution = f"{c.attribution} #{c.doc_id.split(':', 1)[-1]}"
    return cards


ABSTAIN_MESSAGE = (
    "I don't have a sourced fatwa on this. Nothing in the corpus is close enough "
    "to the question to answer from, and answering anyway would mean inventing one."
)

# The other way to end up with nothing: documents were retrieved and read, and
# none of them turned out to be about the question. Worth saying differently -
# "nothing matched" and "what matched was off topic" are different facts about
# the corpus, and only the second one means retrieval fired and was overruled.
OFF_TOPIC_MESSAGE = (
    "I don't have a sourced fatwa on this. The search returned {n} document(s), but "
    "on reading them none actually answers this question - they only share subject "
    "matter with it."
)


@dataclass
class Answer:
    """Everything app.py renders."""

    query: str
    state: str                      # 'abstain' | 'answered'
    cards: list[RulingCard] = field(default_factory=list)
    comparison: Comparison | None = None
    hits: list[Doc] = field(default_factory=list)
    top_score: float = 0.0
    message: str = ""
    detail: str = ""                # the caption under `message`, when abstaining
    llm_calls: int = 0
    discarded: int = 0              # read, then judged off topic. See source_card.

    @property
    def unverified(self) -> list[str]:
        return [q for c in self.cards for q in c.unverified_quotes]

    @property
    def summary(self) -> str:
        """One line of conversation memory, fed back into `condense_query`.

        Deliberately just attributions and verdicts. Feeding the full cards back
        would let an earlier turn's prose leak into a later turn's retrieval, and
        the whole design rests on each card being scoped to one document.
        """
        if self.state == "abstain":
            return "No sourced fatwa in the corpus; declined to answer."
        return "; ".join(f"{c.attribution}: {c.verdict}" for c in self.cards[:4])


def answer(
    query: str,
    idx,
    max_source_cards: int = 3,
    on_progress: Callable[[str], None] | None = None,
) -> Answer:
    """Retrieve, then generate. The one function app.py and the notebook call.

    `max_source_cards` trims how many of the retrieved hits get their own LLM
    call. Retrieval returns TOP_K_ISLAMQA (5); cards below rank 3 rarely add a
    distinct position and each one is a round trip, so the default trades the
    tail for latency. Pass 5 to card every hit.

    `on_progress` receives a short status string at each stage. The whole thing
    takes 15-25s, which is a long time to show a chat user a bare spinner - the
    UI uses this to say which stage it is in.
    """
    say = on_progress or (lambda _s: None)

    say("Searching the corpus...")
    r = idx.retrieve(query)

    if r["abstain"]:
        return Answer(
            query=query, state="abstain", hits=[], top_score=r["top_score"],
            message=ABSTAIN_MESSAGE,
            detail=f"Best match scored {r['top_score']:.2f}, below the "
                   f"{ABSTAIN_THRESHOLD:.2f} threshold. Try rephrasing, or ask about "
                   "prayer, purity, marriage, or finance.",
        )

    cards: list[RulingCard] = []
    calls = 0

    hits = r["hits"][:max_source_cards]
    discarded = 0
    if hits:
        say(f"Reading {len(hits)} candidate sources...")
        # Independent calls over independent documents - the only reason the
        # generation half is not 5x slower than the retrieval half.
        with cf.ThreadPoolExecutor(max_workers=len(hits)) as pool:
            built = list(pool.map(lambda d: source_card(d, query), hits))
        calls += len(hits)
        # None = the model read it and said it does not answer the question.
        kept = [c for c in built if c is not None]
        discarded = len(built) - len(kept)
        cards += kept

    # Everything retrieved was read and rejected. Say so rather than rendering an
    # answer with no cards in it, which reads as a broken page.
    if not cards:
        return Answer(
            query=query, state="abstain", hits=r["hits"], top_score=r["top_score"],
            message=OFF_TOPIC_MESSAGE.format(n=len(hits)),
            detail="Nothing was fabricated to fill the gap - the retrieved documents "
                   "are listed below if you want to judge them yourself.",
            llm_calls=calls, discarded=discarded,
        )

    disambiguate(cards)
    say("Comparing what they agree and differ on...")
    comparison = compare(cards, query)
    if comparison is not None:
        calls += 1

    # Only the documents that survived belong in the hit list - the UI reads it
    # back to resolve each card's source link, and listing the rejects there
    # would put documents on the page that we just decided were off topic.
    kept_ids = {c.doc_id for c in cards}
    return Answer(
        query=query,
        state="answered",
        cards=cards,
        comparison=comparison,
        hits=[h for h in r["hits"] if h.id in kept_ids],
        top_score=r["top_score"],
        llm_calls=calls,
        discarded=discarded,
    )


# ---------------------------------------------------------------------------
def render(a: Answer) -> str:
    """Plain-text rendering, for the CLI and the notebook. app.py renders cards."""
    L = [
        f"=== {a.query}",
        f"    [{a.state}] top={a.top_score:.3f}  llm_calls={a.llm_calls}"
        f"  discarded={a.discarded}",
    ]
    if a.state == "abstain":
        L.append(f"    {a.message}")
        return "\n".join(L)

    for c in a.cards:
        L.append(f"\n  -- {c.attribution}  [{c.verdict}]  ({c.doc_id})")
        L.append(f"     {c.one_line}")
        L.append(f"     {c.summary}")
        for cond in c.conditions:
            L.append(f"     * {cond}")
        for e in c.evidences:
            L.append(f"     [{e['type']}] {e['ref']}: \"{e['quote'][:90]}\"")
        for q in c.unverified_quotes:
            L.append(f"     [UNVERIFIED - not found in source] \"{q[:90]}\"")

    if a.comparison:
        L.append("\n  -- comparison (no overall verdict, by design)")
        for x in a.comparison.agreement:
            L.append(f"     agree: {x}")
        for d in a.comparison.divergence:
            L.append(f"     diverge: {d['point']}")
            for p in d["positions"]:
                L.append(f"        {p['who']}: {p['stance']}")
        L.append(f"     turns on: {a.comparison.turns_on}")
    return "\n".join(L)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from search import Index

    from data.raw.demo_queries import QUERIES

    index = Index()
    for q in sys.argv[1:] or QUERIES[:2]:
        print(render(answer(q, index)), "\n")
