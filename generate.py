"""The generation half of the RAG loop: retrieved Docs -> RulingCards -> Comparison.

    from search import Index
    from generate import answer

    result = answer("Does touching a woman break wudu?", Index())
    result.state        # 'abstain' | 'four_school' | 'single_source'
    result.cards        # list[RulingCard] - one per school or per source
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

THE THREE GENERATION PATHS (plan.md section 3)
----------------------------------------------
  four-school panel  4 cards built DIRECTLY from Doc.positions - no LLM writes
                     the text, because the source already states each school's
                     position. One cheap batched call assigns the verdict enum
                     for colour coding. 1 call total.
  single-source hits one structured call per Doc, scoped to that Doc. <=5 calls.
  comparison         one call over the finished cards. 1 call.

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

SchoolKey = Literal["hanafi", "shafii", "maliki", "hanbali"]


class Evidence(BaseModel):
    type: Literal["quran", "hadith", "scholarly"]
    ref: str = Field(description="Surah:ayah, hadith collection and number, or the scholar/work cited")
    quote: str = Field(description="Verbatim span copied from the document. Do not paraphrase.")


class CardOut(BaseModel):
    """One document's ruling, as the model reports it."""

    verdict: Verdict
    one_line: str = Field(description="The ruling in one sentence, under 25 words.")
    reasoning: str = Field(description="Why this source rules that way, 2-4 sentences, in its own terms.")
    evidences: list[Evidence] = Field(default_factory=list, max_length=4)
    conditions: list[str] = Field(
        default_factory=list,
        description="Qualifications the ruling depends on. Empty if the ruling is unconditional.",
    )


class SchoolVerdict(BaseModel):
    school: SchoolKey
    verdict: Verdict


class VerdictBatch(BaseModel):
    """The one cheap call the four-school path makes: enums only, no prose."""

    verdicts: list[SchoolVerdict]


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
        description="The one factual or interpretive question the disagreement reduces to. "
        "One sentence. If the sources agree, say what they agree on and stop."
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
    "'permissible' is a misquotation of a scholar."
)

CARD_PROMPT = """Document to extract from - this is the ONLY source you may use.

<source>{label} ({orientation})</source>
<title>{title}</title>
<question>{question}</question>
<answer>
{answer}
</answer>

Extract this one document's ruling on the user's question: "{query}"

Pick the verdict enum that this document's own conclusion matches. Use "depends"
when the document makes the ruling conditional rather than stating one outcome.
"""

SCHOOL_VERDICT_PROMPT = """Each block below is one school's stated position on: "{query}"

{blocks}

For each school, choose the verdict enum that its own stated position matches.
Do not rewrite the positions and do not say which school is correct - you are
only labelling text that already exists, so the UI can colour-code the cards.
Use "depends" where a school makes the ruling conditional.
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

COMPARISON_PROMPT = """Below are finished ruling cards, each already attributed to one source
or school, on the question: "{query}"

{cards}

Report where they agree and where they diverge.

You must NOT produce an overall verdict, and there is no field for one. Do not
say which position is stronger, safer, more authoritative, or more widely held.
A reader must finish this able to state each position accurately - not able to
say who won.

`turns_on` is the single underlying question the disagreement reduces to (a
definition, a hadith's authenticity, a condition's scope). If the sources do not
actually disagree, say what they share and stop.
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
# Path 1 - four-school panel. The source already wrote the positions.
# ---------------------------------------------------------------------------
def school_cards(doc: Doc, query: str) -> list[RulingCard]:
    """4 cards from `doc.positions`, with ONE batched call for the verdict enums.

    The position text is copied through verbatim - no model rewrites it. That is
    the whole reason the four-school panel is the trustworthy part of the demo:
    the only thing generated is a six-value enum, and if it is wrong the reader
    can see the mismatch against the text sitting right next to it.
    """
    stated = doc.stated_positions
    if not stated:
        return []

    blocks = "\n\n".join(
        f"<school key=\"{p.school}\" name=\"{p.school_label}\">\n{p.text}\n</school>"
        for p in stated
    )
    batch: VerdictBatch = _parse(
        SCHOOL_VERDICT_PROMPT.format(query=query, blocks=blocks), VerdictBatch
    )
    by_school = {v.school: v.verdict for v in batch.verdicts}

    return [
        RulingCard(
            doc_id=doc.id,
            attribution=f"{p.school_label} school",
            verdict=by_school.get(p.school, "depends"),
            one_line=p.text.strip().split("\n")[0][:200],
            reasoning=p.text.strip(),      # verbatim from the source
            evidences=[],
            conditions=[],
            unverified_quotes=[],
        )
        for p in stated
    ]


# ---------------------------------------------------------------------------
# Path 2 - one structured call per single-source hit.
# ---------------------------------------------------------------------------
def source_card(doc: Doc, query: str) -> RulingCard:
    """One card, scoped to one document, with every quote verified against it."""
    out: CardOut = _parse(
        CARD_PROMPT.format(
            label=doc.source_label,
            orientation=doc.orientation,
            title=doc.title,
            question=doc.question,
            answer=doc.llm_context(),      # truncated to MAX_ANSWER_CHARS_FOR_LLM
            query=query,
        ),
        CardOut,
    )
    card = RulingCard(
        doc_id=doc.id,
        attribution=f"{doc.source_label} ({doc.orientation})",
        verdict=out.verdict,
        one_line=out.one_line,
        reasoning=out.reasoning,
        evidences=[e.model_dump() for e in out.evidences],
        conditions=out.conditions,
    )
    # Verify against the FULL answer, not llm_context() - a quote from the tail
    # of a truncated document is real, and flagging it would be a false alarm.
    return verify_quotes(card, f"{doc.title}\n{doc.question}\n{doc.answer}")


# ---------------------------------------------------------------------------
# Path 3 - the comparison layer.
# ---------------------------------------------------------------------------
def compare(cards: list[RulingCard], query: str) -> Comparison | None:
    """Agreement and divergence across the finished cards. No overall verdict."""
    if len(cards) < 2:
        return None      # nothing to compare; one card is already the answer

    rendered = "\n\n".join(
        f"<card who=\"{c.attribution}\">\n"
        f"verdict: {c.verdict}\n"
        f"{c.one_line}\n{c.reasoning}"
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
    interest-free variant). Left alone, both cards read "IslamQA.info (Salafi)"
    and the comparison then lists that name twice with contradictory stances,
    which looks like the source contradicting itself. Suffixing the fatwa number
    makes them two documents again.
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


@dataclass
class Answer:
    """Everything app.py renders. Mirrors the three UI states in plan.md."""

    query: str
    state: str                      # 'abstain' | 'four_school' | 'single_source'
    cards: list[RulingCard] = field(default_factory=list)
    comparison: Comparison | None = None
    panel_doc: Doc | None = None    # the multi_school record, if the panel fired
    hits: list[Doc] = field(default_factory=list)
    top_score: float = 0.0
    message: str = ""
    llm_calls: int = 0

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
        )

    cards: list[RulingCard] = []
    calls = 0

    if r["show_schools"] and r["panel"]:
        say("Reading the four schools' positions...")
        cards += school_cards(r["panel"], query)
        calls += 1

    hits = r["hits"][:max_source_cards]
    if hits:
        say(f"Extracting {len(hits)} source rulings...")
        # Independent calls over independent documents - the only reason the
        # generation half is not 5x slower than the retrieval half.
        with cf.ThreadPoolExecutor(max_workers=len(hits)) as pool:
            cards += list(pool.map(lambda d: source_card(d, query), hits))
        calls += len(hits)

    disambiguate(cards)
    say("Comparing what they agree and differ on...")
    comparison = compare(cards, query)
    if comparison is not None:
        calls += 1

    return Answer(
        query=query,
        state="four_school" if r["show_schools"] else "single_source",
        cards=cards,
        comparison=comparison,
        panel_doc=r["panel"],
        hits=r["hits"],
        top_score=r["top_score"],
        llm_calls=calls,
    )


# ---------------------------------------------------------------------------
def render(a: Answer) -> str:
    """Plain-text rendering, for the CLI and the notebook. app.py renders cards."""
    L = [f"=== {a.query}", f"    [{a.state}] top={a.top_score:.3f}  llm_calls={a.llm_calls}"]
    if a.state == "abstain":
        L.append(f"    {a.message}")
        return "\n".join(L)

    for c in a.cards:
        L.append(f"\n  -- {c.attribution}  [{c.verdict}]  ({c.doc_id})")
        L.append(f"     {c.one_line}")
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
