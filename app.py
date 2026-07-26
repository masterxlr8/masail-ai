"""Streamlit chat UI over the fatwa corpus.

    streamlit run app.py

WHAT THE UI IS FOR
------------------
The product rule is that we never merge sources into one synthesised ruling, so
the interface cannot look like a chatbot that answers. Every assistant turn is a
**stack of attributed cards** plus an explicit comparison - one card per fatwa,
headed by the mufti or site that issued it, each linking back to the original.

That constraint is what most of the layout below is doing:

  one card per     never a paragraph blending three sources.
  fatwa
  named author     the header is the mufti or site, never "the Hanafi school" -
                   one fatwa is not a madhhab's settled position.
  source link      no uncited claim renders; every card footer links out.
  unverified       quotes that failed the substring check are shown in a warning,
  quotes           not hidden.

READING ORDER
-------------
cards -> comparison -> retrieved documents. The comparison used to lead, on the
theory that a cross-source read orients everything under it. It does not: it is
written entirely in terms of who said what, so leading with it asks the reader to
follow "Hanbalidisciples, IslamQA.org says X, Thehanbalimadhhab, IslamQA.org says
Y" before either name has appeared on the page. Sources first, then the read
across them, then the audit trail - each section only refers back to something
already shown.

The comparison is still emphatically NOT a verdict - `Comparison` has no field
for one - and moving it below the cards makes that harder to misread, not easier.

ONE SOURCES SECTION
-------------------
There used to be two - "The four schools" above "Contemporary sources" - back
when a FiqhQA record supplied all four madhhabs' positions. That produced pages
carrying both "Hanbali school - X" and "IslamQA.org (Hanbali) #154072 - X", which
reads as two different kinds of authority when it was only two different
databases. The FiqhQA records are gone (see config.SOURCE_META) and every card
now names the mufti or site that wrote it, so they all belong in one list.

CONVERSATION
------------
Follow-ups ("what about the Hanafi view?") are rewritten into standalone queries
before retrieval - see `generate.condense_query`. Without that, retrieval
silently degrades to noise after the first message, which is the failure mode
every naive chat-RAG has.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from data.raw.config import LLM_MODEL, RELEVANCE_THRESHOLD
from generate import answer, condense_query

load_dotenv()

st.set_page_config(
    page_title="Masail AI",
    page_icon=":material/menu_book:",
    layout="centered",
)

# Verdict -> (badge colour, icon). Green/red only where the source is
# unambiguous; `depends` is violet rather than a warning colour because a
# conditional ruling is a legitimate answer, not a degraded one.
VERDICT = {
    "permissible":   ("green",  ":material/check_circle:"),
    "recommended":   ("green",  ":material/thumb_up:"),
    "obligatory":    ("blue",   ":material/priority_high:"),
    "disliked":      ("orange", ":material/thumb_down:"),
    "impermissible": ("red",    ":material/block:"),
    "depends":       ("violet", ":material/alt_route:"),
}

# The four schools of law. The corpus also carries `Salafi`, which is a
# methodology rather than a madhhab, so it is worded differently - see
# `leaning_line`.
MADHHABS = {"Hanafi", "Maliki", "Shafi'i", "Hanbali"}

# Label -> query. The label carries an icon so the empty state reads as a set of
# suggestions rather than a wall of sentences.
EXAMPLES = {
    ":material/savings: Conventional mortgages": "Is a conventional mortgage permissible?",
    ":material/set_meal: Seafood other than fish": "Is it permissible to eat seafood other than fish?",
    ":material/currency_bitcoin: Buying and selling bitcoin": "What is the ruling on buying and selling bitcoin?",
    ":material/health_and_safety: Life insurance": "Is life insurance allowed in Islam?",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the corpus and vectors (first run only)...")
def load_index():
    """Corpus + the vector file + the BM25 index, built once per server.

    The warm-up call matters: the embedding model loads lazily on first use and
    takes ~15s. Paying it here means it lands in the cache-resource spinner
    rather than inside the user's first question.
    """
    from search import Index

    idx = Index()
    idx.embed_query("warm up the embedding model")
    return idx


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def leaning_line(doc) -> str:
    """Where the fatwa is published and which school that publisher follows.

    Phrased as the SITE's madhhab, never the ruling's, because that is all the
    corpus records: an islamqaorg fatwa filed under Hanbali is one Hanbali
    scholar answering on a site organised by school, not the Hanbali position.
    Saying "IslamQA.org, which follows the Hanbali madhhab" keeps the school
    visible - the reader wants it, and it is genuinely informative - without
    letting one answer speak for a school. The card header names the actual
    author for the same reason; see generate.attribution_for.

    Salafi is the one orientation in the corpus that is a methodology rather
    than a madhhab, so it does not take the "madhhab" wording.
    """
    o = (doc.orientation or "").strip()
    if o in MADHHABS:
        return f"{doc.source_label}, which follows the {o} madhhab"
    if o and o != "Unspecified":
        return f"{doc.source_label}, {o} in orientation"
    return doc.source_label


def leaning_tag(doc) -> str:
    """Just the school, for places too tight for a sentence. '' if unrecorded."""
    o = (doc.orientation or "").strip()
    if o in MADHHABS:
        return f"{o} madhhab"
    return o if o and o != "Unspecified" else ""


def ruling_card(card, doc) -> None:
    """One document's position, attributed to whoever wrote it."""
    colour, icon = VERDICT.get(card.verdict, ("gray", ":material/help:"))
    with st.container(border=True):
        # Attribution stretches, badge sits at the content width, so the verdict
        # lands hard right and a stack of cards is scannable down one column.
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**{card.attribution}**")
            st.badge(card.verdict, icon=icon, color=colour)

        # Publisher and school directly under the name, before any content: it
        # is how a reader places an answer they have never heard of.
        if doc is not None:
            st.caption(f":material/account_balance: Published on {leaning_line(doc)}")

        # The ruling in bold, then the summary of the fatwa under it. The summary
        # is the point of the card: most readers will not open a 3,000-word fatwa,
        # so this has to leave them knowing what it actually said.
        if card.one_line.strip():
            st.markdown(f"**{card.one_line.strip()}**")
        if card.summary.strip():
            st.markdown(card.summary.strip())

        if card.conditions:
            st.markdown("**Conditions**")
            st.markdown("\n".join(f"- {c}" for c in card.conditions))

        if card.evidences:
            with st.expander(
                f"Evidence ({len(card.evidences)}), verified against the source",
                icon=":material/verified:",
                type="compact",
            ):
                for e in card.evidences:
                    st.caption(f"**{e['type'].title()}** · {e['ref']}")
                    st.markdown(f"> {e['quote']}")

        if card.unverified_quotes:
            st.warning(
                "These quotes could not be found in the source document and may be "
                "fabricated:\n\n"
                + "\n".join(f"- {q}" for q in card.unverified_quotes),
                icon=":material/report:",
            )

        if doc is not None:
            st.caption(f":material/open_in_new: [Read the original]({doc.url})")


def comparison_block(cmp_, leanings: dict[str, str]) -> None:
    """The cross-source read, under the cards it reads across.

    `leanings` maps a card's attribution to its site's school, so a position in
    here carries the same label the card above it does. Without it this section
    is a list of site names with no way to tell a Hanbali answer from a Maliki
    one - which is exactly what it looked like before, since the model copies
    `who` verbatim from the card and the card is where the school was shown.

    No caption under the heading. It used to carry the "no overall verdict" note,
    which the sidebar About already makes, and repeating it on every answer put a
    disclaimer between the reader and the content they asked for.
    """
    if cmp_ is None:
        return

    def who(name: str) -> str:
        tag = leanings.get(name)
        return f"**{name}** ({tag})" if tag else f"**{name}**"

    st.space("small")
    st.subheader("How they compare", anchor=False)
    with st.container(border=True):
        if cmp_.agreement:
            st.markdown(":material/handshake: **They agree that**")
            st.markdown("\n".join(f"- {a}" for a in cmp_.agreement))
        for d in cmp_.divergence:
            st.markdown(f":material/call_split: **They differ on:** {d['point']}")
            st.markdown(
                "\n".join(f"- {who(p['who'])} — {p['stance']}" for p in d["positions"])
            )
        if cmp_.turns_on:
            # The field holds the crux of the disagreement, or - where the sources
            # agree - what they share, so the label has to follow the content.
            label = "Why they differ" if cmp_.divergence else "What it comes down to"
            st.markdown(f":material/key: **{label}** — {cmp_.turns_on}")


def documents_block(a) -> None:
    """The audit trail, last. Only documents that survived both relevance gates."""
    if not a.hits:
        return
    st.space("small")
    with st.expander(
        f"Retrieved documents ({len(a.hits)})",
        icon=":material/manage_search:",
        type="compact",
    ):
        for n, h in enumerate(a.hits, 1):
            st.markdown(
                f"{n}. [{h.title or h.id}]({h.url}) — {h.source_label} "
                f"({h.orientation}) · `{h.score:.3f}`"
            )
        # Ordering is the fused rank; the number is the cosine, which is the
        # readable one but not the one that decided the order. Say so, or the
        # list looks mis-sorted.
        note = (
            f"Ranked by hybrid keyword + vector fusion; the score shown is cosine "
            f"similarity. Below {RELEVANCE_THRESHOLD:.2f} a document is dropped as "
            f"unrelated."
        )
        if a.discarded:
            note += (
                f" {a.discarded} further document(s) were read and discarded for not "
                "answering the question."
            )
        st.caption(note)
        st.caption(f"Top score {a.top_score:.3f} · {a.llm_calls} LLM calls")


def render_answer(a) -> None:
    """The cards, then the read across them, then the documents behind both."""
    if a.state == "abstain":
        with st.container(border=True):
            st.markdown(":material/search_off: **No sourced answer**")
            st.markdown(a.message)
            if a.detail:
                st.caption(a.detail)
        documents_block(a)
        return

    # 1. The sources themselves, in one section - see ONE SOURCES SECTION in the
    #    module docstring. Cards hold a doc_id, not a Doc; look the document back
    #    up for the publisher, school and link.
    by_id = {d.id: d for d in a.hits}
    st.subheader(f"What each source says ({len(a.cards)})", anchor=False)
    for c in a.cards:
        ruling_card(c, by_id.get(c.doc_id))

    # 2. The cross-source read, now that every name in it has been introduced.
    #    The model copies `who` verbatim from the card attribution, so keying the
    #    school lookup on that string is exact rather than fuzzy.
    leanings = {
        c.attribution: leaning_tag(by_id[c.doc_id])
        for c in a.cards
        if c.doc_id in by_id
    }
    comparison_block(a.comparison, leanings)

    # 3. The audit trail.
    documents_block(a)


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
st.title("Masail AI", anchor=False)
st.caption(
    "Ask a question. Get each scholar's and site's published position, side by side."
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error(
        "No `ANTHROPIC_API_KEY` found. Copy `.env.example` to `.env` and add your key, "
        "or set it in the Streamlit Cloud secrets dashboard.",
        icon=":material/key_off:",
    )
    st.stop()

idx = load_index()
st.session_state.setdefault("turns", [])       # [{question, answer, searched_for}]
st.session_state.setdefault("pending", None)   # question awaiting an answer


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("About", anchor=False)
    st.markdown(
        "A research tool over **published fatwas**. It finds what individual "
        "scholars and sites have actually said on your question and shows each "
        "answer separately, with a link to the original."
    )
    st.markdown(
        "**It does not issue rulings.** There is no single answer at the top of "
        "the page, because a merged verdict would be a position no scholar holds."
    )
    st.markdown(
        "**Every card is one fatwa.** A card is what its named author wrote - not "
        "the position of a madhhab, which one answer cannot settle. The school "
        "shown under each name is the *publishing site's*, so you can place the "
        "answer without reading it as the school's last word."
    )

    if st.button(
        "New conversation",
        icon=":material/add_comment:",
        width="stretch",
        disabled=not st.session_state.turns,
    ):
        st.session_state.turns = []
        st.rerun()

    st.space("small")
    with st.container(border=True):
        st.caption(f":material/database: {len(idx.docs):,} fatwas")
        st.caption(f":material/smart_toy: `{LLM_MODEL}`")

    st.caption(
        "Not a substitute for a qualified scholar. Sources are used under their "
        "own licences for non-commercial research."
    )



# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------
# Read the box first. chat_input is pinned to the bottom of the viewport wherever
# it is called, so calling it up here costs nothing visually and lets the empty
# state below see the new question - otherwise the example chips linger through
# the very turn they started. `submit_mode="disable"` because a turn takes
# 15-25s and a second question fired into it would race the first.
if prompt := st.chat_input(
    "Ask about prayer, purity, marriage, finance...", submit_mode="disable"
):
    st.session_state.pending = prompt

# Empty state: give the user something to click rather than a blank box.
if not st.session_state.turns and st.session_state.pending is None:
    st.caption("Try one of these")
    picked = st.pills(
        "Example questions", list(EXAMPLES), label_visibility="collapsed", key="eg"
    )
    if picked:
        st.session_state.pending = EXAMPLES[picked]
        st.rerun()

# Replay the conversation so far.
for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("searched_for"):
            st.caption(f":material/search: Searched for: *{turn['searched_for']}*")
        render_answer(turn["answer"])

# The new turn. Split from the replay above so the spinner appears in the right
# place and the finished turn re-renders identically on the next run.
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=False)
        try:
            history = [(t["question"], t["answer"].summary) for t in st.session_state.turns]
            search_query, is_followup = question, False
            if history:
                status.update(label="Reading the conversation so far...")
                search_query, is_followup = condense_query(history, question)

            result = answer(
                search_query, idx,
                on_progress=lambda s: status.update(label=s),
            )
            status.update(label="Done", state="complete")
        except RuntimeError as e:
            status.update(label="Failed", state="error")
            st.error(str(e), icon=":material/error:")
            st.stop()

        searched_for = search_query if is_followup else None
        if searched_for:
            st.caption(f":material/search: Searched for: *{searched_for}*")
        render_answer(result)

    st.session_state.turns.append(
        {"question": question, "answer": result, "searched_for": searched_for}
    )


# ---------------------------------------------------------------------------
# Focus rings
# ---------------------------------------------------------------------------
# The only injected code in the app, and only because clicking anything left a
# highlight parked on it. Two earlier attempts failed for instructive reasons.
#
# CSS could never have worked: every ring Streamlit draws is already gated on
# :focus-visible or React Aria's [data-focus-visible], so a rule qualified with
# :not(:focus-visible) can only match once the ring is gone. The ring persists
# because the focused NODE persists - Streamlit leaves the previous run's
# elements on screen, greyed as stale, until their replacements arrive, and a
# stale node is still document.activeElement. Restyling focus cannot fix an
# element that is genuinely focused; dropping the focus can.
#
# Blurring on click then failed too, on the sidebar toggle of all things. That
# button UNMOUNTS when pressed - collapse destroys stSidebarCollapseButton and
# mounts stExpandSidebarButton in the toolbar instead - and React puts focus on
# the replacement. So the ring ends up on an element that was never clicked,
# which a click handler by definition cannot see.
#
# Hence a modality shim rather than a click handler: remember whether the last
# input was pointer or keyboard, and on ANY focus arriving during pointer use -
# clicked, restored, or moved by React - drop it. Keyboard users are untouched
# and keep the ring they navigate by, which is the only thing a focus ring is
# for. Fields are excluded outright, as are the roles whose whole interaction
# model is focus-driven: blurring a menu or listbox would break it.
#
# Last in the script, not in the sidebar. st.html occupies a real layout slot
# even when it renders nothing, so in the sidebar it opened a stray gap - and,
# worse, a collapsed sidebar is the one state where the toggle ring is most
# visible and the sidebar's own elements are least certain to be mounted.
st.html(
    """
    <script>
      if (!window.__fatwaFinderFocusShim) {
        window.__fatwaFinderFocusShim = true;
        const NAV_KEYS = new Set([
          'Tab', 'Enter', ' ', 'Escape', 'Home', 'End',
          'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
        ]);
        const KEEP_FOCUS =
          'input, textarea, select, [contenteditable], ' +
          '[role="dialog"], [role="menu"], [role="listbox"], [role="combobox"]';
        let byKeyboard = false;
        addEventListener('keydown', (e) => {
          if (NAV_KEYS.has(e.key)) byKeyboard = true;
        }, true);
        addEventListener('pointerdown', () => { byKeyboard = false; }, true);
        addEventListener('focusin', (e) => {
          const el = e.target;
          if (byKeyboard || !(el instanceof HTMLElement)) return;
          if (el.closest(KEEP_FOCUS)) return;
          requestAnimationFrame(() => {
            if (document.activeElement === el) el.blur();
          });
        }, true);
      }
    </script>
    """,
    unsafe_allow_javascript=True,
)
