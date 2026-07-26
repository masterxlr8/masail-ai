"""Streamlit chat UI over the fatwa corpus.

    streamlit run app.py

WHAT THE UI IS FOR
------------------
The product rule is that we never merge sources into one synthesised ruling, so
the interface cannot look like a chatbot that answers. Every assistant turn is a
**stack of attributed cards** plus an explicit comparison - one card per school
or per source, each linking back to the fatwa it came from.

That constraint is what most of the layout below is doing:

  coverage badge   the four-school panel's ABSENCE is a stated state, not a
                   silent gap. A judge who types their own question and misses
                   the 121 multi-school records must be told so.
  one card per     never a paragraph blending three sources.
  source
  source link      no uncited claim renders; every card footer links out.
  unverified       quotes that failed the substring check are shown in a warning,
  quotes           not hidden.

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

from data.raw.config import ABSTAIN_THRESHOLD, FIQH_THRESHOLD, LLM_MODEL
from generate import answer, condense_query

load_dotenv()

st.set_page_config(page_title="Fatwa Finder", page_icon="📖", layout="centered")

# Verdict -> Streamlit colour. Green/red only where the source is unambiguous;
# `depends` is violet rather than a warning colour because a conditional ruling
# is a legitimate answer, not a degraded one.
VERDICT_COLOUR = {
    "permissible": "green",
    "recommended": "green",
    "obligatory": "blue",
    "disliked": "orange",
    "impermissible": "red",
    "depends": "violet",
}

EXAMPLES = [
    "Does touching a woman break wudu?",
    "Is a conventional mortgage permissible?",
    "Where do you place your hands in prayer?",
    "What is the ruling on buying and selling bitcoin?",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the corpus and vectors (first run only)...")
def load_index():
    """Corpus + both vector files + the BM25 index, built once per server.

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
def badge(state: str) -> None:
    """Coverage, stated explicitly. Silence here reads as a bug."""
    if state == "four_school":
        st.markdown(":green-background[Four-school comparison available]")
    elif state == "single_source":
        st.markdown(
            ":orange-background[Single-source answer] "
            "&nbsp;no four-school record matched this question"
        )
    else:
        st.markdown(":red-background[No sourced answer]")


def ruling_card(card, doc) -> None:
    """One school's or one source's position. Never a merged verdict."""
    colour = VERDICT_COLOUR.get(card.verdict, "gray")
    with st.container(border=True):
        st.markdown(
            f"**{card.attribution}** &nbsp; :{colour}-background[{card.verdict}]"
        )

        # School cards carry the source's own prose, so `one_line` is its first
        # sentence and printing both would read as a stutter.
        body, lead = card.reasoning.strip(), card.one_line.strip()
        if lead and not body.startswith(lead[:50]):
            st.markdown(lead)
        st.markdown(body)

        if card.conditions:
            st.markdown("**Conditions**")
            for c in card.conditions:
                st.markdown(f"- {c}")

        if card.evidences:
            with st.expander(f"Evidence ({len(card.evidences)}) - verified against the source"):
                for e in card.evidences:
                    st.markdown(f"**{e['type']}** &middot; {e['ref']}")
                    st.markdown(f"> {e['quote']}")

        if card.unverified_quotes:
            st.warning(
                "These quotes could not be found in the source document and may be "
                "fabricated:\n\n"
                + "\n".join(f"- {q}" for q in card.unverified_quotes)
            )

        if doc is not None:
            st.caption(f"[{doc.source_label}]({doc.url}) &middot; {doc.orientation}")


def comparison_block(cmp_) -> None:
    if cmp_ is None:
        return
    st.markdown("#### How they compare")
    st.caption(
        "Where the sources agree and where they diverge. There is deliberately no "
        "overall verdict - this tool reports positions, it does not rule between them."
    )
    if cmp_.agreement:
        st.markdown("**They agree that**")
        for a in cmp_.agreement:
            st.markdown(f"- {a}")
    for d in cmp_.divergence:
        st.markdown(f"**They differ on:** {d['point']}")
        for p in d["positions"]:
            st.markdown(f"- *{p['who']}* — {p['stance']}")
    if cmp_.turns_on:
        st.info(f"**What it turns on:** {cmp_.turns_on}")


def render_answer(a) -> None:
    badge(a.state)

    if a.state == "abstain":
        st.markdown(a.message)
        st.caption(
            f"Best match scored {a.top_score:.2f}, below the {ABSTAIN_THRESHOLD:.2f} "
            "threshold. Try rephrasing, or ask about prayer, purity, marriage, or finance."
        )
        return

    # Cards hold a doc_id, not a Doc - look the document back up for the link.
    by_id = {d.id: d for d in a.hits}
    if a.panel_doc is not None:
        by_id[a.panel_doc.id] = a.panel_doc

    # Two groups, because they are not the same kind of claim: school cards are
    # the source's own prose copied through, contemporary cards are extracted.
    panel_cards = [c for c in a.cards if c.attribution.endswith("school")]
    source_cards = [c for c in a.cards if not c.attribution.endswith("school")]

    if panel_cards:
        st.markdown("#### The four schools")
        st.caption(
            "Copied verbatim from the source record - no model rewrote this text. "
            f"Matched `{a.panel_doc.id}`."
        )
        for c in panel_cards:
            ruling_card(c, by_id.get(c.doc_id))

    if source_cards:
        st.markdown("#### Contemporary sources")
        for c in source_cards:
            ruling_card(c, by_id.get(c.doc_id))

    comparison_block(a.comparison)

    with st.expander(f"Retrieved documents ({len(a.hits)})"):
        for h in a.hits:
            st.markdown(
                f"`{h.score:.3f}` [{h.title or h.id}]({h.url}) — "
                f"{h.source_label} ({h.orientation})"
            )
        st.caption(
            f"Top score {a.top_score:.3f} &middot; {a.llm_calls} LLM calls &middot; "
            f"panel threshold {FIQH_THRESHOLD:.2f}"
        )


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
st.title("📖 Fatwa Finder")
st.caption("Ask a question. Get every school's and source's position, side by side.")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error(
        "No `ANTHROPIC_API_KEY` found. Copy `.env.example` to `.env` and add your key, "
        "or set it in the Streamlit Cloud secrets dashboard."
    )
    st.stop()

idx = load_index()
st.session_state.setdefault("turns", [])       # [{question, answer, searched_for}]
st.session_state.setdefault("pending", None)   # question awaiting an answer


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "A research tool over **published fatwas**. It finds what scholars and "
        "schools have actually said on your question and shows each position "
        "separately, with a link to the original."
    )
    st.markdown(
        "**It does not issue rulings.** There is no single answer at the top of "
        "the page, because a merged verdict would be a position no scholar holds."
    )
    st.divider()
    if st.button("New conversation", use_container_width=True):
        st.session_state.turns = []
        st.rerun()
    st.divider()
    st.caption(f"Model `{LLM_MODEL}`")
    st.caption(f"{len(idx.single):,} fatwas &middot; {len(idx.multi)} four-school records")
    st.caption(
        "Not a substitute for a qualified scholar. Sources are used under their "
        "own licences for non-commercial research."
    )


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------
# Read the box first. chat_input is pinned to the bottom of the viewport wherever
# it is called, so calling it up here costs nothing visually and lets the empty
# state below see the new question - otherwise the example buttons linger through
# the very turn they started.
if prompt := st.chat_input("Ask about prayer, purity, marriage, finance..."):
    st.session_state.pending = prompt

# Empty state: give the user something to click rather than a blank box.
if not st.session_state.turns and st.session_state.pending is None:
    st.markdown("**Try one of these**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, use_container_width=True, key=f"eg{i}"):
            st.session_state.pending = q
            st.rerun()

# Replay the conversation so far.
for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("searched_for"):
            st.caption(f"Searched for: *{turn['searched_for']}*")
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
            st.error(str(e))
            st.stop()

        searched_for = search_query if is_followup else None
        if searched_for:
            st.caption(f"Searched for: *{searched_for}*")
        render_answer(result)

    st.session_state.turns.append(
        {"question": question, "answer": result, "searched_for": searched_for}
    )
