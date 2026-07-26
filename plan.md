# Fatwa RAG — Final 8-Hour Build Plan

## Context

One-day AI-native hackathon, Islamic theme, team of 2, 8 hours. Empty starting directory
(`c:\Users\DELL\ummah_project`). Python 3.13 present. Both members stronger in Python →
**Python + Streamlit throughout.**

**Product:** given a question, retrieve relevant fatwas, summarise each source's ruling separately,
and surface where the schools agree and diverge.

**Core principle.** Never merge sources into one synthesised ruling — that manufactures a position no
scholar holds, and it's the first thing a knowledgeable judge probes. Instead: **one ruling card per
source/school, attributed and linked, plus a comparison layer that names the divergence without
picking a winner.**

---

## 1. Datasets (both verified against the HF datasets server)

| Dataset | Rows | Role | Status |
|---|---|---|---|
| `kingkaung/english_islamqainfo` | **19,052** | Retrieval backbone | ✅ Verified clean |
| `MBZUAI/FiqhQA` config `english` | **121** | Four-school comparison layer | ✅ Verified clean |

### `kingkaung/english_islamqainfo` — the backbone

English contemporary fatwas. CC-BY-NC-4.0. Salafi-leaning (Munajjid) — **badge every card with its
orientation**; never present it as neutral.

Columns: `global_id`, `title`, `question`, `answer`, `topic`, `page_url`, `article_url`,
`original_topic_id`, `original_id`.

Cleaning quirks already identified — handle these in `ingest.py`:
- `question` begins with a literal `"Question\n\n"` prefix → strip it
- `answer` begins with `"Praise be to Allah."` (no trailing space) → strip it
- Whitespace and line-wrapping are irregular throughout → collapse
- `article_url` is the real per-fatwa link; `page_url` is only the topic page

### `MBZUAI/FiqhQA` — the comparison layer, **not** the corpus

121 rows only. Too small for retrieval, but each row carries all four Sunni positions pre-separated,
which is exactly the headline feature.

Columns: `Category`, `Title_original` (Arabic), `statement_original` (Arabic), `Agreement`,
`title_en`, `statement_en`, `question_en`, `maliki_ans`, `hanafi_ans`, `shafeii_ans`, `hanbali_ans`.

- `Agreement` is `"Agreement"` or `"Disagreement"` — **ground truth to evaluate your comparison layer**
- Missing positions carry filler text like *"The school does not have an opinion on the topic"* →
  detect and drop, don't render as a real position
- The `english` config is genuinely English; only `Title_original`/`statement_original` are Arabic

> **Provenance caveat.** From the paper *"Sacred or Synthetic? Evaluating LLM Reliability and
> Abstention for Religious Questions."* The per-school fields are likely LLM-rendered from the Arabic
> `statement_original` (which reads like al-Mawsūʿa al-Fiqhiyya). Anchor cards on `statement_en` and
> describe the school fields as structured summaries. **Read the dataset card in the first 10
> minutes** before building the pitch around it.

### Rejected, with reasons (have these ready for judges)

- **`IslamQA/askimam`** (Hanafi) — not corrupt. A webdataset of questions + per-sentence BERT
  embeddings; HF auto-conversion fails because the embedding field is `list<list<list<double>>>` on
  most records but `list<null>` where empty → `ArrowTypeError` → `SplitsNotFoundError`. Recoverable
  by bypassing `load_dataset()` and parsing the 304 MB tar.xz, but answer *text* presence is
  unconfirmed. **Deferred — this is the obvious "what's next" answer.**
- **`IslamQA/hadithanswers`** — 2.12 GB single JSON, same text-presence risk.
- **`ubaid-ai/fiqh-qa-bot-data`, `Usmanbhat/fiqh-database`** — both fail to load on HF.

---

## 2. Architecture

```
ingest.py    HF ──► clean ──► corpus_islamqa.json  (19,052)
                          └─► corpus_fiqhqa.json   (121)

embed.py     ──► vectors_islamqa.npy   (19,052 × 512)
             ──► vectors_fiqhqa.npy    (121 × 512)

search.py    query ──┬─► search_islamqa()  cosine + BM25 → RRF → top 5
                     └─► search_fiqhqa()   cosine over 121 → top 1

generate.py  FiqhQA hit ──► 4 cards built directly from *_ans  (NO LLM)
             IslamQA hits ─► 1 structured LLM call each ──► RulingCard
             all cards ────► comparison layer ──► Comparison

app.py       Streamlit
```

### The two-index decision — this is what makes the feature actually fire

Do **not** put FiqhQA's 121 rows into the same index as IslamQA's 19,052. They would almost never win
top-k, and your headline feature would silently never appear.

Instead run **two searches per query**. Searching 121 vectors is free, so this guarantees the
four-school panel fires whenever there's a reasonable match, regardless of what IslamQA returns.

```python
islamqa_hits = search_islamqa(query, k=5)      # hybrid BM25 + cosine + RRF
fiqh_hit     = search_fiqhqa(query, k=1)       # cosine only; 121 vectors
show_schools = fiqh_hit and fiqh_hit.score > FIQH_THRESHOLD
```

### Coverage must be explicit in the UI

The failure mode to design against: a judge types their own question, misses the 121, and the
four-school panel quietly doesn't render. Make coverage a visible state, not an absence:

- `show_schools` true → **"Four-school comparison available"** badge + the comparison panel
- `show_schools` false → **"Single-source answer (Salafi orientation)"** badge

Being explicit about coverage reads as rigour. Silence reads as a bug.

### No topic filtering anywhere

Embed everything — filtering existed only to cut embedding cost, and the full corpus costs ~$0.20.
And there is **no runtime query→topic classifier**: semantic search already handles topic matching,
while a classifier adds latency plus a failure mode where a misclassified query returns nothing.

### Embeddings

- **`text-embedding-3-small`**, batch 100/request, **`dimensions=512`** → 19,173 × 512 float32 ≈
  **39 MB**, comfortable inside Streamlit Cloud's ~1 GB. ~$0.20, a few minutes.
- **No-key fallback:** `BAAI/bge-small-en-v1.5` via `sentence-transformers` (384-dim, free, CPU-fine,
  better retrieval than MiniLM-L6).
- **Embed `title + question`, not the answer.** Queries *are* questions; question↔question similarity
  is far stronger. Answers are stored for display and summarisation only.
- **Add BM25** (`rank_bm25`) fused by reciprocal rank fusion on the IslamQA index. Embeddings are weak
  on rare domain terms (*riba*, *masah*, *mudarabah*, *istihada*); BM25 nails them. ~15 lines.
- `np.save` L2-normalised float32 → retrieval is one matmul, 1–3 ms.

---

## 3. Tech structure

```
ummah_project/
├── requirements.txt
├── .env                    # OPENAI_API_KEY   (gitignored)
├── config.py               # model names, thresholds, paths
├── schema.py               # Doc, RulingCard, Comparison  ← THE CONTRACT
├── demo_queries.py         # the 7 demo queries
├── ingest.py               # HF → corpus_*.json
├── embed.py                # corpus_*.json → vectors_*.npy
├── search.py               # search_islamqa(), search_fiqhqa()
├── generate.py             # cards + comparison + guardrails
├── app.py                  # Streamlit
├── eval.py                 # retrieval + comparison accuracy
└── data/                   # corpus_*.json, vectors_*.npy  (gitignored if large)
```

**requirements.txt**
```
streamlit
datasets
numpy
rank-bm25
openai
python-dotenv
```

### `schema.py` — freeze this in the first 30 minutes

Everything parallelises once this is fixed. Person B builds against fixtures shaped like this.

```python
@dataclass
class Doc:
    id: str
    source: str              # 'islamqa' | 'fiqhqa'
    source_label: str        # "IslamQA.info" | "FiqhQA (al-Mawsūʿa al-Fiqhiyya)"
    orientation: str         # 'Salafi' | 'Four Sunni schools'
    title: str
    question: str
    answer: str | None       # islamqa only
    positions: dict | None   # fiqhqa only: {hanafi, shafii, maliki, hanbali}
    agreement: str | None    # fiqhqa only: 'Agreement' | 'Disagreement'
    category: str
    url: str
    score: float = 0.0

@dataclass
class RulingCard:
    doc_id: str
    attribution: str         # "Hanafi school" | "IslamQA.info (Salafi)"
    verdict: str             # permissible|impermissible|disliked|recommended|obligatory|depends
    one_line: str
    reasoning: str
    evidences: list          # [{type: quran|hadith|scholarly, ref, quote}]
    conditions: list[str]

@dataclass
class Comparison:
    agreement: list[str]
    divergence: list         # [{point, positions: [{who, stance}]}]
    turns_on: str
    # deliberately NO overall verdict field
```

Also frozen at 0:30 — the search signatures:
```python
def search_islamqa(query: str, k: int = 5) -> list[Doc]: ...
def search_fiqhqa(query: str, k: int = 1) -> list[Doc]: ...
```

### Generation paths

| Path | Work | Cost |
|---|---|---|
| FiqhQA hit | Build 4 `RulingCard`s directly from `hanafi_ans`/`shafeii_ans`/`maliki_ans`/`hanbali_ans`. One cheap batched call to assign verdict enums for colour coding | ~0 LLM |
| IslamQA hits | One structured-output call per doc → `RulingCard`, scoped to that single document | 5 calls |
| Comparison | One call over all cards → `Comparison`. Prompt explicitly forbids adjudicating | 1 call |

### Guardrails — build as demo features, not disclaimers

- **Abstain below an RRF-score threshold** → "I don't have a sourced fatwa on this." Demo this
  deliberately; a RAG system that knows when to shut up beats one that always answers.
- **Citation verification** — assert each `evidences[].quote` appears in its parent doc's text
  (normalised substring match); flag unverified quotes in the UI rather than swallowing them.
- **No uncited claim renders.** Every card carries a source badge linking to `article_url`.
- **No-adjudication check** — grep rendered output for "the correct view is" / "the stronger opinion";
  must never appear.
- Footer: a research tool over published fatwas, not a substitute for a qualified scholar.

---

## 4. Demo queries

**First task in hour 1 (2 minutes):** dump all 121 `question_en` values from FiqhQA. The classical
demo queries must be chosen from *within* that list — this determines your demo, so do it before
anything else. Categories seen so far: Prayer, Purity, Marriage.

**Centrepiece:** *"Does touching a woman break wudu?"* — the canonical four-school split (Hanafi: no;
Shafi'i: yes; Maliki and Hanbali: with desire). Genuine divergence, easy to explain. Confirm it's in
the 121; if not, substitute another `Agreement == "Disagreement"` row.

- **Four-school path (verify against the 121):** touching a woman and wudu · saying "Amin" aloud ·
  one Marriage-category question
- **IslamQA path (contemporary):** conventional mortgages · cryptocurrency trading · life insurance
- **Refusal test:** "What are the rules for prayer on Mars?" → must abstain

---

## 5. Hour-by-hour with delegation

**Person A — Corpus & Retrieval:** `ingest.py`, `embed.py`, `search.py`, `eval.py`
**Person B — Generation & UI:** `generate.py`, `app.py`

| Time | Person A | Person B |
|---|---|---|
| **0:00–0:30** | **Both:** repo + venv + `requirements.txt`, API key working end-to-end, **write and freeze `schema.py`**. A dumps FiqhQA's 121 questions and reads the dataset card; both pick the 7 demo queries into `demo_queries.py` | |
| **0:30–2:00** | `ingest.py` — load both datasets, strip `"Question\n\n"` and `"Praise be to Allah."`, collapse whitespace, drop "no opinion" fillers, unify to `Doc` → `corpus_islamqa.json`, `corpus_fiqhqa.json` | Streamlit shell + `generate.py` against **fixtures**: query box, comparison panel, ruling-card component, coverage badge, source links. Clickable UI by 2:00 |
| **2:00–3:15** | `embed.py` (batch 100, `dimensions=512`) → both `.npy`. `search.py`: cosine + BM25 + RRF for IslamQA, plain cosine for FiqhQA | FiqhQA cards direct from `positions` (no LLM). IslamQA extraction as one structured call. Comparison-layer prompt + abstain path |
| **3:15–3:45** | **INTEGRATION — both.** Fixtures out, real `search_*()` in. One query end-to-end, however ugly. Slipping past 4:00 → start cutting. | |
| **3:45–5:15** | Tune `FIQH_THRESHOLD` so the four-school panel fires on all 3 classical queries and stays silent on the Mars query. `eval.py` over all 7 | Prompt quality, citation verification, no-adjudication check, verdict colour coding |
| **5:15–6:15** | Abstain threshold, **pre-warm a cache of the 7 demo queries** (`@st.cache_data`) | UI polish: loading / abstain / empty states, `st.expander` source drawer, orientation + coverage badges |
| **6:15–7:00** | **Both:** push to public GitHub, deploy to Streamlit Community Cloud, API key via the secrets dashboard. Verify on the **real URL**, not localhost | |
| **7:00** | **FEATURE FREEZE** | |
| **7:00–7:45** | **Both:** rehearse 3× aloud, write the 90-second pitch, screenshot every screen as a wifi fallback | |
| **7:45–8:00** | Buffer | |

---

## 6. Risk register and cut lines

| Risk | Mitigation |
|---|---|
| FiqhQA's 121 rows don't cover a judge's ad-hoc question | Explicit coverage badge — the four-school panel's absence is a *stated state*, not a silent gap |
| FiqhQA provenance weaker than hoped | Confirmed in the first 10 min; anchor cards on `statement_en`, describe school fields as structured summaries |
| Retrieval returns plausible-but-wrong fatwas | Hybrid BM25 + vector, not pure vector. Hand-inspect all 7 queries via `eval.py` by 5:15 |
| Streamlit Cloud RAM (~1 GB) | `dimensions=512` → ~39 MB; `@st.cache_resource` loads vectors once |
| Live API latency/flakiness on stage | Pre-warmed cache of the 7 demo queries. Say plainly it's cached — normal demo practice |
| "Your contemporary source is Salafi" | Every card badged with orientation. Answer honestly; askimam is the stated next step |
| Licensing | FiqhQA is a public research benchmark; IslamQA is CC-BY-NC-4.0; non-commercial demo; every card links to its original |

**Behind at 3:15** — drop the BM25 half of retrieval (pure cosine), drop IslamQA card extraction and
show retrieved answers verbatim with source badges.
**Behind at 5:15** — drop citation verification and the source drawer.
**Never cut** — the four-school comparison panel, the abstain state, per-source attribution.

---

## 7. Judge Q&A — rehearse these five

1. *"Whose ruling is this?"* → We never issue one. Each card is one school's or one source's position,
   attributed and linked. The comparison panel names the divergence and what it turns on.
2. *"How do you prevent hallucination?"* → Four-school rulings come pre-separated from the dataset
   with no generation at all; IslamQA extraction is scoped to a single document per call; quotes are
   verified against source text; and we abstain below a retrieval threshold — here's a live abstain.
3. *"What if the schools disagree?"* → That's the feature. And we measure it against FiqhQA's
   `Agreement` ground-truth column — here's our accuracy on 121 questions.
4. *"Only 121 four-school questions?"* → Yes, and the UI says so per query rather than hiding it.
   Scaling that is the next step: askimam.org is ~10k+ Hanafi fatwas, recoverable by parsing the raw
   archive around a known HF schema bug.
5. *"Where's the data from, can you use it?"* → A public research benchmark plus 19k CC-BY-NC IslamQA
   fatwas; non-commercial; every answer links to its original.

---

## 8. Verification

- **`python eval.py`** — prints top-5 IslamQA titles plus the FiqhQA hit and its score for each of the
  7 demo queries. Run after every retrieval change; judge by eye whether the right fatwas surface.
- **Comparison accuracy** — on FiqhQA hits, compare your layer's agree/diverge call against the
  `Agreement` column across all 121 rows. This is a real accuracy number to quote to judges.
- **Coverage behaviour** — the four-school panel must fire on all 3 classical queries and stay silent
  on the 3 contemporary ones and on Mars.
- **Citation check** — every `evidences[].quote` must appear in its parent doc's text; failures
  surface in the UI.
- **Abstain path** — "rules for prayer on Mars" must abstain, not invent.
- **No-adjudication check** — grep rendered output for "the correct view is" / "the stronger opinion";
  must never appear.
- **End-to-end on the deployed URL** — all 7 queries on Streamlit Cloud, before the 7:00 freeze.