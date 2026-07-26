"""Central configuration. Import from here rather than hardcoding paths or thresholds."""

from pathlib import Path

# This file lives at <project root>/data/raw/config.py, so the project root is
# two levels up. Everything else is derived from it - never hardcode a path.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"

# --- corpus artifacts ---------------------------------------------------------
# One merged corpus. Scraped sources append to this same file via a new adapter.
CORPUS = DATA / "corpus.json"
SCHEMA_DOC = RAW / "schema.json"

# One index over the whole corpus. There used to be a second one for the FiqhQA
# four-school records; see the note under SOURCE_META for why they are gone.
VECTORS_SINGLE = DATA / "vectors_single.npy"

# --- upstream sources ---------------------------------------------------------
HF_PARQUET = {
    "islamqa": "https://huggingface.co/api/datasets/kingkaung/english_islamqainfo/parquet/default/train/0.parquet",
}

# Controlled vocabulary. `orientation` is shown on every card, so it must be a
# fixed set - a scraper inventing "salafi-leaning" would silently split the badge
# into two categories. Adding a new site means adding its orientation HERE first.
ORIENTATIONS = [
    "Salafi",
    "Hanafi",
    "Shafi'i",
    "Maliki",
    "Hanbali",
    "Unspecified",
]

# Register every source here - scraped ones included. `orientation` is shown on
# every card; we never present a single-madhhab source as neutral.
#
# REMOVED: fiqhqa (MBZUAI), 121 multi_school records that were the four-school
# comparison layer. They failed the standard every other source here meets: all
# 121 shared ONE url - the HuggingFace dataset page - so the "source" link on a
# card led to a download, not to a fatwa. No named scholar, no citable original,
# licence "Research benchmark". A card reading "the Hanafi school holds X" with
# nothing behind it a reader can check is exactly the authority this tool is not
# entitled to claim, so the records are out rather than dressed up. The
# multi_school record type survives in schema.py for a future source that can
# cite itself properly.
SOURCE_META = {
    "islamqa": {
        "source_label": "IslamQA.info",
        "source_url": "https://islamqa.info",
        "orientation": "Salafi",
        "scholar": "Muhammad Salih al-Munajjid and team",
        "license": "CC-BY-NC-4.0",
    },
    # --- scraped sources (see scrape.py) ---
    "islamqaorg": {
        "source_label": "IslamQA.org",
        "source_url": "https://islamqa.org",
        # An AGGREGATOR over ~40 darul-iftas spanning all four madhhabs, so the
        # orientation is per-record, read off the URL, not per-source. This
        # default only applies if a record somehow arrives without one.
        "orientation": "Unspecified",
        "scholar": None,        # set per record to the issuing institution
        "license": "Scraped - non-commercial research use",
    },
    "askimam": {
        "source_label": "AskImam (Darul Iftaa)",
        "source_url": "https://askimam.org",
        "orientation": "Hanafi",
        "scholar": "Mufti Ebrahim Desai",
        "license": "Scraped - non-commercial research use",
    },
}

# --- cleaning -----------------------------------------------------------------
# Category labels render on card badges, so source-side typos are visible.
# Applied to every source in make_doc; upstream data is left untouched.
CATEGORY_FIXES = {
    "Mariage": "Marriage",       # FiqhQA misspelling
    "Hajj Omra": "Hajj & Umrah",
}

MIN_ANSWER_CHARS = 50          # drops 7 stub rows in IslamQA
MAX_ANSWER_CHARS_FOR_LLM = 6000  # median answer is ~3k; cap context per card

# --- sampling -----------------------------------------------------------------
# IslamQA is 15,296 unique fatwas - more than a one-day demo needs, and it
# dominates every other source. Take a stratified 10%: proportional per topic
# with a floor of one, so no topic disappears and the mix is unchanged.
# Set to 1.0 to use the full corpus; nothing else needs touching.
ISLAMQA_SAMPLE_FRAC = 0.10
SAMPLE_SEED = 42               # fixed, so the corpus is reproducible

# Sampling is blind to what the demo needs. There is exactly ONE bitcoin fatwa in
# all 15,296, so a 10% sample drops it 90% of the time - and did. Any fatwa whose
# text matches one of these is kept regardless of the sample. Add a term here
# whenever a demo query has thin coverage; it costs a handful of docs.
SAMPLE_PINS = [
    r"bitcoin|cryptocurrenc",
    r"mortgage",
    r"life insurance",
    r"forex|foreign exchange",
]

# --- scraping -----------------------------------------------------------------
# Both scrapers write already-schema-conformant JSON straight into data/.
SCRAPED = {
    "islamqaorg": RAW / "islamqaorg.json",
    "askimam": RAW / "askimam.json",
}

# IslamQA.info contributes ~1,589; the two scrapes make up the balance - and
# shift the corpus off its Salafi centre of gravity, which is the point.
# 1589 islamqa + these two = 3,879, the corpus as it stands since FiqhQA's 121
# were dropped. islamqa.org lands 3 short of its quota (dead urls in the
# sitemap), so askimam carries the remainder.
SCRAPE_TARGETS = {"islamqaorg": 1290, "askimam": 1003}

SCRAPE_DELAY = 0.34            # seconds between requests, per site. Be a good guest.
SCRAPE_TIMEOUT = 45
SCRAPE_RETRIES = 3
USER_AGENT = (
    "Mozilla/5.0 (compatible; FatwaRAG-research/1.0; "
    "non-commercial academic corpus; contact via project repo)"
)

# islamqa.org republishes askimam.org under /hanafi/askimam/. We scrape askimam
# directly, so skip those here rather than paying for them twice.
ISLAMQAORG_SKIP_INSTITUTIONS = {"askimam"}

# The four madhhabs are wildly unequal on islamqa.org (hanafi 91,580 vs hanbali
# 182). Sampling proportionally would rebuild the very bias we are removing, so
# the scraper fills each madhhab as evenly as availability allows instead.
ISLAMQAORG_BALANCE_BY_MADHHAB = True

# --- embeddings ---------------------------------------------------------------
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 512               # 19k x 512 float32 = ~39MB, fits Streamlit Cloud
EMBED_BATCH = 100
LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # no-key fallback (384 dims)

# --- retrieval ----------------------------------------------------------------
TOP_K_ISLAMQA = 5
RRF_K = 60                     # reciprocal rank fusion constant
# CALIBRATED FOR THE LOCAL EMBEDDER (bge-small, 384 dims). Recalibrate if you
# switch to OpenAI - the two models have completely different score floors.
# bge-small scores UNRELATED pairs at 0.60-0.71, so the original 0.45/0.30 sat
# below the noise floor: every demo query fired the four-school panel and nothing
# ever abstained. Measured on data/raw/demo_queries.py, the separating windows
# are (0.710, 0.738] and (0.710, 0.784]; these values sit inside both.
ABSTAIN_THRESHOLD = 0.75       # below this, refuse to answer at all

# Per-hit relevance floor. ABSTAIN_THRESHOLD only ever looked at the BEST hit, so
# a question with one strong match dragged four weak ones along with it and every
# one of them got a card, a citation and a source badge - "Can You Pray in a
# Moving Car?" rendered as a source on where to place your hands, and "Playing
# with and selling Pokemon cards" on bitcoin. Both scored ~0.70, i.e. inside the
# 0.60-0.71 band bge-small gives UNRELATED pairs. This is stage one of two: it
# drops the documents that are cheap to prove irrelevant, before they cost an
# LLM call. Stage two is CardOut.answers_question in generate.py, where the model
# reads the full document and makes the actual judgement.
RELEVANCE_THRESHOLD = 0.72

# The escape hatch that keeps hybrid search hybrid. A rare-term match (riba,
# mudarabah, istihada) is exactly the case where BM25 is right and the embedding
# is weak, so a hit BM25 ranks this highly survives a failing cosine and is left
# for the LLM to judge. Without this, the cosine floor quietly turns the system
# back into pure vector search.
BM25_STRONG_RANK = 5

# --- generation ---------------------------------------------------------------
# Matches test.py, the connection smoke test. One line to change if you move to
# claude-opus-5; generate.py adapts its sampling arguments automatically.
LLM_MODEL = "claude-sonnet-5"

# This is a CEILING, not a target - we are billed on tokens produced. It has to
# be generous because thinking tokens count against it: the model thinks first
# and emits JSON second, so a tight limit truncates the JSON mid-string and the
# structured-output parse fails with a confusing "EOF while parsing" rather than
# an obvious "you ran out of room". 4000 was not enough for the comparison call
# over five cards; measured usage is ~600-2500.
LLM_MAX_TOKENS = 16000

# Temperature 1 - the API default, and what a RAG layer wants here: the model is
# extracting from a supplied document, not inventing, so the grounding comes from
# the prompt and the retrieved context rather than from sampling.
#
# IMPORTANT: this is NOT always passable as a parameter. On claude-sonnet-5 only
# the DEFAULT value is accepted (any other value returns a 400); on
# claude-opus-5 / fable-5 / opus-4.8 / opus-4.7 the parameter is removed outright
# and sending it at all returns a 400. generate.sampling_args() encodes that -
# it passes the value where it is legal and omits it where it is not, which is
# identical behaviour either way because 1.0 IS the default.
LLM_TEMPERATURE = 1.0
LLM_NO_TEMPERATURE_MODELS = {
    "claude-opus-5",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
}
