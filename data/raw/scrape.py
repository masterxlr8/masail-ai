"""Scrape islamqa.org and askimam.org into the common schema.

    python scrape.py                 # both, to the targets in config.SCRAPE_TARGETS
    python scrape.py islamqaorg      # one site
    python scrape.py askimam -n 200  # override the target

Output goes straight to data/islamqaorg.json and data/askimam.json as arrays of
schema-conformant records - the same shape as data/corpus.json, validated before
writing. ingest.py then merges them with the bulk datasets.

WHY THESE TWO SITES
-------------------
The corpus was 93% IslamQA.info, which is Salafi. That is a real bias and it
shows on every card. These two sites are the correction:

  islamqa.org   an AGGREGATOR over ~40 darul-iftas. The madhhab and the issuing
                institution are both in the URL path
                (/hanafi/darul-iftaa-chicago/271036/slug/), so we get all four
                Sunni schools labelled at the record level, for free.
  askimam.org   Mufti Ebrahim Desai's Darul Iftaa, 24,865 fatwas, uniformly Hanafi.

BALANCE, NOT PROPORTION
-----------------------
islamqa.org has 91,580 Hanafi URLs and 182 Hanbali. Sampling proportionally
would rebuild the bias we are removing, so `balanced_quota` fills the smallest
madhhabs to exhaustion first and splits what is left evenly. Hanbali and Maliki
end up fully included; Hanafi and Shafi'i are sampled.

HOW EACH SITE IS REACHED
------------------------
islamqa.org  WordPress. wp-json is 401, so we use sitemap.xml -> 20
             sitemap-posts pages -> 98,637 URLs, then fetch each article and
             parse .entry-content. Institutions each style their own markup,
             so the question/answer split tries three strategies in order
             (see `split_qa`) - measured at 95% on a 40-URL sample.
askimam.org  React SPA over a Django REST API at askimam.org:8000.
             List:   /api/v1/fatwas/fatwas/?page=N   (25/page, 24,865 total)
             Detail: /api/v1/fatwas/fatwas/<id>/     (question, answer, category)
             Pages are sampled evenly across the whole range rather than taken
             from the front, so the sample spans 2000s-2020s rather than only
             the most recent fatwas.

MANNERS
-------
config.SCRAPE_DELAY between requests, one connection, descriptive User-Agent,
exponential backoff on failure, and the run is resumable - re-running skips
records already in the output file. Neither site's robots.txt disallows
anything (both were checked; islamqa.org's is empty, askimam.org has none).
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from datetime import date

import requests
from bs4 import BeautifulSoup

from data.raw.config import (
    DATA,
    ISLAMQAORG_BALANCE_BY_MADHHAB,
    ISLAMQAORG_SKIP_INSTITUTIONS,
    MIN_ANSWER_CHARS,
    RAW,
    SAMPLE_SEED,
    SCRAPE_DELAY,
    SCRAPE_RETRIES,
    SCRAPE_TARGETS,
    SCRAPE_TIMEOUT,
    SCRAPED,
    USER_AGENT,
)
from ingest import clean_text, make_doc
from data.raw.schema import Doc, content_duplicates, validate_docs

TODAY = date.today().isoformat()

MADHHAB_LABEL = {
    "hanafi": "Hanafi",
    "shafii": "Shafi'i",
    "maliki": "Maliki",
    "hanbali": "Hanbali",
}


# --------------------------------------------------------------------------- #
# polite fetching                                                             #
# --------------------------------------------------------------------------- #
class Fetcher:
    """One session, one connection, a fixed delay between requests."""

    def __init__(self, delay: float = SCRAPE_DELAY):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.delay = delay
        self.last = 0.0
        self.n = 0

    def _wait(self) -> None:
        gap = time.monotonic() - self.last
        if gap < self.delay:
            time.sleep(self.delay - gap)
        self.last = time.monotonic()

    def get(self, url: str, **kw):
        """GET with exponential backoff. Returns None if all retries fail."""
        for attempt in range(SCRAPE_RETRIES):
            self._wait()
            try:
                r = self.s.get(url, timeout=SCRAPE_TIMEOUT, **kw)
                self.n += 1
                if r.status_code == 200:
                    return r
                if r.status_code in (404, 401, 403):
                    return None          # genuinely absent, no point retrying
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)     # 1s, 2s, 4s
        return None

    def json(self, url: str, **kw):
        r = self.get(url, **kw)
        if r is None:
            return None
        try:
            return r.json()
        except ValueError:
            return None


# --------------------------------------------------------------------------- #
# shared parsing                                                              #
# --------------------------------------------------------------------------- #
def html_to_text(html: str) -> str:
    """Flatten a fatwa's HTML body to readable plain text."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    return clean_text(soup.get_text("\n", strip=True))


# Answer-start markers, derived by inspecting real pages rather than guessed.
# Institutions open the answer with a bare 'Answer' line, an Arabic al-jawab, a
# basmala, or a return salaam - all four occur.
ANSWER_MARKER = re.compile(
    r"(?m)^[\s\W]{0,4}(?:"
    r"Answer\b[\s:.\-]*$"
    r"|Answer\s*[:.\-]"
    r"|A\s*[:.]"
    r"|الجواب"
    r"|In the name of All?a?h,?\s*Most"
    r"|Wa\s*[-'’]?\s*alaikum"
    r")",
    re.I,
)
QUESTION_HEAD = re.compile(r"(?m)\A[\s\W]{0,4}Question\b[\s:.\-]*$")

CLASS_Q = re.compile(r"(?:^|-)(?:question|ques|q)$", re.I)
CLASS_A = re.compile(r"(?:^|-)(?:answer|ans|a)$", re.I)

# Trailing boilerplate every darul-ifta appends. Cut at the first one.
TAIL = re.compile(
    r"(?m)^[\s\W]{0,4}(?:Original Source Link|Answered by\s*:|Approved by\s*:"
    r"|Checked (?:and )?[Aa]pproved|And All?ah (?:Ta.?ala )?[Kk]nows [Bb]est)",
)


def split_qa(node) -> tuple[str, str]:
    """Separate a fatwa page into (question, answer).

    Three strategies, in order of reliability:
      1. Per-institution CSS classes - dic-question/dic-answer,
         ddb-question/ddb-answer, di-jordan-ques/di-jordan-ans, sb-answer, ...
         All share a suffix, so one regex catches every variant.
      2. An answer-start marker in the text; everything before it is the question.
      3. Neither - the page is an article rather than a Q&A. Everything becomes
         the answer and the question stays empty, which is honest: `embed_text`
         then falls back to the title alone.
    """
    q_node, a_node = node.find(class_=CLASS_Q), node.find(class_=CLASS_A)
    if q_node is not None and a_node is not None:
        return html_to_text(str(q_node)), html_to_text(str(a_node))

    text = html_to_text(str(node))
    text = QUESTION_HEAD.sub("", text).strip()
    m = ANSWER_MARKER.search(text)
    if m and m.start() > 0:
        return text[: m.start()].strip(), text[m.start():].strip()
    return "", text


def trim_tail(answer: str) -> str:
    """Drop the signature block, keeping the first line of it as attribution."""
    m = TAIL.search(answer)
    return answer[: m.start()].strip() if m and m.start() > 200 else answer


# Once the question and answer are separated, their own labels are noise. Both
# repeat because we split ON them, and they'd otherwise be embedded as content.
Q_PREFIX = re.compile(r"^\s*(?:Q(?:uestion)?\s*[:.\u2013-]\s*)+", re.I)
A_PREFIX = re.compile(r"^\s*(?:A(?:nswer)?\s*[:.\u2013-]\s*)+", re.I)

# SeekersGuidance and Qibla prepend the responding scholar to the question body.
# That is provenance, not content: it belongs in `scholar`, and leaving it in
# the question pollutes embed_text for every one of their fatwas.
ANSWERED_BY = re.compile(
    r"\A\s*Answered by\s*[:\-]?\s*(?P<who>[^\n]{3,60}?)\s*"
    r"(?=\bQuestion\b|[\n]|\Z)",
    re.I,
)


# Every askimam answer opens with the basmala and a return salaam - the exact
# analogue of IslamQA.info's 'Praise be to Allah.', which ingest already strips.
# It is on ~100% of records, so it carries no retrieval signal and only costs
# BM25 weight and LLM context. Spelling varies wildly, hence the tolerance.
OPENER = re.compile(
    r"\A\s*(?:"
    r"In the [Nn]ame of All?a?a?h[^\n]{0,70}?(?:Merciful|Raheem)\s*[.,]?\s*"
    r"|As[- ]?salaamu?\s*[`\u2018\u2019']?\s*al[ae]yk?um[^\n]{0,80}?"
    r"(?:Wabarakatoh|wabarakatuh|[Bb]arakatuh|Barakaatuh)\s*[.,]?\s*"
    r")+",
    re.I,
)


def strip_labels(question: str, answer: str) -> tuple[str, str, str | None]:
    """Remove Q:/A: labels and lift any 'Answered by ...' into an attribution."""
    who = None
    answer = OPENER.sub("", answer)
    m = ANSWERED_BY.match(question)
    if m:
        who = m.group("who").strip(" .,:-")
        question = question[m.end():]
    question = Q_PREFIX.sub("", question).strip()
    answer = A_PREFIX.sub("", answer).strip()
    return question, answer, who


# --------------------------------------------------------------------------- #
# balancing                                                                   #
# --------------------------------------------------------------------------- #
def balanced_quota(pools: dict[str, list], total: int) -> dict[str, int]:
    """Allocate `total` across pools as evenly as availability allows.

    Fills the smallest pools to exhaustion first and redistributes what they
    cannot absorb. With hanbali=182, maliki=278, shafii=6,596, hanafi=91,580 and
    total=1,290 this takes all Hanbali and Maliki and splits the rest evenly,
    rather than handing 93% of the budget to the Hanafis.
    """
    quota: dict[str, int] = {}
    left, names = total, sorted(pools, key=lambda k: len(pools[k]))
    for i, name in enumerate(names):
        share = left // (len(names) - i)
        take = min(len(pools[name]), share)
        quota[name] = take
        left -= take
    return quota


# --------------------------------------------------------------------------- #
# islamqa.org                                                                 #
# --------------------------------------------------------------------------- #
SITEMAP_INDEX = "https://islamqa.org/sitemap.xml"
URL_CACHE = RAW / "islamqaorg_urls.txt"


def islamqaorg_urls(f: Fetcher) -> list[str]:
    """All post URLs, from the sitemap index. Cached after the first run."""
    if URL_CACHE.exists():
        return URL_CACHE.read_text(encoding="utf-8").split()

    index = f.get(SITEMAP_INDEX)
    pages = re.findall(r"<loc>([^<]+sitemap-posts[^<]*)</loc>", index.text if index else "")
    urls: list[str] = []
    for page in pages:
        r = f.get(page)
        if r:
            urls += re.findall(r"<loc>([^<]+)</loc>", r.text)
    RAW.mkdir(parents=True, exist_ok=True)
    URL_CACHE.write_text("\n".join(urls), encoding="utf-8")
    return urls


def scrape_islamqaorg(f: Fetcher, target: int, seen: set[str]) -> list[Doc]:
    urls = islamqaorg_urls(f)
    print(f"  sitemap: {len(urls)} post urls")

    # /{madhhab}/{institution}/{id}/{slug}/
    pools: dict[str, list[tuple]] = {}
    for u in urls:
        parts = u.rstrip("/").split("/")
        if len(parts) < 7:
            continue
        madhhab, institution, native = parts[3], parts[4], parts[5]
        if madhhab not in MADHHAB_LABEL or institution in ISLAMQAORG_SKIP_INSTITUTIONS:
            continue
        if not native.isdigit():
            continue
        pools.setdefault(madhhab, []).append((u, madhhab, institution, native))

    print("  available:", {k: len(v) for k, v in sorted(pools.items())})
    rng = random.Random(SAMPLE_SEED)
    for v in pools.values():
        rng.shuffle(v)

    quota = (balanced_quota(pools, target) if ISLAMQAORG_BALANCE_BY_MADHHAB
             else {k: round(target * len(v) / sum(map(len, pools.values())))
                   for k, v in pools.items()})
    print("  quota    :", quota)

    docs: list[Doc] = []
    for madhhab in sorted(quota, key=lambda k: len(pools[k])):
        want, got, cursor = quota[madhhab], 0, 0
        pool = pools[madhhab]
        # Over-fetch: some pages are articles or too short and get dropped.
        while got < want and cursor < len(pool):
            url, mad, institution, native = pool[cursor]
            cursor += 1
            if f"islamqaorg:{native}" in seen:
                continue
            doc = fetch_islamqaorg_one(f, url, mad, institution, native)
            if doc is None:
                continue
            docs.append(doc)
            got += 1
            if got % 50 == 0:
                print(f"    {mad:<8} {got}/{want}")
        print(f"    {madhhab:<8} {got}/{want} done ({cursor} urls visited)")
    return docs


def fetch_islamqaorg_one(f: Fetcher, url, madhhab, institution, native) -> Doc | None:
    r = f.get(url)
    if r is None:
        return None
    soup = BeautifulSoup(r.text, "html.parser")
    body = soup.select_one(".entry-content")
    if body is None:
        return None

    for tag in body.select(".original_source, script, style"):
        tag.decompose()

    title = soup.find("h1")
    title = title.get_text(" ", strip=True) if title else ""
    question, answer = split_qa(body)
    question, answer, answered_by = strip_labels(question, trim_tail(answer))
    if len(answer) < MIN_ANSWER_CHARS:
        return None

    institution_label = institution.replace("-", " ").title()

    # <meta property="article:published_time"> is the only reliable date here.
    meta = soup.find("meta", property="article:published_time")
    published = (meta.get("content") or "")[:10] if meta else None

    return make_doc(
        "islamqaorg", native,
        title=title, question=question, answer=answer,
        categories=[institution_label],
        url=url,
        orientation=MADHHAB_LABEL[madhhab],
        madhhab=madhhab,
        # Named scholar where the page gives one, otherwise the issuing body.
        scholar=f"{answered_by} ({institution_label})" if answered_by else institution_label,
        date_published=published or None,
    )


# --------------------------------------------------------------------------- #
# askimam.org                                                                 #
# --------------------------------------------------------------------------- #
ASKIMAM_API = "https://askimam.org:8000/api/v1/fatwas/fatwas/"
ASKIMAM_WEB = "https://www.askimam.org/public/question_detail/"
ASKIMAM_PAGE_SIZE = 25


def scrape_askimam(f: Fetcher, target: int, seen: set[str]) -> list[Doc]:
    head = f.json(ASKIMAM_API, params={"page": 1})
    if not head:
        print("  askimam api unreachable")
        return []
    total = head["count"]
    print(f"  api: {total} fatwas, {ASKIMAM_PAGE_SIZE} per window")

    # `page` is silently IGNORED - every value returns the same newest 25 rows.
    # `offset` is the one parameter this endpoint actually honours (measured;
    # `limit`, `p`, `page_number`, `start` and `ordering` are all ignored too).
    need = max(1, -(-target // ASKIMAM_PAGE_SIZE)) + 12         # slack for drops
    step = max(ASKIMAM_PAGE_SIZE, (total // need // ASKIMAM_PAGE_SIZE) * ASKIMAM_PAGE_SIZE)
    offsets = list(range(0, total, step))
    random.Random(SAMPLE_SEED).shuffle(offsets)

    docs: list[Doc] = []
    for offset in offsets:
        if len(docs) >= target:
            break
        listing = f.json(ASKIMAM_API, params={"offset": offset})
        for row in (listing or {}).get("results", []):
            if len(docs) >= target:
                break
            # The detail endpoint is keyed by `old_id`, not the listing's `id`.
            # Passing `id` 500s for most rows, which costs three retries each.
            fid = row.get("old_id") or row.get("id")
            if fid is None or f"askimam:{fid}" in seen:
                continue
            seen.add(f"askimam:{fid}")   # windows overlap; never fetch one twice
            doc = fetch_askimam_one(f, fid)
            if doc is not None:
                docs.append(doc)
                if len(docs) % 50 == 0:
                    print(f"    {len(docs)}/{target}")
    print(f"    {len(docs)}/{target} done")
    return docs


def fetch_askimam_one(f: Fetcher, fid) -> Doc | None:
    d = f.json(f"{ASKIMAM_API}{fid}/")
    if not d:
        return None
    ans = d.get("answer") or {}
    q = ans.get("question") or {}

    question, answer, _ = strip_labels(
        html_to_text(q.get("question") or ""),
        trim_tail(html_to_text(ans.get("answer") or "")),
    )
    if len(answer) < MIN_ANSWER_CHARS:
        return None

    category = d.get("category")
    if isinstance(category, dict):
        category = category.get("name")

    return make_doc(
        "askimam", fid,
        title=q.get("title") or "",
        question=question,
        answer=answer,
        categories=[category] if category else [],
        url=f"{ASKIMAM_WEB}{fid}",
        madhhab="hanafi",
        date_published=(ans.get("answer_modified") or "")[:10] or None,
    )


# --------------------------------------------------------------------------- #
SCRAPERS = {"islamqaorg": scrape_islamqaorg, "askimam": scrape_askimam}


def load_existing(path) -> list[Doc]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [Doc.from_dict(d) for d in json.load(fh)]


def main(argv: list[str]) -> int:
    target_override = None
    if "-n" in argv:
        i = argv.index("-n")
        target_override = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]

    wanted = argv or list(SCRAPERS)
    unknown = [s for s in wanted if s not in SCRAPERS]
    if unknown:
        print(f"unknown source(s): {unknown}. available: {list(SCRAPERS)}")
        return 1

    DATA.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        out = SCRAPED[name]
        target = target_override or SCRAPE_TARGETS[name]
        existing = load_existing(out)
        seen = {d.id for d in existing}
        need = target - len(existing)
        print(f"\n{name}: have {len(existing)}, target {target}")
        if need <= 0:
            print("  already at target, nothing to do")
            continue

        f = Fetcher()
        t0 = time.monotonic()
        fresh = SCRAPERS[name](f, need, seen)
        docs = existing + fresh

        errors = validate_docs(docs)
        if errors:
            print(f"  {len(errors)} validation error(s), NOT written:")
            for e in errors[:10]:
                print(f"    {e}")
            return 1

        with open(out, "w", encoding="utf-8") as fh:
            json.dump([d.to_dict() for d in docs], fh, ensure_ascii=False, indent=1)
        dupes = content_duplicates(docs)
        print(f"  +{len(fresh)} new -> {len(docs)} records in {out}")
        print(f"  {f.n} requests in {time.monotonic() - t0:.0f}s"
              + (f", {len(dupes)} content-duplicate group(s)" if dupes else ""))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
