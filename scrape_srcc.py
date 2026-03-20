"""
scrape_srcc.py  —  Stanford Decoupled Profile (SDP) / Drupal 10 edition

Strategy:
  1. JSON:API  — fetches structured content types with known SDP field names.
               stanford_page is intentionally skipped here because it uses
               Layout Builder + paragraph components, not a simple body field.
  2. HTML crawl — picks up stanford_page and any other routes not covered by
               JSON:API. Layout Builder multi-region aware, BigPipe-safe.

Output mirrors magicFile.py conventions (YAML front matter, flat .md files).

Usage:
    python scrape_srcc.py

Environment variables (optional):
    SRCC_OUTPUT_DIR   - output directory       (default: docs/srcc)
    SRCC_MAX_PAGES    - HTML crawl page cap     (default: 500)
    LOG_FILE          - log file path           (default: magicFile.log)
"""

import os
import re
import logging
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL        = "https://srcc.stanford.edu"
OUTPUT_DIR      = Path(os.getenv("SRCC_OUTPUT_DIR", "docs/srcc"))
MAX_PAGES       = int(os.getenv("SRCC_MAX_PAGES", 500))
REQUEST_DELAY   = 0.5
REQUEST_TIMEOUT = 15
LOG_FILE        = os.getenv("LOG_FILE", "magicFile.log")

# ---------------------------------------------------------------------------
# SDP bundle definitions
#
# Each entry defines how to extract meaningful text from a JSON:API response
# for that content type. Field names are based on the Stanford Decoupled
# Profile (SDP) schema.
#
# Fields reference:
#   body                    - standard Drupal body field (processed HTML)
#   su_news_components      - news body paragraphs (wysiwyg blocks)
#   su_person_components    - person bio paragraphs
#   su_event_components     - event body paragraphs (fallback to su_event_dek)
#   su_policy_body          - policy body (processed HTML)
#   su_publication_citation - publication structured citation fields
#   su_course_description   - course description text
#
# 'extra_fields' are appended as plain metadata after the main body.
# ---------------------------------------------------------------------------

BUNDLES = {
    "stanford_news": {
        "body_field":    "su_news_components",   # paragraphs; fallback below
        "body_fallback": "body",
        "extra_fields":  ["su_news_dek"],        # subheadline/summary
        "date_field":    "su_news_publishing_date",
    },
    "stanford_person": {
        "body_field":    "su_person_components",
        "body_fallback": "body",
        "extra_fields":  [
            "su_person_full_title",
            "su_person_email",
            "su_person_short_bio",
        ],
    },

    "stanford_policy": {
        "body_field":    "su_policy_body",       # processed HTML
        "body_fallback": "body",
        "extra_fields":  ["su_policy_updated"],
    },
    "stanford_publication": {
        "body_field":    "body",
        "body_fallback": None,
        "extra_fields":  ["su_publication_citation"],
    },
    "stanford_course": {
        "body_field":    "su_course_description",  # plain text
        "body_fallback": "body",
        "extra_fields":  [
            "su_course_code",
            "su_course_id",
            "su_course_quarters",
            "su_course_subject",
        ],
    },
}

# stanford_page is intentionally absent — it uses Layout Builder paragraph
# components and has no usable body field. The HTML crawl handles it instead.

# ---------------------------------------------------------------------------
# Drupal URL patterns that produce noise — skip during HTML crawl.
# ---------------------------------------------------------------------------

SKIP_PATH_PATTERNS = [
    r"^/user",
    r"^/node/\d+$",
    r"^/search",
    r"^/filter",
    r"^/taxonomy/term",
    r"^/admin",
    r"^/contextual",
    r"^/events",             # skip event pages in HTML crawl
    r"\?.*page=",
    r"\.json$",
]

# ---------------------------------------------------------------------------
# Layout Builder / HTML extraction
# ---------------------------------------------------------------------------

# SDP Layout Builder regions — collect ALL, not just the first.
LAYOUT_BUILDER_SELECTORS = [
    ("div", "layout-builder__region"),
    ("div", "layout__region"),
    ("div", "su-page-components"),         # SDP-specific wrapper
    ("div", "field--name-su-page-components"),
    ("div", "field--name-body"),
]

# Single-element fallbacks if no LB regions found.
CONTENT_SELECTORS = [
    {"class": "node__content"},
    {"class": "layout-container"},
    {"role": "main"},
    {"id": "main-content"},
    {"id": "content"},
]

STRIP_SELECTORS = [
    ("tag",   "nav"),
    ("tag",   "header"),
    ("tag",   "footer"),
    ("tag",   "script"),
    ("tag",   "style"),
    ("tag",   "noscript"),
    ("tag",   "drupal-render-placeholder"),
    ("class", "su-global-footer"),
    ("class", "su-masthead"),
    ("class", "visually-hidden"),
    ("id",    "toolbar-bar"),
    ("id",    "toolbar-administration"),
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def slugify_url(url: str) -> str:
    """https://srcc.stanford.edu/about/team/ -> srcc_about_team.md"""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_").replace("-", "_")
    path = re.sub(r"_+", "_", path)
    name = path if path else "index"
    if parsed.query:
        qs = re.sub(r"[^a-z0-9]", "_", parsed.query.lower())[:30]
        name = f"{name}__{qs}"
    return f"{name}.md"


def write_markdown(output_path: Path, title: str, url: str, markdown_body: str, extra_meta: dict = {}):
    """Write a .md file with YAML front matter. Mirrors magicFile.py conventions."""
    meta = {"title": title, "url": url, "source": url, **extra_meta}
    front_matter = yaml.dump(meta, default_flow_style=False, sort_keys=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"---\n{front_matter}---\n\n{markdown_body.strip()}\n",
        encoding="utf-8",
    )


def normalize(url: str) -> str:
    p = urlparse(url)
    return p._replace(fragment="", query="").geturl().rstrip("/")


def should_skip(url: str) -> bool:
    path = urlparse(url).path
    return any(re.search(pat, path) for pat in SKIP_PATH_PATTERNS)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "SRCC-chatbot-crawler/1.0 (internal)"})
    return s


def html_to_md(html: str) -> str:
    """Convert an HTML string to Markdown, stripping empty output."""
    return md(html, heading_style="ATX").strip()

# ---------------------------------------------------------------------------
# Path 1 — JSON:API (structured content types)
# ---------------------------------------------------------------------------

def resolve_field(attrs: dict, field_name: str) -> str:
    """
    Extract text from an attribute field, handling:
      - processed HTML  ({"value": ..., "processed": ...})
      - plain string
      - None / missing
    Returns a Markdown string (converts HTML if needed).
    """
    val = attrs.get(field_name)
    if not val:
        return ""
    if isinstance(val, dict):
        html = val.get("processed") or val.get("value") or ""
        return html_to_md(html) if html else ""
    if isinstance(val, str):
        return val.strip()
    # Lists (e.g. course quarters) — join as a comma string
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v)
    return str(val)


def build_body_from_attrs(attrs: dict, bundle_cfg: dict) -> str:
    """
    Assemble the main body text for a node using the bundle config.
    Tries body_field first, falls back to body_fallback.
    Note: paragraph component fields (su_news_components etc.) are
    relationships, not attributes — they won't be in attrs unless
    ?include= is used. If empty, we fall back gracefully so the HTML
    crawl can pick up the page instead.
    """
    body = resolve_field(attrs, bundle_cfg["body_field"])
    if not body and bundle_cfg.get("body_fallback"):
        body = resolve_field(attrs, bundle_cfg["body_fallback"])
    return body


def build_extra_meta(attrs: dict, bundle_cfg: dict) -> dict:
    """Collect extra fields into a flat dict for front matter."""
    meta = {}
    for field in bundle_cfg.get("extra_fields", []):
        val = resolve_field(attrs, field)
        if val:
            meta[field] = val
    if bundle_cfg.get("date_field"):
        date_val = attrs.get(bundle_cfg["date_field"])
        if date_val:
            meta["date"] = date_val
    return meta


def fetch_jsonapi_bundle(session: requests.Session, bundle: str, bundle_cfg: dict) -> list[dict]:
    """
    Fetch all nodes for a bundle via JSON:API with cursor-based pagination.
    Requests only the fields we actually need to keep payloads small.
    """
    needed_fields = set()
    needed_fields.add(bundle_cfg["body_field"])
    if bundle_cfg.get("body_fallback"):
        needed_fields.add(bundle_cfg["body_fallback"])
    needed_fields.update(bundle_cfg.get("extra_fields", []))
    if bundle_cfg.get("date_field"):
        needed_fields.add(bundle_cfg["date_field"])
    needed_fields.update(["title", "path", "status"])

    endpoint = f"{BASE_URL}/jsonapi/node/{bundle}"
    params = {
        f"fields[node--{bundle}]": ",".join(needed_fields),
        "filter[status]":          1,     # published only
        "page[limit]":             50,
    }
    results = []

    while endpoint:
        try:
            resp = session.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 404:
                logging.info(f"JSON:API bundle not found (skipping): {bundle}")
                return []
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logging.error(f"JSON:API error for '{bundle}': {e}")
            return results

        for node in data.get("data", []):
            attrs = node.get("attributes", {})
            title = attrs.get("title", "Untitled")
            path  = (attrs.get("path") or {}).get("alias") or f"/node/{node['id']}"
            url   = f"{BASE_URL}{path}"
            body  = build_body_from_attrs(attrs, bundle_cfg)
            extra = build_extra_meta(attrs, bundle_cfg)
            results.append({"title": title, "url": url, "body": body, "extra": extra})

        endpoint = (data.get("links") or {}).get("next", {}).get("href")
        params   = {}
        time.sleep(REQUEST_DELAY)

    return results


def run_jsonapi(session: requests.Session, written_urls: set) -> int:
    total = 0
    print("\n── JSON:API pass ──")
    print("   (stanford_page skipped — handled by HTML crawl)\n")

    for bundle, cfg in BUNDLES.items():
        nodes = fetch_jsonapi_bundle(session, bundle, cfg)
        if not nodes:
            print(f"  {bundle}: 0 nodes (bundle not found or empty)")
            continue

        written = 0
        skipped = 0
        for node in nodes:
            url = normalize(node["url"])
            if url in written_urls:
                skipped += 1
                continue

            body = node["body"]
            if not body:
                # No body via JSON:API — HTML crawl will pick this up
                logging.info(f"No JSON:API body for {url}, deferring to HTML crawl")
                skipped += 1
                continue

            filename = slugify_url(url)
            write_markdown(
                OUTPUT_DIR / filename,
                node["title"],
                url,
                body,
                node.get("extra", {}),
            )
            written_urls.add(url)
            written += 1
            total += 1

        print(f"  {bundle:<30} written: {written:>4}   deferred to HTML: {skipped:>4}")

    return total

# ---------------------------------------------------------------------------
# Path 2 — HTML crawl (stanford_page + Layout Builder)
# ---------------------------------------------------------------------------

def strip_noise(soup: BeautifulSoup):
    for kind, value in STRIP_SELECTORS:
        if kind == "tag":
            for el in soup.find_all(value):
                el.decompose()
        elif kind == "class":
            for el in soup.find_all(class_=value):
                el.decompose()
        elif kind == "id":
            el = soup.find(id=value)
            if el:
                el.decompose()


def extract_layout_builder_content(soup: BeautifulSoup) -> str:
    """
    SDP Layout Builder assembles pages from multiple region divs.
    Collect all of them, deduplicate, and concatenate.
    """
    regions = []
    for tag, cls in LAYOUT_BUILDER_SELECTORS:
        regions.extend(soup.find_all(tag, class_=cls))

    if regions:
        seen = set()
        unique = [r for r in regions if not (id(r) in seen or seen.add(id(r)))]
        return html_to_md("\n\n".join(str(r) for r in unique))

    # Single-element fallbacks
    for attrs in CONTENT_SELECTORS:
        el = soup.find(attrs=attrs)
        if el:
            return html_to_md(str(el))

    body = soup.find("body") or soup
    return html_to_md(str(body))


def extract_title(soup: BeautifulSoup, url: str) -> str:
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    if soup.title:
        return soup.title.get_text(strip=True).split("|")[0].strip()
    return urlparse(url).path.strip("/").replace("/", " › ").title() or "SRCC"


def is_crawlable(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
        return False
    if re.search(r"\.(pdf|zip|docx?|xlsx?|pptx?|png|jpe?g|gif|svg|css|js)$",
                  parsed.path, re.IGNORECASE):
        return False
    return True


def run_html_crawl(session: requests.Session, written_urls: set) -> int:
    session.headers.update({"Accept": "text/html"})
    queue   = [normalize(BASE_URL)]
    visited = set(written_urls)
    total   = 0

    print("\n── HTML crawl pass (stanford_page + Layout Builder) ──\n")

    while queue and total < MAX_PAGES:
        url  = queue.pop(0)
        norm = normalize(url)
        if norm in visited or should_skip(norm):
            continue
        visited.add(norm)
        time.sleep(REQUEST_DELAY)

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
        except requests.RequestException as e:
            logging.error(f"HTML fetch failed: {url} — {e}")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            abs_url = normalize(urljoin(url, a["href"]))
            if is_crawlable(abs_url) and abs_url not in visited:
                queue.append(abs_url)

        strip_noise(soup)
        title         = extract_title(soup, url)
        markdown_body = extract_layout_builder_content(soup)

        if not markdown_body.strip():
            logging.warning(f"No content extracted (HTML): {url}")
            continue

        filename = slugify_url(url)
        write_markdown(OUTPUT_DIR / filename, title, url, markdown_body)
        written_urls.add(norm)
        total += 1
        print(f"  [{total:>4}] {url:<65} -> {filename}")

    return total

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def crawl():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session      = make_session()
    written_urls: set[str] = set()

    print(f"Target  : {BASE_URL}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Strategy: JSON:API (structured types) + HTML crawl (stanford_page)\n")

    jsonapi_count = run_jsonapi(session, written_urls)
    html_count    = run_html_crawl(session, written_urls)

    total = jsonapi_count + html_count
    print(f"\n✅  Done.  JSON:API: {jsonapi_count}  |  HTML crawl: {html_count}  |  Total: {total}")
    print(f"   Files written to: {OUTPUT_DIR}")
    logging.info(f"SRCC crawl complete. jsonapi={jsonapi_count}, html={html_count}, total={total}")


def process_srcc():
    """Hook for magicFile.py: `from scrape_srcc import process_srcc`"""
    crawl()


if __name__ == "__main__":
    crawl()
