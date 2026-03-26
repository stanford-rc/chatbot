"""
scrape_carina.py  —  Static site edition

docs.carina.stanford.edu is a static site (Jekyll/similar) with:
  - Full server-side rendering, no JS required
  - Clean single-column content under <main id="page-content">
  - All pages linked from the nav present on every page
  - No JSON:API or CMS backend needed

Strategy: seed from the nav links on the homepage, then crawl
any additional internal links discovered along the way.

Output mirrors magicFile.py conventions (YAML front matter, flat .md files).

Usage:
    python scrape_carina.py

Environment variables (optional):
    CARINA_OUTPUT_DIR  - output directory   (default: docs/carina)
    CARINA_MAX_PAGES   - page cap           (default: 200)
    LOG_FILE           - log file path      (default: magicFile.log)
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

BASE_URL        = "https://docs.carina.stanford.edu"
OUTPUT_DIR      = Path(os.getenv("CARINA_OUTPUT_DIR", "docs/carina"))
MAX_PAGES       = int(os.getenv("CARINA_MAX_PAGES", 200))
REQUEST_DELAY   = 0.5
REQUEST_TIMEOUT = 15
LOG_FILE        = os.getenv("LOG_FILE", "logs/scrapers.log")

# Elements to strip before content extraction.
# The site wraps chrome in <header>, <nav>, and <footer> — easy to isolate.
STRIP_SELECTORS = [
    ("tag",   "header"),
    ("tag",   "nav"),
    ("tag",   "footer"),
    ("tag",   "script"),
    ("tag",   "style"),
    ("tag",   "noscript"),
    ("class", "site-nav"),           # top navigation bar
    ("class", "page-sidebar"),       # "On This Page" / "See Also" sidebars
    ("class", "sidebar"),
    ("id",    "skip-nav"),
]

# Skip these path patterns — external-pointing anchors and utility pages.
SKIP_PATH_PATTERNS = [
    r"^/404",
    r"^/search",
    r"#",                            # in-page anchors
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
# Helpers
# ---------------------------------------------------------------------------

def slugify_url(url: str) -> str:
    """
    https://docs.carina.stanford.edu/slurm-carina -> carina_slurm_carina.md
    https://docs.carina.stanford.edu/             -> carina_index.md
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_").replace("-", "_")
    name = f"carina_{path}" if path else "carina_index"
    return f"{name}.md"


def write_markdown(output_path: Path, title: str, url: str, markdown_body: str):
    """Write a .md file with YAML front matter. Mirrors magicFile.py conventions."""
    meta = {"title": title, "url": url, "source": url}
    front_matter = yaml.dump(meta, default_flow_style=False, sort_keys=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"---\n{front_matter}---\n\n{markdown_body.strip()}\n",
        encoding="utf-8",
    )


def normalize(url: str) -> str:
    """Strip fragment and trailing slash for deduplication."""
    p = urlparse(url)
    # remove fragment only — preserve path/query
    clean = p._replace(fragment="").geturl().rstrip("/")
    return clean


def should_skip(url: str) -> bool:
    path = urlparse(url).path
    return any(re.search(pat, path) for pat in SKIP_PATH_PATTERNS)


def is_internal(url: str) -> bool:
    """Only follow links within docs.carina.stanford.edu."""
    parsed = urlparse(url)
    # relative links have no netloc
    if not parsed.netloc:
        return True
    return parsed.netloc == urlparse(BASE_URL).netloc


def is_html_resource(url: str) -> bool:
    """Skip non-HTML assets and non-HTTP schemes (mailto:, tel:, etc.)."""
    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        return False
    if not parsed.netloc and not parsed.path:
        return False
    return not re.search(
        r"\.(pdf|zip|docx?|xlsx?|pptx?|png|jpe?g|gif|svg|css|js|ico|txt|xml)$",
        parsed.path, re.IGNORECASE
    )


def strip_noise(soup: BeautifulSoup):
    """Remove chrome and sidebar elements in-place."""
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


def extract_content(soup: BeautifulSoup) -> str:
    """
    Pull main content from the page.
    The site uses <main id="page-content"> to wrap the article body,
    which cleanly excludes nav, header, and sidebars.
    """
    main = (
        soup.find("main", id="page-content")
        or soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find("body")
        or soup
    )
    return md(str(main), heading_style="ATX").strip()


def extract_title(soup: BeautifulSoup, url: str) -> str:
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    if soup.title:
        return soup.title.get_text(strip=True).split("|")[0].strip()
    return urlparse(url).path.strip("/").replace("/", " › ").title() or "Carina"


def seed_urls_from_nav(soup: BeautifulSoup) -> list[str]:
    """
    Extract all nav links from the page. Since the full site nav is
    present on every page, seeding from the homepage nav gives us the
    complete page list upfront without needing a full crawl.
    """
    nav = soup.find("nav") or soup.find(class_="site-nav")
    if not nav:
        return []
    return [
        normalize(urljoin(BASE_URL, a["href"]))
        for a in nav.find_all("a", href=True)
        if is_internal(a["href"]) and is_html_resource(a["href"])
    ]

# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

def crawl():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "SRCC-chatbot-crawler/1.0 (internal)"})

    visited: set[str] = set()
    total   = 0
    errors  = 0

    print(f"Target  : {BASE_URL}")
    print(f"Output  : {OUTPUT_DIR}\n")

    # --- Seed from homepage nav ---
    try:
        resp = session.get(BASE_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        home_soup = BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"FATAL: Could not fetch homepage — {e}")
        logging.critical(f"Carina homepage fetch failed: {e}")
        return

    nav_urls  = seed_urls_from_nav(home_soup)
    queue     = [normalize(BASE_URL)] + nav_urls
    print(f"Seeded {len(nav_urls)} URLs from nav. Starting crawl...\n")

    while queue and total < MAX_PAGES:
        url  = queue.pop(0)
        norm = normalize(url)

        if norm in visited or should_skip(norm):
            continue
        visited.add(norm)

        time.sleep(REQUEST_DELAY)
        logging.info(f"Fetching: {url}")

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            if "text/html" not in resp.headers.get("Content-Type", ""):
                logging.info(f"Skipping non-HTML: {url}")
                continue
        except requests.RequestException as e:
            logging.error(f"Fetch failed: {url} — {e}")
            errors += 1
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Discover any additional internal links not already in the nav
        for a in soup.find_all("a", href=True):
            abs_url = normalize(urljoin(url, a["href"]))
            if (
                is_internal(abs_url)
                and is_html_resource(abs_url)
                and abs_url not in visited
                and not should_skip(abs_url)
            ):
                queue.append(abs_url)

        strip_noise(soup)
        title   = extract_title(soup, url)
        content = extract_content(soup)

        if not content:
            logging.warning(f"No content extracted: {url}")
            continue

        filename = slugify_url(url)
        write_markdown(OUTPUT_DIR / filename, title, url, content)
        total += 1
        print(f"  [{total:>3}] {url:<60} -> {filename}")

    print(f"\n✅  Done. Pages scraped: {total}  |  Errors: {errors}")
    print(f"   Files written to: {OUTPUT_DIR}")
    logging.info(f"Carina crawl complete. total={total}, errors={errors}")


def process_carina():
    """Hook for magicFile.py: `from scrape_carina import process_carina`"""
    crawl()


if __name__ == "__main__":
    crawl()
