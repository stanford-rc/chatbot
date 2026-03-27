"""
scrape_static_docs.py  —  Generic Stanford static docs site scraper

Handles any Stanford static documentation site sharing the common template
(Jekyll/similar, <main id="page-content">, Stanford footer chrome).

Currently configured for:
  - docs.carina.stanford.edu
  - nero-docs.stanford.edu

To add a new site, add an entry to SITES at the bottom of this file.

Output mirrors magicFile.py conventions (YAML front matter, flat .md files).

Usage:
    # Scrape all configured sites:
    python scrape_static_docs.py

    # Scrape a specific site by key:
    python scrape_static_docs.py carina
    python scrape_static_docs.py nero

Environment variables (optional):
    STATIC_DOCS_OUTPUT_DIR  - base output directory  (default: docs)
    STATIC_DOCS_MAX_PAGES   - page cap per site       (default: 200)
    LOG_FILE                - log file path            (default: logs/scrapers.log)
"""

import os
import re
import sys
import logging
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# ---------------------------------------------------------------------------
# Site definitions
# ---------------------------------------------------------------------------

SITES = {
    "carina": {
        "base_url":    "https://docs.carina.stanford.edu",
        "output_dir":  "carina",
        "slug_prefix": "carina",
    },
    "nero": {
        "base_url":    "https://nero-docs.stanford.edu",
        "output_dir":  "nero",
        "slug_prefix": "nero",
    },
    # Add more sites here:
    # "example": {
    #     "base_url":    "https://example-docs.stanford.edu",
    #     "output_dir":  "example",
    #     "slug_prefix": "example",
    # },
}

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

BASE_OUTPUT_DIR = Path(os.getenv("STATIC_DOCS_OUTPUT_DIR", "docs"))
MAX_PAGES       = int(os.getenv("STATIC_DOCS_MAX_PAGES", 200))
REQUEST_DELAY   = 0.5
REQUEST_TIMEOUT = 15
LOG_FILE        = os.getenv("LOG_FILE", "logs/scrapers.log")

# ---------------------------------------------------------------------------
# Selectors (shared across all sites — same Stanford template)
# ---------------------------------------------------------------------------

STRIP_SELECTORS = [
    ("tag",   "header"),
    ("tag",   "nav"),
    ("tag",   "footer"),
    ("tag",   "script"),
    ("tag",   "style"),
    ("tag",   "noscript"),
    ("class", "site-nav"),
    ("class", "page-sidebar"),
    ("class", "sidebar"),
    ("id",    "skip-nav"),
]

SKIP_PATH_PATTERNS = [
    r"^/404",
    r"^/search",
    r"#",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Compiled patterns (module-level to avoid recompiling in loops)
# ---------------------------------------------------------------------------

BARE_HOST_RE = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}$", re.IGNORECASE)
SKIP_EXT_RE  = re.compile(
    r"\.(pdf|zip|docx?|xlsx?|pptx?|png|jpe?g|gif|svg|css|js|ico|txt|xml)$",
    re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify_url(url: str, prefix: str) -> str:
    """
    https://docs.carina.stanford.edu/slurm-carina -> carina_slurm_carina.md
    https://nero-docs.stanford.edu/jupyter.html   -> nero_jupyter.md
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    path = re.sub(r"\.html?$", "", path)
    path = path.replace("/", "_").replace("-", "_")
    path = re.sub(r"_+", "_", path)
    name = f"{prefix}_{path}" if path else f"{prefix}_index"
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
    return urlparse(url)._replace(fragment="").geturl().rstrip("/")


def should_skip(url: str) -> bool:
    path = urlparse(url).path
    return any(re.search(pat, path) for pat in SKIP_PATH_PATTERNS)


def is_internal(url: str, base_netloc: str) -> bool:
    parsed = urlparse(url)
    if not parsed.netloc:
        return True
    return parsed.netloc == base_netloc


def is_html_resource(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        return False
    if not parsed.netloc and not parsed.path:
        return False
    if not parsed.scheme and "@" in parsed.path:
        return False
    if not parsed.scheme and BARE_HOST_RE.match(parsed.path):
        return False
    return not SKIP_EXT_RE.search(parsed.path)


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


def extract_content(soup: BeautifulSoup) -> str:
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
    return urlparse(url).path.strip("/").replace("/", " › ").title() or "Docs"


def seed_urls_from_nav(soup: BeautifulSoup, base_url: str, base_netloc: str) -> list[str]:
    nav = soup.find("nav") or soup.find(class_="site-nav")
    if not nav:
        return []
    return [
        normalize(urljoin(base_url, a["href"]))
        for a in nav.find_all("a", href=True)
        if "@" not in a["href"]
        and is_internal(a["href"], base_netloc)
        and is_html_resource(a["href"])
    ]

# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

def crawl_site(site_key: str, site_cfg: dict):
    base_url    = site_cfg["base_url"]
    base_netloc = urlparse(base_url).netloc
    output_dir  = BASE_OUTPUT_DIR / site_cfg["output_dir"]
    slug_prefix = site_cfg["slug_prefix"]

    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "SRCC-chatbot-crawler/1.0 (internal)"})

    visited: set[str] = set()
    total   = 0
    errors  = 0

    print(f"\n{'='*20} {site_key} {'='*20}")
    print(f"Target  : {base_url}")
    print(f"Output  : {output_dir}\n")

    try:
        resp = session.get(base_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        home_soup = BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"FATAL: Could not fetch homepage — {e}")
        logging.critical(f"{site_key} homepage fetch failed: {e}")
        return

    nav_urls = seed_urls_from_nav(home_soup, base_url, base_netloc)
    queue    = [normalize(base_url)] + nav_urls
    print(f"Seeded {len(nav_urls)} URLs from nav. Starting crawl...\n")

    while queue and total < MAX_PAGES:
        url  = queue.pop(0)
        norm = normalize(url)

        if norm in visited or should_skip(norm):
            continue
        visited.add(norm)

        time.sleep(REQUEST_DELAY)
        logging.info(f"[{site_key}] Fetching: {url}")

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            if "text/html" not in resp.headers.get("Content-Type", ""):
                continue
        except requests.RequestException as e:
            logging.error(f"[{site_key}] Fetch failed: {url} — {e}")
            errors += 1
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "@" in href:
                continue
            abs_url = normalize(urljoin(url, href))
            if (
                is_internal(abs_url, base_netloc)
                and is_html_resource(abs_url)
                and abs_url not in visited
                and not should_skip(abs_url)
            ):
                queue.append(abs_url)

        strip_noise(soup)
        title   = extract_title(soup, url)
        content = extract_content(soup)

        if not content:
            logging.warning(f"[{site_key}] No content extracted: {url}")
            continue

        filename = slugify_url(url, slug_prefix)
        write_markdown(output_dir / filename, title, url, content)
        total += 1
        print(f"  [{total:>3}] {url:<65} -> {filename}")

    print(f"\n✅  {site_key} done. Pages: {total}  |  Errors: {errors}")
    logging.info(f"{site_key} crawl complete. total={total}, errors={errors}")


def crawl_all():
    for key, cfg in SITES.items():
        crawl_site(key, cfg)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def process_static_docs():
    """Hook for magicFile.py: `from scrape_static_docs import process_static_docs`"""
    crawl_all()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        key = sys.argv[1]
        if key not in SITES:
            print(f"Unknown site '{key}'. Available: {', '.join(SITES)}")
            sys.exit(1)
        crawl_site(key, SITES[key])
    else:
        crawl_all()
