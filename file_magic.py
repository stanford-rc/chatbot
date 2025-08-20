import os
import shutil
import subprocess
import yaml
import csv
import re
import sys
import logging
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import requests
from pathlib import Path
from urllib.parse import urljoin
from pprint import pprint
from typing import Iterator, Dict, Any, Optional
from var_clean_up import process_markdown_file
# --- CONFIGURATION ---
REPO_URL = "https://github.com/stanford-rc/www.sherlock.stanford.edu.git" 
LOCAL_REPO_PATH = "temp_repo"
FLAT_OUTPUT_DIR = "sherlock"
OUTPUT_CSV_FILE = "url_map.txt"
TARGETS = [
    {
        "url": "https://www.sherlock.stanford.edu/docs/tech/facts/",
        "file": "sherlock/facts.md",
        "title": "Sherlock Facts"
    },
    {
        "url": "https://www.sherlock.stanford.edu/docs/tech/",
        "file": "sherlock/tech.md",
        "title": "Sherlock Facts"

    },
    {
        "url": "https://www.sherlock.stanford.edu/docs/software/list/",
        "file": "sherlock/list.md",
        "title": "Sherlock Software List"
    },
    {
        "url": "https://www.sherlock.stanford.edu/docs/",
        "file": "sherlock/index.md",
        "title": "Welcome to Sherlock"
    },
    {
        "url": "https://www.sherlock.stanford.edu",
        "file": "sherlock/home.md",
        "title": "Sherlock"
    },
]

# Configure logging
logging.basicConfig(level=logging.INFO,filename='magicFile.log' ,format='%(asctime)s - %(levelname)s - %(message)s')
# --- END CONFIGURATION ---

# --- General-Purpose Tolerant YAML Loader ---

def ignore_and_warn_on_unknown_tags(loader, tag_prefix, node):
    """
    A generic YAML constructor that gets called for any unrecognized tag.
    It prints a warning and returns None, allowing parsing to continue.
    """
    # We can optionally print a warning to be aware of what's being ignored.
    # print(f"  [YAML Warning] Ignoring unknown tag '{node.tag}'")
    return None

class TolerantSafeLoader(yaml.SafeLoader):
    """
    A custom SafeLoader that is tolerant of unknown tags. Instead of crashing,
    it calls a default constructor that returns None for any tag it doesn't
    recognize. This is essential for parsing complex mkdocs.yml files.
    """
    pass

# Register our generic handler for ANY tag starting with '!'
# This single line handles !!python/name, !!python/object/apply, !ENV, and any others.
TolerantSafeLoader.add_multi_constructor('!', ignore_and_warn_on_unknown_tags)
TolerantSafeLoader.add_multi_constructor('tag:yaml.org,2002:python', ignore_and_warn_on_unknown_tags)
# --- End of Custom Loader Definition ---


def clone_repo(repo_url: str, local_path: str):
    """Clones a GitHub repository, ensuring a fresh start."""
    print(f"Checking out repository: {repo_url}")
    if os.path.exists(local_path):
        print(f"Removing existing directory: {local_path}")
        shutil.rmtree(local_path)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, local_path],
            check=True, capture_output=True, text=True
        )
        print("Repository checked out successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e.stderr}")
        raise

def generate_url_from_path(relative_path: str, base_url: str) -> str:
    """Generates a clean, 'pretty' URL from a file path."""
    if not base_url.endswith('/'): base_url += '/'
    url_path = relative_path.replace('\\', '/')
    if url_path.endswith('index.md'):
        url_path = os.path.dirname(url_path) + '/'
    else:
        url_path = url_path.rsplit('.', 1)[0] + '/'
    if url_path in ('./', '/'):
        url_path = ''
    return urljoin(base_url, url_path)

def parse_nav_generator(
    node: Any,
    docs_dir: Path,
    current_category: Optional[str] = None
) -> Iterator[Dict[str, Any]]:
    """
    Recursively parses a navigation node and yields a dictionary for each document,
    including special handling for standalone index files.

    Args:
        node: The current piece of the navigation structure to parse.
        docs_dir: The root path of the documentation source files.
        current_category: The title of the parent category, used for index files.
    
    Yields:
        A dictionary for each document found, matching the specified format.
    """
    # Case 1: The node is a list. Iterate and yield from recursive calls.
    if isinstance(node, list):
        for item in node:
            yield from parse_nav_generator(item, docs_dir, current_category)

    # Case 2: The node is a dictionary.
    elif isinstance(node, dict):
        for title, path_or_list in node.items():
            # This is a new category with sub-items (e.g., "Home: [...]").
            # The key is the new category title for its children.
            if isinstance(path_or_list, list):
                yield from parse_nav_generator(
                    path_or_list, docs_dir, current_category=title
                )
            # This is a standard title-file pair (e.g., "Concepts: ...").
            elif isinstance(path_or_list, str):
                yield {
                    "title": title,
                    "relative_path": path_or_list,
                    "source_path": docs_dir / path_or_list,
                    "file_name": os.path.basename(path_or_list)
                }

    # Case 3: The node is a string (a standalone file path).
    elif isinstance(node, str):
        if current_category:
            # Use the parent category as the title.
            yield {
                "title": current_category,
                "relative_path": node,
                "source_path": docs_dir / node,
                "file_name": os.path.basename(node)
            }

def scrape_url_to_file(url: str, output_path: Path, title: str):
    """
    Scrapes the main content from a single URL, converts it to Markdown,
    and saves it to the specified local file.

    Args:
        url (str): The URL of the page to scrape.
        output_path (Path): A Path object for the destination file.
        title (str): The title of the page (used for metadata)
    """
    print(f"-> Processing: {url}")

    try:
        # 1. Fetch the HTML content from the URL
        print("   - Fetching page content...")
        response = requests.get(url, timeout=15)
        response.raise_for_status() # Check for HTTP errors

        # 2. Parse the HTML and find the main content
        print("   - Parsing and finding main content...")
        soup = BeautifulSoup(response.text, 'html.parser')

        # This selector is specific to sites built with MkDocs Material theme.
        # It might need to be adjusted for other websites.
        main_content_div = soup.find('div', role='main')

        if not main_content_div:
            main_content_div = soup.find('article')

        if not main_content_div:
            main_content_div = soup.find('section', id='info')

        if not main_content_div:
            print(f"   - ERROR: Could not find a suitable main content container on the page.", file=sys.stderr)
            # We continue to the next item instead of exiting
            return

        # 3. Convert the isolated HTML to Markdown
        print("   - Converting HTML to Markdown...")
        markdown_content = md(str(main_content_div), heading_style="ATX")

        # 4. Write the Markdown to the target file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"   - Writing content to '{output_path}'...")
        output_path.write_text(markdown_content, encoding='utf-8')
        metadata_to_add = {'title': title, 'url': url}
        add_metadata_to_file(output_path, output_path, metadata_to_add)
        
        print(f"✅ Success for: {output_path}")

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR fetching {url}: {e}")
    except IOError as e:
        print(f"❌ ERROR writing to {output_path}: {e}")
    except Exception as e:
        print(f"❌ An unexpected ERROR occurred for {url}: {e}")
    

def handle_duplicate_filename(filename: str, used_filenames: set) -> str:
    """Checks for and resolves duplicate filenames by appending a counter."""
    if filename not in used_filenames:
        used_filenames.add(filename)
        return filename
    base, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_filename = f"{base}-{counter}{ext}"
        if new_filename not in used_filenames:
            used_filenames.add(new_filename)
            return new_filename
        counter += 1

def add_metadata_to_file(source_path: Path, dest_path: Path, meta: dict):
    """Reads a source file, adds YAML front matter, and writes to destination."""
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  [Warning] Source file not found: {source_path}. Skipping.")
        return

    front_matter_pattern = re.compile(r'^---\s*\n(.*?\n)---\s*\n', re.DOTALL)
    match = front_matter_pattern.match(content)
    existing_metadata = {}
    main_content = content
    if match:
        try:
            existing_metadata = yaml.load(match.group(1), Loader=TolerantSafeLoader) or {}
        except yaml.YAMLError as e:
            print(f"  [Warning] Could not parse existing front matter in {source_path}: {e}")
        main_content = content[match.end():]
    
    existing_metadata.update(meta)
    new_yaml_front_matter = yaml.dump(existing_metadata, default_flow_style=False, sort_keys=False)
    new_content = f"---\n{new_yaml_front_matter}---\n\n{main_content.lstrip()}"
    with open(dest_path, 'w', encoding='utf-8') as f: f.write(new_content)

def variable_clean_up(dest_path: Path):
    """Reads a source file, does some variable magic, and writes to destination."""
    processed_text = process_markdown_file(dest_path)
    with open(dest_path, 'w', encoding='utf-8') as f: f.write(processed_text)

def remove_temp_repo_directory():
    print(f"Removing temp directory: {LOCAL_REPO_PATH}")
    shutil.rmtree(LOCAL_REPO_PATH)

def main():
    """Main execution function."""
    print("--- Document Processing Script Started ---")

    try:
        clone_repo(REPO_URL, LOCAL_REPO_PATH)
    except Exception:
        print("Halting script due to repository checkout failure."); return

    repo_path = Path(LOCAL_REPO_PATH)
    config_file = repo_path / "mkdocs.yml"
    if not config_file.exists():
        print("Error: 'mkdocs.yml' not found in the repo root."); return

    print(f"Loading configuration from: {config_file} using a general-purpose tolerant loader.")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            # Use our new, highly tolerant loader
            config = yaml.load(f, Loader=TolerantSafeLoader)
    except yaml.YAMLError as e:
        print(f"\nFATAL: The 'mkdocs.yml' file has a syntax error that could not be handled.\nError details: {e}\n"); return

    base_site_url = config.get('site_url')
    docs_dir_name = config.get('docs_dir', 'docs')
    nav = config.get('nav')
    
    if not base_site_url: print("Error: 'site_url' not found in mkdocs.yml."); return
    if not nav: print("Error: 'nav' section not found in mkdocs.yml."); return

    docs_dir = repo_path / docs_dir_name
    print(f"Successfully parsed config. Site URL: {base_site_url}, Docs dir: {docs_dir_name}")

    flat_dir_path = Path(FLAT_OUTPUT_DIR); flat_dir_path.mkdir(exist_ok=True)
    used_filenames = set()
    processed_count = 0

    with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['url', 'file_name', 'title']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()

        print("\nStarting single-pass processing of documents from navigation...")
        
        for doc_info in parse_nav_generator(nav, docs_dir):
            logging.info(f"Doc info: {doc_info}")
            doc_info['url'] = generate_url_from_path(doc_info['relative_path'], base_site_url)
            doc_info['file_name'] = handle_duplicate_filename(doc_info['file_name'], used_filenames)
            csv_writer.writerow({k: doc_info[k] for k in fieldnames})
            destination_path = flat_dir_path / doc_info['file_name']
            print(f"  - Processing: {doc_info['relative_path']:<40} -> {doc_info['file_name']}")
            metadata_to_add = {'title': doc_info['title'], 'url': doc_info['url']}
            add_metadata_to_file(doc_info['source_path'], destination_path, metadata_to_add)
            variable_clean_up(destination_path)
            processed_count += 1

    print(f"\nSuccessfully processed {processed_count} documents.")
    print(f"Index written to {OUTPUT_CSV_FILE}")
    print(f"Modified documents saved in {FLAT_OUTPUT_DIR}")
    remove_temp_repo_directory()

    if not TARGETS:
        print("No targets defined in the 'TARGETS' list. Exiting.")
        return

    print("--- Starting Scraper ---")
    total_targets = len(TARGETS)
    
    for i, target in enumerate(TARGETS, 1):
        print(f"\n[{i}/{total_targets}]" + "-" * 40)
        url = target.get("url")
        file_path_str = target.get("file")
        title = target.get("title")

        if not url or not file_path_str:
            print(f"Skipping invalid target entry: {target}")
            continue

        scrape_url_to_file(url=url, output_path=Path(file_path_str), title = title)
        
    print("\n--- All tasks completed! ---")

    print("--- Script Finished Successfully ---")


if __name__ == "__main__":
    main()