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
from urllib.parse import urljoin, urlparse
from pprint import pprint
from typing import Iterator, Dict, Any, Optional

# Assuming var_clean_up.py exists and has the process_markdown_file function
# from var_clean_up import process_markdown_file

# --- MOCK function for demonstration if var_clean_up is not available ---
def process_markdown_file(file_path):
    """Placeholder function. Replace with your actual implementation."""
    logging.info(f"Variable cleanup would run on: {file_path}")
    # Example: return text.replace("{{ some_var }}", "some_value")
    # For this rework, we'll just log and do nothing to the file.
    return 
# --- END MOCK ---

# --- CONFIGURATION ---
# Define all sources to be processed in this list.
# To add a new repository, simply add a new dictionary to the SOURCES list.
SOURCES = [
    {
        "repo_url": "git@github.com:stanford-rc/farmshare-docs.git",
        "repo_name": "farmshare",  # A short name for creating directories
    },
    {
        "repo_url": "git@github.com:stanford-rc/docs.elm.stanford.edu.git",
        "repo_name": "elm",  # A short name for creating directories
    },
     {
        "repo_url": "git@github.com:stanford-rc/docs.oak.stanford.edu.git",
        "repo_name": "oak",  # A short name for creating directories
    },   
    {
        "repo_url": "git@github.com:stanford-rc/www.sherlock.stanford.edu.git",
        "repo_name": "sherlock",  # A short name for creating directories
        "scraper_targets": [
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
    },
    # {
    #     "repo_url": "https://github.com/another/example-repo.git",
    #     "repo_name": "example",
    #     "scraper_targets": [
    #          { "url": "...", "file": "...", "title": "..." }
    #     ]
    # }
]

# Configure logging
LOG_FILE = 'magicFile.log'
logging.basicConfig(level=logging.INFO,
                    filename=LOG_FILE,
                    filemode='w',  # Start with a fresh log file each run
                    format='%(asctime)s - %(levelname)s - %(message)s')
# --- END CONFIGURATION ---

# --- General-Purpose Tolerant YAML Loader ---
def ignore_and_warn_on_unknown_tags(loader, tag_prefix, node):
    """A generic YAML constructor that gets called for any unrecognized tag."""
    logging.warning(f"Ignoring unknown YAML tag '{node.tag}'")
    return None

class TolerantSafeLoader(yaml.SafeLoader):
    """A custom SafeLoader that is tolerant of unknown tags."""
    pass

TolerantSafeLoader.add_multi_constructor('!', ignore_and_warn_on_unknown_tags)
TolerantSafeLoader.add_multi_constructor('tag:yaml.org,2002:python', ignore_and_warn_on_unknown_tags)
# --- End of Custom Loader Definition ---


def clone_repo(repo_url: str, local_path: Path):
    """Clones a GitHub repository, ensuring a fresh start."""
    logging.info(f"Cloning repository: {repo_url} into {local_path}")
    if local_path.exists():
        logging.info(f"Removing existing directory: {local_path}")
        shutil.rmtree(local_path)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(local_path)],
            check=True, capture_output=True, text=True
        )
        logging.info("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error cloning repository: {e.stderr}")
        raise

def generate_url_from_path(relative_path: str, base_url: str) -> str:
    """Generates a clean, 'pretty' URL from a file path."""
    if not base_url.endswith('/'):
        base_url += '/'
    url_path = Path(relative_path).as_posix() # Use posix paths for URLs
    if url_path.endswith('index.md'):
        url_path = str(Path(url_path).parent) + '/'
    else:
        url_path = str(Path(url_path).with_suffix('')) + '/'
    if url_path in ('./', '/'):
        url_path = ''
    return urljoin(base_url, url_path)

def parse_nav_generator(node: Any, docs_dir: Path, current_category: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Recursively parses a navigation node and yields a dictionary for each document."""
    if isinstance(node, list):
        for item in node:
            yield from parse_nav_generator(item, docs_dir, current_category)
    elif isinstance(node, dict):
        for title, path_or_list in node.items():
            if isinstance(path_or_list, list):
                yield from parse_nav_generator(path_or_list, docs_dir, current_category=title)
            elif isinstance(path_or_list, str):
                yield {
                    "title": title,
                    "relative_path": path_or_list,
                    "source_path": docs_dir / path_or_list,
                    "file_name": Path(path_or_list).name
                }
    elif isinstance(node, str):
        if current_category:
            yield {
                "title": current_category,
                "relative_path": node,
                "source_path": docs_dir / node,
                "file_name": Path(node).name
            }

def scrape_url_to_file(url: str, output_path: Path, title: str):
    """Scrapes content from a URL, converts to Markdown, and saves to a file."""
    logging.info(f"-> Processing URL: {url}")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        main_content_div = soup.find('div', role='main') or soup.find('article') or soup.find('section', id='info')

        if not main_content_div:
            logging.error(f"Could not find a suitable main content container on {url}")
            return

        markdown_content = md(str(main_content_div), heading_style="ATX")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown_content, encoding='utf-8')
        
        metadata_to_add = {'title': title, 'url': url}
        add_metadata_to_file(output_path, output_path, metadata_to_add)
        logging.info(f"✅ Success for: {output_path}")

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ ERROR fetching {url}: {e}")
    except IOError as e:
        logging.error(f"❌ ERROR writing to {output_path}: {e}")
    except Exception as e:
        logging.error(f"❌ An unexpected ERROR occurred for {url}: {e}")

def handle_duplicate_filename(filename: str, used_filenames: set) -> str:
    """Checks for and resolves duplicate filenames by appending a counter."""
    if filename not in used_filenames:
        used_filenames.add(filename)
        return filename
    
    p = Path(filename)
    base, ext = p.stem, p.suffix
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
        content = source_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        logging.warning(f"Source file not found: {source_path}. Skipping metadata addition.")
        return

    front_matter_pattern = re.compile(r'^---\s*\n(.*?\n)---\s*\n', re.DOTALL)
    match = front_matter_pattern.match(content)
    
    existing_metadata = {}
    main_content = content
    if match:
        try:
            existing_metadata = yaml.load(match.group(1), Loader=TolerantSafeLoader) or {}
        except yaml.YAMLError as e:
            logging.warning(f"Could not parse existing front matter in {source_path}: {e}")
        main_content = content[match.end():]
    
    existing_metadata.update(meta)
    new_yaml_front_matter = yaml.dump(existing_metadata, default_flow_style=False, sort_keys=False)
    new_content = f"---\n{new_yaml_front_matter}---\n\n{main_content.lstrip()}"
    dest_path.write_text(new_content, encoding='utf-8')

def variable_clean_up(dest_path: Path):
    """Wrapper for the variable cleanup process."""
    try:
        process_markdown_file(dest_path)
    except Exception as e:
        logging.error(f"Failed to run variable cleanup on {dest_path}: {e}")


def cleanup_directory(dir_path: Path):
    """Removes a directory if it exists."""
    if dir_path.exists():
        logging.info(f"Removing temporary directory: {dir_path}")
        shutil.rmtree(dir_path)

def process_repository(config: dict):
    """
    Main processing logic for a single repository.
    Clones, parses mkdocs.yml, processes files, and runs the scraper.
    """
    repo_url = config["repo_url"]
    repo_name = config["repo_name"]
    scraper_targets = config.get("scraper_targets", [])

    print(f"\n{'='*20} Processing Repository: {repo_name} {'='*20}")
    logging.info(f"Starting processing for repository: {repo_url}")

    # Define dynamic paths based on repo_name
    local_repo_path = Path(f"temp_repo_{repo_name}")
    flat_output_dir = Path(f"{repo_name}")
    output_csv_file = Path(f"{repo_name}_url_map.csv")

    try:
        clone_repo(repo_url, local_repo_path)
    except Exception:
        print(f"FATAL: Halting processing for {repo_name} due to repository clone failure.")
        logging.critical(f"Clone failure for {repo_name}. Aborting its processing.")
        cleanup_directory(local_repo_path)
        return

    config_file = local_repo_path / "mkdocs.yml"
    if not config_file.exists():
        print(f"Error: 'mkdocs.yml' not found in {repo_name}. Skipping mkdocs processing.")
        logging.error(f"'mkdocs.yml' not found for {repo_name}.")
    else:
        process_mkdocs_repo(config_file, local_repo_path, flat_output_dir, output_csv_file)

    # --- Scraper ---
    if not scraper_targets:
        print("No scraper targets defined for this repository.")
    else:
        print("\n--- Starting Scraper ---")
        for i, target in enumerate(scraper_targets, 1):
            print(f"\n[{i}/{len(scraper_targets)}] Scraping '{target.get('title')}'")
            url = target.get("url")
            file_path_str = target.get("file")
            title = target.get("title")

            if not all([url, file_path_str, title]):
                logging.warning(f"Skipping invalid scraper target entry: {target}")
                continue
            
            # Ensure the output file is inside the main output directory for this repo
            scrape_output_path = flat_output_dir / Path(file_path_str).name
            scrape_url_to_file(url=url, output_path=scrape_output_path, title=title)

    cleanup_directory(local_repo_path)
    print(f"--- Finished processing for {repo_name} ---")

def process_mkdocs_repo(config_file: Path, repo_path: Path, flat_dir_path: Path, output_csv_file: Path):
    """Handles the part of the processing specific to mkdocs repos."""
    print(f"Loading configuration from: {config_file}")
    try:
        config = yaml.load(config_file.read_text(encoding='utf-8'), Loader=TolerantSafeLoader)
    except yaml.YAMLError as e:
        print(f"FATAL: YAML syntax error in {config_file}. Details: {e}")
        logging.critical(f"YAML syntax error in {config_file}: {e}")
        return

    base_site_url = config.get('site_url')
    docs_dir_name = config.get('docs_dir', 'docs')
    nav = config.get('nav')

    if not all([base_site_url, nav]):
        print("Error: 'site_url' and/or 'nav' not found in mkdocs.yml. Skipping doc generation.")
        logging.error("'site_url' and/or 'nav' missing in mkdocs.yml.")
        return

    docs_dir = repo_path / docs_dir_name
    print(f"Successfully parsed config. Site URL: {base_site_url}, Docs dir: {docs_dir}")

    flat_dir_path.mkdir(exist_ok=True)
    used_filenames = set()
    processed_count = 0

    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['url', 'file_name', 'title']
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()

        print("\nStarting processing of documents from navigation...")
        for doc_info in parse_nav_generator(nav, docs_dir):
            doc_info['url'] = generate_url_from_path(doc_info['relative_path'], base_site_url)
            doc_info['file_name'] = handle_duplicate_filename(doc_info['file_name'], used_filenames)
            csv_writer.writerow({k: doc_info[k] for k in fieldnames})
            
            destination_path = flat_dir_path / doc_info['file_name']
            print(f"  - Processing: {doc_info['relative_path']:<40} -> {destination_path}")
            
            metadata_to_add = {'title': doc_info['title'], 'url': doc_info['url']}
            add_metadata_to_file(doc_info['source_path'], destination_path, metadata_to_add)
            variable_clean_up(destination_path)
            processed_count += 1
    
    print(f"\nSuccessfully processed {processed_count} documents from mkdocs.yml.")
    print(f"URL map written to {output_csv_file}")
    print(f"Markdown documents saved in {flat_dir_path}")

def main():
    """Main execution function to loop through all configured sources."""
    print("--- Document Processing Script Started ---")
    logging.info("Script started.")
    
    if not SOURCES:
        print("No sources configured in the 'SOURCES' list. Exiting.")
        logging.warning("SOURCES list is empty. Nothing to do.")
        return
        
    for source_config in SOURCES:
        try:
            process_repository(source_config)
        except Exception as e:
            repo_name = source_config.get("repo_name", "Unknown")
            print(f"An unexpected error occurred while processing {repo_name}. See log for details.")
            logging.error(f"CRITICAL FAILURE during processing of {repo_name}: {e}", exc_info=True)
    
    print("\n--- All configured jobs have been processed. ---")
    logging.info("Script finished successfully.")


if __name__ == "__main__":
    main()
