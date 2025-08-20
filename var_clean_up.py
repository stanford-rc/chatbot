#!/usr/bin/env python3

import argparse
import re
import sys

# --- Configuration ---
# This dictionary is used by the processing function.
# You can modify it here or pass a custom dictionary to the helper functions.
LOCAL_VARS = {
    "purge_days": "90",
    "support_email": "srcc-support@stanford.edu",
    "url_list":"https://www.sherlock.stanford.edu/docs/software/list/",
    "url_contact": "mailto:srcc-support@stanford.edu",
    "url_check_quotas": "https://www.sherlock.stanford.edu/docs/storage/#checking-quotas",
    "url_snapshots": "https://www.sherlock.stanford.edu/docs/storage/data-protection/#snapshots",
    "url_gdrive": "https://www.sherlock.stanford.edu/docs/storage/data-transfer/#google-drive",
    "url_cpu": "https://www.sherlock.stanford.edu/docs/glossary/#cpu",
    "url_prereq": "https://www.sherlock.stanford.edu/docs/getting-started/#windows",
    "url_connecting": "https://www.sherlock.stanford.edu/docs/getting-started/connecting/#authentication",
    "url_prereq": "https://www.sherlock.stanford.edu/docs/#prerequisites",
    "url_account": "https://www.sherlock.stanford.edu/docs/#how-to-request-an-account",
    "url_dtn": "https://www.sherlock.stanford.edu/docs/storage/data-transfer/#data-transfer-nodes-dtns",
    "url_clugens": "https://www.sherlock.stanford.edu/docs/concepts/#cluster-generations",
    "url_owners": "https://www.sherlock.stanford.edu/docs/concepts/#investing-in-sherlock",
    "url_srcc": "//srcc.stanford.edu",
    "url_sh_part": "https://www.sherlock.stanford.edu/docs/user-guide/running-jobs/#available-resources",
    "url_maintenances-and-upgrades":"https://www.sherlock.stanford.edu/docs/concepts/#maintenances-and-upgrades",
    "title: Home": "title: Welcome to Sherlock",
}

# --- Helper Functions ---

def _substitute_variables(content, variables):
    """Substitutes {{var_name}} placeholders with values from a dictionary."""
    pattern = re.compile(r'\{\{\s*(\w+)\s*\}\}')

    def replacer(match):
        var_name = match.group(1)
        return variables.get(var_name, match.group(0))

    return pattern.sub(replacer, content)

def _remove_footer_block(content):
    """Removes link definition blocks and includes from the end of the file."""
    footer_pattern = re.compile(r"^(?:\[comment\]: #|--8<---)", re.MULTILINE)
    match = footer_pattern.search(content)

    if match:
        print("Found and removed footer block.", file=sys.stderr)
        return content[:match.start()].rstrip()
    return content

# --- Main Processing Function ---

def process_markdown_file(file_path):
    """
    Reads, cleans, and substitutes variables in a markdown file.

    This is the main reusable function. It orchestrates the entire process:
    1. Reads the content from the given file path.
    2. Removes the footer block (link definitions, includes).
    3. Substitutes {{variable}} placeholders.

    Args:
        file_path (str): The path to the input markdown file.

    Returns:
        str: The processed content of the markdown file as a string.
    
    Raises:
        FileNotFoundError: If the file_path does not exist.
        Exception: For other file reading errors.
    """
    # 1. Read the file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'", file=sys.stderr)
        raise  # Re-raise the exception for the caller to handle
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
        raise

    # 2. Remove the footer block
    cleaned_content = _remove_footer_block(content)

    # 3. Perform variable substitution using the global LOCAL_VARS
    final_content = _substitute_variables(cleaned_content, LOCAL_VARS)

    return final_content


# --- Command-Line Interface ---

def main_cli():
    """
    Handles command-line execution. This function is called when the script
    is run directly from the terminal.
    """
    parser = argparse.ArgumentParser(
        description="Parses a markdown file, performs variable substitution, and removes footer blocks.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        help="The path to the input markdown file."
    )
    parser.add_argument(
        "-o", "--output",
        metavar="OUTPUT_FILE",
        help="The path to the output file. If not provided, prints to standard output."
    )

    args = parser.parse_args()

    try:
        # Call the main processing function
        print(f"Processing '{args.input_file}'...", file=sys.stderr)
        processed_content = process_markdown_file(args.input_file)

        # Handle the output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            print(f"Successfully wrote output to '{args.output}'", file=sys.stderr)
        else:
            print(processed_content)

    except Exception:
        # The processing function already prints detailed errors.
        # We just exit with an error code.
        sys.exit(1)


if __name__ == "__main__":
    main_cli()