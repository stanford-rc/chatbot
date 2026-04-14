#!/usr/bin/env python3
"""
generate_manifests.py — Write .content_manifest.json for each docs subdirectory.

Run after all scrapers finish.  Each manifest maps filename -> SHA-256 hash.
The RAG service compares these manifests at startup to detect changed docs
and selectively invalidate stale semantic cache entries.

Usage:
    python generate_manifests.py              # default: docs/
    python generate_manifests.py /path/to/docs
"""

import hashlib
import json
import sys
from pathlib import Path


def write_manifest(doc_dir: Path) -> int:
    """Write .content_manifest.json for a single directory.

    Returns number of .md files hashed.
    """
    manifest = {}
    for md_file in sorted(doc_dir.glob("*.md")):
        content = md_file.read_bytes()
        manifest[md_file.name] = hashlib.sha256(content).hexdigest()

    if not manifest:
        return 0

    manifest_path = doc_dir / ".content_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(manifest)


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs")

    if not base.is_dir():
        print(f"ERROR: {base} is not a directory")
        sys.exit(1)

    total_files = 0
    total_dirs = 0

    # Process all subdirectories that contain .md files
    for subdir in sorted(base.iterdir()):
        if subdir.is_dir() and any(subdir.glob("*.md")):
            count = write_manifest(subdir)
            if count:
                print(f"  {subdir.name}: {count} files")
                total_files += count
                total_dirs += 1

    # Also process the base directory itself if it has .md files
    if any(base.glob("*.md")):
        count = write_manifest(base)
        if count:
            print(f"  {base.name}/: {count} files")
            total_files += count
            total_dirs += 1

    print(f"\nManifests written: {total_dirs} directories, {total_files} total files")


if __name__ == "__main__":
    main()
