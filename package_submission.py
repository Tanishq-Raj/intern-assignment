"""Helper script to bundle a clean submission zip archive for the assignment."""

from __future__ import annotations

import os
from pathlib import Path
import zipfile

OUTPUT_ZIP = Path("alpha_law_submission.zip")
EXCLUDED_EXTENSIONS = {".pyc", ".pyo", ".db", ".pkl", ".npz", ".log"}
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".venv", "venv", ".git", ".ruff_cache"}

ROOT_FILES = [
    "README.md",
    "NOTES.md",
    "STANDARDS.md",
    "requirements.txt",
    ".gitignore",
    ".env.example",
    "pytest.ini",
    "ingest.py",
    "run_tests.py",
]

TARGET_DIRECTORIES = ["api", "rag", "tests"]


def create_submission_zip() -> None:
    """Bundle project files into a clean archive without binaries or caches."""
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    included_entries: list[str] = []

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add root configuration and documentation files
        for filename in ROOT_FILES:
            file_path = Path(filename)
            if file_path.exists():
                zf.write(file_path, arcname=filename)
                included_entries.append(filename)

        # Add original source dataset only
        dataset_csv = Path("data/netflix_titles.csv")
        if dataset_csv.exists():
            zf.write(dataset_csv, arcname="data/netflix_titles.csv")
            included_entries.append("data/netflix_titles.csv")

        # Add source code and test trees
        for dir_name in TARGET_DIRECTORIES:
            base_dir = Path(dir_name)
            for root, dirs, files in os.walk(base_dir):
                # Filter out excluded subdirectories in place
                dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
                for f in files:
                    file_path = Path(root) / f
                    if file_path.suffix.lower() not in EXCLUDED_EXTENSIONS:
                        rel_path = file_path.as_posix()
                        zf.write(file_path, arcname=rel_path)
                        included_entries.append(rel_path)

    print("=" * 60)
    print(f"Submission archive created: {OUTPUT_ZIP.name}")
    print(f"Archive size: {OUTPUT_ZIP.stat().st_size:,} bytes")
    print(f"Total bundled files: {len(included_entries)}")
    print("=" * 60)
    for entry in sorted(included_entries):
        print(f"  [+] {entry}")
    print("=" * 60)


if __name__ == "__main__":
    create_submission_zip()
