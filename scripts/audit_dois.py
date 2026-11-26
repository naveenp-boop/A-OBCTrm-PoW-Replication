"""Dynamically audit all DOI references in main.tex against the CrossRef API."""
from __future__ import annotations
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def fetch_crossref(doi: str, retries: int = 3) -> dict:
    url = f"https://api.crossref.org/works/{doi}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CitationAudit/1.0 (mailto:naveen_p@srmap.edu.in)"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read().decode("utf-8"))["message"]
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise


def main() -> int:
    tex_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[2] / "main.tex"
    )
    if not tex_path.exists():
        print(f"ERROR: TeX file not found: {tex_path}")
        return 1

    text = tex_path.read_text(encoding="utf-8")
    if r"\begin{thebibliography}" not in text:
        print("ERROR: Missing bibliography section")
        return 1

    bib = text.split(r"\begin{thebibliography}", 1)[1]
    entries = re.split(r"\\bibitem\{([^}]+)\}", bib)
    total = 0
    doi_count = 0
    errors: list[str] = []

    for idx, j in enumerate(range(1, len(entries), 2), 1):
        total += 1
        key = entries[j]
        body = " ".join(entries[j + 1].split())
        dm = re.search(r"DOI:\s*(10\.\d{4,9}/[^\s]+)", body)
        if not dm:
            print(f"[{idx:2d}] {key:22s} (no DOI)")
            continue

        doi = dm.group(1).replace(r"\_", "_").rstrip(".")
        doi_count += 1
        try:
            d = fetch_crossref(doi)
            authors = [a.get("family", "") for a in d.get("author", [])[:2]]
            title = (d.get("title") or [""])[0]
            print(f"[{idx:2d}] {key:22s} OK  {doi} -> {authors} | {title[:60]}")
        except Exception as e:
            msg = f"[{idx:2d}] {key} ({doi}) -> ERROR: {e}"
            print(msg)
            errors.append(msg)
        time.sleep(0.15)

    print(f"Summary: {total} references checked, {doi_count} DOIs queried, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
