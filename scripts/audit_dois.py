"""Audit all DOI references in main.tex against CrossRef API."""
import json
import urllib.request

DOIS = [
    ("8", "10.1007/s12083-026-02206-w"),
    ("14", "10.1007/s11042-024-18874-w"),
    ("20", "10.1007/s11276-024-03851-w"),
    ("22", "10.1109/TCE.2025.3647809"),
    ("31", "10.1109/JSEN.2026.3678320"),
    ("34", "10.1002/spy2.70052"),
    ("35", "10.1109/ICCMC69250.2026.11625095"),
    ("37", "10.1002/cpe.70077"),
    ("40", "10.1038/s41598-026-37402-4"),
    ("43", "10.1109/ICACIC59454.2023.10435217"),
    ("47", "10.1109/OCIT65031.2024.00102"),
    ("51", "10.1002/cpe.8211"),
    ("59", "10.1109/TKDE.2026.3677426"),
    ("60", "10.1109/TSMC.2025.3646727"),
    ("61", "10.1109/TMC.2025.3571431"),
]

for num, doi in DOIS:
    try:
        req = urllib.request.Request(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "CitationAudit/1.0 (mailto:naveen_p@srmap.edu.in)"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read().decode("utf-8"))["message"]
            authors = [f"{a.get('family', '')}, {a.get('given', '')}" for a in d.get("author", [])]
            title = d.get("title", [""])[0]
            print(f"[{num}] {doi} | Authors={authors} | Title={title}")
    except Exception as e:
        print(f"[{num}] {doi} -> ERROR: {e}")
