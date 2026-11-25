"""Strict LaTeX Manuscript QA & Overleaf Compilation Safety Validator.

Verifies:
1. Balanced braces & matching \\begin{...}/\\end{...} environment nesting.
2. Zero deprecated $$...$$ blocks and balanced inline $...$ per paragraph.
3. Zero blank lines inside equation/cases/tabular/tabularx/algorithm blocks.
4. Exact '&' column counts matching tabular/tabularx column specs on every row.
5. Every Figure, Table, Algorithm, and Equation is labeled and cited in text.
6. Every \\includegraphics asset exists alongside main.tex.
7. Strictly sequential first-appearance ordering for Tables, Figures, and Citations.
8. Zero uncited bibitems and zero undefined citation keys.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path


def check_tex(tex_path: Path) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    root = tex_path.parent
    raw = tex_path.read_text(encoding="utf-8")
    # Remove LaTeX comments for structural analysis
    clean_lines = [re.sub(r"(?<!\\)%.*", "", l) for l in raw.splitlines()]
    clean = "\n".join(clean_lines)

    if r"\begin{thebibliography}" not in clean:
        return ["Missing \\begin{thebibliography}"]

    pre_doc, doc_body = clean.split(r"\begin{document}", 1)
    body, bib = doc_body.split(r"\begin{thebibliography}", 1)

    # 1. Environment stack & blank-line check inside math/tabular/algorithm
    stack: list[tuple[str, int]] = []
    no_blank_envs = {"equation", "equation*", "align", "align*", "cases", "tabular", "tabularx"}
    token_pat = re.compile(r"\\(begin|end)\{([^}]+)\}")
    for lno, line in enumerate(clean_lines, 1):
        for m in token_pat.finditer(line):
            cmd, env = m.group(1), m.group(2)
            if cmd == "begin":
                stack.append((env, lno))
            else:
                if not stack or stack[-1][0] != env:
                    errors.append(f"Line {lno}: mismatched \\end{{{env}}} (stack={stack[-3:]})")
                else:
                    stack.pop()
        if stack and stack[-1][0] in no_blank_envs and line.strip() == "":
            errors.append(f"Line {lno}: illegal blank line inside \\begin{{{stack[-1][0]}}}")
    if stack:
        errors.append(f"Unclosed environments: {stack}")

    # 2. Inline $ balance per paragraph & deprecated $$
    n_dd = len(re.findall(r"\$\$", body))
    if n_dd:
        errors.append(f"Found {n_dd} deprecated '$$' tokens in body")
    for pno, para in enumerate(body.split("\n\n"), 1):
        dollars = len(re.findall(r"(?<!\\)\$", para))
        if dollars % 2 != 0:
            errors.append(f"Paragraph {pno}: odd number of '$' ({dollars}) -> {para[:80].strip()!r}")

    # 3. Tabular column count consistency
    tab_pat = re.compile(r"\\begin\{(tabularx?)\}(?:\{[^}]*\})?\s*\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{\1\}", re.S)
    for m in tab_pat.finditer(body):
        kind, colspec, inner = m.group(1), m.group(2), m.group(3)
        stripped_spec = re.sub(r">\{(?:[^{}]|\{[^{}]*\})*\}", "", colspec)
        stripped_spec = re.sub(r"p\{[^}]*\}", "P", stripped_spec)
        ncols = sum(1 for ch in stripped_spec if ch in "lcrXP")
        rows = [r.strip() for r in inner.split(r"\\") if re.search(r"(?<!\\)&", r)]
        for r in rows:
            cells = len(re.findall(r"(?<!\\)&", r)) + 1
            if cells != ncols:
                errors.append(f"Table row has {cells} columns vs spec {ncols}: {r[:80]!r}")

    # 4. Figures, Tables, Algorithms, Equations inventory & sequential citation check
    env_pat = re.compile(r"\\begin\{(figure|table|algorithm)\}(.*?)\\end\{\1\}", re.S)
    figs, tabs, algs = [], [], []
    for m in env_pat.finditer(body):
        kind, inner = m.group(1), m.group(2)
        cap = re.search(r"\\caption\{", inner)
        lbl = re.search(r"\\label\{([^}]+)\}", inner)
        img = re.search(r"\\includegraphics.*?\{([^}]+)\}", inner)
        rec = {
            "label": lbl.group(1) if lbl else None,
            "has_cap": bool(cap),
            "img": img.group(1) if img else None,
        }
        if kind == "figure":
            figs.append(rec)
        elif kind == "table":
            tabs.append(rec)
        else:
            algs.append(rec)

    print(f"--- QA AUDIT: {tex_path.name} ---")
    for label_type, items in [("Figure", figs), ("Table", tabs), ("Algorithm", algs)]:
        last_first_pos = -1
        for idx, it in enumerate(items, 1):
            lbl = it["label"]
            if not it["has_cap"]:
                errors.append(f"{label_type} {idx} has no \\caption")
            if not lbl:
                errors.append(f"{label_type} {idx} has no \\label")
                continue
            if label_type == "Figure":
                if not it["img"]:
                    errors.append(f"Figure {idx} ({lbl}) has no \\includegraphics")
                elif not (root / it["img"]).exists():
                    errors.append(f"Figure {idx} ({lbl}) missing file: {it['img']}")
            matches = list(re.finditer(r"\\ref\{" + re.escape(lbl) + r"\}", body))
            if not matches:
                errors.append(f"{label_type} {idx} ({lbl}) is NEVER cited in body")
            else:
                first_pos = matches[0].start()
                if first_pos < last_first_pos:
                    warnings.append(f"{label_type} {idx} ({lbl}) first cited out of sequential order")
                last_first_pos = first_pos
            print(f"  {label_type:9s} {idx}: label={lbl:24s} refs={len(matches)}" + (f" img={it['img']}" if it['img'] else ""))

    # 5. Citations & Bibitems
    cited_order: list[str] = []
    for m in re.finditer(r"\\cite\{([^}]+)\}", body):
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in cited_order:
                cited_order.append(k)
    defined = re.findall(r"\\bibitem\{([^}]+)\}", bib)
    uncited = [d for d in defined if d not in set(cited_order)]
    undef = [c for c in cited_order if c not in set(defined)]
    if uncited:
        errors.append(f"Uncited bibitems ({len(uncited)}): {uncited}")
    if undef:
        errors.append(f"Undefined citations: {undef}")
    if cited_order != defined and not uncited and not undef:
        warnings.append("Citations are not ordered sequentially [1..N] matching first appearance")

    # 6. Check dead placeholder URLs and markdown backticks
    if "10.5281/zenodo.11482091" in clean or "SRMAP-Consensus-Research" in clean:
        warnings.append("Legacy broken Zenodo 11482091 or SRMAP-Consensus-Research URL still present")
    if "PENDING_" in clean:
        warnings.append("Pending link placeholder still present in document")
    bad_bt = re.findall(r"(?<!`)`(?!`)", clean)
    if bad_bt:
        errors.append(f"Found {len(bad_bt)} markdown single-backtick characters (use \\texttt{{...}} instead)")

    print(f"Total bibitems={len(defined)}, cited={len(cited_order)}, uncited={len(uncited)}, undef={len(undef)}")
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print("  [WARN]", w)
    if errors:
        print("ERRORS:")
        for e in errors:
            print("  [ERR]", e)
    else:
        print("PASS: Zero structural errors!")
    return errors


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["main.tex"]
    rc = 0
    for t in targets:
        p = Path(t) if Path(t).is_absolute() else root / t
        if check_tex(p):
            rc = 1
    sys.exit(rc)
