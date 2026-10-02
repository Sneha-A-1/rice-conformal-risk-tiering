"""Static consistency checks for the manuscript (run after make_tables.py and a LaTeX build).

Reports: generated macros used but not defined, placeholders left in the text, citation keys missing from the
bibliography, labels referenced but not defined, unused bibliography entries, and LaTeX warnings about undefined
references/citations in manuscript/main.log.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1] / "manuscript"


def expand(path, seen=None):
    seen = seen or set()
    text = path.read_text()
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            continue
        m = re.search(r"\\input\{([^}]+)\}", line)
        if m and not m.group(1).startswith("generated/numbers"):
            p = ROOT / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
            if p.exists() and p not in seen:
                seen.add(p)
                out.append(expand(p, seen))
                continue
        out.append(line)
    return "\n".join(out)


def main():
    tex = expand(ROOT / "main.tex")
    gen = (ROOT / "generated" / "numbers.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", gen))
    local = set(re.findall(r"\\(?:def|newcommand\{)\\?([A-Za-z]+)", tex))
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)", tex))
    latex_builtin = {"Require", "State", "If", "ElsIf", "Else", "EndIf", "For", "EndFor", "While", "EndWhile", "Return",
                     "Procedure", "EndProcedure", "Comment", "Authors", "Address", "TeX", "LaTeX", "Pr", "Huge", "Large",
                     "LARGE", "Big", "Bigg", "Delta", "Gamma", "Lambda", "Omega", "Phi", "Pi", "Psi", "Sigma", "Theta",
                     "Upsilon", "Xi", "Leftrightarrow", "Rightarrow", "Leftarrow"}
    missing = sorted(u for u in used - defined - local - latex_builtin if not u.startswith("Begin"))
    print("undefined generated macros:", missing or "none")
    ph = re.findall(r"\[\[[^\]]*\]\]", tex)
    print("placeholders:", len(ph), ph[:10])
    bib = (ROOT / "references.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cites = set()
    for c in re.findall(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", tex):
        cites |= {k.strip() for k in c.split(",")}
    print("citations missing from bib:", sorted(cites - keys) or "none")
    print("bib entries never cited:", sorted(keys - cites) or "none")
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    for f in (ROOT / "generated").glob("tab_*.tex"):
        labels |= set(re.findall(r"\\label\{([^}]+)\}", f.read_text()))
    refs = set()
    for r in re.findall(r"\\(?:eq)?ref\{([^}]+)\}", tex):
        refs.add(r)
    print("refs to undefined labels:", sorted(refs - labels) or "none")
    log = ROOT / "main.log"
    if log.exists():
        L = log.read_text(errors="ignore")
        und = re.findall(r"(?:Reference|Citation) `([^']+)' .*undefined", L)
        print("LaTeX undefined refs/cites:", sorted(set(und)) or "none")
        print("overfull hboxes > 5pt:", len([x for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt", L) if float(x) > 5]))
    words = len(re.findall(r"[A-Za-z]+", re.sub(r"\\[A-Za-z]+|\$[^$]*\$", " ", tex)))
    print("approximate word count (all text incl. captions):", words)
    return 1 if missing or ph else 0


if __name__ == "__main__":
    sys.exit(main())
