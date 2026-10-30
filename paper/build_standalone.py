"""Inline generated results for a self-contained LaTeX preview.

Run from any directory: python paper/build_standalone.py
The source remains main.tex. The generated timegym-v2.tex needs no BibTeX,
external figures, or additional files and is suitable for the app compiler.
"""

from pathlib import Path
import re


def inline_macros(manuscript: str, paper_dir: Path) -> str:
    """Resolve the caption macro input, including its optional existence guard."""
    name = "generated-macros.tex"
    token = r"\input{" + name + "}"
    if token not in manuscript:
        return manuscript
    path = paper_dir / name
    if not path.is_file():
        raise SystemExit(f"Missing paper/{name}; render the supplement first")
    macros = path.read_text(encoding="utf-8")

    def group(start):
        while start < len(manuscript) and manuscript[start].isspace():
            start += 1
        if start == len(manuscript) or manuscript[start] != "{":
            raise SystemExit("Malformed generated-macros existence guard")
        depth, end = 1, start + 1
        while end < len(manuscript) and depth:
            if manuscript[end] in "{}" and manuscript[end-1] != "\\":
                depth += 1 if manuscript[end] == "{" else -1
            end += 1
        if depth:
            raise SystemExit("Unclosed generated-macros existence guard")
        return manuscript[start+1:end-1], end

    guard = re.search(r"\\IfFileExists\s*\{generated-macros\.tex\}", manuscript)
    if guard:
        present, end = group(guard.end())
        _, end = group(end)
        manuscript = manuscript[:guard.start()] + present + manuscript[end:]
    return manuscript.replace(token, macros)


def build() -> Path:
    paper_dir = Path(__file__).resolve().parent
    manuscript = (paper_dir / "main.tex").read_text(encoding="utf-8")
    standalone = inline_macros(manuscript, paper_dir)
    for name, marker in (("generated-results.tex", "RESULTS"),
                         ("generated-supplement.tex", "SUPPLEMENT")):
        results_path = paper_dir / name
        if not results_path.is_file():
            raise SystemExit(f"Missing paper/{name}. Render both frozen studies before building the manuscript.")
        start = f"% BEGIN GENERATED {marker}"
        end = f"% END GENERATED {marker}"
        if standalone.count(start) != 1 or standalone.count(end) != 1:
            raise SystemExit(f"Expected exactly one generated-{marker.lower()} block in main.tex")
        before, remaining = standalone.split(start, 1)
        _, after = remaining.split(end, 1)
        standalone = before + start + "\n" + results_path.read_text(encoding="utf-8") + "\n" + end + after
    # Keep vector figures self-contained: the editor compiler deliberately has
    # no access to companion graphics files. Conventional main.tex uses inputs.
    names = re.findall(r"% BEGIN VECTOR FIGURE ([a-z-]+)", standalone)
    if len(names) != len(set(names)):
        raise SystemExit("Duplicate vector-figure identifiers")
    for name in names:
        start, end = f"% BEGIN VECTOR FIGURE {name}", f"% END VECTOR FIGURE {name}"
        if standalone.count(start) != 1 or standalone.count(end) != 1:
            raise SystemExit(f"Expected one complete vector-figure block for {name}")
        figure = paper_dir / "figures" / f"{name}.tikz"
        if not figure.is_file():
            raise SystemExit(f"Missing {figure.name}; run python paper/render_figures.py first")
        before, remaining = standalone.split(start, 1)
        _, after = remaining.split(end, 1)
        standalone = before + start + "\n" + figure.read_text(encoding="utf-8") + "\n" + end + after
    output = paper_dir / "timegym-v2.tex"
    output.write_text(standalone, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build())
