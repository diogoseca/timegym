#!/usr/bin/env python3
"""Print the abstract as plain text, with the generated numbers filled in, for the arXiv form.

    python paper/plain_abstract.py

The abstract of main.tex quotes numbers through macros of generated-macros.tex. arXiv takes the
abstract as plain text of at most 1,920 characters, so this script fills the macros in, prints
the text and its length, and fails when a macro is undefined, when LaTeX markup is left, or
when the text is too long. It uses only the Python standard library.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
ARXIV_LIMIT = 1920


def plain_abstract(paper_dir: Path = PAPER_DIR) -> str:
    manuscript = (paper_dir / "main.tex").read_text(encoding="utf-8")
    if manuscript.count(r"\begin{abstract}") != 1 or manuscript.count(r"\end{abstract}") != 1:
        raise ValueError("Expected one abstract in main.tex")
    body = manuscript.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                             (paper_dir / "generated-macros.tex").read_text(encoding="utf-8")))

    def fill(match):
        if match.group(1) not in macros:
            raise ValueError(f"The abstract uses \\{match.group(1)}, which generated-macros.tex does not define")
        return macros[match.group(1)]

    text = " ".join(re.sub(r"\\([A-Za-z]+)\{\}", fill, body).split())
    left = re.findall(r"\\[A-Za-z]+|[{}$~]", text)
    if left:
        raise ValueError("LaTeX markup left in the abstract: " + ", ".join(sorted(set(left))))
    if len(text) > ARXIV_LIMIT:
        raise ValueError(f"The abstract has {len(text):,} characters; arXiv allows {ARXIV_LIMIT:,}")
    return text


def main() -> int:
    try:
        text = plain_abstract()
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(text)
    print(f"\n{len(text):,} characters (arXiv allows {ARXIV_LIMIT:,})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
