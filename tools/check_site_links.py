"""Report relative links in a built MkDocs site that point at missing files.

`mkdocs build --strict` validates Markdown links but not the HTML that
mkdocs-jupyter renders from notebooks. Run this after a build:

    python tools/check_site_links.py site

It exits 1 and lists each broken link when any relative `href` or `src`
resolves to nothing inside the site directory.
"""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


class _Targets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.targets.append(value)


def broken_links(site: Path) -> list[tuple[str, str]]:
    site = site.resolve()
    broken = []
    for page in sorted(site.rglob("*.html")):
        parser = _Targets()
        parser.feed(page.read_text(encoding="utf-8"))
        for target in sorted(set(parser.targets)):
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
                continue
            resolved = (page.parent / unquote(parts.path)).resolve()
            if parts.path.endswith("/"):
                resolved = resolved / "index.html"
            if not resolved.is_relative_to(site) or not resolved.exists():
                broken.append((page.relative_to(site).as_posix(), target))
    return broken


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("site", nargs="?", default="site", type=Path)
    args = parser.parse_args(argv)
    if not any(args.site.rglob("*.html")):
        print(f"No HTML pages under {args.site}; build the site first", file=sys.stderr)
        return 1
    broken = broken_links(args.site)
    for page, target in broken:
        print(f"{page}: {target}")
    if broken:
        print(f"{len(broken)} broken relative links", file=sys.stderr)
        return 1
    print(f"No broken relative links in {args.site}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
