"""Stage linked repository documents so local and built-site links both work.

MkDocs invokes this hook; it never executes notebook code or changes the repo.
Only documentation and its transitive local file links enter the temporary tree.
"""
from __future__ import annotations

import atexit
from pathlib import Path
from html.parser import HTMLParser
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

from markdown import markdown
from mkdocs.exceptions import PluginError


_STAGING = []


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        attribute = {"a": "href", "img": "src"}.get(tag)
        if attribute:
            target = dict(attrs).get(attribute)
            if target:
                self.targets.append(target)


def local_link_targets(source: str) -> list[str]:
    # Markdown escapes fenced/inline code and separates optional link titles.
    parser = _Links()
    parser.feed(markdown(source, extensions=["fenced_code"]))
    return parser.targets


def on_config(config):
    root = Path(config.config_file_path).resolve().parent
    temporary = TemporaryDirectory(prefix="timegym-docs-")
    _STAGING.append(temporary)
    atexit.register(temporary.cleanup)
    stage = Path(temporary.name)
    pending = [root / "README.md", *sorted((root / "docs").glob("*.md")),
               *sorted((root / "notebooks").glob("00_*.ipynb")), *sorted((root / "notebooks").glob("2[0-9]_*.ipynb")),
               *sorted((root / "notebooks").glob("3[0-9]_*.ipynb"))]
    copied = set()
    while pending:
        source = pending.pop()
        resolved = source.resolve()
        if not resolved.is_relative_to(root) or source.is_symlink():
            raise PluginError(f"Documentation link leaves the source tree: {source}")
        if not source.is_file():
            raise PluginError(f"Missing linked documentation file: {source}")
        if resolved in copied:
            continue
        copied.add(resolved)
        destination = stage / resolved.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolved, destination)
        if source.suffix == ".md":
            for target in local_link_targets(source.read_text(encoding="utf-8")):
                parts = urlsplit(target)
                if parts.scheme or parts.netloc or not parts.path:
                    continue
                pending.append(source.parent / unquote(parts.path))
    config["docs_dir"] = str(stage)
    # `serve` otherwise watches only the temporary copies. Watch original
    # directories as well, so edits and new linked files trigger fresh staging.
    watch = list(config.get("watch", []))
    for path in (root / "README.md", root / "docs", root / "notebooks",
                 root / "examples", root / "experiments", root / "paper",
                 root / "tools", root / "CHANGELOG.md", root / "CITATION.cff"):
        if path.exists() and str(path) not in watch:
            watch.append(str(path))
    config["watch"] = watch
    return config
