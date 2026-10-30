"""Stage linked repository documents so local and built-site links both work.

MkDocs invokes this hook; it never executes notebook code or changes the repo.
Only documentation and its transitive local file links enter the temporary tree.
"""
from __future__ import annotations

import atexit
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

from mkdocs.exceptions import PluginError


_STAGING = []


def on_config(config):
    root = Path(config.config_file_path).resolve().parent
    temporary = TemporaryDirectory(prefix="timegym-docs-")
    _STAGING.append(temporary)
    atexit.register(temporary.cleanup)
    stage = Path(temporary.name)
    pending = [root / "README.md", *sorted((root / "docs").glob("*.md"))]
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
            for target in re.findall(r"\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
                parts = urlsplit(target.strip("<>"))
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
