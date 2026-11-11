"""Stage linked repository documents so local and built-site links both work.

MkDocs invokes this hook; it never executes notebook code or changes the repo.
Only documentation and its transitive local file links enter the temporary tree.
Rendered notebooks keep their repository-relative links, so the hook also
points those links at the built pages, or at GitHub for files the site does not
render. Pages that moved keep their old URLs as small redirect pages.
"""
from __future__ import annotations

import atexit
from pathlib import Path, PurePosixPath
from html.parser import HTMLParser
import posixpath
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.parse import quote, unquote, urlsplit

from markdown import markdown
from mkdocs.exceptions import PluginError
from mkdocs.utils import get_relative_url


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
    pending = [root / "README.md", *[path for path in [root / "CHANGELOG.md"] if path.exists()],
               *sorted((root / "docs").rglob("*.md")), *sorted((root / "docs").rglob("*.css")),
               *sorted((root / "tutorials").glob("[0-9][0-9]_*.ipynb"))]
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
                if raw_prefix(config) and target.startswith(raw_prefix(config)):
                    # README images use raw GitHub URLs so PyPI can show them;
                    # the site serves its own copy, which must exist.
                    pending.append(root / unquote(target[len(raw_prefix(config)):]))
                    continue
                parts = urlsplit(target)
                if parts.scheme or parts.netloc or not parts.path:
                    continue
                pending.append(source.parent / unquote(parts.path))
    config["docs_dir"] = str(stage)
    # `serve` otherwise watches only the temporary copies. Watch original
    # directories as well, so edits and new linked files trigger fresh staging.
    watch = list(config.get("watch", []))
    for path in (root / "README.md", root / "docs", root / "tutorials", root / "source",
                 root / "examples", root / "experiments", root / "paper",
                 root / "tools", root / "CHANGELOG.md", root / "CITATION.cff"):
        if path.exists() and str(path) not in watch:
            watch.append(str(path))
    config["watch"] = watch
    return config


def raw_prefix(config):
    """The raw.githubusercontent.com address of the main branch, from repo_url."""
    repo = (config.get("repo_url") or "").rstrip("/")
    if not repo.startswith("https://github.com/"):
        return ""
    return repo.replace("https://github.com/", "https://raw.githubusercontent.com/", 1) + "/main/"


def on_page_markdown(markdown, page, config, files):
    """Serve README images from the site rather than from GitHub."""
    prefix = raw_prefix(config)
    if not prefix or prefix not in markdown:
        return markdown
    here = posixpath.dirname(page.file.src_uri)
    return re.sub(
        re.escape(prefix) + r"([^)\s\"']+)",
        lambda match: posixpath.relpath(match.group(1), here or "."),
        markdown,
    )


_HREF = re.compile(r'(<a\b[^>]*?\bhref=")([^"]*)(")')


def notebook_link(href, page_src, page_url, files, repo_url):
    """Return the built-site form of a link written relative to a notebook.

    A link to a rendered page becomes a relative page URL; a link to any other
    repository file becomes a GitHub URL. External, absolute and in-page links
    are returned unchanged.
    """
    parts = urlsplit(href)
    if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
        return href
    target = posixpath.normpath(str(PurePosixPath(page_src).parent / unquote(parts.path)))
    if target == ".." or target.startswith("../"):
        return href
    fragment = f"#{parts.fragment}" if parts.fragment else ""
    linked = files.get_file_from_path(target)
    if linked is not None and linked.is_documentation_page():
        return get_relative_url(linked.url, page_url) + fragment
    if not repo_url:
        return href
    return f"{repo_url.rstrip('/')}/blob/main/{quote(target)}{fragment}"


def on_page_content(html, page, config, files):
    if not page.file.src_uri.endswith(".ipynb"):
        return html
    repo_url = config.get("repo_url")
    return _HREF.sub(
        lambda match: match.group(1)
        + notebook_link(match.group(2), page.file.src_uri, page.url, files, repo_url)
        + match.group(3),
        html,
    )


# Tutorials lived in notebooks/ with other numbers until the tutorials and the
# source notebooks were split; old links (the 1.1.0 PyPI page among them) keep working.
MOVED_PAGES = {
    "notebooks/00_quick_start/": "tutorials/01_quick_start/",
    "notebooks/28_custom_model/": "tutorials/02_custom_model/",
    "notebooks/23_faults_repairs_and_replay/": "tutorials/03_faults_repairs_and_replay/",
    "notebooks/29_target_transforms/": "tutorials/04_target_transforms/",
    "notebooks/30_reductions/": "tutorials/05_reductions/",
    "notebooks/21_multivariate_forecasting/": "tutorials/06_multivariate_forecasting/",
    "notebooks/24_real_data_forecasts/": "tutorials/07_real_data_forecasts/",
    "notebooks/20_univariate_uncertainty/": "tutorials/08_univariate_uncertainty/",
    "notebooks/22_multivariate_uncertainty/": "tutorials/09_multivariate_uncertainty/",
    "notebooks/25_capability_sweeps/": "tutorials/10_capability_sweeps/",
    "notebooks/26_composed_and_financial_stresses/": "tutorials/11_composed_and_financial_stresses/",
    "notebooks/27_joint_financial_uncertainty/": "tutorials/12_joint_financial_uncertainty/",
    "docs/notebooks/": "docs/tutorials/",
}

_REDIRECT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Moved</title>
<meta name="robots" content="noindex">
<link rel="canonical" href="{absolute}">
<meta http-equiv="refresh" content="0; url={relative}">
</head>
<body><p>This page moved to <a href="{relative}">{relative}</a>.</p></body>
</html>
"""


def on_post_build(config):
    site = Path(config["site_dir"])
    base = (config.get("site_url") or "").rstrip("/") + "/"
    for old, new in MOVED_PAGES.items():
        if not (site / new / "index.html").is_file():
            raise PluginError(f"Redirect target is not a built page: {new}")
        page = site / old / "index.html"
        if page.exists():
            raise PluginError(f"Redirect would overwrite a built page: {old}")
        page.parent.mkdir(parents=True, exist_ok=True)
        relative = posixpath.relpath(new, old) + "/"
        page.write_text(_REDIRECT.format(absolute=base + new, relative=relative), encoding="utf-8")
