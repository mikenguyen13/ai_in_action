"""Give every page of the rendered book one canonical address.

Quarto writes a sitemap from ``book.site-url`` but no ``<link rel="canonical">``.
This script adds one to the head of each HTML page, pointing at the page's URL
on the published site (``book.site-url`` in _quarto.yml), so copies served from
other hosts or paths (``index.html`` versus ``/``, http, mirrors, previews)
consolidate on that address. It also rewrites the sitemap's ``.../index.html``
entry to the site root, so the sitemap and the canonical link agree.

Idempotent; runs as a Quarto post-render step, or by hand over an existing site:

    python scripts/canonical_links.py docs
"""
from __future__ import annotations

import html
import os
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- canonical:begin -->", "<!-- canonical:end -->"
# Pages that should not be indexed under a canonical address.
SKIP = {"404.html", "search.html"}


def site_url() -> str:
    book = yaml.safe_load((ROOT / "_quarto.yml").read_text(encoding="utf-8"))["book"]
    return book["site-url"].rstrip("/") + "/"


def page_url(site: str, rel: str) -> str:
    if rel == "index.html":
        return site
    if rel.endswith("/index.html"):
        return site + rel[: -len("index.html")]
    return site + rel


def process(path: pathlib.Path, out_dir: pathlib.Path, site: str) -> bool:
    rel = path.relative_to(out_dir).as_posix()
    if rel in SKIP or rel.startswith("site_libs/"):
        return False
    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    if "</head>" not in text:
        return False
    text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", "", text, flags=re.S)
    tag = f'{BEGIN}\n<link rel="canonical" href="{html.escape(page_url(site, rel), quote=True)}">\n{END}'
    text = text.replace("</head>", tag + "\n</head>", 1)
    path.write_text(text, encoding="utf-8", newline="")
    return True


def fix_sitemap(out_dir: pathlib.Path, site: str) -> None:
    sitemap = out_dir / "sitemap.xml"
    if not sitemap.exists():
        return
    with open(sitemap, encoding="utf-8", newline="") as fh:
        text = fh.read()
    # Quarto merges a re-render into the existing sitemap by <loc>, so after an
    # earlier run there can be both a ".../" and a ".../index.html" entry: keep
    # one root entry, at the first position, with the latest <lastmod>.
    pattern = re.compile(r"[ \t]*<url>\s*<loc>(.*?)</loc>(.*?)</url>\n?", re.S)
    roots = {f"{site}index.html", site}
    entries = [m for m in pattern.finditer(text) if m.group(1) in roots]
    if not entries:
        return
    lastmods = [re.search(r"<lastmod>(.*?)</lastmod>", m.group(2)) for m in entries]
    lastmod = max((lm.group(1) for lm in lastmods if lm), default=None)
    root = f"  <url>\n    <loc>{site}</loc>\n" + (f"    <lastmod>{lastmod}</lastmod>\n" if lastmod else "") + "  </url>\n"
    first = entries[0]
    fixed = text[: first.start()] + root + pattern.sub(
        lambda m: "" if m.group(1) in roots else m.group(0), text[first.end():]
    )
    if fixed != text:
        sitemap.write_text(fixed, encoding="utf-8", newline="")


def main() -> None:
    site = site_url()
    if len(sys.argv) > 1:
        out_dir = pathlib.Path(sys.argv[1]).resolve()
        files = sorted(out_dir.rglob("*.html"))
    else:
        # Quarto post-render: only the pages this render produced.
        out_dir = ROOT / os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "docs")
        listed = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "")
        files = [pathlib.Path(p) for p in listed.splitlines() if p.endswith(".html")] or sorted(out_dir.rglob("*.html"))
        files = [p if p.is_absolute() else ROOT / p for p in files]
    out_dir = out_dir.resolve()
    done = sum(process(p.resolve(), out_dir, site) for p in files if p.exists())
    fix_sitemap(out_dir, site)
    print(f"canonical_links: {done} pages")


if __name__ == "__main__":
    main()
