"""Make every page of the rendered book citable.

Reads the canonical citation from CITATION.cff and injects into each HTML page:

* Highwire Press ``citation_*`` and Dublin Core meta tags (what Google Scholar,
  Zotero, Mendeley, and other reference managers read),
* a ``rel="cite-as"`` link to the DOI (RFC 8574) once a DOI exists,
* schema.org JSON-LD for the chapter as part of the book,
* a visible "How to cite" box (APA and BibTeX) at the end of the page.

The preface (index.html) is cited as the book; chapters are cited as chapters
in the book, so every citation points at the same work and DOI. The script is
idempotent and runs as a Quarto post-render step; it can also be run by hand
over an existing site without re-rendering:

    python scripts/citation_meta.py docs
"""
from __future__ import annotations

import html
import json
import os
import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- citation-meta:begin -->", "<!-- citation-meta:end -->"
BOX_BEGIN, BOX_END = "<!-- citation-box:begin -->", "<!-- citation-box:end -->"
# Pages that are not part of the citable work.
SKIP = {"about.html", "privacy.html", "cookies.html", "search.html", "404.html"}


def load_work() -> dict:
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    pref = cff.get("preferred-citation", cff)
    authors = [
        {
            "family": a["family-names"],
            "given": a["given-names"],
            "orcid": (a.get("orcid") or "").rstrip("/"),
        }
        for a in pref["authors"]
    ]
    return {
        "title": pref["title"],
        "authors": authors,
        "year": str(pref.get("year") or cff["date-released"][:4]),
        "date": cff["date-released"],
        "url": pref.get("url") or cff["url"],
        "doi": pref.get("doi") or cff.get("doi") or "",
        "abstract": cff.get("abstract", ""),
        "keywords": cff.get("keywords", []),
    }


def page_title(text: str) -> str | None:
    """Chapter title from the rendered h1, without the (unstable) number."""
    m = re.search(r'<h1 class="title">(.*?)</h1>', text, re.S)
    if not m:
        return None
    inner = m.group(1)
    t = re.search(r'<span class="chapter-title">(.*?)</span>', inner, re.S)
    raw = t.group(1) if t else inner
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def apa_authors(authors: list[dict]) -> str:
    names = [f"{a['family']}, {' '.join(p[0] + '.' for p in a['given'].split())}" for a in authors]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def bib_key(work: dict, chapter: str | None, stem: str) -> str:
    base = f"{work['authors'][0]['family'].lower()}{work['year']}aiinaction"
    return base if chapter is None else f"{base}-{re.sub(r'[^a-z0-9]+', '', stem.lower())}"


def references(work: dict, chapter: str | None, url: str, stem: str) -> tuple[str, str]:
    """APA (HTML) and BibTeX for the book or for one chapter."""
    link = f"https://doi.org/{work['doi']}" if work["doi"] else work["url"]
    auth = apa_authors(work["authors"])
    if chapter is None:
        apa = f"{auth} ({work['year']}). <em>{html.escape(work['title'])}</em>. {link}"
    else:
        apa = (
            f"{auth} ({work['year']}). {html.escape(chapter)}. In <em>{html.escape(work['title'])}</em>. "
            f"{url}" + (f" ({link})" if work["doi"] else "")
        )
    who = " and ".join(f"{a['family']}, {a['given']}" for a in work["authors"])
    fields = [("author", who)]
    if chapter is None:
        kind, fields = "book", fields + [("title", "{" + work["title"] + "}")]
    else:
        kind = "incollection"
        fields += [("title", "{" + chapter + "}"), ("booktitle", "{" + work["title"] + "}")]
    fields += [("year", work["year"])]
    if work["doi"]:
        fields += [("doi", work["doi"])]
    fields += [("url", url), ("note", "Online book")]
    width = max(len(k) for k, _ in fields)
    body = ",\n".join(f"  {k.ljust(width)} = {{{v}}}" for k, v in fields)
    return apa, f"@{kind}{{{bib_key(work, chapter, stem)},\n{body}\n}}"


def head_block(work: dict, chapter: str | None, url: str) -> str:
    m = []

    def meta(name: str, value: str) -> None:
        m.append(f'<meta name="{name}" content="{html.escape(value, quote=True)}">')

    meta("citation_title", chapter or work["title"])
    for a in work["authors"]:
        meta("citation_author", f"{a['family']}, {a['given']}")
        if a["orcid"]:
            meta("citation_author_orcid", a["orcid"])
    if chapter is not None:
        meta("citation_inbook_title", work["title"])
        meta("citation_book_title", work["title"])
    meta("citation_publication_date", work["year"])
    meta("citation_online_date", work["date"].replace("-", "/"))
    meta("citation_publisher", f"{work['authors'][0]['given']} {work['authors'][0]['family']}")
    meta("citation_fulltext_html_url", url)
    meta("citation_abstract_html_url", url)
    meta("citation_language", "en")
    if work["keywords"]:
        meta("citation_keywords", "; ".join(work["keywords"]))
    if work["doi"]:
        meta("citation_doi", work["doi"])
        m.append(f'<link rel="cite-as" href="https://doi.org/{work["doi"]}">')
    meta("DC.title", chapter or work["title"])
    for a in work["authors"]:
        meta("DC.creator", f"{a['family']}, {a['given']}")
    meta("DC.date", work["year"])
    meta("DC.type", "Text")
    meta("DC.identifier", f"https://doi.org/{work['doi']}" if work["doi"] else url)
    if chapter is not None:
        meta("DC.relation.isPartOf", work["title"])

    person = [
        {"@type": "Person", "name": f"{a['given']} {a['family']}", **({"sameAs": a["orcid"]} if a["orcid"] else {})}
        for a in work["authors"]
    ]
    book = {
        "@type": "Book",
        "name": work["title"],
        "url": work["url"],
        "author": person,
        "datePublished": work["date"],
        **({"identifier": f"https://doi.org/{work['doi']}", "sameAs": f"https://doi.org/{work['doi']}"} if work["doi"] else {}),
    }
    if chapter is None:
        ld = {"@context": "https://schema.org", **book, "description": work["abstract"]}
    else:
        ld = {"@context": "https://schema.org", "@type": "Chapter", "name": chapter, "url": url, "author": person, "isPartOf": book}
    m.append('<script type="application/ld+json">' + json.dumps(ld, ensure_ascii=False) + "</script>")
    return BEGIN + "\n" + "\n".join(m) + "\n" + END


def cite_box(work: dict, chapter: str | None, url: str, stem: str) -> str:
    apa, bib = references(work, chapter, url, stem)
    what = "this book" if chapter is None else "this chapter"
    return f"""{BOX_BEGIN}
<section id="cite-this-page" class="cite-this-page">
<h2 class="anchored">How to cite {what}</h2>
<p class="cite-apa">{apa}</p>
<details><summary>BibTeX</summary>
<pre class="cite-bibtex"><code>{html.escape(bib)}</code></pre>
</details>
</section>
<style>
.cite-this-page {{ margin-top: 2.5rem; padding: 1rem 1.25rem; border-left: 4px solid var(--bs-primary, #2780e3);
  background: rgba(127, 127, 127, .07); border-radius: .25rem; }}
.cite-this-page h2 {{ font-size: 1.15rem; margin-top: 0; border-bottom: 0; }}
.cite-this-page .cite-apa {{ margin-bottom: .5rem; padding-left: 2em; text-indent: -2em; }}
.cite-this-page summary {{ cursor: pointer; }}
.cite-this-page pre, .cite-this-page pre code {{ margin: .5rem 0 0; white-space: pre-wrap; word-break: break-all; }}
</style>
{BOX_END}"""


def strip(text: str, begin: str, end: str) -> str:
    return re.sub(re.escape(begin) + r".*?" + re.escape(end) + r"\n?", "", text, flags=re.S)


def process(path: pathlib.Path, out_dir: pathlib.Path, work: dict) -> bool:
    rel = path.relative_to(out_dir).as_posix()
    if rel in SKIP or "/" in rel:
        return False
    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    if "</head>" not in text or "</main>" not in text:
        return False
    text = strip(strip(text, BEGIN, END), BOX_BEGIN, BOX_END)
    chapter = None if rel == "index.html" else page_title(text)
    if rel != "index.html" and not chapter:
        return False
    url = work["url"] if rel == "index.html" else work["url"].rstrip("/") + "/" + rel
    text = text.replace("</head>", head_block(work, chapter, url) + "\n</head>", 1)
    text = text.replace("</main>", cite_box(work, chapter, url, path.stem) + "\n</main>", 1)
    path.write_text(text, encoding="utf-8", newline="")
    return True


def main() -> None:
    work = load_work()
    if len(sys.argv) > 1:
        out_dir = pathlib.Path(sys.argv[1]).resolve()
        files = sorted(out_dir.glob("*.html"))
    else:
        # Quarto post-render: only the pages this render produced.
        out_dir = ROOT / os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "docs")
        listed = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "")
        files = [pathlib.Path(p) for p in listed.splitlines() if p.endswith(".html")] or sorted(out_dir.glob("*.html"))
        files = [p if p.is_absolute() else ROOT / p for p in files]
    done = sum(process(p.resolve(), out_dir.resolve(), work) for p in files if p.exists())
    print(f"citation_meta: {done} pages made citable")


if __name__ == "__main__":
    main()
