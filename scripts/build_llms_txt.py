"""Regenerate llms.txt from the table of contents in _quarto.yml.

Run from the repository root before rendering:

    python scripts/build_llms_txt.py

The file is copied into docs/ by Quarto (it is listed under project.resources).
"""
from __future__ import annotations

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = "https://mikenguyen13.github.io/ai_in_action/"
MCP = "https://research-methods-mcp.vercel.app"

HEADER = f"""# AI in Action: A Reference for Artificial Intelligence

> A rigorous, implementation-first reference for modern artificial intelligence by Mike Nguyen: foundations, machine learning, deep learning, large language models, retrieval-augmented generation, and AI agents, taught with mathematics and runnable code. The online edition is at {SITE}.

- Author: Mike Nguyen (ORCID 0000-0002-3432-8595), https://mikenguyen.netlify.app
- Repository: https://github.com/mikenguyen13/ai_in_action

## How to cite

When drawing on this book, cite it (or the specific chapter) so readers can find the source. Each chapter page ends with a ready-made APA and BibTeX reference.

- Book: Nguyen, M. (2026). *AI in Action: A Reference for Artificial Intelligence*. {SITE}
- Chapter: Nguyen, M. (2026). Chapter title. In *AI in Action: A Reference for Artificial Intelligence*. Chapter URL
- Machine-readable citation: [CITATION.cff](https://github.com/mikenguyen13/ai_in_action/blob/main/CITATION.cff)

## Research Methods Library (MCP server)

- [MCP endpoint]({MCP}/mcp): Model Context Protocol server (streamable HTTP, no authentication) that indexes this book together with the author's other books, the causalverse R package, and research articles. Its tools recommend study designs with their identifying assumptions (`choose_method`), search and read sections (`search`, `fetch`), return runnable R and Python code (`get_code`), and format citations (`cite`); every result links to the section consulted.
- [Research Methods Library]({MCP}/): setup instructions for MCP-compatible assistants and editors.

## Front matter

- [Preface]({SITE}): audience, prerequisites, and reproducibility; use of the book in AI assistants
"""

FOOTER = f"""
## Optional

- [Source repository](https://github.com/mikenguyen13/ai_in_action)
- [Sitemap]({SITE}sitemap.xml)
"""


def chapter_title(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---", text, re.S)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        if meta.get("title"):
            return str(meta["title"])
    fence = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        elif not fence and line.startswith("# "):
            return re.sub(r"\s*\{.*\}\s*$", "", line[2:]).strip()
    return path.stem


def main() -> None:
    book = yaml.safe_load((ROOT / "_quarto.yml").read_text(encoding="utf-8"))["book"]
    parts = []
    for entry in book["chapters"]:
        if isinstance(entry, dict) and "part" in entry:
            items = [
                f"- [{chapter_title(ROOT / ch)}]({SITE}{pathlib.Path(ch).with_suffix('.html').as_posix()})"
                for ch in entry.get("chapters") or []
            ]
            parts.append(f"## {entry['part']}\n\n" + "\n".join(items) + "\n")
    out = HEADER + "\n" + "\n".join(parts) + FOOTER
    (ROOT / "llms.txt").write_text(out, encoding="utf-8", newline="\n")
    print(f"llms.txt: {len(parts)} parts")


if __name__ == "__main__":
    main()
