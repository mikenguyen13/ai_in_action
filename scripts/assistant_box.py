"""Add a small "Use this book with your AI assistant" box to every chapter.

The box sits at the end of each chapter (after the text, before the previous/next
links) and gives the Research Methods Library MCP endpoint with this book's source
label (``?src=book-ai-in-action``, counted per source by the server), a copy button,
Cursor and VS Code install buttons, and a link to the setup page. The home page
carries the same content in its "Use this book in your AI assistant" section
(index.qmd), so index.html is skipped.

Idempotent; runs as a Quarto post-render step, or by hand over an existing site:

    python scripts/assistant_box.py docs
"""
from __future__ import annotations

import base64
import html
import json
import os
import pathlib
import re
import sys
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- assistant-box:begin -->", "<!-- assistant-box:end -->"
SKIP = {"index.html", "about.html", "privacy.html", "cookies.html", "search.html", "404.html"}

SERVER = "https://research-methods-mcp.vercel.app"
ENDPOINT = f"{SERVER}/mcp?src=book-ai-in-action"
CURSOR = "https://cursor.com/en/install-mcp?name=research-methods&config=" + urllib.parse.quote(
    base64.b64encode(json.dumps({"url": ENDPOINT}, separators=(",", ":")).encode()).decode(), safe=""
)
VSCODE = "https://insiders.vscode.dev/redirect/mcp/install?name=research-methods&config=" + urllib.parse.quote(
    json.dumps({"type": "http", "url": ENDPOINT}, separators=(",", ":")), safe=""
)
CURSOR_BADGE = "https://cursor.com/deeplink/mcp-install-dark.svg"
VSCODE_BADGE = "https://img.shields.io/badge/VS_Code-Install_server-0098FF?logo=visualstudiocode&logoColor=white"


def box() -> str:
    e = html.escape
    return f"""{BEGIN}
<aside class="assistant-box" aria-label="Use this book with your AI assistant">
<p class="assistant-lead"><strong>Use this book with your AI assistant.</strong> Ask questions about this book in Claude, Cursor, VS Code, or ChatGPT, with answers that cite the section.</p>
<p class="assistant-endpoint"><code>{e(ENDPOINT)}</code> <button type="button" class="assistant-copy" data-copy="{e(ENDPOINT, quote=True)}">Copy</button></p>
<p class="assistant-install"><a href="{e(CURSOR, quote=True)}"><img src="{e(CURSOR_BADGE, quote=True)}" alt="Add to Cursor" height="20" loading="lazy"></a>
<a href="{e(VSCODE, quote=True)}"><img src="{e(VSCODE_BADGE, quote=True)}" alt="Install in VS Code" height="20" loading="lazy"></a>
<a href="{SERVER}/">Setup for Claude, ChatGPT, and other assistants</a></p>
</aside>
<style>
.assistant-box {{ margin-top: 1.5rem; padding: .6rem .9rem; font-size: .85rem; line-height: 1.45;
  border: 1px solid rgba(127, 127, 127, .25); border-radius: .25rem; }}
.assistant-box p {{ margin: .25rem 0; }}
.assistant-box code {{ font-size: .8rem; word-break: break-all; }}
.assistant-box .assistant-copy {{ font-size: .75rem; padding: 0 .45rem; border: 1px solid rgba(127, 127, 127, .4);
  border-radius: .2rem; background: transparent; color: inherit; }}
.assistant-box .assistant-install {{ display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; }}
.assistant-box .assistant-install img {{ height: 20px; width: auto; vertical-align: middle; }}
</style>
<script>
document.querySelectorAll(".assistant-copy").forEach(function (b) {{
  b.addEventListener("click", function () {{
    var done = function () {{ b.textContent = "Copied"; setTimeout(function () {{ b.textContent = "Copy"; }}, 1500); }};
    if (navigator.clipboard) {{ navigator.clipboard.writeText(b.dataset.copy).then(done); }}
  }});
}});
</script>
{END}"""


def process(path: pathlib.Path, out_dir: pathlib.Path, block: str) -> bool:
    rel = path.relative_to(out_dir).as_posix()
    if rel in SKIP or "/" in rel:
        return False
    with open(path, encoding="utf-8", newline="") as fh:
        text = fh.read()
    if "</main>" not in text or '<h1 class="title">' not in text:
        return False
    text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", "", text, flags=re.S)
    text = text.replace("</main>", block + "\n</main>", 1)
    path.write_text(text, encoding="utf-8", newline="")
    return True


def main() -> None:
    block = box()
    if len(sys.argv) > 1:
        out_dir = pathlib.Path(sys.argv[1]).resolve()
        files = sorted(out_dir.glob("*.html"))
    else:
        # Quarto post-render: only the pages this render produced.
        out_dir = ROOT / os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "docs")
        listed = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "")
        files = [pathlib.Path(p) for p in listed.splitlines() if p.endswith(".html")] or sorted(out_dir.glob("*.html"))
        files = [p if p.is_absolute() else ROOT / p for p in files]
    out_dir = out_dir.resolve()
    done = sum(process(p.resolve(), out_dir, block) for p in files if p.exists())
    print(f"assistant_box: {done} pages")


if __name__ == "__main__":
    main()
