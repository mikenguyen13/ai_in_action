# AI in Action

Building artificial intelligence systems from the ground up, with **Python** as the working language and **mathematics** as the organizing principle.

This is the source for a [Quarto](https://quarto.org) book aimed at advanced undergraduates, graduate students, and practitioners who want a rigorous, implementation-focused view of AI. Rather than treating AI as a collection of library calls, every chapter stays close to the underlying math while always returning to executable Python you can run, modify, and test.

Each chapter follows the same pattern: introduce a concept intuitively and mathematically, derive the core algorithm step by step, implement it in Python, analyze the implementation line by line, then explore variations and failure modes on realistic data.

## Reading the book

The rendered book is published via GitHub Pages from the [`docs/`](docs/) directory. Source and issues live at <https://github.com/mikenguyen13/ai_in_action>.

## Building locally

The project uses [`uv`](https://docs.astral.sh/uv/) for Python (3.10) and [Quarto](https://quarto.org) for rendering.

```bash
# 1. Install dependencies into a local virtual environment
uv sync

# 2. Render the whole book to docs/
quarto render

# 3. Or live-preview while editing
quarto preview
```

`quarto render` writes HTML output to `docs/`. Computational chapters are cached via `jupyter-cache`, so re-renders only re-execute changed code.

## Repository layout

| Path | Contents |
|------|----------|
| `_quarto.yml` | Book configuration and table of contents. The full outline is extensive; chapters not yet written are commented out. |
| `*.qmd` | Chapter sources (Quarto Markdown). Some are numbered by their slot in the outline (e.g. `519-model-context-protocol-mcp.qmd`); others are descriptively named. |
| `index.qmd` | Preface, audience, prerequisites, and how the book is organized. |
| `references.bib` | Shared bibliography. |
| `docs/` | Rendered site (GitHub Pages output). Generated, do not edit by hand. |
| `_archive/` | Reference material salvaged from an earlier draft (notebooks, a GNN writeup, an applied example). The leading `_` makes Quarto ignore it during render; it is **not** part of the book. |
| `pyproject.toml` / `uv.lock` | Python dependencies and lockfile. |

## Contributing a chapter

1. Add the `.qmd` file at the repository root.
2. Reference it in the appropriate part under `book.chapters` in `_quarto.yml` (uncomment its planned slot, or add a new entry).
3. Run `quarto inspect` to confirm the configuration resolves, then `quarto render` to build.

## License

See the repository for license details.
