<p align="center">
  <img src="cover.png" alt="AI in Action cover" width="220">
</p>

<h1 align="center">AI in Action</h1>

<p align="center">
  <strong>A Reference for Artificial Intelligence</strong><br>
  Mike Nguyen
</p>

<p align="center">
  <a href="https://mikenguyen13.github.io/ai_in_action/"><img alt="Read online" src="https://img.shields.io/badge/read-online-2b4c9b?style=flat-square"></a>
  <a href="https://github.com/mikenguyen13/ai_in_action/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/mikenguyen13/ai_in_action/ci.yml?branch=main&label=tests&style=flat-square"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square">
  <img alt="Julia" src="https://img.shields.io/badge/julia-1.10%2B-9558b2?style=flat-square">
  <img alt="Rust" src="https://img.shields.io/badge/rust-stable-b7410e?style=flat-square">
  <img alt="Built with Quarto" src="https://img.shields.io/badge/built%20with-Quarto-75aadb?style=flat-square">
</p>

---

**AI in Action** is a rigorous, implementation-first reference for modern artificial intelligence. It covers the mathematical foundations, classical machine learning, deep learning, transformers and large language models, retrieval and knowledge graphs, AI agents, and the frontier methods of 2024 to 2026. Every chapter derives the ideas carefully and then makes them run.

📖 **Read the book:** <https://mikenguyen13.github.io/ai_in_action/>

## What makes it different

- **Mathematics first, never hand-waved.** Algorithms are derived step by step, with the assumptions and failure modes stated.
- **Code that actually runs.** Python examples are executed at build time, so every figure and table in the book is produced by the code shown next to it.
- **Three languages.** Core methods are implemented in Python, Julia, and Rust, with shared test fixtures so the three versions agree.
- **Grounded in the literature.** Chapters cite peer-reviewed sources with DOIs and integrate recent research as it is published.
- **Open-source tooling.** Examples favor mature, freely available libraries over proprietary services.

## Contents

The book is organized in eighteen parts:

| Part | Topic |
|:----:|:------|
| I | Foundations of Intelligence |
| II | Mathematical Foundations |
| III | Data Foundations |
| IV | Classical Machine Learning |
| V | Unsupervised Learning |
| VI | Model Evaluation and Selection |
| VII | Neural Networks Foundations |
| VIII | The Transformer Revolution |
| IX | Natural Language Processing |
| X | Economics, Business, and Finance |
| XI | Knowledge Graphs and GraphRAG |
| XII | AI Agents and Autonomy |
| XIII | Computer Vision and Multimodal AI |
| XIV | Responsible and Trustworthy AI |
| XV | Frontier Models and Methods (2024 to 2026) |
| XVI | Transfer Learning and Domain Adaptation |
| XVII | ML Engineering and Production Monitoring |
| XVIII | AI Safety and Robustness |

A machine-readable chapter index is published at [`llms.txt`](https://mikenguyen13.github.io/ai_in_action/llms.txt).

## Companion libraries

The reference implementations behind the chapters are packaged as installable, tested libraries:

| Language | Package | Location | Tests |
|:---------|:--------|:---------|:------|
| Python | `aiinaction` | [`src/aiinaction/`](src/aiinaction/) | `pytest` |
| Julia | `AIInAction` | [`julia/AIInAction/`](julia/AIInAction/) | `Pkg.test()` |
| Rust | `aiinaction` | [`rust/aiinaction/`](rust/aiinaction/) | `cargo test` |

Continuous integration runs all three test suites on every push to `main`.

```bash
# Python
pip install -e ".[dev]" && pytest -q

# Julia
julia --project=julia/AIInAction -e 'using Pkg; Pkg.test()'

# Rust
cd rust/aiinaction && cargo test
```

## Building the book

Requirements: [Quarto](https://quarto.org) 1.5 or newer, [uv](https://docs.astral.sh/uv/), and Julia 1.11 or newer for the chapters that execute Julia.

```bash
uv sync                                   # create .venv with the pinned dependencies
export QUARTO_PYTHON=.venv/bin/python     # on Windows: .venv\Scripts\python.exe
quarto render                             # build the full book into docs/
quarto preview                            # live preview while editing
```

Executed results are cached in `_freeze/`, so after the first build only chapters whose code changed are re-run. A single chapter can be rendered with `quarto render <chapter>.qmd`.

## Repository layout

| Path | Contents |
|:-----|:---------|
| `*.qmd` | Chapter sources (Quarto Markdown) |
| `_quarto.yml` | Book configuration and table of contents |
| `references.bib` | Shared bibliography |
| `src/`, `tests/` | Python library and tests |
| `julia/` | Julia library, tests, and the notebook environment for Julia-executed chapters |
| `rust/` | Rust library and tests |
| `theme.scss`, `theme-dark.scss` | Site typography and color themes |
| `scripts/` | Build helpers, including the `llms.txt` generator |
| `docs/` | Rendered site served by GitHub Pages (generated; do not edit by hand) |

## Feedback

Corrections and suggestions are welcome. Please [open an issue](https://github.com/mikenguyen13/ai_in_action/issues), or use the *Report an issue* link on any page of the online book.

## Citation

If you use this book, please cite it. GitHub's *Cite this repository* button (from [`CITATION.cff`](CITATION.cff)) gives APA and BibTeX, and every chapter page of the online book ends with a reference for that chapter.

```bibtex
@book{nguyen2026aiinaction,
  author = {Nguyen, Mike},
  title  = {{AI in Action: A Reference for Artificial Intelligence}},
  year   = {2026},
  url    = {https://mikenguyen13.github.io/ai_in_action/},
  note   = {Online book}
}
```

## License

© Mike Nguyen. All rights reserved. Please contact the author before reproducing the book text or figures.
