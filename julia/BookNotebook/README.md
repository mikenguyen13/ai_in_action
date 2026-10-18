# BookNotebook: the Julia environment for executed Julia in chapters

Chapters whose Julia code executes use Quarto's native `julia` engine and this
environment. Each such chapter declares, in its frontmatter,

```yaml
engine: julia
julia:
  exeflags: ["--project=julia/BookNotebook"]
```

and includes the shared setup file right after its title:

```markdown
{{< include _julia-setup.qmd >}}
```

The setup cell (hidden) instantiates this environment if needed, points PythonCall at
the interpreter named by `QUARTO_PYTHON` (falling back to the project `.venv`) so the
chapter's `{python}` cells run with exactly the packages they use under the Jupyter
engine, selects matplotlib's non-interactive `Agg` backend, makes `plt.show()` hand every
open figure to Quarto, as the Jupyter engine does, and makes Python logging handlers
write to the current cell's stderr (each cell's captured stream is closed when the
cell ends, so a handler created in an earlier cell would otherwise fail). It also
switches joblib (and so scikit-learn's `n_jobs`) to its threading backend, because
joblib's default process pool cannot start workers from an interpreter embedded in
Julia.

The environment holds the book's `AIInAction` package (resolved from `../AIInAction`
through the `[sources]` table, so no committed Manifest is needed), PythonCall, and the
open-source Julia packages the chapters demonstrate (Distributions, DataFrames, GLM,
MLJ and its model interfaces, DecisionTree, XGBoost, LightGBM, CatBoost, LIBSVM,
Flux, Zygote, ForwardDiff, JuMP with Ipopt, Optim, Clustering, UMAP, and others).

## Rendering

Rendering requires Julia 1.11 or newer. Point Quarto at it with `QUARTO_JULIA` (for
example, a juliaup install: `juliaup add 1.13.1`, then set `QUARTO_JULIA` to that
version's `bin\julia.exe`). Keep the Julia depot on a short path (the default `~/.julia`
works): QuartoNotebookRunner installs deeply nested files and fails on Windows when the
depot path is long. On Windows the engine starts its server through PowerShell, so
PowerShell must be on `PATH` (it is in a normal shell; add it when launching renders
from a minimal POSIX shell).

## Writing Python cells in these chapters

The engine passes Python source to PythonCall inside a Julia string, so a `{python}`
cell must not contain triple double quotes; use `'''` for docstrings. Everything else
(figures, pandas tables, printed output, cross-references) behaves as under Jupyter.
