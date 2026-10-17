# BookNotebook: the Julia environment for executed Julia in chapters

Chapters that execute Julia use Quarto's native `julia` engine and this environment:

```yaml
engine: julia
julia:
  exeflags: ["--project=julia/BookNotebook"]
```

The environment holds the book's `AIInAction` package (resolved from `../AIInAction`
through the `[sources]` table, so no committed Manifest is needed) and PythonCall,
which runs the chapter's `{python}` cells inside the same notebook. Each such
chapter starts with a hidden Julia setup cell that instantiates this environment if
needed and points PythonCall at the interpreter named by `QUARTO_PYTHON` (falling
back to the project `.venv`), so Python cells see exactly the packages they see
under the Jupyter engine.

Rendering requires Julia 1.11 or newer. Point Quarto at it with `QUARTO_JULIA`
(for example, a juliaup install: `juliaup add 1.13.1`, then set `QUARTO_JULIA` to
that version's `bin\julia.exe`). Keep the Julia depot on a short path (the default
`~/.julia` works): QuartoNotebookRunner installs deeply nested files and fails on
Windows when the depot path is long.
