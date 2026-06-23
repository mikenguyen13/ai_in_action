using Test
using AIInAction

# Run every test file matching test_*.jl. Each file is included into its own
# freshly generated module so that top-level `const` fixtures in different files
# (e.g. SOFTMAX_Z, PTS, X_BLOBS) cannot collide with one another. Each test file
# brings in whatever it needs with its own `using` statements.
const TEST_DIR = @__DIR__

test_files = sort(filter(readdir(TEST_DIR)) do f
    f != "runtests.jl" && startswith(f, "test_") && endswith(f, ".jl")
end)

for f in test_files
    modname = Symbol("TestModule_" * replace(f, ".jl" => ""))
    path = joinpath(TEST_DIR, f)
    @eval module $modname
        using Test
        using AIInAction
        include($path)
    end
end
