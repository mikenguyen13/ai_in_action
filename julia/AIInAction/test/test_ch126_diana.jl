using Test
using AIInAction.Ch126Diana

# Shared fixture: 1-D points {1,2,3,20,21,22}, d = absolute difference.
# Identical data to the Python and Rust suites. Member indices here are 1-based
# (so Python's splinter {0,1,2} appears as {1,2,3}); flat labels stay 0-based.
const PTS = [1.0, 2.0, 3.0, 20.0, 21.0, 22.0]
const N = length(PTS)
const DIST = [abs(PTS[i] - PTS[j]) for i in 1:N, j in 1:N]

@testset "DIANA parity fixtures" begin
    @test diameter(DIST, collect(1:N)) ≈ 21.0
    @test diameter(DIST, [4]) == 0.0

    spl, rem = macnaughton_smith_split(DIST, collect(1:N))
    @test spl == [1, 2, 3]
    @test rem == [4, 5, 6]

    splits = diana(DIST)
    @test length(splits) == N - 1
    first = splits[1]
    @test first.parent == [1, 2, 3, 4, 5, 6]
    @test first.splinter == [1, 2, 3]
    @test first.remainder == [4, 5, 6]
    @test first.diameter ≈ 21.0
    diams = [s.diameter for s in splits]
    @test diams == sort(diams; rev = true)

    @test diana_labels(DIST, 2) == [0, 0, 0, 1, 1, 1]
    @test diana_labels(DIST, 3) == [0, 1, 1, 2, 2, 2]
    @test diana_labels(DIST, N) == [0, 1, 2, 3, 4, 5]
    @test diana_labels(DIST, 1) == [0, 0, 0, 0, 0, 0]
end

@testset "DIANA edge cases" begin
    spl, rem = macnaughton_smith_split(DIST, [3])
    @test isempty(spl)
    @test rem == [3]

    @test diana(reshape([0.0], 1, 1)) == Ch126Diana.Split[]
    @test diana_labels(reshape([0.0], 1, 1), 1) == [0]

    @test_throws ArgumentError diana([0.0 1.0; 2.0 0.0])              # asymmetric
    @test_throws ArgumentError diana([1.0 1.0; 1.0 1.0])              # nonzero diag
    @test_throws ArgumentError diana([0.0 -1.0; -1.0 0.0])            # negative
    @test_throws ArgumentError macnaughton_smith_split(DIST, [1, 99]) # bad index
    @test_throws ArgumentError macnaughton_smith_split(DIST, [1, 1])  # duplicate
    @test_throws ArgumentError diana_labels(DIST, 0)
    @test_throws ArgumentError diana_labels(DIST, N + 1)
end
