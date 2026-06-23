using Test
using AIInAction.Ch071Eda

# Shared fixtures: identical to the Python and Rust test suites.
const SENTENCE = "the quick movie was good and fast"
const TOKENS = ["the", "quick", "movie", "was", "good", "and", "fast"]

@testset "Ch071 LCG parity" begin
    rng = Lcg(42)
    @test [next_u32!(rng) for _ in 1:5] == [705894, 1126542223, 1579310009, 565444343, 807934826]

    rng = Lcg(42)
    floats = [next_float!(rng) for _ in 1:3]
    expected = [0.000328707, 0.5245871018, 0.7354235321]
    @test all(abs(g - e) < 1e-9 for (g, e) in zip(floats, expected))

    rng = Lcg(7)
    @test [randint!(rng, 10) for _ in 1:6] == [9, 3, 6, 5, 9, 7]

    @test next_u32!(Lcg(0)) != 0
    @test_throws ArgumentError Lcg(-1)
    @test_throws ArgumentError randint!(Lcg(1), 0)
end

@testset "Ch071 tokenize" begin
    @test tokenize(SENTENCE) == TOKENS
    @test_throws ArgumentError tokenize("   ")
end

@testset "Ch071 operations parity" begin
    @test synonym_replacement(TOKENS, 2, Lcg(1)) ==
          ["the", "quick", "feature", "was", "good", "and", "rapid"]
    @test random_insertion(TOKENS, 2, Lcg(2)) ==
          ["fine", "great", "the", "quick", "movie", "was", "good", "and", "fast"]
    @test random_swap(TOKENS, 2, Lcg(3)) ==
          ["the", "quick", "movie", "good", "was", "and", "fast"]
    @test random_deletion(TOKENS, 0.3, Lcg(4)) == ["quick", "was", "and"]
end

@testset "Ch071 operation edge cases" begin
    @test synonym_replacement(["xx", "yy", "zz"], 3, Lcg(5)) == ["xx", "yy", "zz"]
    @test random_swap(["solo"], 5, Lcg(6)) == ["solo"]
    @test random_deletion(["solo"], 1.0, Lcg(6)) == ["solo"]
    @test length(random_deletion(TOKENS, 1.0, Lcg(8))) == 1
    @test_throws ArgumentError synonym_replacement(String[], 1, Lcg(1))
    @test_throws ArgumentError random_swap(String[], 1, Lcg(1))
    @test_throws ArgumentError random_deletion(TOKENS, 2.0, Lcg(1))
end

@testset "Ch071 eda parity" begin
    @test eda(SENTENCE; seed=123, num_aug=4) == [
        "the quick feature was good and fast",
        "the quick movie was good rapid and fast",
        "the quick movie was good and fast",
        "the quick movie was good and fast",
    ]

    @test eda("a sleek and surprisingly fast car"; seed=99, num_aug=6,
              alpha_sr=0.2, alpha_ri=0.2, alpha_rs=0.2, p_rd=0.2) == [
        "a sleek and surprisingly fast auto",
        "auto a sleek and surprisingly fast car",
        "car sleek and surprisingly fast a",
        "a sleek and surprisingly fast car",
        "a sleek and surprisingly fast automobile",
        "a sleek and surprisingly fast car speedy",
    ]

    @test eda(SENTENCE; seed=7, num_aug=5) == eda(SENTENCE; seed=7, num_aug=5)
    @test length(eda(SENTENCE; seed=1, num_aug=10)) == 10
    @test eda(SENTENCE; seed=1, num_aug=0) == String[]
end

@testset "Ch071 eda validation" begin
    @test_throws ArgumentError eda("   "; seed=1)
    @test_throws ArgumentError eda(SENTENCE; num_aug=-1)
    @test_throws ArgumentError eda(SENTENCE; alpha_sr=1.5)
    @test_throws ArgumentError eda(SENTENCE; p_rd=-0.1)
end
