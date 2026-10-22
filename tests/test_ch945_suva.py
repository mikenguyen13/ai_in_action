"""Tests for aiinaction.ch945_suva, including shared cross-language fixtures.

The fixtures here are the single source of truth: the Julia and Rust test suites
assert against the same numbers, which keeps the three implementations at parity.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aiinaction import ch945_suva as s

# ---------------------------------------------------------------------------
# Shared fixtures (mirrored in julia/AIInAction/test and rust/aiinaction/src)
# ---------------------------------------------------------------------------
CDF_POINTS = [(-1.96, 0.02499789514822045), (0.0, 0.5), (0.5, 0.691462461274013),
              (2.5, 0.9937903346742238), (-8.0, 6.220960520142783e-16), (10.0, 1.0)]

GRID_COUNT = 240
GRID_FIRST = (0, 0, 0, 200)
GRID_LAST = (600, 600, 600, 400)
RECIP_COUNT = 120
RECIP_FIRST = (0, 0, 200, 0)
RECIP_LAST = (600, 600, 400, 600)
RECIP_01 = [(0, 0, 1, 0), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1), (1, 1, 0, 1)]

MOTIVES = [((200, 400, 600, 200), (1, 1, 1, -1)), ((0, 600, 600, 0), (1, 1, 0, 0)),
           ((400, 400, 200, 600), (-1, -1, 1, 0))]

OLS_X = [[1.0, float(i)] for i in range(8)]
OLS_Y = [1.0, 2.5, 2.0, 4.5, 4.0, 6.5, 5.5, 8.0]
OLS_CLUSTERS = [0, 0, 1, 1, 2, 2, 3, 3]
OLS_COEF = (1.041666666666666, 0.916666666666667)
OLS_SE = (0.25027848001436165, 0.05787698374909847)

DEMEAN_VALUES = [1.0, 3.0, 2.0, 6.0, 10.0]
DEMEAN_GROUPS = ["a", "a", "b", "b", "b"]
DEMEAN_OUT = [-1.0, 1.0, -4.0, 0.0, 4.0]

DIFF_PROP = ((30, 50, 18, 50), 0.24, 0.09699484522385712)

KAPPA_A = [0, 0, 1, 1, 2, 2, 0, 1]
KAPPA_B = [0, 1, 1, 1, 2, 0, 0, 1]
KAPPA = 0.6097560975609756

AUC_SCORES = [0.1, 0.4, 0.35, 0.8, 0.4, 0.2]
AUC_LABELS = [0, 0, 1, 1, 1, 0]
AUC = 0.8333333333333334

D_X = [5.0, 6.0, 7.0, 8.0]
D_Y = [3.0, 4.0, 5.0]
COHENS_D = 2.1128856368212916

TOST_PASS = ((0.02, 0.04, 0.1), 0.02275013194817923)
TOST_FAIL = ((0.12, 0.04, 0.1), 0.691462461274013)

DEFF_5_02 = 1.8
N_045_030 = 160
N_045_030_DEFF = 288
N_05_04_POWER90 = 515
Z_90 = 1.2815515655446004

DPO_01 = 2.1972245773362196
DPO_025_BETA2 = 0.5493061443340549

THRESHOLD_CASES = [((0.5, 0.2, 0.25), 1), ((0.2, 0.5, 0.25), -1), ((0.5, 0.4, 0.25), 0), ((0.3, 0.3, 0.0), 0)]
TRANSITIVE_WEAK = (13, 27, 6, 8)  # weak orders of 3 items; strict-only profiles
TRANSITIVE_STRICT_PART = 19  # strict partial orders of 3 items
TRANSITIVE_CASES = [((1, 1, 1), True, True), ((1, -1, 1), False, False),
                    ((0, 1, 0), False, True), ((0, 0, 1), False, True), ((0, 0, 0), True, True)]

CELL_HUMAN = [1.0, 2.0, 5.0, 6.0]
CELL_LLM = [2.2, 1.8, 6.2, 5.8]
CELL_GROUPS = ["a", "a", "b", "b"]
CELL_POOLED = (0.9411946707617073, 0.9176470588235291)
CELL_WITHIN = (-1.0, -0.4)

COLLAPSE_IN = ["u", "u", "s", "s", "f", "s"]
COLLAPSE_OUT = ["u", "s", "f", "s"]

TREE_PATHS = [(["understanding", "self_interest", "self_interest"], "B1"),
              (["understanding", "self_interest"], "B2"),
              (["understanding", "fairness"], "B2"),
              (["self_interest"], "B1")]
TREE_ROOT_COUNT = 4
TREE_UNDERSTANDING_COUNT = 3
TREE_UNDERSTANDING_ACTIONS = {"B1": 1, "B2": 2}
TREE_PATH_PROB = 0.25  # understanding -> self_interest -> B1 = 3/4 * 2/3 * 1/2


# ---------------------------------------------------------------------------
# Numeric core
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("x,expected", CDF_POINTS)
def test_normal_cdf_fixture(x, expected):
    assert s.normal_cdf(x) == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_normal_cdf_matches_erfc():
    for i in range(-900, 901):
        x = i / 100
        assert abs(s.normal_cdf(x) - 0.5 * math.erfc(-x / math.sqrt(2))) < 1e-14


def test_payoff_grid_fixtures():
    g = s.payoff_grid()
    assert len(g) == GRID_COUNT and g[0] == GRID_FIRST and g[-1] == GRID_LAST
    r = s.payoff_grid(game="reciprocity")
    assert len(r) == RECIP_COUNT and r[0] == RECIP_FIRST and r[-1] == RECIP_LAST
    assert s.payoff_grid((0, 1), "reciprocity") == RECIP_01
    assert len(s.payoff_grid((0, 1))) == 12


def test_payoff_grid_reciprocity_has_no_selfserving_help():
    for a1, b1, a2, b2 in s.payoff_grid(game="reciprocity"):
        # the option better for A is never strictly better for B
        if a1 > a2:
            assert b1 <= b2
        else:
            assert a1 < a2 and b1 >= b2


def test_payoff_grid_validation():
    with pytest.raises(ValueError):
        s.payoff_grid(game="ultimatum")
    with pytest.raises(ValueError):
        s.payoff_grid((1, 1))


@pytest.mark.parametrize("payoffs,expected", MOTIVES)
def test_motive_features(payoffs, expected):
    assert s.motive_features(*payoffs) == expected


def test_ols_cluster_fixture_and_statsmodels():
    r = s.ols_cluster(OLS_X, OLS_Y, OLS_CLUSTERS)
    assert r.coef == pytest.approx(OLS_COEF, rel=1e-12)
    assert r.se == pytest.approx(OLS_SE, rel=1e-10)
    assert r.n == 8 and r.n_clusters == 4
    sm = pytest.importorskip("statsmodels.api")
    f = sm.OLS(np.array(OLS_Y), np.array(OLS_X)).fit(cov_type="cluster", cov_kwds={"groups": np.array(OLS_CLUSTERS)})
    assert r.se == pytest.approx(tuple(f.bse), rel=1e-10)


def test_ols_cluster_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 2))
    y = 0.5 + 2.0 * x[:, 0] - 1.0 * x[:, 1] + rng.normal(scale=0.1, size=500)
    r = s.ols_cluster(np.column_stack([np.ones(500), x]).tolist(), y.tolist(), list(range(500)))
    assert r.coef == pytest.approx((0.5, 2.0, -1.0), abs=0.03)
    lo, hi = r.ci()[1]
    assert lo < 2.0 < hi


def test_ols_cluster_rejects_singular_and_bad_input():
    with pytest.raises(ValueError):
        s.ols_cluster([[1.0, 1.0]] * 5, [1, 2, 3, 4, 5], [0, 1, 0, 1, 0])
    with pytest.raises(ValueError):
        s.ols_cluster([[1.0]] * 3, [1.0, 2.0, 3.0], [0, 0, 0])
    with pytest.raises(ValueError):
        s.ols_cluster([[1.0]], [1.0], [0])


def test_demean_by_group():
    assert s.demean_by_group(DEMEAN_VALUES, DEMEAN_GROUPS) == pytest.approx(DEMEAN_OUT)


def test_within_estimator_equals_dummy_regression():
    """Frisch-Waugh-Lovell: demeaned slopes equal slopes with group dummies."""
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(20), 10)
    w = rng.integers(0, 2, size=200).astype(float)
    y = 0.3 * w + rng.normal(size=20)[g] + rng.normal(scale=0.2, size=200)
    within = s.ols_cluster([[v] for v in s.demean_by_group(w.tolist(), g.tolist())],
                           s.demean_by_group(y.tolist(), g.tolist()), g.tolist())
    dummies = (g[:, None] == np.arange(20)[None, :]).astype(float)
    full, *_ = np.linalg.lstsq(np.column_stack([w, dummies]), y, rcond=None)
    assert within.coef[0] == pytest.approx(full[0], rel=1e-10)


def test_diff_proportions():
    (k1, n1, k0, n0), d, se = DIFF_PROP
    out = s.diff_proportions(k1, n1, k0, n0)
    assert out == pytest.approx((d, se), rel=1e-12)
    with pytest.raises(ValueError):
        s.diff_proportions(5, 4, 1, 4)


def test_cohen_kappa():
    assert s.cohen_kappa(KAPPA_A, KAPPA_B) == pytest.approx(KAPPA, rel=1e-12)
    assert s.cohen_kappa(["x", "y"], ["x", "y"]) == 1.0
    assert s.cohen_kappa(["x", "x"], ["x", "x"]) == 1.0


def test_auc_score_with_ties_and_sklearn():
    assert s.auc_score(AUC_SCORES, AUC_LABELS) == pytest.approx(AUC, rel=1e-12)
    metrics = pytest.importorskip("sklearn.metrics")
    assert s.auc_score(AUC_SCORES, AUC_LABELS) == pytest.approx(metrics.roc_auc_score(AUC_LABELS, AUC_SCORES))
    with pytest.raises(ValueError):
        s.auc_score([0.1, 0.2], [1, 1])


def test_cohens_d():
    assert s.cohens_d(D_X, D_Y) == pytest.approx(COHENS_D, rel=1e-12)


def test_tost():
    args, p = TOST_PASS
    assert s.tost_p_value(*args) == pytest.approx(p, rel=1e-12)
    args, p = TOST_FAIL
    assert s.tost_p_value(*args) == pytest.approx(p, rel=1e-12)
    with pytest.raises(ValueError):
        s.tost_p_value(0.0, 0.0, 0.1)


def test_power_planning():
    assert s.design_effect(5, 0.2) == pytest.approx(DEFF_5_02)
    assert s.samples_per_arm(0.45, 0.30) == N_045_030
    assert s.samples_per_arm(0.45, 0.30, deff=DEFF_5_02) == N_045_030_DEFF
    assert s.samples_per_arm(0.5, 0.4, z_beta=Z_90) == N_05_04_POWER90
    with pytest.raises(ValueError):
        s.samples_per_arm(0.3, 0.3)


def test_dpo_smoothed_margin_and_optimum():
    assert s.dpo_smoothed_margin(0.1) == pytest.approx(DPO_01, rel=1e-14)
    assert s.dpo_smoothed_margin(0.25, 2.0) == pytest.approx(DPO_025_BETA2, rel=1e-14)
    sp = pytest.importorskip("sympy")
    m, e = sp.symbols("m epsilon", positive=True)
    loss = -(1 - e) * sp.log(1 / (1 + sp.exp(-m))) - e * sp.log(1 / (1 + sp.exp(m)))
    root = sp.log((1 - e) / e)
    assert sp.simplify(sp.diff(loss, m).subs(m, root)) == 0


@pytest.mark.parametrize("args,expected", THRESHOLD_CASES)
def test_threshold_choice(args, expected):
    assert s.threshold_choice(*args) == expected


def test_threshold_choice_validation():
    with pytest.raises(ValueError):
        s.threshold_choice(0.0, 1.0, -0.1)
    with pytest.raises(ValueError):
        s.threshold_choice(math.nan, 1.0, 0.1)


@pytest.mark.parametrize("answers,weak,strict", TRANSITIVE_CASES)
def test_pairwise_transitive_cases(answers, weak, strict):
    assert s.pairwise_transitive(*answers) is weak
    assert s.pairwise_transitive(*answers, strict_only=True) is strict


def test_pairwise_transitive_counts():
    import itertools
    profiles = list(itertools.product((-1, 0, 1), repeat=3))
    strict = [p for p in profiles if 0 not in p]
    n_t, n, n_ts, n_s = TRANSITIVE_WEAK
    assert (sum(s.pairwise_transitive(*p) for p in profiles), len(profiles)) == (n_t, n)
    assert (sum(s.pairwise_transitive(*p) for p in strict), len(strict)) == (n_ts, n_s)
    assert sum(s.pairwise_transitive(*p, strict_only=True) for p in profiles) == TRANSITIVE_STRICT_PART
    with pytest.raises(ValueError):
        s.pairwise_transitive(2, 0, 0)


def test_threshold_respondents_have_transitive_strict_part():
    """A semiorder respondent can fail the weak-order test but never the strict one."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        u = rng.normal(size=3)
        a = [s.threshold_choice(u[i], u[j], 0.6) for i, j in ((0, 1), (0, 2), (1, 2))]
        assert s.pairwise_transitive(*a, strict_only=True)
        assert s.pairwise_transitive(*[s.threshold_choice(u[i], u[j], 0.0)
                                       for i, j in ((0, 1), (0, 2), (1, 2))])


def test_cell_agreement():
    assert s.cell_agreement(CELL_LLM, CELL_HUMAN) == pytest.approx(CELL_POOLED, rel=1e-12)
    assert s.cell_agreement(CELL_LLM, CELL_HUMAN, CELL_GROUPS) == pytest.approx(CELL_WITHIN, rel=1e-12)
    # The correlation is blind to exaggeration; the slope is not.
    r, b = s.cell_agreement([1.0 + 3.0 * h for h in CELL_HUMAN], CELL_HUMAN)
    assert (r, b) == pytest.approx((1.0, 3.0))
    with pytest.raises(ValueError):
        s.cell_agreement([1.0, 1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        s.cell_agreement([1.0], [1.0])


def test_collapse_and_tree():
    assert s.collapse_repeats(COLLAPSE_IN) == COLLAPSE_OUT
    t = s.ReasoningTree.from_paths(TREE_PATHS)
    assert t.root.count == TREE_ROOT_COUNT
    node = t.node(["understanding"])
    assert node.count == TREE_UNDERSTANDING_COUNT
    assert dict(node.actions) == TREE_UNDERSTANDING_ACTIONS
    assert s.path_probability(t, ["understanding", "self_interest", "B1"]) == pytest.approx(TREE_PATH_PROB)
    assert s.path_probability(t, ["understanding", "altruism"]) == 0.0
    # Probabilities of complete paths sum to one.
    total = sum(s.path_probability(t, list(c) + [a]) for c, a in
                {(tuple(s.collapse_repeats(c)), a) for c, a in TREE_PATHS})
    assert total == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def test_state_prompt_and_validation():
    st = s.State("group", (200, 400, 600, 200), ingroup=True, cue="hometown")
    p = st.prompt()
    assert "same hometown" in p and "person A receives 200" in p and "<answer>" in p
    assert st.config_id == "200-400-600-200"
    with pytest.raises(ValueError):
        s.State("group", (0, 0, 0, 200))
    with pytest.raises(ValueError):
        s.State("direct", (0, 0, 200, 0), prior="maybe")
    assert len(s.make_states("group")) == 240 * 4 * 2
    assert len(s.make_states("direct")) == 240


def test_parse_response_and_coder():
    cot, act = s.parse_response("<reason>B2 gives the largest total for the two of us. "
                                "Person A would be better off with B2.</reason><answer> B2 </answer>")
    assert act == "B2" and len(cot) == 2
    assert s.LexiconCoder().code_all(cot) == ["social_welfare", "altruism"]
    assert s.parse_response("<reason>x.</reason> no answer")[1] is None
    assert s.LexiconCoder().code("Let me compare the two options.") == "understanding"


def test_llm_coder_prompt_roundtrip():
    prompt = s.build_coding_prompt("I keep more for myself.")
    assert "self_interest" in prompt and "<codes>" in prompt
    assert s.parse_coding_output("ok <codes>['self_interest', 'understanding']</codes>") == ["self_interest", "understanding"]
    with pytest.raises(ValueError):
        s.parse_coding_output("<codes>['greed']</codes>")


def test_simulator_lexicon_agreement_is_high_but_imperfect():
    m = s.SimulatedLLM(s.Profile("t"), seed=3)
    s.run_states(m, s.make_states("distributional")[:60], reps=3)
    coder = s.LexiconCoder()
    k = s.cohen_kappa([g for g, _ in m.gold], [coder.code(t) for _, t in m.gold])
    assert 0.7 < k < 0.99


def test_audit_recovers_ground_truth_directions():
    prof = s.Profile("t", w_self=0.5, w_other=0.35, temperature=1.2, ingroup_other=0.35, recip_direct=0.4, stance_other=0.0)
    recs = s.run_states(s.SimulatedLLM(prof, 1), s.make_states("distributional"), reps=3)
    eq3 = s.distributional_preferences(recs)
    assert eq3.coef[1] > 0.2 and eq3.coef[4] > 0.2          # self interest and welfare both matter
    grp = s.run_states(s.SimulatedLLM(prof, 2), s.make_states("group"), reps=1)
    eq4 = s.group_identity_effects(grp)
    assert eq4.coef[8] > 0 and eq4.coef[5] < 0               # ingroup raises welfare, lowers self interest
    rec, se = s.reciprocity(s.run_states(s.SimulatedLLM(prof, 3), s.make_states("direct"), reps=4))
    assert rec > 3 * se


def test_decorative_cot_has_no_interventional_effect():
    prof = s.Profile("deco", delta={k: 0.0 for k in s.Profile("x").delta}, stance_other=0.9, temperature=1.2)
    states = s.make_states("distributional")
    eff, se = s.intervention_effect(s.SimulatedLLM(prof, 5), states, "social_welfare", reps=8)
    assert abs(eff) < 3 * se + 0.02


def test_counterfactual_dpo_removes_ingroup_gap_only():
    prof = s.Profile("t", w_self=0.5, w_other=0.35, temperature=1.2, ingroup_other=0.6, stance_other=0.0)
    grp = s.make_states("group")[:400]
    pairs = s.counterfactual_pairs(s.SimulatedLLM(prof, 7), grp, reps=4)
    ad = s.fit_dpo_adapter(pairs, epsilon=0.0, scope="ingroup", steps=600)
    assert len(ad.theta) == 4
    out_state = s.State("distributional", (200, 400, 600, 200))
    assert ad.offset(out_state) == 0.0
    before = s.group_identity_effects(s.run_states(s.SimulatedLLM(prof, 8), grp, reps=3))
    after = s.group_identity_effects(s.run_states(s.AlignedLLM(prof, ad, 8), grp, reps=3))
    assert abs(after.coef[8]) < abs(before.coef[8])


def test_evaluate_spec():
    spec = s.AuditSpec("t", (s.Requirement("welfare", "w", "at_least", 0.3),
                             s.Requirement("blind", "g", "equivalent", 0.1),
                             s.Requirement("cap", "c", "at_most", 0.5)))
    rows = s.evaluate_spec(spec, {"w": (0.5, 0.05), "g": (0.02, 0.04), "c": (0.6, 0.01)})
    assert [r["pass"] for r in rows] == [True, True, False]
    with pytest.raises(KeyError):
        s.evaluate_spec(spec, {"w": (0.5, 0.05)})
    with pytest.raises(ValueError):
        s.Requirement("x", "y", "about", 1.0)


def test_composition_conversion_is_exact():
    prof = s.Profile("t", temperature=1.2)
    states = s.make_states("distributional")[:120]
    a = s.run_states(s.SimulatedLLM(prof, 1), states, reps=3)
    b = s.run_states(s.SimulatedLLM(s.Profile("u", temperature=1.2, p_prosocial=0.8), 2), states, reps=3)
    out = s.composition_conversion(a, b, lambda r: "W" if "social_welfare" in r.codes else "-")
    assert out["composition"] + out["conversion"] == pytest.approx(out["total"], abs=1e-12)
    same = s.composition_conversion(a, a, lambda r: "x")
    assert same["total"] == 0.0 and same["composition"] == 0.0


def test_reciprocity_is_difference_of_proportions_with_clustered_se():
    recs = s.run_states(s.SimulatedLLM(s.Profile("t", recip_direct=0.4), 4), s.make_states("direct")[:80], reps=6)
    est, se = s.reciprocity(recs)
    k = {"help": 0, "harm": 0}
    n = {"help": 0, "harm": 0}
    for r in recs:
        f = s.prosocial_flag(r)
        if f is not None:
            k[r.state.prior] += f
            n[r.state.prior] += 1
    d, se_iid = s.diff_proportions(k["help"], n["help"], k["harm"], n["harm"])
    assert est == pytest.approx(d, abs=1e-12)
    assert se > 0 and se_iid > 0
