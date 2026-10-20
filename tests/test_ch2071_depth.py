"""Tests for aiinaction.ch2071_depth, including the shared cross-language fixtures.

The fixtures below are the single source of truth: the Julia and Rust test
suites assert against the same numbers, which is what keeps the three
implementations at parity. Every quantity is deterministic (closed forms,
composite Simpson quadrature with fixed nodes, and fixed step RK4), so the
ports agree to floating point tolerance.

The two dimensional quadrature is slow in pure Python, so the suite calls
``correlation_map`` at its default resolution only twice.
"""
from __future__ import annotations

import math

import pytest

from aiinaction import ch2071_depth as d

TOL = 1e-9

# Quadrature fixtures.
GE_Z2 = 0.9999999999999986
GE_Z4 = 2.9999999999999942
GE_COS = 0.6065306597126348
GE_COS_N40_LIM6 = 0.6065306577125951
GE2_SQ_N40_LIM6 = 1.999999841083864
GE2_COSPROD_N20_LIM5 = 0.08206418533372323

# Mean field length and correlation maps.
LM_RELU = 0.9999999999999994
LM_TANH = 0.6414417355967605
LM_LINEAR = 0.9399999999999971
LM_TANH_Q2 = 1.2999393641598669
QSTAR_TANH = 0.41803720053418575
QSTAR_RELU = 0.40000000000256286
QSTAR_TANH_CHAOS = 2.195493928033813
CM_RELU = 0.6089975654758696
CM_TANH = 0.54974715475882
CM_TANH_N40 = 0.37652618700891255
CM_LINEAR_N40 = -0.27000003475872764
CHI1_TANH = 0.9386362681983815
CHI1_RELU = 0.9986701923986636
CHI1_LINEAR = 0.8999999999999972
XI_09 = 9.491221581029905

# ReLU kernel, residual networks and the continuous depth limit.
RCM_INPUTS = [-1.0, -0.5, 0.0, 0.5, 0.9, 1.0]
RCM = [0.0, 0.1089977810442294, 0.3183098861837907, 0.6089977810442293, 0.9095383988446721, 1.0]
TRAJ_PLAIN = [0.2, 0.4246975638632654, 0.5598220427417753, 0.6495461163619815, 0.7129656727390657, 0.7598110813130384]
TRAJ_RES = [0.2, 0.24493951277265308, 0.2860269749375514, 0.3237085998221533, 0.35836528648096716, 0.39032419037801597]
TRAJ_NEG = [-0.9, 0.009538398844672072, 0.3230935658001504, 0.49662004436592994]
COLLAPSE_PLAIN = 0.44413219804902115
COLLAPSE_RES = 11.103304951225528
ODE = 0.3839566908351801
ODE_COARSE = 0.2991227756181538
RV1 = 1024.0
RV2 = 1.3524069147107642

# Deep linear learning time.
DLT_DEPTHS = [1, 2, 3, 5, 10]
DLT = [0.0, 0.5493061443340553, 1.0179617754504038, 2.4934025172272927, 32.14439124998376]
DLT_OPTS = 2.9016436644053982
DLA_DEPTHS = [3, 5, 10]
DLA = [2.0, 2.6666666666666665, 32.0]
DLA_S = 2.777777777777778

# Residual paths and the sawtooth.
PLD4 = [0.0625, 0.25, 0.375, 0.25, 0.0625]
EPD = [0.19753086419753096, 0.3950617283950618, 0.29629629629629634, 0.09876543209876541, 0.012345679012345677]
EPD_ZERO = [1.0, 0.0, 0.0, 0.0]
SAW_XS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
SAW = [0.0, 0.8, 0.3999999999999999, 0.0, 0.40000000000000036, 0.0]


# --------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------

def test_activation_values():
    assert d.ACTIVATIONS == ("linear", "relu", "tanh")
    assert d.activation("linear", -1.5) == -1.5
    assert d.activation("relu", -1.5) == 0.0
    assert d.activation("relu", 2.0) == 2.0
    assert d.activation("relu", 0.0) == 0.0
    assert d.activation("tanh", 0.5) == pytest.approx(math.tanh(0.5), abs=1e-15)


def test_activation_derivative_values():
    assert d.activation_derivative("linear", 3.0) == 1.0
    assert d.activation_derivative("relu", 0.0) == 0.0
    assert d.activation_derivative("relu", 0.1) == 1.0
    assert d.activation_derivative("relu", -0.1) == 0.0
    assert d.activation_derivative("tanh", 0.5) == pytest.approx(1.0 - math.tanh(0.5) ** 2, abs=1e-15)


def test_activation_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown activation"):
        d.activation("sigmoid", 0.0)
    with pytest.raises(ValueError, match="unknown activation"):
        d.activation_derivative("gelu", 0.0)
    with pytest.raises(ValueError, match="unknown activation"):
        d.length_map(1.0, 1.0, 0.0, "softplus")


# --------------------------------------------------------------------------
# Gaussian quadrature
# --------------------------------------------------------------------------

def test_gaussian_expectation_matches_fixture_and_moments():
    assert d.gaussian_expectation(lambda z: z * z) == pytest.approx(GE_Z2, abs=TOL)
    assert d.gaussian_expectation(lambda z: z**4) == pytest.approx(GE_Z4, abs=TOL)
    assert d.gaussian_expectation(math.cos) == pytest.approx(GE_COS, abs=TOL)
    assert d.gaussian_expectation(math.cos, n=40, lim=6.0) == pytest.approx(GE_COS_N40_LIM6, abs=TOL)
    # Exact Gaussian moments: E[z^2] = 1, E[z^4] = 3, E[cos z] = exp(-1/2).
    assert GE_Z2 == pytest.approx(1.0, abs=1e-12)
    assert GE_Z4 == pytest.approx(3.0, abs=1e-12)
    assert GE_COS == pytest.approx(math.exp(-0.5), abs=1e-12)


def test_gaussian_expectation_2d_matches_fixture():
    got = d.gaussian_expectation_2d(lambda a, b: (a + b) ** 2, n=40, lim=6.0)
    assert got == pytest.approx(GE2_SQ_N40_LIM6, abs=TOL)
    got = d.gaussian_expectation_2d(lambda a, b: math.cos(a) * math.cos(2.0 * b), n=20, lim=5.0)
    assert got == pytest.approx(GE2_COSPROD_N20_LIM5, abs=TOL)
    # E[(z1 + z2)^2] = 2 for independent standard normals.
    assert GE2_SQ_N40_LIM6 == pytest.approx(2.0, abs=1e-6)


def test_quadrature_rejects_odd_or_tiny_panel_counts():
    with pytest.raises(ValueError, match="even integer"):
        d.gaussian_expectation(math.cos, n=3)
    with pytest.raises(ValueError, match="even integer"):
        d.gaussian_expectation(math.cos, n=0)
    with pytest.raises(ValueError, match="even integer"):
        d.gaussian_expectation_2d(lambda a, b: a * b, n=5)


# --------------------------------------------------------------------------
# Length map and its fixed point
# --------------------------------------------------------------------------

def test_length_map_matches_fixture():
    assert d.length_map(1.0, 2.0, 0.0, "relu") == pytest.approx(LM_RELU, abs=TOL)
    assert d.length_map(1.0, 1.5, 0.05, "tanh") == pytest.approx(LM_TANH, abs=TOL)
    assert d.length_map(0.7, 1.2, 0.1, "linear") == pytest.approx(LM_LINEAR, abs=TOL)
    assert d.length_map(2.0, 2.5, 0.0, "tanh") == pytest.approx(LM_TANH_Q2, abs=TOL)


def test_relu_length_map_is_critical_at_sigma_w2_two():
    # E[relu(sqrt(q) z)^2] = q / 2, so sigma_w^2 = 2 preserves the length.
    for q in (0.25, 1.0, 3.0):
        assert d.length_map(q, 2.0, 0.0, "relu") == pytest.approx(q, rel=1e-12)
    # A linear network maps q to sigma_w^2 q + sigma_b^2 exactly.
    assert d.length_map(0.7, 1.2, 0.1, "linear") == pytest.approx(1.2 * 0.7 + 0.1, abs=1e-12)


def test_length_map_rejects_invalid():
    with pytest.raises(ValueError, match="sigma_w2 > 0"):
        d.length_map(1.0, 0.0, 0.0, "relu")
    with pytest.raises(ValueError, match="sigma_w2 > 0"):
        d.length_map(1.0, 1.0, -0.1, "relu")
    with pytest.raises(ValueError, match="q must be nonnegative"):
        d.length_map(-1.0, 1.0, 0.0, "relu")


def test_length_fixed_point_matches_fixture():
    assert d.length_fixed_point(1.5, 0.05, "tanh") == pytest.approx(QSTAR_TANH, abs=TOL)
    assert d.length_fixed_point(1.5, 0.1, "relu") == pytest.approx(QSTAR_RELU, abs=TOL)
    assert d.length_fixed_point(4.0, 0.05, "tanh", q0=0.5) == pytest.approx(QSTAR_TANH_CHAOS, abs=TOL)


def test_length_fixed_point_is_a_fixed_point():
    # ReLU with sigma_w^2 < 2: q* = sigma_b^2 / (1 - sigma_w^2 / 2).
    assert QSTAR_RELU == pytest.approx(0.1 / (1.0 - 0.75), abs=1e-9)
    assert d.length_map(QSTAR_TANH, 1.5, 0.05, "tanh") == pytest.approx(QSTAR_TANH, abs=1e-10)


def test_length_fixed_point_raises_on_divergence_or_slow_convergence():
    with pytest.raises(RuntimeError, match="diverged"):
        d.length_fixed_point(3.0, 0.0, "relu")
    with pytest.raises(RuntimeError, match="did not converge"):
        d.length_fixed_point(1.5, 0.05, "tanh", max_iter=2)


# --------------------------------------------------------------------------
# Correlation map, chi_1 and the depth scale
# --------------------------------------------------------------------------

def test_correlation_map_matches_fixture_and_relu_closed_form():
    got = d.correlation_map(0.5, 1.0, 2.0, 0.0, "relu")
    assert got == pytest.approx(CM_RELU, abs=TOL)
    # Quadrature agrees with the arc cosine kernel at criticality.
    assert got == pytest.approx(d.relu_correlation_map(0.5), abs=1e-6)
    assert d.correlation_map(0.5, QSTAR_TANH, 1.5, 0.05, "tanh") == pytest.approx(CM_TANH, abs=TOL)


def test_correlation_map_coarse_grid_matches_fixture():
    got = d.correlation_map(0.3, QSTAR_TANH, 1.5, 0.05, "tanh", n=40)
    assert got == pytest.approx(CM_TANH_N40, abs=TOL)
    got = d.correlation_map(-0.4, 0.8, 1.3, 0.2, "linear", n=40)
    assert got == pytest.approx(CM_LINEAR_N40, abs=TOL)
    # Linear network: c' = (sigma_w^2 q c + sigma_b^2) / q.
    assert got == pytest.approx((1.3 * 0.8 * -0.4 + 0.2) / 0.8, abs=1e-6)


def test_correlation_map_rejects_invalid():
    with pytest.raises(ValueError, match="correlation must lie"):
        d.correlation_map(1.5, 1.0, 2.0, 0.0, "relu", n=4)
    with pytest.raises(ValueError, match="sigma_w2 > 0"):
        d.correlation_map(0.5, 1.0, -2.0, 0.0, "relu", n=4)
    with pytest.raises(ValueError, match="q must be positive"):
        d.correlation_map(0.5, 0.0, 2.0, 0.0, "relu", n=4)


def test_chi1_matches_fixture():
    assert d.chi1(QSTAR_TANH, 1.5, "tanh") == pytest.approx(CHI1_TANH, abs=TOL)
    assert d.chi1(1.0, 2.0, "relu") == pytest.approx(CHI1_RELU, abs=TOL)
    assert d.chi1(1.0, 0.9, "linear") == pytest.approx(CHI1_LINEAR, abs=TOL)
    # ReLU at sigma_w^2 = 2 is critical, chi_1 = 1 up to the quadrature error
    # of integrating a step function; linear chi_1 is sigma_w^2.
    assert CHI1_RELU == pytest.approx(1.0, abs=2e-3)
    assert CHI1_LINEAR == pytest.approx(0.9, abs=1e-12)


def test_chi1_rejects_invalid():
    with pytest.raises(ValueError, match="sigma_w2 > 0"):
        d.chi1(1.0, 0.0, "tanh")
    with pytest.raises(ValueError, match="q >= 0"):
        d.chi1(-1.0, 1.0, "tanh")


def test_correlation_depth_scale():
    assert d.correlation_depth_scale(0.9) == pytest.approx(XI_09, abs=TOL)
    assert d.correlation_depth_scale(1.0) == math.inf
    assert d.correlation_depth_scale(0.5) == pytest.approx(1.0 / math.log(2.0), abs=1e-15)
    with pytest.raises(ValueError, match="chi_c must be positive"):
        d.correlation_depth_scale(0.0)


# --------------------------------------------------------------------------
# ReLU correlation kernel
# --------------------------------------------------------------------------

def test_relu_correlation_map_matches_fixture():
    got = [d.relu_correlation_map(c) for c in RCM_INPUTS]
    assert got == pytest.approx(RCM, abs=TOL)
    assert d.relu_correlation_map(0.0) == pytest.approx(1.0 / math.pi, abs=1e-15)
    with pytest.raises(ValueError, match="correlation must lie"):
        d.relu_correlation_map(1.01)
    with pytest.raises(ValueError, match="correlation must lie"):
        d.relu_correlation_map(-1.01)


def test_relu_correlation_trajectory_matches_fixture():
    assert d.relu_correlation_trajectory(0.2, 5) == pytest.approx(TRAJ_PLAIN, abs=TOL)
    assert d.relu_correlation_trajectory(0.2, 5, alpha=0.5) == pytest.approx(TRAJ_RES, abs=TOL)
    assert d.relu_correlation_trajectory(-0.9, 3) == pytest.approx(TRAJ_NEG, abs=TOL)
    assert d.relu_correlation_trajectory(0.3, 0) == [0.3]
    # alpha = 0 switches every residual branch off.
    assert d.relu_correlation_trajectory(0.3, 4, alpha=0.0) == pytest.approx([0.3] * 5, abs=1e-15)


def test_relu_correlation_trajectory_rejects_invalid():
    with pytest.raises(ValueError, match="depth must be nonnegative"):
        d.relu_correlation_trajectory(0.2, -1)
    with pytest.raises(ValueError, match="alpha must be nonnegative"):
        d.relu_correlation_trajectory(0.2, 3, alpha=-0.1)


def test_plain_relu_correlation_collapses_like_inverse_square():
    traj = d.relu_correlation_trajectory(0.2, 2000)
    ratios = [(1.0 - traj[l]) / d.relu_collapse_asymptotic(l) for l in (100, 1000, 2000)]
    assert ratios[0] < ratios[1] < ratios[2] < 1.0
    assert ratios[2] == pytest.approx(1.0, abs=0.03)


def test_relu_collapse_asymptotic():
    assert d.relu_collapse_asymptotic(10) == pytest.approx(COLLAPSE_PLAIN, abs=TOL)
    assert d.relu_collapse_asymptotic(10, alpha=0.5) == pytest.approx(COLLAPSE_RES, abs=TOL)
    # Residual law with alpha = 1 is four times the plain law.
    assert d.relu_collapse_asymptotic(7.5, alpha=1.0) == pytest.approx(4.0 * d.relu_collapse_asymptotic(7.5), rel=1e-14)
    with pytest.raises(ValueError, match="depth must be positive"):
        d.relu_collapse_asymptotic(0)
    with pytest.raises(ValueError, match="alpha must be positive"):
        d.relu_collapse_asymptotic(10, alpha=0.0)


def test_relu_kernel_ode_matches_fixture():
    assert d.relu_kernel_ode(0.2, 1.0) == pytest.approx(ODE, abs=TOL)
    assert d.relu_kernel_ode(-0.3, 2.0, steps=50) == pytest.approx(ODE_COARSE, abs=TOL)
    assert d.relu_kernel_ode(0.4, 0.0) == 0.4
    with pytest.raises(ValueError, match="steps > 0"):
        d.relu_kernel_ode(0.2, 1.0, steps=0)
    with pytest.raises(ValueError, match="t >= 0"):
        d.relu_kernel_ode(0.2, -1.0)


def test_scaled_residual_trajectory_approaches_kernel_ode():
    target = d.relu_kernel_ode(0.2, 1.0)
    errors = []
    for depth in (10, 100, 1000):
        traj = d.relu_correlation_trajectory(0.2, depth, alpha=1.0 / math.sqrt(depth))
        errors.append(abs(traj[-1] - target))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 2e-4
    # The error is O(1 / L): each tenfold increase in depth cuts it about tenfold.
    assert errors[1] / errors[2] == pytest.approx(10.0, rel=0.05)


# --------------------------------------------------------------------------
# Residual variance
# --------------------------------------------------------------------------

def test_residual_variance():
    assert d.residual_variance(1.0, 10, 1.0) == pytest.approx(RV1, abs=TOL)
    assert d.residual_variance(0.5, 100, 0.1) == pytest.approx(RV2, abs=TOL)
    # alpha = 1 / sqrt(L) gives q_0 (1 + 1/L)^L, which tends to q_0 e.
    assert d.residual_variance(1.0, 100_000, 1.0 / math.sqrt(100_000)) == pytest.approx(math.e, rel=1e-4)
    with pytest.raises(ValueError, match="q0 >= 0"):
        d.residual_variance(-1.0, 10, 1.0)
    with pytest.raises(ValueError, match="depth >= 0"):
        d.residual_variance(1.0, -1, 1.0)
    with pytest.raises(ValueError, match="alpha >= 0"):
        d.residual_variance(1.0, 10, -1.0)


# --------------------------------------------------------------------------
# Deep linear networks
# --------------------------------------------------------------------------

def test_deep_linear_learning_time_matches_fixture():
    got = [d.deep_linear_learning_time(depth, 0.5) for depth in DLT_DEPTHS]
    assert got == pytest.approx(DLT, abs=TOL)
    assert d.deep_linear_learning_time(4, 0.3, s=2.0, frac=0.9, n=200) == pytest.approx(DLT_OPTS, abs=TOL)
    # Already past the target: no time needed.
    assert d.deep_linear_learning_time(1, 0.9) == 0.0


def test_deep_linear_learning_time_closed_forms():
    # N = 1: du/dt = s - u, so T = log((s - u0) / (s - frac s)).
    assert d.deep_linear_learning_time(1, 0.2) == pytest.approx(math.log(0.8 / 0.5), rel=1e-10)
    # N = 2: du/dt = 2 u (s - u), so T = (1/2) log(u1 (s - u0) / (u0 (s - u1))).
    u0 = 0.25
    assert DLT[1] == pytest.approx(0.5 * math.log(0.5 * (1.0 - u0) / (u0 * 0.5)), rel=1e-10)


def test_deep_linear_learning_time_approaches_asymptotic():
    ratios = [
        d.deep_linear_learning_time(depth, 0.5) / d.deep_linear_time_asymptotic(depth, 0.5)
        for depth in (5, 10, 20)
    ]
    gaps = [abs(r - 1.0) for r in ratios]
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-4


def test_deep_linear_learning_time_rejects_invalid():
    with pytest.raises(ValueError, match="depth must be at least 1"):
        d.deep_linear_learning_time(0, 0.5)
    with pytest.raises(ValueError, match="a0 > 0"):
        d.deep_linear_learning_time(3, 0.0)
    with pytest.raises(ValueError, match="s > 0"):
        d.deep_linear_learning_time(3, 0.5, s=0.0)
    with pytest.raises(ValueError, match="0 < frac < 1"):
        d.deep_linear_learning_time(3, 0.5, frac=1.0)
    with pytest.raises(ValueError, match="even integer"):
        d.deep_linear_learning_time(3, 0.5, n=7)


def test_deep_linear_time_asymptotic():
    got = [d.deep_linear_time_asymptotic(depth, 0.5) for depth in DLA_DEPTHS]
    assert got == pytest.approx(DLA, abs=TOL)
    assert d.deep_linear_time_asymptotic(4, 0.3, s=2.0) == pytest.approx(DLA_S, abs=TOL)
    with pytest.raises(ValueError, match="depth > 2"):
        d.deep_linear_time_asymptotic(2, 0.5)
    with pytest.raises(ValueError, match="a0 > 0"):
        d.deep_linear_time_asymptotic(3, -0.5)


# --------------------------------------------------------------------------
# Residual networks as ensembles of paths
# --------------------------------------------------------------------------

def test_path_length_distribution():
    assert d.path_length_distribution(4) == pytest.approx(PLD4, abs=TOL)
    assert d.path_length_distribution(0) == [1.0]
    for n in (1, 7, 54):
        dist = d.path_length_distribution(n)
        assert len(dist) == n + 1
        assert sum(dist) == pytest.approx(1.0, abs=1e-12)
        mean = sum(k * p for k, p in enumerate(dist))
        assert mean == pytest.approx(n / 2.0, abs=1e-10)
    with pytest.raises(ValueError, match="n_blocks must be nonnegative"):
        d.path_length_distribution(-1)


def test_effective_path_distribution():
    assert d.effective_path_distribution(4, 0.5) == pytest.approx(EPD, abs=TOL)
    assert d.effective_path_distribution(3, 0.0) == pytest.approx(EPD_ZERO, abs=TOL)
    # rho = 1 recovers the raw path count law.
    assert d.effective_path_distribution(6, 1.0) == pytest.approx(d.path_length_distribution(6), abs=1e-15)
    for n, rho in ((4, 0.5), (54, 0.2), (10, 3.0)):
        dist = d.effective_path_distribution(n, rho)
        assert sum(dist) == pytest.approx(1.0, abs=1e-12)
        mean = sum(k * p for k, p in enumerate(dist))
        assert mean == pytest.approx(n * rho / (1.0 + rho), abs=1e-10)
    with pytest.raises(ValueError, match="n_blocks >= 0"):
        d.effective_path_distribution(-1, 0.5)
    with pytest.raises(ValueError, match="rho >= 0"):
        d.effective_path_distribution(4, -0.5)


# --------------------------------------------------------------------------
# Telgarsky sawtooth
# --------------------------------------------------------------------------

def test_sawtooth_matches_fixture():
    assert [d.sawtooth(x, 3) for x in SAW_XS] == pytest.approx(SAW, abs=TOL)
    assert d.sawtooth(0.37, 0) == 0.37
    # One layer is the tent map.
    assert d.sawtooth(0.25, 1) == pytest.approx(0.5, abs=1e-15)
    assert d.sawtooth(0.75, 1) == pytest.approx(0.5, abs=1e-15)
    with pytest.raises(ValueError, match="depth must be nonnegative"):
        d.sawtooth(0.5, -1)


def test_sawtooth_has_two_to_the_depth_pieces():
    for depth in range(1, 7):
        grid = 4 * 2**depth
        ys = [d.sawtooth(i / grid, depth) for i in range(grid + 1)]
        assert d.count_linear_pieces(ys) == 2**depth


def test_count_linear_pieces():
    xs = [i / 64 for i in range(65)]
    assert d.count_linear_pieces([2.0 * x + 1.0 for x in xs]) == 1
    # A breakpoint inside a grid cell still counts once.
    assert d.count_linear_pieces([abs(x - 0.33) for x in xs]) == 2
    assert d.count_linear_pieces([1.0, 2.0, 3.0]) == 1
    with pytest.raises(ValueError, match="at least three samples"):
        d.count_linear_pieces([1.0, 2.0])
