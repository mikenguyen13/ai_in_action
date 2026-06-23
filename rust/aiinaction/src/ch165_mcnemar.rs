//! McNemar's test for comparing two classifiers on a shared test set (Rust).
//!
//! Mirrors the Python module `aiinaction.ch165_mcnemar` and the Julia module
//! `AIInAction.Ch165Mcnemar`. The shared fixtures in the tests below match the
//! Python/Julia suites, which is what keeps the three implementations at parity.
//!
//! Given two classifiers evaluated on the same test set, the 2x2 correctness table
//! is
//!
//! ```text
//!                  B correct   B wrong
//!     A correct        a           b
//!     A wrong          c           d
//! ```
//!
//! Only the discordant cells `b` and `c` carry information. Under the null
//! hypothesis of equal error rates, `b ~ Binomial(b + c, 1/2)`. Two variants are
//! provided: the chi-squared approximation with Edwards' continuity correction, and
//! the exact two-sided binomial test on `min(b, c)`.
//!
//! This is a std-only implementation. The chi-squared survival function is computed
//! from the regularized upper incomplete gamma function `Q(1/2, x/2)` (series plus
//! Lentz continued fraction), which agrees with the Python/Julia results to machine
//! precision.

/// The 2x2 correctness contingency table for two classifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContingencyTable {
    /// Both classifiers correct.
    pub a: u64,
    /// `A` correct, `B` wrong (discordant).
    pub b: u64,
    /// `A` wrong, `B` correct (discordant).
    pub c: u64,
    /// Both classifiers wrong.
    pub d: u64,
}

impl ContingencyTable {
    /// Total number of examples.
    pub fn n(&self) -> u64 {
        self.a + self.b + self.c + self.d
    }
    /// Number of discordant examples `b + c`.
    pub fn n_discordant(&self) -> u64 {
        self.b + self.c
    }
}

/// The outcome of McNemar's test.
#[derive(Clone, Debug, PartialEq)]
pub struct McNemarResult {
    /// Chi-squared statistic, or `min(b, c)` as f64 for the exact variant.
    pub statistic: f64,
    /// Two-sided p-value under `H_0`.
    pub p_value: f64,
    /// Either `"chi2"` or `"exact"`.
    pub method: &'static str,
    /// Discordant count favouring `A`.
    pub b: u64,
    /// Discordant count favouring `B`.
    pub c: u64,
}

/// Builds the 2x2 correctness table from two per-example correctness vectors.
pub fn contingency_table(
    correct_a: &[bool],
    correct_b: &[bool],
) -> Result<ContingencyTable, String> {
    if correct_a.len() != correct_b.len() {
        return Err(format!(
            "length mismatch: len(correct_a)={} != len(correct_b)={}",
            correct_a.len(),
            correct_b.len()
        ));
    }
    if correct_a.is_empty() {
        return Err("inputs must be non-empty".to_string());
    }
    let (mut a, mut b, mut c, mut d) = (0u64, 0u64, 0u64, 0u64);
    for (&ca, &cb) in correct_a.iter().zip(correct_b.iter()) {
        match (ca, cb) {
            (true, true) => a += 1,
            (true, false) => b += 1,
            (false, true) => c += 1,
            (false, false) => d += 1,
        }
    }
    Ok(ContingencyTable { a, b, c, d })
}

/// Natural log of the gamma function (Lanczos approximation).
fn ln_gamma(x: f64) -> f64 {
    const COF: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_9e-2,
        -0.539_523_938_495_3e-5,
    ];
    let mut y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000_000_000_190_015;
    for &c in COF.iter() {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

/// Regularized upper incomplete gamma function `Q(s, x) = 1 - P(s, x)`.
fn gammq(s: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < s + 1.0 {
        // Series expansion for the lower regularized gamma P, then return 1 - P.
        let mut ap = s;
        let mut sum = 1.0 / s;
        let mut del = sum;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        let p = sum * (-x + s * x.ln() - ln_gamma(s)).exp();
        1.0 - p
    } else {
        // Lentz's continued fraction for the upper regularized gamma Q.
        let tiny = 1e-300;
        let mut b = x + 1.0 - s;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - s);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        (-x + s * x.ln() - ln_gamma(s)).exp() * h
    }
}

/// Survival function of the chi-squared distribution with one degree of freedom.
fn chi2_sf_1dof(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gammq(0.5, x / 2.0)
}

/// Probability mass `C(n, k) (1/2)^n`, evaluated in log space for stability.
fn binom_pmf_half(k: u64, n: u64) -> f64 {
    let log_coef = ln_gamma((n + 1) as f64)
        - ln_gamma((k + 1) as f64)
        - ln_gamma((n - k + 1) as f64);
    (log_coef - (n as f64) * 2.0_f64.ln()).exp()
}

/// Two-sided exact binomial p-value for `min(b, c)` under `Binomial(b + c, 1/2)`.
fn exact_two_sided_p(b: u64, c: u64) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    let mut lower_tail = 0.0;
    for i in 0..=k {
        lower_tail += binom_pmf_half(i, n);
    }
    (2.0 * lower_tail).min(1.0)
}

/// Runs McNemar's test on the discordant counts `b` and `c`.
///
/// `exact = None` selects the exact test when `b + c < 25` and the chi-squared
/// approximation otherwise. `correction` toggles Edwards' continuity correction on
/// the chi-squared statistic (ignored by the exact variant).
pub fn mcnemar_test(
    b: u64,
    c: u64,
    exact: Option<bool>,
    correction: bool,
) -> Result<McNemarResult, String> {
    let n = b + c;
    if n == 0 {
        return Err(
            "McNemar's test is undefined when b + c = 0 (no discordant pairs)".to_string(),
        );
    }
    let use_exact = match exact {
        Some(v) => v,
        None => n < 25,
    };

    if use_exact {
        let p = exact_two_sided_p(b, c);
        return Ok(McNemarResult {
            statistic: b.min(c) as f64,
            p_value: p,
            method: "exact",
            b,
            c,
        });
    }

    let diff = if b >= c { b - c } else { c - b } as f64;
    let delta = if correction { (diff - 1.0).max(0.0) } else { diff };
    let chi2 = (delta * delta) / (n as f64);
    let p = chi2_sf_1dof(chi2);
    Ok(McNemarResult {
        statistic: chi2,
        p_value: p,
        method: "chi2",
        b,
        c,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // Shared fixtures: identical to the Python and Julia test suites.
    fn correctness() -> (Vec<bool>, Vec<bool>) {
        let a_int: [u8; 12] = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1];
        let b_int: [u8; 12] = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0];
        (
            a_int.iter().map(|&v| v != 0).collect(),
            b_int.iter().map(|&v| v != 0).collect(),
        )
    }

    #[test]
    fn contingency_table_matches_fixture() {
        let (ca, cb) = correctness();
        let t = contingency_table(&ca, &cb).unwrap();
        assert_eq!((t.a, t.b, t.c, t.d), (2, 6, 3, 1));
        assert_eq!(t.n(), 12);
        assert_eq!(t.n_discordant(), 9);
    }

    #[test]
    fn chi2_with_correction_matches_fixture() {
        let cases: [(u64, u64, f64, f64); 3] = [
            (12, 5, 2.1176470588235294, 0.14561009539686698),
            (30, 15, 4.355555555555555, 0.03688842570704986),
            (3, 1, 0.25, 0.6170750774519738),
        ];
        for (b, c, stat, p) in cases {
            let r = mcnemar_test(b, c, Some(false), true).unwrap();
            assert_eq!(r.method, "chi2");
            assert!((r.statistic - stat).abs() < TOL);
            assert!((r.p_value - p).abs() < TOL);
        }
    }

    #[test]
    fn chi2_without_correction_matches_fixture() {
        let cases: [(u64, u64, f64, f64); 3] = [
            (12, 5, 2.8823529411764706, 0.08955507441364255),
            (30, 15, 5.0, 0.025347318677468256),
            (3, 1, 1.0, 0.31731050786291404),
        ];
        for (b, c, stat, p) in cases {
            let r = mcnemar_test(b, c, Some(false), false).unwrap();
            assert!((r.statistic - stat).abs() < TOL);
            assert!((r.p_value - p).abs() < TOL);
        }
    }

    #[test]
    fn exact_matches_fixture() {
        let cases: [(u64, u64, f64); 3] = [
            (12, 5, 0.1434631347656256),
            (30, 15, 0.03569780355519456),
            (3, 1, 0.6249999999999994),
        ];
        for (b, c, p) in cases {
            let r = mcnemar_test(b, c, Some(true), true).unwrap();
            assert_eq!(r.method, "exact");
            assert!((r.statistic - (b.min(c) as f64)).abs() < TOL);
            assert!((r.p_value - p).abs() < TOL);
        }
    }

    #[test]
    fn auto_method_selection() {
        assert_eq!(mcnemar_test(12, 5, None, true).unwrap().method, "exact");
        assert_eq!(mcnemar_test(30, 15, None, true).unwrap().method, "chi2");
        assert_eq!(mcnemar_test(3, 1, None, true).unwrap().method, "exact");
    }

    #[test]
    fn symmetry_in_b_and_c() {
        let r1 = mcnemar_test(30, 15, Some(false), true).unwrap();
        let r2 = mcnemar_test(15, 30, Some(false), true).unwrap();
        assert!((r1.p_value - r2.p_value).abs() < TOL);
        assert!((r1.statistic - r2.statistic).abs() < TOL);
        let e1 = mcnemar_test(12, 5, Some(true), true).unwrap();
        let e2 = mcnemar_test(5, 12, Some(true), true).unwrap();
        assert!((e1.p_value - e2.p_value).abs() < TOL);
    }

    #[test]
    fn equal_discordant_counts_give_p_one() {
        let r = mcnemar_test(10, 10, Some(false), true).unwrap();
        assert!((r.statistic - 0.0).abs() < TOL);
        assert!((r.p_value - 1.0).abs() < TOL);
        let e = mcnemar_test(10, 10, Some(true), true).unwrap();
        assert!((e.p_value - 1.0).abs() < TOL);
    }

    #[test]
    fn zero_discordant_errors() {
        assert!(mcnemar_test(0, 0, None, true).is_err());
    }

    #[test]
    fn table_length_mismatch_errors() {
        assert!(contingency_table(&[true, false, true], &[true, false]).is_err());
    }

    #[test]
    fn table_empty_errors() {
        assert!(contingency_table(&[], &[]).is_err());
    }
}
