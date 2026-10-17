//! Human-in-the-loop sequential experiments for drug discovery (Rust).
//!
//! Mirrors the Python module `aiinaction.ch526_hitl_discovery` and the Julia module
//! `AIInAction.Ch526HitlDiscovery`, after He, Hua, Wang, and Zheng (2026),
//! "Collaborative Intelligence in Sequential Experiments: A Human-in-the-Loop
//! Framework for Drug Discovery" (Information Systems Research 37(3), 1390-1415).
//! The shared fixtures in the tests below match the Python/Julia suites, which
//! keeps the implementations at parity (indices are 0-based, as in Python).
//!
//! The problem: a pool of `N` candidate molecules, a budget of `B` lab tests per
//! round for `R` rounds, and a "target" is a molecule whose `K` binary properties
//! are all positive. Each round a surrogate trained on the labels so far hands
//! the expert two shortlists: an **uncertainty list** (highest epistemic
//! uncertainty, [`decompose_uncertainty`]) and a **search list** (highest upper
//! confidence bound, [`ucb_scores`]). The expert picks the `B` molecules that are
//! sent to the lab ([`expert_select`]).
//!
//! Ported here: the pure numeric core. [`binary_entropy`],
//! [`decompose_uncertainty`], [`ucb_scores`], [`top_k`], [`merge_shortlists`],
//! [`Expert`] with [`Expert::reveal`] and [`expert_select`], [`is_target`],
//! [`hit_rate`], [`recall_rate`], [`BootstrapLogisticEnsemble`], and the nine
//! [`Strategy`] variants run by [`simulate`].
//!
//! Python only (not ported): SATPdb download and loading, ESM2 embeddings, task
//! construction from SATPdb, and the torch `BayesianMLP` surrogate.
//!
//! This is a std-only implementation. Random numbers come from a small hand
//! written SplitMix64 generator ([`Rng`]), so simulations are deterministic for a
//! seed but are not bit-identical to NumPy's generator; the integration tests
//! are statistical. Errors are `Err(String)` where Python raises `ValueError` or
//! `AssertionError`.

use std::f64::consts::LN_2;

/// Clipping constant for the entropy, matching the Python `EPS`.
pub const EPS: f64 = 1e-12;

/// Natural log of 2, re-exported for callers that scale entropy to `[0, 1]`.
pub const LN2: f64 = LN_2;

/// A posterior sample tensor of shape `S x N x K` (draws, molecules, properties).
pub type Samples = Vec<Vec<Vec<f64>>>;

/// An `N x K` matrix stored as rows.
pub type Matrix = Vec<Vec<f64>>;

// --------------------------------------------------------------------------
// Deterministic RNG
// --------------------------------------------------------------------------

/// SplitMix64 pseudo random generator (Steele, Lea, Flood 2014).
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Creates a generator from a seed.
    pub fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform draw in `[0, 1)` with 53 bits of resolution.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `[0, n)`; `n` must be positive.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Standard normal draw (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = 1.0 - self.next_f64(); // in (0, 1]
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// In-place Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i + 1);
            items.swap(i, j);
        }
    }

    /// `size` distinct elements of `items`, sampled without replacement.
    pub fn choice(&mut self, items: &[usize], size: usize) -> Vec<usize> {
        let size = size.min(items.len());
        let mut pool = items.to_vec();
        for i in 0..size {
            let j = i + self.below(pool.len() - i);
            pool.swap(i, j);
        }
        pool.truncate(size);
        pool
    }
}

// --------------------------------------------------------------------------
// Uncertainty decomposition and scores
// --------------------------------------------------------------------------

/// Entropy of a Bernoulli(`p`) in nats, with `0 log 0 = 0` (via clipping).
pub fn binary_entropy(p: f64) -> f64 {
    let p = p.clamp(EPS, 1.0 - EPS);
    -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
}

/// Split predictive uncertainty into data and model parts.
///
/// `prob_samples[s][n][k]` is draw `s` of P(property `k` is positive) for
/// molecule `n`. Returns `(mu, aleatoric, epistemic)`, each `N x K`:
/// the posterior mean, the expected entropy `E H[p]` (label noise), and the
/// mutual information `H[E p] - E H[p]` (what more data can teach the model).
pub fn decompose_uncertainty(prob_samples: &[Vec<Vec<f64>>]) -> Result<(Matrix, Matrix, Matrix), String> {
    let s = prob_samples.len();
    if s == 0 {
        return Err("prob_samples must contain at least one draw".to_string());
    }
    let n = prob_samples[0].len();
    let k = prob_samples[0].first().map_or(0, |r| r.len());
    for draw in prob_samples {
        if draw.len() != n || draw.iter().any(|r| r.len() != k) {
            return Err("prob_samples must have a consistent S x N x K shape".to_string());
        }
    }
    let sf = s as f64;
    let mut mu = vec![vec![0.0; k]; n];
    let mut ale = vec![vec![0.0; k]; n];
    let mut epi = vec![vec![0.0; k]; n];
    for draw in prob_samples {
        for i in 0..n {
            for j in 0..k {
                mu[i][j] += draw[i][j];
                ale[i][j] += binary_entropy(draw[i][j]);
            }
        }
    }
    for i in 0..n {
        for j in 0..k {
            mu[i][j] /= sf;
            ale[i][j] /= sf;
            epi[i][j] = (binary_entropy(mu[i][j]) - ale[i][j]).max(0.0);
        }
    }
    Ok((mu, ale, epi))
}

/// Search score `sum_k [mu_k + sqrt(beta) * sigma_d_k]` per molecule (paper
/// Eq. 16-18). For a Bernoulli head pass `sigma_d = sqrt(mu (1 - mu))`.
pub fn ucb_scores(mu: &[Vec<f64>], sigma_d: &[Vec<f64>], beta: f64) -> Result<Vec<f64>, String> {
    if !(beta.is_finite() && beta >= 0.0) {
        return Err(format!("beta must be a non-negative finite number, got {}", beta));
    }
    if mu.len() != sigma_d.len() || mu.iter().zip(sigma_d).any(|(a, b)| a.len() != b.len()) {
        return Err("mu and sigma_d must have the same shape".to_string());
    }
    let sb = beta.sqrt();
    Ok(mu
        .iter()
        .zip(sigma_d)
        .map(|(m, s)| m.iter().zip(s).fold(0.0, |acc, (&a, &b)| acc + (a + sb * b)))
        .collect())
}

/// Indices of the `k` largest scores, ties broken by lower index.
///
/// `exclude` is a mask of molecules that are unavailable (already tested).
/// Fewer than `k` indices come back only if fewer are available.
pub fn top_k(scores: &[f64], k: usize, exclude: Option<&[bool]>) -> Vec<usize> {
    let s: Vec<f64> = scores
        .iter()
        .enumerate()
        .map(|(i, &v)| match exclude {
            Some(m) if m.get(i).copied().unwrap_or(false) => f64::NEG_INFINITY,
            _ => v,
        })
        .collect();
    let n_avail = s.iter().filter(|v| v.is_finite()).count();
    let k = k.min(n_avail);
    let mut order: Vec<usize> = (0..s.len()).collect();
    order.sort_by(|&a, &b| {
        s[b].partial_cmp(&s[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.truncate(k);
    order
}

/// Union of the two shortlists, search list first, order preserved, no repeats.
pub fn merge_shortlists(uncertainty_idx: &[usize], search_idx: &[usize]) -> Vec<usize> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for &i in search_idx.iter().chain(uncertainty_idx) {
        if seen.insert(i) {
            out.push(i);
        }
    }
    out
}

// --------------------------------------------------------------------------
// The expert
// --------------------------------------------------------------------------

/// A simulated domain expert with partial, private knowledge.
///
/// The expert knows the true target status of a molecule with probability
/// `knowledge`, decided once per molecule ([`Expert::reveal`]). `explore_share`
/// is the fraction of the unknown molecules filled from the uncertainty list
/// rather than the search list.
#[derive(Clone, Debug, PartialEq)]
pub struct Expert {
    /// Probability that the expert knows a given molecule.
    pub knowledge: f64,
    /// Share of unknown slots filled from the uncertainty list.
    pub explore_share: f64,
    /// Which molecules the expert knows (set by [`Expert::reveal`]).
    pub known_mask: Option<Vec<bool>>,
    /// True target status of every molecule (set by [`Expert::reveal`]).
    pub known_target: Option<Vec<bool>>,
}

impl Expert {
    /// Creates an expert with no revealed knowledge yet.
    pub fn new(knowledge: f64, explore_share: f64) -> Self {
        Expert {
            knowledge,
            explore_share,
            known_mask: None,
            known_target: None,
        }
    }

    /// Fixes, once, which molecules the expert has private knowledge about.
    pub fn reveal(&mut self, is_target_vec: &[bool], rng: &mut Rng) {
        self.known_mask = Some(
            (0..is_target_vec.len())
                .map(|_| rng.next_f64() < self.knowledge)
                .collect(),
        );
        self.known_target = Some(is_target_vec.to_vec());
    }
}

/// Round half to even, like Python's `round`.
fn round_half_even(x: f64) -> usize {
    let f = x.floor();
    let d = x - f;
    let r = if d > 0.5 || (d == 0.5 && f % 2.0 != 0.0) {
        f + 1.0
    } else {
        f
    };
    r.max(0.0) as usize
}

/// Choose the molecules sent to the lab.
///
/// Rules, in order: (1) every molecule the expert knows to be a target is taken
/// first; (2) known non-targets are dropped; (3) the remaining slots go to
/// unknown molecules, split between the search and uncertainty lists by
/// `explore_share`; (4) if slots are still empty, they are filled from the
/// dropped molecules so the budget is always spent. Errors if
/// [`Expert::reveal`] has not been called.
pub fn expert_select(
    expert: &Expert,
    search_list: &[usize],
    uncertainty_list: &[usize],
    budget: usize,
) -> Result<Vec<usize>, String> {
    let (km, kt) = match (&expert.known_mask, &expert.known_target) {
        (Some(m), Some(t)) => (m, t),
        _ => return Err("call Expert::reveal first".to_string()),
    };
    let pool = merge_shortlists(uncertainty_list, search_list);
    let mut chosen: Vec<usize> = pool
        .iter()
        .copied()
        .filter(|&i| km[i] && kt[i])
        .take(budget)
        .collect();
    let mut taken: std::collections::HashSet<usize> = chosen.iter().copied().collect();

    let unknown = |seq: &[usize], taken: &std::collections::HashSet<usize>| -> Vec<usize> {
        seq.iter()
            .copied()
            .filter(|i| !taken.contains(i) && !km[*i])
            .collect()
    };

    let srch = unknown(search_list, &taken);
    let unc = unknown(uncertainty_list, &taken);
    let n_left = budget - chosen.len();
    let n_unc = round_half_even(expert.explore_share * n_left as f64).min(n_left);
    let n_srch = n_left - n_unc;
    for (lst, mut n) in [(&srch, n_srch), (&unc, n_unc)] {
        for &i in lst.iter() {
            if n == 0 {
                break;
            }
            if taken.insert(i) {
                chosen.push(i);
                n -= 1;
            }
        }
    }
    // Top up from whatever is left among unknowns, then from known non-targets.
    let both: Vec<usize> = search_list.iter().chain(uncertainty_list).copied().collect();
    for i in unknown(&both, &taken) {
        if chosen.len() >= budget {
            break;
        }
        if taken.insert(i) {
            chosen.push(i);
        }
    }
    for &i in &pool {
        if chosen.len() >= budget {
            break;
        }
        if taken.insert(i) {
            chosen.push(i);
        }
    }
    chosen.truncate(budget);
    Ok(chosen)
}

// --------------------------------------------------------------------------
// Metrics
// --------------------------------------------------------------------------

/// A molecule is a target when every one of its `K` properties is positive.
pub fn is_target(labels: &[Vec<u8>]) -> Vec<bool> {
    labels.iter().map(|row| row.iter().all(|&v| v != 0)).collect()
}

/// Final hit rate `S / (B R)`: targets found per lab test spent.
pub fn hit_rate(n_hits: usize, budget_per_round: usize, rounds: usize) -> Result<f64, String> {
    if budget_per_round == 0 || rounds == 0 {
        return Err("budget_per_round and rounds must be positive".to_string());
    }
    Ok(n_hits as f64 / (budget_per_round * rounds) as f64)
}

/// Share of all targets in the pool found so far.
pub fn recall_rate(n_hits: usize, n_targets_in_pool: usize) -> Result<f64, String> {
    if n_targets_in_pool == 0 {
        return Err("n_targets_in_pool must be positive".to_string());
    }
    Ok(n_hits as f64 / n_targets_in_pool as f64)
}

// --------------------------------------------------------------------------
// Bootstrap logistic ensemble surrogate
// --------------------------------------------------------------------------

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Bootstrap ensemble of L2-regularised logistic regressions, one per property.
///
/// `n_members` members are fitted to bootstrap resamples, so their disagreement
/// is a crude posterior over parameters. [`predict_samples`](Self::predict_samples)
/// returns the `S x N x K` tensor that [`decompose_uncertainty`] consumes.
#[derive(Clone, Debug)]
pub struct BootstrapLogisticEnsemble {
    /// Number of bootstrap members `S`.
    pub n_members: usize,
    /// L2 penalty.
    pub l2: f64,
    /// Gradient-descent step size.
    pub lr: f64,
    /// Full-batch gradient steps per member.
    pub epochs: usize,
    rng: Rng,
    /// Fitted weights, shape `S x K x (D + 1)` (last column is the intercept).
    w: Option<Vec<Vec<Vec<f64>>>>,
}

impl BootstrapLogisticEnsemble {
    /// Creates an ensemble. The Python defaults are `n_members = 10`,
    /// `l2 = 1e-2`, `lr = 0.5`, `epochs = 200`, `seed = 0`.
    pub fn new(n_members: usize, l2: f64, lr: f64, epochs: usize, seed: u64) -> Self {
        BootstrapLogisticEnsemble {
            n_members,
            l2,
            lr,
            epochs,
            rng: Rng::new(seed),
            w: None,
        }
    }

    /// An ensemble with the Python default hyperparameters.
    pub fn with_seed(seed: u64) -> Self {
        Self::new(10, 1e-2, 0.5, 200, seed)
    }

    /// Fits every member to its own bootstrap resample of `(x, y)`.
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[Vec<u8>]) -> Result<(), String> {
        let n = x.len();
        if n == 0 || n != y.len() {
            return Err("x and y must be non-empty and have the same length".to_string());
        }
        let d = x[0].len();
        let k = y[0].len();
        if x.iter().any(|r| r.len() != d) || y.iter().any(|r| r.len() != k) {
            return Err("x and y must be rectangular".to_string());
        }
        let nf = n as f64;
        let mut all = Vec::with_capacity(self.n_members);
        for _ in 0..self.n_members {
            let idx: Vec<usize> = (0..n).map(|_| self.rng.below(n)).collect();
            let mut ws = vec![vec![0.0; d + 1]; k];
            for _ in 0..self.epochs {
                let mut grad = vec![vec![0.0; d + 1]; k];
                for &r in &idx {
                    let xr = &x[r];
                    for (j, gj) in grad.iter_mut().enumerate() {
                        let mut z = ws[j][d];
                        for t in 0..d {
                            z += xr[t] * ws[j][t];
                        }
                        let e = sigmoid(z) - y[r][j] as f64;
                        for t in 0..d {
                            gj[t] += e * xr[t];
                        }
                        gj[d] += e;
                    }
                }
                for j in 0..k {
                    for t in 0..=d {
                        let g = grad[j][t] / nf + self.l2 * ws[j][t];
                        ws[j][t] -= self.lr * g;
                    }
                }
            }
            all.push(ws);
        }
        self.w = Some(all);
        Ok(())
    }

    /// Posterior draws of P(positive), shape `S x N x K`.
    pub fn predict_samples(&self, x: &[Vec<f64>]) -> Result<Samples, String> {
        let w = self
            .w
            .as_ref()
            .ok_or_else(|| "call fit before predict_samples".to_string())?;
        let mut out = Vec::with_capacity(w.len());
        for ws in w {
            let d = ws[0].len() - 1;
            let mut draw = Vec::with_capacity(x.len());
            for xr in x {
                if xr.len() != d {
                    return Err(format!("expected {} features, got {}", d, xr.len()));
                }
                draw.push(
                    ws.iter()
                        .map(|wj| {
                            let mut z = wj[d];
                            for t in 0..d {
                                z += xr[t] * wj[t];
                            }
                            sigmoid(z)
                        })
                        .collect(),
                );
            }
            out.push(draw);
        }
        Ok(out)
    }
}

// --------------------------------------------------------------------------
// The sequential-experiment simulator
// --------------------------------------------------------------------------

/// Maps a feature matrix to an `S x N x K` tensor of posterior draws.
pub type Predictor = Box<dyn Fn(&[Vec<f64>]) -> Samples>;

/// The nine selection strategies compared in the chapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Strategy {
    /// B random untested molecules each round.
    Random,
    /// Top B by posterior mean.
    Search,
    /// Top B by epistemic uncertainty.
    ActiveLearning,
    /// Top B by the search score.
    Ucb,
    /// Random for `warmup_rounds`, then UCB.
    RandomThenUcb,
    /// Active learning for `warmup_rounds`, then UCB.
    AlThenUcb,
    /// The expert screens `human_view` random molecules, using private knowledge only.
    HumanOnly,
    /// Half of B from the search list, the rest from the uncertainty list, no override.
    AlgorithmOnly,
    /// The proposed framework: the expert chooses B from the union of the shortlists.
    Hitl,
}

impl Strategy {
    /// All strategies, in the order of the Python `STRATEGIES` tuple.
    pub const ALL: [Strategy; 9] = [
        Strategy::Random,
        Strategy::Search,
        Strategy::ActiveLearning,
        Strategy::Ucb,
        Strategy::RandomThenUcb,
        Strategy::AlThenUcb,
        Strategy::HumanOnly,
        Strategy::AlgorithmOnly,
        Strategy::Hitl,
    ];

    /// The Python string name of the strategy.
    pub fn as_str(&self) -> &'static str {
        match self {
            Strategy::Random => "random",
            Strategy::Search => "search",
            Strategy::ActiveLearning => "active_learning",
            Strategy::Ucb => "ucb",
            Strategy::RandomThenUcb => "random_then_ucb",
            Strategy::AlThenUcb => "al_then_ucb",
            Strategy::HumanOnly => "human_only",
            Strategy::AlgorithmOnly => "algorithm_only",
            Strategy::Hitl => "hitl",
        }
    }

    /// Parses a Python strategy name.
    pub fn parse(name: &str) -> Result<Strategy, String> {
        Strategy::ALL
            .iter()
            .copied()
            .find(|s| s.as_str() == name)
            .ok_or_else(|| format!("unknown strategy {:?}", name))
    }
}

/// Simulation settings; [`Default`] gives the Python keyword defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct SimConfig {
    /// Lab tests per round `B`.
    pub budget: usize,
    /// Number of rounds `R`.
    pub rounds: usize,
    /// Uncertainty-list length `q`.
    pub q: usize,
    /// Search-list length `h`.
    pub h: usize,
    /// UCB exploration weight.
    pub beta: f64,
    /// Warm-up rounds for the two `*_then_ucb` strategies.
    pub warmup_rounds: usize,
    /// Probability the expert knows a molecule.
    pub knowledge: f64,
    /// Share of unknown slots filled from the uncertainty list.
    pub explore_share: f64,
    /// Molecules the `human_only` expert screens per round.
    pub human_view: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            budget: 50,
            rounds: 50,
            q: 50,
            h: 50,
            beta: 1.0,
            warmup_rounds: 10,
            knowledge: 0.5,
            explore_share: 0.3,
            human_view: 200,
            seed: 0,
        }
    }
}

/// Cumulative hits per round plus the pool facts needed for the metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct SimResult {
    /// The strategy that was run.
    pub strategy: Strategy,
    /// Cumulative number of targets found after each round.
    pub hits_by_round: Vec<usize>,
    /// Number of targets in the pool.
    pub n_targets: usize,
    /// Lab tests per round.
    pub budget: usize,
    /// Number of rounds.
    pub rounds: usize,
}

impl SimResult {
    /// Final hit rate `S / (B R)`.
    pub fn hit_rate(&self) -> Result<f64, String> {
        hit_rate(
            self.hits_by_round.last().copied().unwrap_or(0),
            self.budget,
            self.rounds,
        )
    }

    /// Recall after each round (errors if the pool has no target).
    pub fn recall_by_round(&self) -> Result<Vec<f64>, String> {
        self.hits_by_round
            .iter()
            .map(|&h| recall_rate(h, self.n_targets))
            .collect()
    }
}

/// Run one sequential experiment and return cumulative hits per round.
///
/// `fit_predict(x_labeled, y_labeled)` returns a [`Predictor`] mapping a feature
/// matrix to an `S x N x K` tensor of posterior draws of P(positive). Strategy
/// semantics are those of the Python docstring.
pub fn simulate<F>(
    strategy: Strategy,
    x: &[Vec<f64>],
    labels: &[Vec<u8>],
    mut fit_predict: F,
    cfg: &SimConfig,
) -> Result<SimResult, String>
where
    F: FnMut(&[Vec<f64>], &[Vec<u8>]) -> Result<Predictor, String>,
{
    if x.len() != labels.len() {
        return Err("x and labels must have the same length".to_string());
    }
    if cfg.budget == 0 || cfg.rounds == 0 {
        return Err("budget and rounds must be positive".to_string());
    }
    let mut rng = Rng::new(cfg.seed);
    let n = x.len();
    let tgt = is_target(labels);
    let mut tested = vec![false; n];
    let mut hits = Vec::with_capacity(cfg.rounds);
    let mut total = 0usize;
    let mut expert: Option<Expert> = None;
    if matches!(strategy, Strategy::HumanOnly | Strategy::Hitl) {
        let mut e = Expert::new(cfg.knowledge, cfg.explore_share);
        e.reveal(&tgt, &mut rng);
        expert = Some(e);
    }

    let mut predictor: Option<Predictor> = None;
    for r in 0..cfg.rounds {
        let have_labels = tested.iter().any(|&t| t);
        if have_labels && !matches!(strategy, Strategy::Random | Strategy::HumanOnly) {
            let idx: Vec<usize> = (0..n).filter(|&i| tested[i]).collect();
            let xl: Vec<Vec<f64>> = idx.iter().map(|&i| x[i].clone()).collect();
            let yl: Vec<Vec<u8>> = idx.iter().map(|&i| labels[i].clone()).collect();
            predictor = Some(fit_predict(&xl, &yl)?);
        }
        let avail: Vec<usize> = (0..n).filter(|&i| !tested[i]).collect();
        let random_pick = |rng: &mut Rng| rng.choice(&avail, cfg.budget.min(avail.len()));

        let pick: Vec<usize> = if strategy == Strategy::Random
            || (predictor.is_none() && strategy != Strategy::HumanOnly)
        {
            random_pick(&mut rng)
        } else if strategy == Strategy::HumanOnly {
            let view = rng.choice(&avail, cfg.human_view.min(avail.len()));
            let e = expert.as_ref().expect("expert exists for human_only");
            let km = e.known_mask.as_ref().expect("revealed");
            let kt = e.known_target.as_ref().expect("revealed");
            let mut known_pos: Vec<usize> =
                view.iter().copied().filter(|&i| km[i] && kt[i]).collect();
            let mut rest: Vec<usize> = view.iter().copied().filter(|&i| !km[i]).collect();
            rng.shuffle(&mut rest);
            known_pos.extend(rest);
            known_pos.truncate(cfg.budget);
            known_pos
        } else {
            let pred = predictor.as_ref().expect("predictor exists");
            let (mu, _, epi) = decompose_uncertainty(&pred(x))?;
            let sigma_d: Matrix = mu
                .iter()
                .map(|row| row.iter().map(|&m| (m * (1.0 - m)).max(0.0).sqrt()).collect())
                .collect();
            let search_score = ucb_scores(&mu, &sigma_d, cfg.beta)?;
            let unc_score: Vec<f64> = epi.iter().map(|row| row.iter().sum()).collect();
            let strat = match strategy {
                Strategy::RandomThenUcb => {
                    if r < cfg.warmup_rounds {
                        Strategy::Random
                    } else {
                        Strategy::Ucb
                    }
                }
                Strategy::AlThenUcb => {
                    if r < cfg.warmup_rounds {
                        Strategy::ActiveLearning
                    } else {
                        Strategy::Ucb
                    }
                }
                s => s,
            };
            match strat {
                Strategy::Random => random_pick(&mut rng),
                Strategy::Search => {
                    let ms: Vec<f64> = mu.iter().map(|row| row.iter().sum()).collect();
                    top_k(&ms, cfg.budget, Some(&tested))
                }
                Strategy::ActiveLearning => top_k(&unc_score, cfg.budget, Some(&tested)),
                Strategy::Ucb => top_k(&search_score, cfg.budget, Some(&tested)),
                Strategy::AlgorithmOnly => {
                    let s_list = top_k(&search_score, cfg.h, Some(&tested));
                    let u_list = top_k(&unc_score, cfg.q, Some(&tested));
                    let half = cfg.budget / 2;
                    let mut a: Vec<usize> = s_list.iter().copied().take(half).collect();
                    let in_a: std::collections::HashSet<usize> = a.iter().copied().collect();
                    let need = cfg.budget - a.len();
                    a.extend(u_list.into_iter().filter(|i| !in_a.contains(i)).take(need));
                    a
                }
                Strategy::Hitl => {
                    let s_list = top_k(&search_score, cfg.h, Some(&tested));
                    let u_list = top_k(&unc_score, cfg.q, Some(&tested));
                    let e = expert.as_ref().expect("expert exists for hitl");
                    expert_select(e, &s_list, &u_list, cfg.budget)?
                }
                Strategy::RandomThenUcb | Strategy::AlThenUcb | Strategy::HumanOnly => {
                    unreachable!("resolved above")
                }
            }
        };
        for &i in &pick {
            tested[i] = true;
            if tgt[i] {
                total += 1;
            }
        }
        hits.push(total);
    }
    Ok(SimResult {
        strategy,
        hits_by_round: hits,
        n_targets: tgt.iter().filter(|&&t| t).count(),
        budget: cfg.budget,
        rounds: cfg.rounds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Shared fixtures: identical to the Python and Julia test suites.
    const H_HALF: f64 = std::f64::consts::LN_2; // binary_entropy(0.5) = 0.6931471805599453
    const H_09: f64 = 0.3250829733914482; // binary_entropy(0.9)

    // S=3 draws, N=2 molecules, K=2 properties; PROBS[s][n][k]
    const PROBS: [[[f64; 2]; 2]; 3] = [
        [[0.9, 0.2], [0.5, 0.5]],
        [[0.1, 0.2], [0.5, 0.5]],
        [[0.5, 0.2], [0.9, 0.1]],
    ];
    const MU: [[f64; 2]; 2] = [[0.5, 0.2], [0.6333333333333333, 0.3666666666666667]];
    const ALEATORIC: [[f64; 2]; 2] = [
        [0.44777104244761395, 0.5004024235381879],
        [0.5704591115037796, 0.5704591115037796],
    ];
    const EPISTEMIC: [[f64; 2]; 2] = [
        [0.24537613811233133, 0.0],
        [0.08669864999356092, 0.08669864999356092],
    ];
    const UCB_BETA2: [f64; 2] = [1.9727922061357857, 2.363003220017392];

    const TOPK_SCORES: [f64; 5] = [0.3, 0.9, 0.9, 0.1, 0.5];
    const TOPK_3: [usize; 3] = [1, 2, 4];
    const TOPK_3_EXCLUDING_1: [usize; 3] = [2, 4, 0];

    const MERGE_UNC: [usize; 3] = [5, 2, 7];
    const MERGE_SRCH: [usize; 3] = [2, 9, 5];
    const MERGED: [usize; 4] = [2, 9, 5, 7];

    const TARGETS: [u8; 10] = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0];
    const KNOWN: [u8; 10] = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0];
    const SEARCH_LIST: [usize; 6] = [0, 1, 3, 4, 6, 7];
    const UNC_LIST: [usize; 5] = [2, 5, 8, 9, 3];
    const EXPERT_PICK_5: [usize; 5] = [0, 3, 6, 2, 5]; // explore_share 0.5, budget 5

    const HIT_30_OF_2500: f64 = 0.012;
    const RECALL_30_OF_148: f64 = 0.20270270270270271;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {} within {} of {}",
            actual,
            tol,
            expected
        );
    }

    /// pytest's default `approx`: relative 1e-6 or absolute 1e-12.
    fn assert_approx(actual: f64, expected: f64) {
        assert_close(actual, expected, (1e-6 * expected.abs()).max(1e-12));
    }

    fn probs() -> Samples {
        PROBS
            .iter()
            .map(|d| d.iter().map(|r| r.to_vec()).collect())
            .collect()
    }

    fn bools(v: &[u8]) -> Vec<bool> {
        v.iter().map(|&b| b != 0).collect()
    }

    fn revealed_expert(knowledge: f64, explore: f64, known: &[bool], target: &[bool]) -> Expert {
        let mut e = Expert::new(knowledge, explore);
        e.known_mask = Some(known.to_vec());
        e.known_target = Some(target.to_vec());
        e
    }

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut r = Rng::new(7);
        for _ in 0..1000 {
            let u = r.next_f64();
            assert!((0.0..1.0).contains(&u));
            assert!(r.below(13) < 13);
        }
        let items: Vec<usize> = (0..50).collect();
        let pick = r.choice(&items, 20);
        let set: std::collections::HashSet<_> = pick.iter().collect();
        assert_eq!(pick.len(), 20);
        assert_eq!(set.len(), 20);
        let (mut s, mut s2) = (0.0, 0.0);
        let m = 20000;
        for _ in 0..m {
            let z = r.normal();
            s += z;
            s2 += z * z;
        }
        assert!((s / m as f64).abs() < 0.05);
        assert!((s2 / m as f64 - 1.0).abs() < 0.05);
    }

    #[test]
    fn binary_entropy_values() {
        assert_approx(binary_entropy(0.5), H_HALF);
        assert_approx(binary_entropy(0.9), H_09);
        assert_close(binary_entropy(0.0), 0.0, 1e-9);
        assert_approx(binary_entropy(0.1), binary_entropy(0.9));
    }

    #[test]
    fn decompose_uncertainty_fixture() {
        let (mu, al, ep) = decompose_uncertainty(&probs()).unwrap();
        for n in 0..2 {
            for k in 0..2 {
                assert_close(mu[n][k], MU[n][k], 1e-12);
                assert_close(al[n][k], ALEATORIC[n][k], 1e-9);
                assert_close(ep[n][k], EPISTEMIC[n][k], 1e-9);
            }
        }
        assert!(decompose_uncertainty(&[]).is_err());
        let ragged = vec![vec![vec![0.5]], vec![vec![0.5], vec![0.5]]];
        assert!(decompose_uncertainty(&ragged).is_err());
    }

    #[test]
    fn decomposition_identity() {
        let mut rng = Rng::new(0);
        let ps: Samples = (0..8)
            .map(|_| {
                (0..30)
                    .map(|_| (0..3).map(|_| 0.01 + 0.98 * rng.next_f64()).collect())
                    .collect()
            })
            .collect();
        let (mu, al, ep) = decompose_uncertainty(&ps).unwrap();
        for n in 0..30 {
            for k in 0..3 {
                assert_close(al[n][k] + ep[n][k], binary_entropy(mu[n][k]), 1e-9);
                assert!(ep[n][k] >= 0.0);
            }
        }
        // identical draws carry no model uncertainty
        let same: Samples = (0..8).map(|_| ps[0].clone()).collect();
        let (_, _, ep) = decompose_uncertainty(&same).unwrap();
        let max = ep.iter().flatten().cloned().fold(0.0, f64::max);
        assert!(max < 1e-9);
    }

    #[test]
    fn ucb_scores_fixture() {
        let mu: Matrix = MU.iter().map(|r| r.to_vec()).collect();
        let sd: Matrix = mu
            .iter()
            .map(|r| r.iter().map(|&m| (m * (1.0 - m)).sqrt()).collect())
            .collect();
        let got = ucb_scores(&mu, &sd, 2.0).unwrap();
        for (a, e) in got.iter().zip(UCB_BETA2.iter()) {
            assert_close(*a, *e, 1e-12);
        }
        // beta = 0 reduces to the summed mean
        let g0 = ucb_scores(&mu, &sd, 0.0).unwrap();
        for (a, row) in g0.iter().zip(&mu) {
            assert_close(*a, row.iter().sum::<f64>(), 1e-12);
        }
        assert!(ucb_scores(&mu, &sd, -1.0).is_err());
        assert!(ucb_scores(&mu, &sd[..1], 1.0).is_err());
    }

    #[test]
    fn top_k_ties_and_exclusion() {
        assert_eq!(top_k(&TOPK_SCORES, 3, None), TOPK_3.to_vec());
        let mut mask = vec![false; 5];
        mask[1] = true;
        assert_eq!(
            top_k(&TOPK_SCORES, 3, Some(&mask)),
            TOPK_3_EXCLUDING_1.to_vec()
        );
        assert_eq!(top_k(&TOPK_SCORES, 10, Some(&mask)).len(), 4);
    }

    #[test]
    fn merge_shortlists_fixture() {
        assert_eq!(merge_shortlists(&MERGE_UNC, &MERGE_SRCH), MERGED.to_vec());
    }

    #[test]
    fn expert_select_fixture() {
        let e = revealed_expert(1.0, 0.5, &bools(&KNOWN), &bools(&TARGETS));
        assert_eq!(
            expert_select(&e, &SEARCH_LIST, &UNC_LIST, 5).unwrap(),
            EXPERT_PICK_5.to_vec()
        );
    }

    #[test]
    fn expert_requires_reveal() {
        let e = Expert::new(0.5, 0.3);
        assert!(expert_select(&e, &[0], &[1], 1).is_err());
    }

    #[test]
    fn expert_overrides_known_negatives_and_spends_budget() {
        let e = revealed_expert(1.0, 0.0, &[true; 10], &[false; 10]);
        let pick = expert_select(&e, &[0, 1, 2], &[3, 4, 5], 4).unwrap();
        let set: std::collections::HashSet<_> = pick.iter().collect();
        assert_eq!(pick.len(), 4);
        assert_eq!(set.len(), 4); // budget spent from the dropped ones
    }

    #[test]
    fn reveal_is_deterministic_and_matches_knowledge() {
        let tgt = vec![true; 2000];
        let mut e = Expert::new(0.25, 0.3);
        e.reveal(&tgt, &mut Rng::new(5));
        let mut e2 = Expert::new(0.25, 0.3);
        e2.reveal(&tgt, &mut Rng::new(5));
        assert_eq!(e, e2);
        let frac = e.known_mask.as_ref().unwrap().iter().filter(|&&b| b).count() as f64 / 2000.0;
        assert!((frac - 0.25).abs() < 0.05);
        let mut all = Expert::new(1.0, 0.3);
        all.reveal(&tgt, &mut Rng::new(1));
        assert!(all.known_mask.unwrap().iter().all(|&b| b));
        let mut none = Expert::new(0.0, 0.3);
        none.reveal(&tgt, &mut Rng::new(1));
        assert!(none.known_mask.unwrap().iter().all(|&b| !b));
    }

    #[test]
    fn metrics_fixture() {
        assert_approx(hit_rate(30, 50, 50).unwrap(), HIT_30_OF_2500);
        assert_approx(recall_rate(30, 148).unwrap(), RECALL_30_OF_148);
        assert_eq!(
            is_target(&[vec![1, 1, 1], vec![1, 0, 1]]),
            vec![true, false]
        );
        assert!(hit_rate(1, 0, 5).is_err());
        assert!(recall_rate(1, 0).is_err());
    }

    #[test]
    fn strategy_names_round_trip() {
        assert_eq!(Strategy::ALL.len(), 9);
        for s in Strategy::ALL {
            assert_eq!(Strategy::parse(s.as_str()).unwrap(), s);
        }
        assert!(Strategy::parse("nope").is_err());
    }

    /// Two-property pool where the target is the intersection of two half-spaces.
    fn synthetic(n: usize, d: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<Vec<u8>>) {
        let mut rng = Rng::new(seed);
        let x: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| rng.normal()).collect())
            .collect();
        let y = x
            .iter()
            .map(|r| {
                vec![
                    (r[0] + 0.3 * r[2] > 1.2) as u8,
                    (r[1] - 0.3 * r[3] > 1.0) as u8,
                ]
            })
            .collect();
        (x, y)
    }

    fn fp(seed: u64) -> impl FnMut(&[Vec<f64>], &[Vec<u8>]) -> Result<Predictor, String> {
        move |xl, yl| {
            let mut ens = BootstrapLogisticEnsemble::new(6, 1e-2, 0.5, 80, seed);
            ens.fit(xl, yl)?;
            Ok(Box::new(move |x: &[Vec<f64>]| {
                ens.predict_samples(x).expect("fitted")
            }))
        }
    }

    fn cfg(budget: usize, rounds: usize, knowledge: f64, seed: u64) -> SimConfig {
        SimConfig {
            budget,
            rounds,
            q: budget,
            h: budget,
            knowledge,
            seed,
            ..SimConfig::default()
        }
    }

    #[test]
    fn bootstrap_ensemble_learns() {
        let (x, y) = synthetic(1500, 8, 0);
        let mut ens = BootstrapLogisticEnsemble::new(8, 1e-2, 0.5, 150, 1);
        assert!(ens.predict_samples(&x[600..]).is_err());
        ens.fit(&x[..600], &y[..600]).unwrap();
        let ps = ens.predict_samples(&x[600..]).unwrap();
        assert_eq!((ps.len(), ps[0].len(), ps[0][0].len()), (8, 900, 2));
        let (mu, _, _) = decompose_uncertainty(&ps).unwrap();
        let mut correct = 0usize;
        for (m, t) in mu.iter().zip(&y[600..]) {
            for k in 0..2 {
                if (m[k] > 0.5) as u8 == t[k] {
                    correct += 1;
                }
            }
        }
        assert!(correct as f64 / (900.0 * 2.0) > 0.85);
        assert!(BootstrapLogisticEnsemble::with_seed(0).fit(&[], &[]).is_err());
    }

    #[test]
    fn simulation_shapes_and_budget() {
        let (x, y) = synthetic(1500, 8, 0);
        let r = simulate(Strategy::Hitl, &x, &y, fp(0), &cfg(20, 6, 0.5, 3)).unwrap();
        assert_eq!(r.hits_by_round.len(), 6);
        assert!(r.hits_by_round.windows(2).all(|w| w[1] >= w[0]));
        assert!(*r.hits_by_round.last().unwrap() <= r.n_targets);
        assert_approx(
            r.hit_rate().unwrap(),
            *r.hits_by_round.last().unwrap() as f64 / (20.0 * 6.0),
        );
        let rec = r.recall_by_round().unwrap();
        assert_eq!(rec.len(), 6);
        assert!(rec.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn every_strategy_runs_and_is_monotone() {
        let (x, y) = synthetic(600, 8, 2);
        for s in Strategy::ALL {
            let c = SimConfig {
                budget: 10,
                rounds: 5,
                q: 10,
                h: 10,
                warmup_rounds: 2,
                human_view: 50,
                seed: 1,
                ..SimConfig::default()
            };
            let r = simulate(s, &x, &y, fp(1), &c).unwrap();
            assert_eq!(r.strategy, s);
            assert_eq!(r.hits_by_round.len(), 5, "{}", s.as_str());
            assert!(r.hits_by_round.windows(2).all(|w| w[1] >= w[0]));
            assert!(*r.hits_by_round.last().unwrap() <= r.n_targets);
        }
    }

    #[test]
    fn simulation_is_deterministic_and_validates() {
        let (x, y) = synthetic(500, 8, 4);
        let c = cfg(10, 4, 0.5, 9);
        let a = simulate(Strategy::Hitl, &x, &y, fp(2), &c).unwrap();
        let b = simulate(Strategy::Hitl, &x, &y, fp(2), &c).unwrap();
        assert_eq!(a, b);
        assert!(simulate(Strategy::Random, &x, &y[..10], fp(0), &c).is_err());
        let bad = SimConfig { rounds: 0, ..c };
        assert!(simulate(Strategy::Random, &x, &y, fp(0), &bad).is_err());
    }

    #[test]
    fn hitl_beats_random_on_synthetic_pool() {
        let (x, y) = synthetic(1500, 8, 0);
        let mean = |s: Strategy, k: f64| -> f64 {
            (0..4u64)
                .map(|seed| {
                    let c = cfg(20, 10, k, seed);
                    *simulate(s, &x, &y, fp(seed), &c)
                        .unwrap()
                        .hits_by_round
                        .last()
                        .unwrap() as f64
                })
                .sum::<f64>()
                / 4.0
        };
        let rnd = mean(Strategy::Random, 0.5);
        let hit = mean(Strategy::Hitl, 0.75);
        assert!(hit > 2.0 * rnd, "hitl {} vs random {}", hit, rnd);
    }

    #[test]
    fn more_expert_knowledge_does_not_hurt() {
        let (x, y) = synthetic(1500, 8, 0);
        let avg = |k: f64| -> f64 {
            (0..5u64)
                .map(|seed| {
                    let c = cfg(20, 10, k, seed);
                    *simulate(Strategy::Hitl, &x, &y, fp(seed), &c)
                        .unwrap()
                        .hits_by_round
                        .last()
                        .unwrap() as f64
                })
                .sum::<f64>()
                / 5.0
        };
        assert!(avg(1.0) >= avg(0.0));
    }
}
