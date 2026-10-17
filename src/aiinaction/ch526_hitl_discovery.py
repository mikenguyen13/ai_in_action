"""Human-in-the-loop sequential experiments for drug discovery.

Reference implementation for Chapter 526, after He, Hua, Wang, and Zheng (2026),
*Collaborative Intelligence in Sequential Experiments: A Human-in-the-Loop
Framework for Drug Discovery* (Information Systems Research 37(3), 1390-1415).

The problem: a pool of N candidate molecules, a budget of B lab tests per round
for R rounds, and a "target" molecule is one whose K binary properties are all
positive. Targets are rare (0.2 to 1 percent of the pool) and no labelled data
exists at the start. Each round the algorithm trains a Bayesian surrogate on the
labels gathered so far and hands the human expert two shortlists plus its own
uncertainty (the "meta-knowledge"):

* an **uncertainty list** of the ``q`` molecules with the highest model
  (epistemic) uncertainty, which teach the algorithm the most;
* a **search list** of the ``h`` molecules with the highest upper confidence
  bound, which are the best bets for a target.

The expert, who holds private domain knowledge, picks the ``B`` molecules that
are actually sent to the lab. Labels flow back and the surrogate is retrained.

The numeric core (uncertainty decomposition, scores, selection, the expert
model, metrics, and a bootstrap-ensemble logistic surrogate) is pure NumPy and
mirrors ``AIInAction.Ch526HitlDiscovery`` in Julia and
``aiinaction::ch526_hitl_discovery`` in Rust. The torch Bayesian neural network
(``BayesianMLP``), the SATPdb loader, and ESM2 embedding are Python only.
"""
from __future__ import annotations

import math
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "binary_entropy",
    "decompose_uncertainty",
    "ucb_scores",
    "top_k",
    "merge_shortlists",
    "Expert",
    "expert_select",
    "hit_rate",
    "recall_rate",
    "is_target",
    "BootstrapLogisticEnsemble",
    "Strategy",
    "SimResult",
    "simulate",
    "SATPDB_URL",
    "SATPDB_CLASSES",
    "fetch_satpdb",
    "load_satpdb",
    "build_task",
    "embed_esm2",
    "load_embeddings",
    "BayesianMLP",
]

LN2 = math.log(2.0)
EPS = 1e-12


# ---------------------------------------------------------------------------
# Uncertainty decomposition and scores
# ---------------------------------------------------------------------------
def binary_entropy(p: np.ndarray | float) -> np.ndarray:
    """Entropy of a Bernoulli(p) in nats, with 0 log 0 = 0."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def decompose_uncertainty(prob_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split predictive uncertainty into data and model parts.

    ``prob_samples`` has shape ``(S, N, K)`` (or ``(S, N)`` for one property):
    ``S`` posterior draws of P(property k is positive) for ``N`` molecules.

    Returns ``(mu, aleatoric, epistemic)`` with shape ``(N, K)``:

    * ``mu`` is the posterior predictive mean,
    * ``aleatoric`` is the expected entropy ``E_theta H[p(y|x,theta)]``
      (label noise that more data cannot remove),
    * ``epistemic`` is the mutual information ``H[E p] - E H[p]`` between the
      label and the parameters (what more data can teach the model).
    """
    ps = np.asarray(prob_samples, dtype=float)
    if ps.ndim == 2:
        ps = ps[:, :, None]
    mu = ps.mean(axis=0)
    aleatoric = binary_entropy(ps).mean(axis=0)
    epistemic = np.maximum(binary_entropy(mu) - aleatoric, 0.0)
    return mu, aleatoric, epistemic


def ucb_scores(mu: np.ndarray, sigma_d: np.ndarray, beta: float) -> np.ndarray:
    """Search score: ``sum_k [mu_k + sqrt(beta) * sigma_d_k]`` (paper Eq. 16-18).

    Data uncertainty (not model uncertainty) is the bonus, so a molecule is not
    rewarded merely for lying in a region the surrogate has not seen. For a
    Bernoulli head the data standard deviation is ``sqrt(mu (1 - mu))``; pass
    that as ``sigma_d`` or use the entropy-based value from
    :func:`decompose_uncertainty` scaled to ``[0, 1]`` by ``1/ln 2``.
    """
    mu = np.atleast_2d(np.asarray(mu, dtype=float))
    sd = np.atleast_2d(np.asarray(sigma_d, dtype=float))
    return (mu + math.sqrt(beta) * sd).sum(axis=1)


def top_k(scores: np.ndarray, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Indices of the ``k`` largest scores, ties broken by lower index.

    ``exclude`` is a boolean mask of molecules that are not available (already
    tested). Fewer than ``k`` indices come back only if fewer are available.
    """
    s = np.asarray(scores, dtype=float).copy()
    if exclude is not None:
        s[np.asarray(exclude, dtype=bool)] = -np.inf
    n_avail = int(np.isfinite(s).sum())
    k = min(k, n_avail)
    # stable: sort by (-score, index)
    order = np.lexsort((np.arange(len(s)), -s))
    return order[:k]


def merge_shortlists(uncertainty_idx: Sequence[int], search_idx: Sequence[int]) -> np.ndarray:
    """Union of the two shortlists, search list first, order preserved, no repeats."""
    seen: set[int] = set()
    out: list[int] = []
    for i in list(search_idx) + list(uncertainty_idx):
        if i not in seen:
            seen.add(int(i))
            out.append(int(i))
    return np.asarray(out, dtype=int)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------
@dataclass
class Expert:
    """A simulated domain expert with partial, private knowledge.

    The expert knows the true target status of a molecule with probability
    ``knowledge`` (decided once per molecule, so it is consistent across
    rounds). This mirrors the paper's knowledge-disclosure experiment, where a
    volunteer saw the labels of a fraction p of the pool.

    ``explore_share`` is the fraction of the molecules the expert has no
    private information about that are filled from the uncertainty list rather
    than the search list. It is the expert's budget split between learning and
    searching, which the paper identifies as a key decision the algorithm
    cannot make for itself.
    """

    knowledge: float
    explore_share: float = 0.3
    known_mask: np.ndarray | None = field(default=None, repr=False)
    known_target: np.ndarray | None = field(default=None, repr=False)

    def reveal(self, is_target_vec: np.ndarray, rng: np.random.Generator) -> None:
        """Fix, once, which molecules the expert has private knowledge about."""
        n = len(is_target_vec)
        self.known_mask = rng.random(n) < self.knowledge
        self.known_target = np.asarray(is_target_vec, dtype=bool)


def expert_select(
    expert: Expert,
    search_list: Sequence[int],
    uncertainty_list: Sequence[int],
    budget: int,
) -> np.ndarray:
    """Choose the molecules sent to the lab.

    Rules, in order:

    1. every molecule the expert *knows* to be a target is taken first;
    2. molecules the expert knows to be non-targets are dropped (override);
    3. the remaining slots go to molecules the expert knows nothing about,
       alternating between the search list and the uncertainty list in the
       proportion ``explore_share``;
    4. if slots are still empty (everything left is a known non-target), they
       are filled from the dropped molecules so the budget is always spent.
    """
    km, kt = expert.known_mask, expert.known_target
    assert km is not None and kt is not None, "call Expert.reveal first"
    pool = merge_shortlists(uncertainty_list, search_list)
    chosen: list[int] = [int(i) for i in pool if km[i] and kt[i]][:budget]
    taken = set(chosen)

    def unknown(seq: Sequence[int]) -> list[int]:
        return [int(i) for i in seq if int(i) not in taken and not km[i]]

    srch, unc = unknown(search_list), unknown(uncertainty_list)
    n_left = budget - len(chosen)
    n_unc = int(round(expert.explore_share * n_left))
    n_srch = n_left - n_unc
    for lst, n in ((srch, n_srch), (unc, n_unc)):
        for i in lst:
            if n == 0:
                break
            if i not in taken:
                chosen.append(i)
                taken.add(i)
                n -= 1
    # top up from whatever is left among unknowns, then from known non-targets
    for i in unknown(list(search_list) + list(uncertainty_list)):
        if len(chosen) >= budget:
            break
        if i not in taken:
            chosen.append(i)
            taken.add(i)
    for i in pool:
        if len(chosen) >= budget:
            break
        if int(i) not in taken:
            chosen.append(int(i))
            taken.add(int(i))
    return np.asarray(chosen[:budget], dtype=int)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def is_target(labels: np.ndarray) -> np.ndarray:
    """A target has every one of its K properties positive. ``labels`` is (N, K)."""
    return np.asarray(labels).astype(bool).all(axis=1)


def hit_rate(n_hits: int, budget_per_round: int, rounds: int) -> float:
    """Final hit rate ``S / (B R)``: targets found per lab test spent."""
    return n_hits / float(budget_per_round * rounds)


def recall_rate(n_hits: int, n_targets_in_pool: int) -> float:
    """Share of all targets in the pool found so far."""
    return n_hits / float(n_targets_in_pool)


# ---------------------------------------------------------------------------
# A dependency-light surrogate (shared across the three languages)
# ---------------------------------------------------------------------------
class BootstrapLogisticEnsemble:
    """Bootstrap ensemble of L2-regularised logistic regressions, one per property.

    A deliberately small stand-in for a Bayesian neural network: ``S`` members
    are fitted to bootstrap resamples, so their disagreement is a (crude)
    posterior over parameters. ``predict_samples`` returns the ``(S, N, K)``
    array that :func:`decompose_uncertainty` consumes.
    """

    def __init__(self, n_members: int = 10, l2: float = 1e-2, lr: float = 0.5,
                 epochs: int = 200, seed: int = 0):
        self.n_members, self.l2, self.lr, self.epochs = n_members, l2, lr, epochs
        self.rng = np.random.default_rng(seed)
        self.w: np.ndarray | None = None  # (S, K, D+1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BootstrapLogisticEnsemble":
        n, d = x.shape
        k = y.shape[1]
        xb = np.hstack([x, np.ones((n, 1))])
        w = np.zeros((self.n_members, k, d + 1))
        for s in range(self.n_members):
            idx = self.rng.integers(0, n, n)
            xs, ys = xb[idx], y[idx].astype(float)
            ws = np.zeros((k, d + 1))
            for _ in range(self.epochs):
                p = 1.0 / (1.0 + np.exp(-xs @ ws.T))
                grad = ((p - ys).T @ xs) / n + self.l2 * ws
                ws -= self.lr * grad
            w[s] = ws
        self.w = w
        return self

    def predict_samples(self, x: np.ndarray) -> np.ndarray:
        assert self.w is not None
        xb = np.hstack([x, np.ones((len(x), 1))])
        z = np.einsum("nd,skd->snk", xb, self.w)
        return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# The sequential-experiment simulator
# ---------------------------------------------------------------------------
Strategy = str  # one of STRATEGIES
STRATEGIES = (
    "random",
    "search",
    "active_learning",
    "ucb",
    "random_then_ucb",
    "al_then_ucb",
    "human_only",
    "algorithm_only",
    "hitl",
)

Surrogate = Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]


@dataclass
class SimResult:
    strategy: str
    hits_by_round: list[int]
    n_targets: int
    budget: int
    rounds: int

    @property
    def hit_rate(self) -> float:
        return hit_rate(self.hits_by_round[-1], self.budget, self.rounds)

    @property
    def recall_by_round(self) -> list[float]:
        return [recall_rate(h, self.n_targets) for h in self.hits_by_round]


def simulate(
    strategy: str,
    x: np.ndarray,
    labels: np.ndarray,
    fit_predict: Surrogate,
    *,
    budget: int = 50,
    rounds: int = 50,
    q: int = 50,
    h: int = 50,
    beta: float = 1.0,
    warmup_rounds: int = 10,
    knowledge: float = 0.5,
    explore_share: float = 0.3,
    human_view: int = 200,
    seed: int = 0,
) -> SimResult:
    """Run one sequential experiment and return cumulative hits per round.

    ``fit_predict(x_labeled, y_labeled)`` must return a function mapping a
    feature matrix to an ``(S, N, K)`` array of posterior draws of P(positive).

    Strategies:

    ``random``           B random untested molecules each round.
    ``search``           top B by posterior mean.
    ``active_learning``  top B by epistemic uncertainty.
    ``ucb``              top B by the search score.
    ``random_then_ucb``  random for ``warmup_rounds``, then ``ucb``.
    ``al_then_ucb``      ``active_learning`` for ``warmup_rounds``, then ``ucb``.
    ``human_only``       the expert screens ``human_view`` random molecules and
                         picks B using private knowledge alone.
    ``algorithm_only``   half of B from the search list, half from the
                         uncertainty list, no human override.
    ``hitl``             the proposed framework: the expert chooses B from the
                         union of the two shortlists.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    tgt = is_target(labels)
    tested = np.zeros(n, dtype=bool)
    hits: list[int] = []
    total = 0
    expert = None
    if strategy in ("human_only", "hitl"):
        expert = Expert(knowledge, explore_share)
        expert.reveal(tgt, rng)

    predictor = None
    for r in range(rounds):
        have_labels = tested.any()
        if have_labels and strategy not in ("random", "human_only"):
            predictor = fit_predict(x[tested], labels[tested])
        avail = np.flatnonzero(~tested)

        def random_pick() -> np.ndarray:
            return rng.choice(avail, size=min(budget, len(avail)), replace=False)

        if strategy == "random" or (predictor is None and strategy != "human_only"):
            pick = random_pick()
        elif strategy == "human_only":
            view = rng.choice(avail, size=min(human_view, len(avail)), replace=False)
            km, kt = expert.known_mask, expert.known_target
            known_pos = [int(i) for i in view if km[i] and kt[i]]
            rest = [int(i) for i in view if not km[i]]
            rng.shuffle(rest)
            pick = np.asarray((known_pos + rest)[:budget], dtype=int)
        else:
            mu, _, epi = decompose_uncertainty(predictor(x))
            sigma_d = np.sqrt(mu * (1.0 - mu))
            search_score = ucb_scores(mu, sigma_d, beta)
            unc_score = epi.sum(axis=1)
            strat = strategy
            if strat == "random_then_ucb":
                strat = "random" if r < warmup_rounds else "ucb"
            if strat == "al_then_ucb":
                strat = "active_learning" if r < warmup_rounds else "ucb"
            if strat == "random":
                pick = random_pick()
            elif strat == "search":
                pick = top_k(mu.sum(axis=1), budget, tested)
            elif strat == "active_learning":
                pick = top_k(unc_score, budget, tested)
            elif strat == "ucb":
                pick = top_k(search_score, budget, tested)
            else:
                s_list = top_k(search_score, h, tested)
                u_list = top_k(unc_score, q, tested)
                if strat == "algorithm_only":
                    half = budget // 2
                    a = list(s_list[:half])
                    b = [int(i) for i in u_list if int(i) not in set(a)]
                    pick = np.asarray(a + b[: budget - len(a)], dtype=int)
                else:  # hitl
                    pick = expert_select(expert, s_list, u_list, budget)
        tested[pick] = True
        total += int(tgt[pick].sum())
        hits.append(total)
    return SimResult(strategy, hits, int(tgt.sum()), budget, rounds)


# ---------------------------------------------------------------------------
# SATPdb data (Python only)
# ---------------------------------------------------------------------------
SATPDB_URL = "https://webs.iiitd.edu.in/raghava/satpdb"
SATPDB_CLASSES = (
    "antibacterial", "anticancer", "antifungal", "antimicrobial", "antiviral",
    "antihypertensive", "antiparasitic", "toxic", "cell_cell_communication",
    "drug_delivery_vehicle",
)


def fetch_satpdb(dest: str | Path = "data/satpdb") -> Path:
    """Download the per-class SATPdb FASTA files (about 1 MB) if not cached."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in SATPDB_CLASSES:
        f = dest / f"{name}.fasta"
        if not f.exists():
            urllib.request.urlretrieve(f"{SATPDB_URL}/{name}.fasta", f)
    return dest


def _read_fasta_sequences(path: Path) -> set[str]:
    seqs: set[str] = set()
    for rec in path.read_text().split(">")[1:]:
        lines = rec.strip().split("\n")
        if len(lines) >= 2:
            seqs.add("".join(lines[1:]).strip())
    return seqs


def load_satpdb(dest: str | Path = "data/satpdb") -> tuple[list[str], np.ndarray, tuple[str, ...]]:
    """Load unique peptides and their multi-label property matrix.

    Returns ``(sequences, Y, class_names)`` where ``Y[i, j] = 1`` if peptide
    ``i`` appears in the FASTA file of class ``j``.
    """
    dest = fetch_satpdb(dest)
    per_class = {c: _read_fasta_sequences(dest / f"{c}.fasta") for c in SATPDB_CLASSES}
    seqs = sorted(set().union(*per_class.values()))
    y = np.array([[s in per_class[c] for c in SATPDB_CLASSES] for s in seqs], dtype=np.int8)
    return seqs, y, SATPDB_CLASSES


def build_task(
    y: np.ndarray, class_names: Sequence[str], properties: Sequence[str],
    pool_size: int = 14000, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick the K task properties and a random search pool.

    Returns ``(pool_idx, test_idx, labels)`` where ``labels`` is the ``(pool, K)``
    sub-matrix for the pool; ``test_idx`` is the held-out remainder.
    """
    cols = [list(class_names).index(p) for p in properties]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    pool_idx, test_idx = np.sort(perm[:pool_size]), np.sort(perm[pool_size:])
    return pool_idx, test_idx, y[pool_idx][:, cols]


def embed_esm2(
    seqs: Sequence[str], model_name: str = "facebook/esm2_t12_35M_UR50D",
    batch_size: int = 128, device: str | None = None,
) -> np.ndarray:
    """Mean-pooled ESM2 embeddings, one row per sequence."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            enc = tok(list(seqs[i:i + batch_size]), return_tensors="pt", padding=True).to(device)
            hid = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((hid * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.vstack(out)


def load_embeddings(seqs: Sequence[str], cache: str | Path = "data/satpdb/esm2_35M.npy",
                    **kwargs) -> np.ndarray:
    """Embed once with ESM2 and cache to ``cache`` (about 30 MB, not committed)."""
    cache = Path(cache)
    if cache.exists():
        x = np.load(cache)
        if len(x) == len(seqs):
            return x
    x = embed_esm2(seqs, **kwargs).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, x)
    return x


class BayesianMLP:
    """Bayes-by-backprop MLP surrogate (variational inference), K sigmoid heads.

    A compact version of the paper's Bayesian neural network: mean-field
    Gaussian posteriors over every weight, a ``N(0, prior_std^2)`` prior,
    optimised on the ELBO with Adam. ``predict_samples`` draws ``S`` weight
    samples and returns the ``(S, N, K)`` probability array.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int] = (128, 64, 32),
                 prior_std: float = 0.1, epochs: int = 40, lr: float = 5e-3,
                 n_samples: int = 10, kl_weight: float = 1.0, device: str | None = None,
                 seed: int = 0):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dims = [in_dim, *hidden, out_dim]
        self.prior_std, self.epochs, self.lr = prior_std, epochs, lr
        self.n_samples, self.kl_weight = n_samples, kl_weight
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        torch.manual_seed(seed)
        self.params = self._init_params()

    def _init_params(self):
        t = self.torch
        ps = []
        for i, o in zip(self.dims[:-1], self.dims[1:]):
            mu = (t.randn(i, o, device=self.device) * (1.0 / math.sqrt(i))).requires_grad_()
            rho = t.full((i, o), -5.0, device=self.device, requires_grad=True)
            bmu = t.zeros(o, device=self.device, requires_grad=True)
            brho = t.full((o,), -5.0, device=self.device, requires_grad=True)
            ps.append((mu, rho, bmu, brho))
        return ps

    def _forward(self, x, sample=True):
        t = self.torch
        kl = x.new_zeros(())
        for li, (mu, rho, bmu, brho) in enumerate(self.params):
            sw, sb = t.nn.functional.softplus(rho), t.nn.functional.softplus(brho)
            w = mu + sw * t.randn_like(mu) if sample else mu
            b = bmu + sb * t.randn_like(bmu) if sample else bmu
            x = x @ w + b
            if li < len(self.params) - 1:
                x = t.relu(x)
            ps2 = self.prior_std**2
            kl = kl + 0.5 * ((sw**2 + mu**2) / ps2 - 1.0 - 2 * t.log(sw / self.prior_std)).sum()
            kl = kl + 0.5 * ((sb**2 + bmu**2) / ps2 - 1.0 - 2 * t.log(sb / self.prior_std)).sum()
        return x, kl

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BayesianMLP":
        t = self.torch
        xt = t.as_tensor(x, dtype=t.float32, device=self.device)
        yt = t.as_tensor(y, dtype=t.float32, device=self.device)
        self.params = self._init_params()
        flat = [p for layer in self.params for p in layer]
        opt = t.optim.Adam(flat, lr=self.lr)
        n = len(xt)
        # class imbalance: weight positives so the surrogate does not collapse to 0
        pos = yt.mean(0).clamp(min=1e-3)
        pw = ((1 - pos) / pos).clamp(max=50.0)
        for _ in range(self.epochs):
            opt.zero_grad()
            logits, kl = self._forward(xt)
            nll = t.nn.functional.binary_cross_entropy_with_logits(
                logits, yt, pos_weight=pw, reduction="sum")
            loss = (nll + self.kl_weight * kl) / n
            loss.backward()
            opt.step()
        return self

    def predict_samples(self, x: np.ndarray) -> np.ndarray:
        t = self.torch
        xt = t.as_tensor(x, dtype=t.float32, device=self.device)
        with t.no_grad():
            draws = [t.sigmoid(self._forward(xt)[0]) for _ in range(self.n_samples)]
        return t.stack(draws).cpu().numpy()
