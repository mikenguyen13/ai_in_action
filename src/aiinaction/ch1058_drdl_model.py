"""DRDL in PyTorch: model, benchmarks, simulated multiview panel, and training (Chapter 1058).

A from-the-paper implementation of the discrete and regularized deep learning
(DRDL) method of Wang, Che, Jiang, and Zhao (2026, Information Systems Research
37(3)) for dynamic financial risk prediction, plus the benchmark families the paper
compares against and the ablations of its Table 8.

The paper's view-specific embedding details and hyperparameters live in an online
appendix that is not reproduced here; the choices below (a per-feature numerical
tokenizer followed by one transformer encoder layer per view, codebook size 64,
commitment weight 0.25, and so on) are documented defaults, not the authors'.

Building blocks shared with the numeric core ``aiinaction.ch1058_drdl``:
``risk_accumulation`` (Eq. 8), the redundancy mask (Eq. 3), the orthogonal loss
(Eq. 7), the focal survival loss (Eqs. 9-10), and adaptive gradient balancing
(Eq. 11). Their torch versions here are differentiable equivalents and are tested
against the numpy reference.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F_

from . import ch1058_drdl as core

__all__ = [
    "MultiviewPanel",
    "make_multiview_risk_panel",
    "torch_risk_accumulation",
    "torch_focal_survival_loss",
    "torch_multiperiod_bce",
    "torch_orthogonal_loss",
    "first_occurrence_mask_torch",
    "DRDLConfig",
    "DRDL",
    "BaselineNet",
    "DeepHitNet",
    "CoxNet",
    "TrainConfig",
    "fit",
    "predict_cumulative",
    "evaluate",
]


# --------------------------------------------------------------------------
# Simulated multiview company panel
# --------------------------------------------------------------------------


@dataclass
class MultiviewPanel:
    """A simulated panel of companies observed through several data views."""

    views: list[np.ndarray]
    times: np.ndarray  # horizon index of event or censoring, 1..m
    events: np.ndarray  # 1 if the first risk event happened at `times`
    factors: np.ndarray  # latent risk factors (n x K), for diagnostics only
    view_names: list[str]
    view_factors: list[list[int]]  # which latent factors each view carries
    m: int

    @property
    def n(self) -> int:
        return len(self.times)

    def subset(self, idx: np.ndarray) -> "MultiviewPanel":
        return MultiviewPanel(
            views=[v[idx] for v in self.views],
            times=self.times[idx],
            events=self.events[idx],
            factors=self.factors[idx],
            view_names=self.view_names,
            view_factors=self.view_factors,
            m=self.m,
        )


# Latent factors: 0 solvency, 1 profitability, 2 liquidity, 3 legal and governance,
# 4 investor sentiment, 5 interfirm contagion, 6 nuisance (shared, no risk signal).
_FACTOR_NAMES = ["solvency", "profitability", "liquidity", "legal", "sentiment", "contagion", "nuisance"]
# Horizon-specific loadings: fundamentals act slowly, event-driven signals act fast.
_BETA = np.array(
    [
        # solv  prof  liq   legal  sent  cont  nuis
        [0.35, 0.35, 0.60, 0.70, 0.90, 0.80, 0.0],  # horizon 1
        [0.60, 0.60, 0.55, 0.50, 0.45, 0.45, 0.0],  # horizon 2
        [0.90, 0.85, 0.40, 0.30, 0.20, 0.25, 0.0],  # horizon 3
    ]
)
_VIEWS = {
    "financial": [0, 1, 2, 6],  # solvency unique; profitability and liquidity shared
    "report_text": [1, 3, 4, 6],  # legal and sentiment unique; profitability shared
    "network": [2, 5, 6],  # contagion unique; liquidity shared
}
_VIEW_DIMS = {"financial": 12, "report_text": 16, "network": 8}


def make_multiview_risk_panel(
    n: int,
    m: int = 3,
    seed: int = 0,
    base_rates: tuple[float, ...] = (0.06, 0.09, 0.08),
    noise: float = 0.8,
    drift: float = 0.0,
    width: int = 1,
) -> MultiviewPanel:
    """Simulate companies, their multiview data, and first-occurrence risk events.

    Each latent factor drives the discrete-time hazard with horizon-specific
    weights (``_BETA``), plus an interaction (weak profitability amplifies
    illiquidity). Each view observes only some factors through a random nonlinear
    mixing with noise, so views share redundant information (profitability,
    liquidity, and a nuisance factor with no risk signal) and also carry
    complementary information found nowhere else. ``drift`` shifts the factor means
    and base hazards to mimic an out-of-time cohort. ``width`` multiplies every
    view's dimension (more entangled, higher-dimensional views at the same signal).
    """
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    if m != 3:
        raise ValueError("the simulator is calibrated for m = 3 horizons")
    if n < 10:
        raise ValueError("n must be at least 10")
    rng = np.random.default_rng(seed)
    k = len(_FACTOR_NAMES)
    shift = np.zeros(k)
    if drift:
        shift[[0, 2, 4]] = drift * np.array([0.5, -0.4, 0.6])
    z = rng.normal(size=(n, k)) + shift

    alpha = np.log(np.array(base_rates) / (1 - np.array(base_rates))) + 0.35 * drift
    eta = z @ _BETA.T + alpha[None, :] + 0.5 * np.maximum(0.0, z[:, [1]]) * np.maximum(0.0, z[:, [2]])
    hazard = 1.0 / (1.0 + np.exp(-eta))

    u = rng.random(size=(n, m))
    fired = u < hazard
    times = np.full(n, m, dtype=np.int64)
    events = np.zeros(n, dtype=np.int64)
    for i in range(n):
        hit = np.flatnonzero(fired[i])
        if hit.size:
            times[i] = hit[0] + 1
            events[i] = 1

    # Fixed (seed-independent) mixing matrices so every cohort shares the same views.
    mix_rng = np.random.default_rng(1058)
    views, names, carried = [], [], []
    for name, fac in _VIEWS.items():
        d = _VIEW_DIMS[name] * width
        w = mix_rng.normal(size=(len(fac), d)) / math.sqrt(len(fac))
        w2 = mix_rng.normal(size=(len(fac), d)) / math.sqrt(len(fac))
        lin = z[:, fac] @ w
        x = lin + 0.5 * np.tanh(z[:, fac] @ w2) + noise * rng.normal(size=(n, d))
        views.append(x.astype(np.float32))
        names.append(name)
        carried.append(fac)
    return MultiviewPanel(views, times, events, z, names, carried, m)


def standardize(train: MultiviewPanel, *others: MultiviewPanel) -> list[MultiviewPanel]:
    """Standardize every view with training-set means and standard deviations."""
    stats = [(v.mean(axis=0), v.std(axis=0) + 1e-6) for v in train.views]
    out = []
    for p in (train, *others):
        q = copy.copy(p)
        q.views = [((v - mu) / sd).astype(np.float32) for v, (mu, sd) in zip(p.views, stats)]
        out.append(q)
    return out


# --------------------------------------------------------------------------
# Differentiable building blocks
# --------------------------------------------------------------------------


def torch_risk_accumulation(o: torch.Tensor) -> torch.Tensor:
    """Eq. 8: ``p_u = 1 - exp(-cumsum(exp(o)))``, monotone in u by construction."""
    return -torch.expm1(-torch.cumsum(torch.exp(o.clamp(max=20.0)), dim=-1))


def torch_multiperiod_bce(f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum over horizons of BCE on cumulative labels, averaged over companies."""
    p = f.clamp(1e-7, 1 - 1e-7)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).sum(dim=1).mean()


def _comparable_matrix(times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    ti, tj = times[:, None], times[None, :]
    ei, ej = events[:, None], events[None, :]
    comp = (ei == 1) & ((ti < tj) | ((ti == tj) & (ej == 0)))
    return comp & ~torch.eye(len(times), dtype=torch.bool, device=times.device)


def torch_focal_survival_loss(f: torch.Tensor, times: torch.Tensor, events: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Eqs. 9-10: mean of ``sigma(max(0, F_j(A_i) - F_i(A_i)))`` over comparable pairs."""
    comp = _comparable_matrix(times, events)
    if not comp.any():
        return f.sum() * 0.0
    idx = (times - 1).long()
    at_ti = f[:, idx].T  # at_ti[i, j] = F_j(A_i)
    own = at_ti.diagonal()[:, None]  # F_i(A_i)
    viol = (at_ti - own).clamp(min=0.0)
    sig = 2.0 / (1.0 + torch.exp(-(viol + 1e-12) ** gamma)) - 1.0
    return (sig * comp).sum() / comp.sum()


def torch_ranking_loss(f: torch.Tensor, times: torch.Tensor, events: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    """DeepHit-style ranking loss ``exp(-(F_i(A_i) - F_j(A_i)) / scale)`` over comparable pairs."""
    comp = _comparable_matrix(times, events)
    if not comp.any():
        return f.sum() * 0.0
    idx = (times - 1).long()
    at_ti = f[:, idx].T
    own = at_ti.diagonal()[:, None]
    return (torch.exp(-(own - at_ti) / scale) * comp).sum() / comp.sum()


def torch_monotonicity_loss(f: torch.Tensor) -> torch.Tensor:
    """Penalty on decreases of cumulative risk across horizons (the 'Mono' loss)."""
    return F_.relu(f[:, :-1] - f[:, 1:]).sum(dim=1).mean()


def torch_orthogonal_loss(codebook: torch.Tensor) -> torch.Tensor:
    """Eq. 7: Frobenius norm of the strict upper triangle of the normalized Gram matrix."""
    u = F_.normalize(codebook, dim=1)
    g = u @ u.T
    return torch.sqrt((torch.triu(g, diagonal=1) ** 2).sum() + 1e-12)


def first_occurrence_mask_torch(codes: torch.Tensor) -> torch.Tensor:
    """Eq. 3 applied row-wise: 1 for the first occurrence of each code in a row, else 0."""
    n, f = codes.shape
    eq = codes[:, :, None] == codes[:, None, :]  # eq[b, i, j]: z_i == z_j
    earlier = torch.tril(torch.ones(f, f, dtype=torch.bool, device=codes.device), diagonal=-1)
    seen_before = (eq & earlier[None]).any(dim=2)
    return (~seen_before).to(torch.float32)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class NumericalTokenizer(nn.Module):
    """Per-feature linear embedding ``x_j w_j + b_j`` (Gorishniy et al., 2022)."""

    def __init__(self, n_features: int, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d) / math.sqrt(d))
        self.bias = nn.Parameter(torch.zeros(n_features, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, None] * self.weight[None] + self.bias[None]


class ViewEmbedding(nn.Module):
    """View-specific embedding: tokenizer plus one transformer encoder layer."""

    def __init__(self, n_features: int, d: int, use_transformer: bool = True):
        super().__init__()
        self.tok = NumericalTokenizer(n_features, d)
        self.enc = (
            nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=2 * d, dropout=0.1, batch_first=True)
            if use_transformer
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.tok(x)
        return self.enc(t) if self.enc is not None else t


@dataclass
class DRDLConfig:
    view_dims: list[int]
    m: int = 3
    d: int = 16
    codebook_size: int = 64
    hidden: int = 64
    commitment: float = 0.25
    orth_weight: float = 0.1
    fsl_weight: float = 1.0
    gamma: float = 1.0
    use_mask: bool = True  # M4 ablation when False
    use_orth: bool = True  # M5 ablation when False
    use_view_embedding: bool = True  # M6 ablation when False (tokenizer only)
    use_risk_accumulation: bool = True  # M3 ablation when False (sigmoid per horizon)
    use_fsl: bool = True  # M2 ablation when False
    use_agb: bool = True  # M1 ablation when False


class DRDL(nn.Module):
    """Discrete and regularized deep learning for dynamic financial risk prediction."""

    def __init__(self, cfg: DRDLConfig):
        super().__init__()
        self.cfg = cfg
        self.views = nn.ModuleList(ViewEmbedding(k, cfg.d, cfg.use_view_embedding) for k in cfg.view_dims)
        self.codebook = nn.Parameter(torch.randn(cfg.codebook_size, cfg.d) * 0.5)
        n_tokens = sum(cfg.view_dims)
        self.head = nn.Sequential(
            nn.Linear(n_tokens * cfg.d, cfg.hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(cfg.hidden, cfg.m)
        )
        self.last_layer = self.head[-1]
        self.aux: dict[str, torch.Tensor] = {}

    def encode(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([emb(x) for emb, x in zip(self.views, xs)], dim=1)  # (n, f, d)

    def quantize(self, e: torch.Tensor):
        flat = e.reshape(-1, e.shape[-1])
        d2 = (flat**2).sum(1, keepdim=True) - 2 * flat @ self.codebook.T + (self.codebook**2).sum(1)[None]
        codes = d2.argmin(dim=1).reshape(e.shape[:2])  # Eq. 2
        matched = self.codebook[codes]  # E^(match)
        return codes, matched

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        e = self.encode(xs)
        codes, matched = self.quantize(e)
        codebook_loss = F_.mse_loss(matched, e.detach())  # Eq. 5
        commit_loss = F_.mse_loss(e, matched.detach())  # Eq. 6
        z = e + (matched - e).detach()  # straight-through estimator
        if self.cfg.use_mask:
            mask = first_occurrence_mask_torch(codes)  # Eq. 3
            z = z * mask[:, :, None]  # Eq. 4
        else:
            mask = torch.ones(codes.shape, device=codes.device)
        o = self.head(z.flatten(1))
        f = torch_risk_accumulation(o) if self.cfg.use_risk_accumulation else torch.sigmoid(o)
        self.aux = {
            "codes": codes,
            "mask": mask,
            "vq_loss": codebook_loss + self.cfg.commitment * commit_loss,
            "orth_loss": torch_orthogonal_loss(self.codebook) if self.cfg.use_orth else torch.zeros(()),
        }
        return f


class BaselineNet(nn.Module):
    """Multitask base architectures: concatenation (CML) or view attention (ATT).

    Outputs independent sigmoid probabilities per horizon; monotonicity and
    ranking are then encouraged only through optional loss terms (Mono, Rank).
    """

    def __init__(self, view_dims: list[int], m: int = 3, hidden: int = 64, fusion: str = "concat"):
        super().__init__()
        self.fusion = fusion
        if fusion == "concat":
            self.body = nn.Sequential(nn.Linear(sum(view_dims), hidden), nn.ReLU(), nn.Dropout(0.1))
        elif fusion == "attention":
            self.view_nets = nn.ModuleList(nn.Sequential(nn.Linear(k, hidden), nn.ReLU()) for k in view_dims)
            self.score = nn.Linear(hidden, 1)
        else:
            raise ValueError("fusion must be 'concat' or 'attention'")
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, m))
        self.attn_weights: torch.Tensor | None = None

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        if self.fusion == "concat":
            h = self.body(torch.cat(xs, dim=1))
        else:
            hs = torch.stack([net(x) for net, x in zip(self.view_nets, xs)], dim=1)  # (n, V, h)
            w = torch.softmax(self.score(hs).squeeze(-1), dim=1)
            self.attn_weights = w.detach()
            h = (w[:, :, None] * hs).sum(dim=1)
        return torch.sigmoid(self.out(h))


class DeepHitNet(nn.Module):
    """DeepHit (Lee et al., 2018) for a single risk: softmax PMF over m horizons plus 'later'."""

    def __init__(self, view_dims: list[int], m: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sum(view_dims), hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, m + 1),
        )

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        pmf = torch.softmax(self.net(torch.cat(xs, dim=1)), dim=1)
        self.pmf = pmf
        return torch.cumsum(pmf[:, :-1], dim=1)


class CoxNet(nn.Module):
    """Linear Cox proportional hazards with a Breslow baseline (Cox, 1972)."""

    def __init__(self, view_dims: list[int], m: int = 3):
        super().__init__()
        self.lin = nn.Linear(sum(view_dims), 1, bias=False)
        self.m = m
        self.register_buffer("base_cumhaz", torch.zeros(m))

    def risk(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return self.lin(torch.cat(xs, dim=1)).squeeze(-1)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        r = torch.exp(self.risk(xs))
        return 1.0 - torch.exp(-self.base_cumhaz[None, :] * r[:, None])

    @torch.no_grad()
    def fit_baseline(self, xs: list[torch.Tensor], times: torch.Tensor, events: torch.Tensor) -> None:
        r = torch.exp(self.risk(xs))
        inc = torch.zeros(self.m)
        for u in range(1, self.m + 1):
            d_u = ((times == u) & (events == 1)).sum()
            at_risk = r[times >= u].sum()
            inc[u - 1] = d_u / at_risk
        self.base_cumhaz = torch.cumsum(inc, dim=0)


def cox_partial_likelihood(risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Negative Breslow partial log-likelihood."""
    loss = risk.sum() * 0.0
    n_events = events.sum().clamp(min=1)
    for u in torch.unique(times[events == 1]):
        fail = (times == u) & (events == 1)
        at_risk = times >= u
        loss = loss - (risk[fail].sum() - fail.sum() * torch.logsumexp(risk[at_risk], dim=0))
    return loss / n_events


# --------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    seed: int = 0
    loss_extras: tuple[str, ...] = ()  # for baselines: subset of ("mono", "rank")
    log_conflicts: bool = False
    history: dict = field(default_factory=dict)


def _tensors(panel: MultiviewPanel):
    xs = [torch.from_numpy(v) for v in panel.views]
    t = torch.from_numpy(panel.times)
    e = torch.from_numpy(panel.events)
    y = torch.from_numpy(core.cumulative_labels(panel.times, panel.events, panel.m)).float()
    return xs, t, e, y


@torch.no_grad()
def predict_cumulative(model: nn.Module, panel: MultiviewPanel) -> np.ndarray:
    model.eval()
    xs, _, _, _ = _tensors(panel)
    return model(xs).double().numpy()


def evaluate(f: np.ndarray, panel: MultiviewPanel) -> dict[str, float]:
    out = {"c_index": core.td_concordance(f, panel.times, panel.events)}
    for h in range(1, panel.m + 1):
        out[f"auc_{h}"] = core.td_auc(f, panel.times, panel.events, h)
    out["ibs"] = float(core.brier_scores(f, panel.times, panel.events).mean())
    out["nonmonotone_share"] = float((np.diff(f, axis=1) < -1e-12).any(axis=1).mean())
    return out


def _flat_grad(params) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel()) for p in params])


def fit(model: nn.Module, train: MultiviewPanel, valid: MultiviewPanel, cfg: TrainConfig) -> nn.Module:
    """Train with early stopping on validation time-dependent C-index."""
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    xs, t, e, y = _tensors(train)
    n = len(t)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, bad = -1.0, copy.deepcopy(model.state_dict()), 0
    cos_log: list[float] = []
    is_drdl = isinstance(model, DRDL)

    if isinstance(model, CoxNet):
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=cfg.weight_decay)
        for _ in range(300):
            opt.zero_grad()
            cox_partial_likelihood(model.risk(xs), t, e).backward()
            opt.step()
        model.fit_baseline(xs, t, e)
        return model

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, cfg.batch_size):
            b = perm[start : start + cfg.batch_size]
            xb, tb, eb, yb = [x[b] for x in xs], t[b], e[b], y[b]
            opt.zero_grad()
            f = model(xb)
            if isinstance(model, DeepHitNet):
                # Likelihood: PMF at the event horizon, or survival beyond the window.
                pmf = model.pmf.clamp_min(1e-7)
                ll_event = torch.log(pmf[torch.arange(len(tb)), tb - 1])
                ll_cens = torch.log(pmf[:, -1])
                nll = -torch.where(eb == 1, ll_event, ll_cens).mean()
                loss = nll + torch_ranking_loss(f, tb, eb)
                loss.backward()
            elif is_drdl:
                c = model.cfg
                bce = torch_multiperiod_bce(f, yb)
                fsl = torch_focal_survival_loss(f, tb, eb, c.gamma) * c.fsl_weight if c.use_fsl else bce * 0.0
                reg = model.aux["vq_loss"] + c.orth_weight * model.aux["orth_loss"]
                last = list(model.last_layer.parameters())
                if c.use_fsl and (c.use_agb or cfg.log_conflicts):
                    g_bce = torch.autograd.grad(bce, last, retain_graph=True)
                    g_fsl = torch.autograd.grad(fsl, last, retain_graph=True)
                    gb = torch.cat([g.reshape(-1) for g in g_bce])
                    gf = torch.cat([g.reshape(-1) for g in g_fsl])
                    cos_log.append(float(F_.cosine_similarity(gb, gf, dim=0)))
                (bce + fsl + reg).backward()
                if c.use_fsl and c.use_agb:
                    # Eq. 11 on the last layer of the prediction layer.
                    gf_proj = torch.from_numpy(core.project_conflicting(gb.detach().numpy(), gf.detach().numpy())).float()
                    new = gb + gf_proj
                    offset = 0
                    for p in last:
                        k = p.numel()
                        p.grad = new[offset : offset + k].reshape(p.shape).clone()
                        offset += k
            else:
                loss = torch_multiperiod_bce(f, yb)
                if "mono" in cfg.loss_extras:
                    loss = loss + torch_monotonicity_loss(f)
                if "rank" in cfg.loss_extras:
                    loss = loss + torch_ranking_loss(f, tb, eb)
                loss.backward()
            opt.step()
        score = core.td_concordance(predict_cumulative(model, valid), valid.times, valid.events)
        if score > best + 1e-4:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    cfg.history = {"best_valid_c": best, "epochs": epoch + 1, "cosines": cos_log}
    return model
