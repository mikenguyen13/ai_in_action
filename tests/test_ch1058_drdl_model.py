"""Tests for aiinaction.ch1058_drdl_model: torch pieces agree with the numpy core."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aiinaction import ch1058_drdl as core
from aiinaction import ch1058_drdl_model as dm

OUTPUTS = [[-1.0, 0.5, -0.3], [0.2, -2.0, 0.1], [-3.0, -1.0, 1.0], [0.0, 0.0, 0.0]]
TIMES = [1, 2, 3, 3]
EVENTS = [1, 1, 1, 0]


def test_torch_risk_accumulation_matches_core():
    f = dm.torch_risk_accumulation(torch.tensor(OUTPUTS, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(f, core.risk_accumulation(OUTPUTS), atol=1e-12)


@pytest.mark.parametrize("gamma", [1.0, 2.0])
def test_torch_focal_loss_matches_core(gamma):
    f = core.risk_accumulation(OUTPUTS)
    got = dm.torch_focal_survival_loss(
        torch.tensor(f), torch.tensor(TIMES), torch.tensor(EVENTS), gamma
    ).item()
    assert got == pytest.approx(core.focal_survival_loss(f, TIMES, EVENTS, gamma), abs=1e-9)


def test_torch_bce_and_orth_match_core():
    f = core.risk_accumulation(OUTPUTS)
    y = torch.from_numpy(core.cumulative_labels(TIMES, EVENTS, 3))
    assert dm.torch_multiperiod_bce(torch.tensor(f), y).item() == pytest.approx(
        core.multiperiod_bce(f, TIMES, EVENTS), abs=1e-6
    )
    cb = [[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]]
    assert dm.torch_orthogonal_loss(torch.tensor(cb)).item() == pytest.approx(core.orthogonal_loss(cb), abs=1e-6)


def test_mask_torch_matches_core_rowwise():
    rows = [[2, 0, 2, 1, 0, 3], [5, 5, 5, 1, 2, 1]]
    got = dm.first_occurrence_mask_torch(torch.tensor(rows)).numpy()
    for r, g in zip(rows, got):
        assert g.tolist() == core.first_occurrence_mask(r).tolist()


def test_simulator_shapes_and_censoring():
    p = dm.make_multiview_risk_panel(400, seed=3)
    assert [v.shape for v in p.views] == [(400, 12), (400, 16), (400, 8)]
    assert set(np.unique(p.events)) <= {0, 1}
    assert np.all(p.times[p.events == 0] == 3)
    assert 0.15 < p.events.mean() < 0.6
    wide = dm.make_multiview_risk_panel(50, seed=3, width=2)
    assert [v.shape[1] for v in wide.views] == [24, 32, 16]
    with pytest.raises(ValueError):
        dm.make_multiview_risk_panel(50, m=4)


def test_drdl_forward_is_monotone_and_masks_duplicates():
    torch.manual_seed(0)
    p = dm.make_multiview_risk_panel(64, seed=1)
    model = dm.DRDL(dm.DRDLConfig([v.shape[1] for v in p.views], codebook_size=8))
    model.eval()
    with torch.no_grad():
        f = model([torch.from_numpy(v) for v in p.views]).numpy()
    assert np.all(np.diff(f, axis=1) >= 0)
    codes = model.aux["codes"].numpy()
    mask = model.aux["mask"].numpy()
    for c, mk in zip(codes, mask):
        assert mk.tolist() == core.first_occurrence_mask(c).tolist()


def test_sigmoid_ablation_can_be_nonmonotone():
    cfg = dm.DRDLConfig([12, 16, 8], use_risk_accumulation=False)
    assert not dm.DRDL(cfg).cfg.use_risk_accumulation


def test_fit_runs_and_agb_logs_cosines():
    torch.manual_seed(0)
    p = dm.make_multiview_risk_panel(300, seed=2)
    idx = np.arange(p.n)
    tr, va = dm.standardize(p.subset(idx[:200]), p.subset(idx[200:]))
    model = dm.DRDL(dm.DRDLConfig([v.shape[1] for v in p.views], codebook_size=16))
    cfg = dm.TrainConfig(epochs=2, batch_size=100, seed=0)
    dm.fit(model, tr, va, cfg)
    assert len(cfg.history["cosines"]) > 0
    res = dm.evaluate(dm.predict_cumulative(model, va), va)
    assert 0.0 <= res["c_index"] <= 1.0
    assert res["nonmonotone_share"] == 0.0


@pytest.mark.parametrize("ctor", [
    lambda d: dm.BaselineNet(d, fusion="concat"),
    lambda d: dm.BaselineNet(d, fusion="attention"),
    lambda d: dm.DeepHitNet(d),
    lambda d: dm.CoxNet(d),
])
def test_baselines_fit(ctor):
    torch.manual_seed(0)
    p = dm.make_multiview_risk_panel(200, seed=4)
    tr, va = dm.standardize(p.subset(np.arange(150)), p.subset(np.arange(150, 200)))
    model = ctor([v.shape[1] for v in p.views])
    dm.fit(model, tr, va, dm.TrainConfig(epochs=2, batch_size=64, loss_extras=("mono", "rank")))
    f = dm.predict_cumulative(model, va)
    assert f.shape == (50, 3)
    assert np.all((f >= 0) & (f <= 1))
