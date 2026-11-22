import numpy as np
import pytest

from broilerseg.experiments import WeightExperiment
from broilerseg.io import load_flocklabel_export
from broilerseg.massmodel import (
    MassModel,
    adjust_mask,
    build_feature_table,
    build_scale_table,
    estimate_density,
    fit_ab,
    fit_mass_model,
    shape_volume_px,
    trim_ends,
)
from broilerseg.synthetic import ColorSegmenter, make_demo_data
from broilerseg.volume import estimate_volume


def rect(h=40, w=100):
    m = np.zeros((h + 20, w + 20), bool)
    m[10:10 + h, 10:10 + w] = True
    return m


def test_offset_and_trim():
    m = rect()
    assert adjust_mask(m, 2).sum() > m.sum() > adjust_mask(m, -2).sum()
    assert (adjust_mask(m, 0) == m).all()
    t = trim_ends(m, 0.1)
    assert t.any(axis=0).sum() == 80  # 10% of 100 columns removed at each end


def test_shape_volume_matches_repository_volume():
    m = np.zeros((200, 200), np.uint8)
    import cv2

    cv2.ellipse(m, (100, 100), (70, 30), 25, 0, 360, 1, -1)
    p = 0.2
    ref = estimate_volume(m.astype(bool), p).volume_cm3
    assert shape_volume_px(m) * p ** 3 == pytest.approx(ref, rel=1e-9)


def test_fit_ab_recovers_power_law_and_fixed_b():
    rng = np.random.default_rng(0)
    v = rng.uniform(500, 5000, 200)
    m = 0.002 * v ** 0.9
    a, b = fit_ab(v, m, ridge=0.0)
    assert b == pytest.approx(0.9, abs=1e-6) and a == pytest.approx(0.002, rel=1e-6)
    a1, b1 = fit_ab(v, 0.0008 * v, fix_b=True)
    assert b1 == 1.0 and a1 == pytest.approx(0.0008)
    model = MassModel(a1, 1.0)
    assert model.effective_density() == pytest.approx(800)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    p = make_demo_data(tmp_path_factory.mktemp("mm"), n_events=6, frames_per_event=5)
    exp = load_flocklabel_export(p["export"])
    we = WeightExperiment(ColorSegmenter())
    we.run(exp.samples)
    return we


def test_fit_mass_model_beats_repository_baseline(experiment):
    f = build_feature_table(experiment, (-2, 0, 2), (0.0, 0.1), (None, 10), progress=False)
    s = build_scale_table(experiment)
    rep = fit_mass_model(f, s, progress=False)
    comp = rep.comparison.set_index("model").rmse_kg
    assert comp.iloc[-1] < comp.iloc[0] / 3          # fitting removes most of the error
    assert not rep.nested_folds.empty and len(rep.nested_folds) == 6
    dens = estimate_density(rep, n_boot=200)
    assert dens["ci95"][0] <= dens["density_kg_m3"] <= dens["ci95"][1]
    # with the repository's shape settings the density is physical: synthetic birds are
    # ellipsoids with depth 0.8 x width at 1000 kg/m^3 -> about 800 effective
    f0 = f[(f.offset_px == 0) & (f.trim == 0.0) & (f.slice_count == "per-column")]
    rep0 = fit_mass_model(f0, s, fix_b=True, progress=False, nested=False)
    assert 700 < estimate_density(rep0, n_boot=100)["density_kg_m3"] < 900
    assert len(rep.oof_events) == 6 and (rep.cv_table.cv_rmse_kg.diff().dropna() > -1e-6).all()


def test_identical_weights_fix_the_exponent(experiment):
    f = build_feature_table(experiment, (0,), (0.0,), (None,), progress=False)
    rep = fit_mass_model(f.assign(actual_kg=0.45), build_scale_table(experiment), progress=False)
    assert not rep.b_searched and rep.model.b == 1.0
    assert "too similar" in rep.b_status()


def test_fitted_records_and_estimate(experiment):
    from broilerseg.massmodel import fitted_records

    f = build_feature_table(experiment, (0, 2), (0.0, 0.1), (None, 10), progress=False)
    rep = fit_mass_model(f, build_scale_table(experiment), progress=False, nested=False)
    fr = fitted_records(rep, experiment.records)
    assert len(fr) == len(experiment.records) and fr.pred_kg.notna().all()
    assert np.allclose(fr.pred_final_kg, rep.model.predict_from_volume(fr.volume_cm3))
    sid = int(fr.sample_id.iloc[0])
    p = float(fr.pixel_to_cm.iloc[0])
    est = rep.model.estimate(experiment.birds[sid].mask, p)
    assert est.mass_kg == pytest.approx(rep.model.predict(experiment.birds[sid].mask, p), rel=1e-9)
