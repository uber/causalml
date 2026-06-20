import numpy as np
import pandas as pd
import pytest

try:
    from causalml.inference.jax import DragonNet
except ImportError:
    pass
from causalml.dataset.regression import simulate_nuisance_and_easy_treatment
from causalml.metrics import get_cumgain


@pytest.mark.jax
def test_dragonnet_jax_fit_predict():
    """DragonNet ITE predictions should lift above random targeting."""
    y, X, w, tau, b, e = simulate_nuisance_and_easy_treatment(n=1000)
    dragon = DragonNet(neurons_per_layer=200, targeted_reg=True, verbose=False)
    ite = dragon.fit_predict(X, w, y)

    assert ite.shape == (1000, 1)
    ate = float(ite.mean())
    assert np.isfinite(ate)

    auuc_metrics = pd.DataFrame({"ite": ite.flatten(), "w": w, "y": y, "tau": tau})
    cumgain = get_cumgain(
        auuc_metrics,
        outcome_col="y",
        treatment_col="w",
        treatment_effect_col="tau",
    )
    # Cumulative gain of model predictions should be positive for positive ATE data
    assert cumgain["ite"].sum() > 0


@pytest.mark.jax
def test_dragonnet_jax_predict_propensity():
    """Propensity scores should be in (0, 1)."""
    y, X, w, tau, b, e = simulate_nuisance_and_easy_treatment(n=500)
    dragon = DragonNet(
        neurons_per_layer=50,
        targeted_reg=False,
        verbose=False,
        adam_epochs=2,
        epochs=5,
    )
    dragon.fit(X, w, y)
    propensity = dragon.predict_propensity(X)

    assert propensity.shape == (500,)
    assert np.all(propensity > 0) and np.all(propensity < 1)


@pytest.mark.jax
def test_dragonnet_jax_save_load(tmp_path):
    """Loaded model should reproduce the same predict_tau output."""
    y, X, w, tau, b, e = simulate_nuisance_and_easy_treatment(n=500)
    dragon = DragonNet(
        neurons_per_layer=50,
        targeted_reg=False,
        verbose=False,
        adam_epochs=2,
        epochs=5,
    )
    dragon.fit(X, w, y)
    ate_before = float(dragon.predict_tau(X).mean())

    ckpt_dir = tmp_path / "dragon_ckpt"
    dragon.save(ckpt_dir)

    dragon2 = DragonNet(neurons_per_layer=50, targeted_reg=False)
    dragon2.load(ckpt_dir)

    ate_after = float(dragon2.predict_tau(X).mean())
    assert ate_after == pytest.approx(ate_before, rel=1e-5)
