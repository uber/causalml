try:
    from causalml.inference.tf import DragonNet
except ImportError:
    pass
from causalml.dataset.regression import simulate_nuisance_and_easy_treatment
import pytest


@pytest.mark.tf
def test_save_load_dragonnet(tmp_path):
    y, X, w, tau, b, e = simulate_nuisance_and_easy_treatment(n=1000)

    dragon = DragonNet(neurons_per_layer=200, targeted_reg=True, verbose=False)
    dragon_ite = dragon.fit_predict(X, w, y, return_components=False)
    dragon_ate = dragon_ite.mean()

    model_file = tmp_path / "smaug.h5"
    dragon.save(model_file)

    smaug = DragonNet()
    smaug.load(model_file)

    assert smaug.predict_tau(X).mean() == dragon_ate
