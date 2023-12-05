from causalml.inference.tf import DragonNet
from causalml.dataset.regression import *
import shutil


def test_save_load_dragonnet():
    y, X, w, tau, b, e = simulate_nuisance_and_easy_treatment(n=1000)

    dragon = DragonNet(neurons_per_layer=200, targeted_reg=True, verbose=False)
    dragon_ite = dragon.fit_predict(X, w, y, return_components=False)
    dragon_ate = dragon_ite.mean()
    dragon.save("smaug")

    smaug = DragonNet()
    smaug.load("smaug")
    shutil.rmtree("smaug")

    assert smaug.predict_tau(X).mean() == dragon_ate
