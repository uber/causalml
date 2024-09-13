import pandas as pd
import pytest

try:
    import torch
    from causalml.inference.torch import CEVAE
except ImportError:
    pass
from causalml.dataset import simulate_hidden_confounder
from causalml.metrics import get_cumgain


@pytest.mark.torch
def test_CEVAE():
    y, X, treatment, tau, b, e = simulate_hidden_confounder(
        n=10000, p=5, sigma=1.0, adj=0.0
    )

    outcome_dist = "normal"
    latent_dim = 20
    hidden_dim = 200
    num_epochs = 50
    batch_size = 100
    learning_rate = 1e-3
    learning_rate_decay = 0.1

    cevae = CEVAE(
        outcome_dist=outcome_dist,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
    )

    cevae.fit(
        X=torch.tensor(X, dtype=torch.float),
        treatment=torch.tensor(treatment, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.float),
    )

    # check the accuracy of the ite accuracy
    ite = cevae.predict(X).flatten()

    auuc_metrics = pd.DataFrame(
        {"ite": ite, "W": treatment, "y": y, "treatment_effect_col": tau}
    )

    cumgain = get_cumgain(
        auuc_metrics, outcome_col="y", treatment_col="W", treatment_effect_col="tau"
    )

    # Check if the cumulative gain when using the model's prediction is
    # higher than it would be under random targeting
    assert cumgain["ite"].sum() > cumgain["Random"].sum()
