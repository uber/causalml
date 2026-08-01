import numpy as np
import pandas as pd
import pytest

try:
    import torch
    from causalml.inference.torch import CEVAE
except ImportError:
    pass
from causalml.dataset import simulate_hidden_confounder
from causalml.metrics import get_cumgain

from .const import RANDOM_SEED


@pytest.mark.torch
def test_CEVAE():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    y, X, treatment, tau, _, _ = simulate_hidden_confounder(
        n=2000, p=5, sigma=1.0, adj=0.0
    )

    outcome_dist = "normal"
    latent_dim = 20
    hidden_dim = 64
    num_epochs = 20
    batch_size = 100
    learning_rate = 1e-3
    learning_rate_decay = 0.1
    num_samples = 200

    cevae = CEVAE(
        outcome_dist=outcome_dist,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        num_samples=num_samples,
    )

    cevae.fit(
        X=torch.tensor(X, dtype=torch.float),
        treatment=torch.tensor(treatment, dtype=torch.float),
        y=torch.tensor(y, dtype=torch.float),
    )

    ite = cevae.predict(X).flatten()

    auuc_metrics = pd.DataFrame({"ite": ite, "W": treatment, "y": y, "tau": tau})

    cumgain = get_cumgain(
        auuc_metrics, outcome_col="y", treatment_col="W", treatment_effect_col="tau"
    )

    # Compare the model's cumulative gain with random targeting.
    random_gain = np.linspace(
        cumgain["ite"].iloc[0], cumgain["ite"].iloc[-1], cumgain.shape[0]
    )
    assert cumgain["ite"].sum() > random_gain.sum()
