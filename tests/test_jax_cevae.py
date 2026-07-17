"""Tests for the JAX/flax.nnx CEVAE implementation.

Run with:
    pytest tests/test_jax_cevae.py --runjax -v

The parity test additionally requires ``--runtorch`` so both backends can be
executed side-by-side. To avoid conflicts between pyro's Torch backend and
JAX inside a single Python process (OMP / threading segfaults on macOS), the
parity test spawns each backend in its own subprocess.
"""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

try:
    from causalml.inference.jax import CEVAE as CEVAEJax
except ImportError:  # pragma: no cover - guarded by --runjax marker
    pass

from causalml.dataset import simulate_hidden_confounder


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a).ravel() - np.asarray(b).ravel())))


def _run_backend(backend: str, seed: int = 7) -> float:
    """Trains one CEVAE backend in a fresh subprocess and returns MAE(ITE, tau)."""
    script = textwrap.dedent(f"""
        import json
        import numpy as np
        np.random.seed({seed})
        from causalml.dataset import simulate_hidden_confounder
        y, X, t, tau, _, _ = simulate_hidden_confounder(
            n=2000, p=5, sigma=1.0, adj=0.0
        )
        common = dict(
            outcome_dist='normal', latent_dim=20, hidden_dim=64,
            num_layers=3, num_epochs=20, batch_size=100, num_samples=200,
            learning_rate=1e-3, learning_rate_decay=0.1, weight_decay=1e-4,
        )
        if {backend!r} == 'jax':
            from causalml.inference.jax import CEVAE
        else:
            from causalml.inference.torch import CEVAE
        model = CEVAE(**common)
        ite = model.fit_predict(X, t, y).ravel()
        mae = float(np.mean(np.abs(ite - tau)))
        print('__RESULT__' + json.dumps({{'mae': mae}}))
        """)
    env_extra = {"KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1"}
    import os as _os

    env = {**_os.environ, **env_extra}
    out = subprocess.check_output(
        [sys.executable, "-c", script], env=env, stderr=subprocess.STDOUT
    ).decode()
    for line in out.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__") :])["mae"]
    raise RuntimeError(f"backend {backend} did not report a result:\n{out}")


@pytest.mark.jax
def test_cevae_jax_fit_predict_shapes():
    """`fit_predict` should return an ITE with shape (n,) and finite values."""
    np.random.seed(0)
    y, X, treatment, tau, _, _ = simulate_hidden_confounder(
        n=1000, p=5, sigma=1.0, adj=0.0
    )
    cevae = CEVAEJax(
        outcome_dist="normal",
        latent_dim=8,
        hidden_dim=32,
        num_layers=2,
        num_epochs=10,
        batch_size=100,
        num_samples=50,
    )
    ite = cevae.fit_predict(X, treatment, y)
    assert ite.shape == (1000,)
    assert np.all(np.isfinite(ite))


@pytest.mark.jax
def test_cevae_jax_recovers_positive_ate():
    """On simulated data with a positive treatment effect the estimated ATE
    should be positive and within a reasonable band around the truth."""
    np.random.seed(42)
    y, X, treatment, tau, _, _ = simulate_hidden_confounder(
        n=2000, p=5, sigma=1.0, adj=0.0
    )
    cevae = CEVAEJax(
        outcome_dist="normal",
        latent_dim=20,
        hidden_dim=64,
        num_layers=3,
        num_epochs=25,
        batch_size=100,
        num_samples=100,
    )
    cevae.fit(X, treatment, y)
    ite = cevae.predict(X)

    ate_true = float(np.mean(tau))
    ate_pred = float(np.mean(ite))
    assert ate_pred > 0, f"Expected positive ATE, got {ate_pred}"
    # Loose bound so this stays robust to seeds / mini-batch shuffle order.
    assert abs(ate_pred - ate_true) / max(abs(ate_true), 1e-6) < 0.5


@pytest.mark.jax
def test_cevae_jax_save_load(tmp_path):
    """Round-trip through orbax should preserve predictions bit-for-bit."""
    np.random.seed(0)
    y, X, treatment, _, _, _ = simulate_hidden_confounder(
        n=400, p=4, sigma=1.0, adj=0.0
    )
    cevae = CEVAEJax(
        outcome_dist="normal",
        latent_dim=6,
        hidden_dim=16,
        num_layers=2,
        num_epochs=3,
        batch_size=100,
        num_samples=20,
    )
    cevae.fit(X, treatment, y)
    ate_before = float(np.mean(cevae.predict(X)))

    ckpt_dir = tmp_path / "cevae_ckpt"
    cevae.save(ckpt_dir)

    cevae2 = CEVAEJax(
        outcome_dist="normal",
        latent_dim=6,
        hidden_dim=16,
        num_layers=2,
        num_samples=20,
        batch_size=100,
    )
    cevae2.load(ckpt_dir, X.shape[1])
    ate_after = float(np.mean(cevae2.predict(X)))
    assert ate_after == pytest.approx(ate_before, rel=1e-5)


@pytest.mark.jax
@pytest.mark.torch
def test_cevae_jax_matches_pyro():
    """The JAX and pyro implementations should reach comparable MAE(ITE, tau).

    Each backend is trained in its own subprocess (jax and torch can conflict
    over OMP threads inside a single process) with identical hyperparameters
    and the same simulated dataset seed. Asserts that the JAX MAE stays
    within 25% of the pyro MAE."""
    mae_torch = _run_backend("torch", seed=7)
    mae_jax = _run_backend("jax", seed=7)
    # Only fail when JAX is *worse* than pyro by more than 25% -- a smaller
    # MAE than the reference is a win, not a regression.
    rel_signed = (mae_jax - mae_torch) / max(mae_torch, 1e-8)
    assert rel_signed < 0.25, (
        f"CEVAE JAX MAE={mae_jax:.4f} vs pyro MAE={mae_torch:.4f} "
        f"(signed diff {rel_signed:+.2%}) exceeds +25% tolerance."
    )
