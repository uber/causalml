"""JAX/flax.nnx port of the Causal Effect Variational Autoencoder (CEVAE).

Provides a scikit-learn style :class:`CEVAE` wrapper whose public API mirrors
:class:`causalml.inference.torch.cevae.CEVAE`. Training uses ``optax.adamw``
with an exponential learning-rate decay matching the pyro
:class:`~pyro.optim.ClippedAdam` schedule (``learning_rate * lrd ** step``,
where ``lrd = learning_rate_decay ** (1 / num_steps)``). Weight decay follows
the decoupled AdamW formulation, which differs slightly from the
L2-in-gradient coupling of pyro's ``ClippedAdam``.

References:
    [1] C. Louizos, U. Shalit, J. Mooij, D. Sontag, R. Zemel, M. Welling
        (2017). Causal Effect Inference with Deep Latent-Variable Models.
"""

from __future__ import annotations

import logging

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from causalml.inference.meta.utils import convert_pd_to_np
from causalml.inference.jax.cevae.losses import cevae_loss
from causalml.inference.jax.cevae.modeling import Guide, Model, PreWhitener

logger = logging.getLogger(__name__)


class _CEVAENet(nnx.Module):
    """Container holding model + guide + whitener as a single nnx tree."""

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        outcome_dist: str,
        *,
        rngs: nnx.Rngs,
    ):
        self.model = Model(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            outcome_dist=outcome_dist,
            rngs=rngs,
        )
        self.guide = Guide(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            outcome_dist=outcome_dist,
            rngs=rngs,
        )
        # Whitener is initialised with zeros/ones so nnx.state includes it;
        # it is (re)fitted from data at the beginning of `fit`.
        placeholder = jnp.zeros((1, feature_dim))
        self.whiten = PreWhitener(placeholder)


def _make_train_step(loss_fn):
    """Returns a jit-compiled step that updates ``model`` in-place."""

    @nnx.jit
    def train_step(net, optimizer, key, x, t, y):
        def _loss(net):
            return loss_fn(net.model, net.guide, key, x, t, y)

        loss_val, grads = nnx.value_and_grad(_loss)(net)
        optimizer.update(net, grads)
        return loss_val

    return train_step


class CEVAE:
    """JAX/flax.nnx CEVAE for treatment-effect estimation.

    Mirrors the API of :class:`causalml.inference.torch.cevae.CEVAE`.

    Args:
        outcome_dist: Outcome distribution as one of ``"bernoulli"``,
            ``"exponential"``, ``"laplace"``, ``"normal"``, ``"studentt"``.
        latent_dim: Dimension of the latent variable ``z``.
        hidden_dim: Width of the hidden layers of the fully-connected nets.
        num_epochs: Number of training epochs.
        num_layers: Number of hidden layers in the fully-connected nets.
        batch_size: Mini-batch size.
        learning_rate: Initial Adam learning rate.
        learning_rate_decay: Overall LR decay across training; per-step decay
            is ``learning_rate_decay ** (1 / num_steps)``.
        num_samples: Number of Monte Carlo samples used by :meth:`predict`.
        weight_decay: Decoupled (AdamW-style) weight decay coefficient.
        seed: PRNG seed for parameter initialization and mini-batch shuffling.
    """

    def __init__(
        self,
        outcome_dist: str = "studentt",
        latent_dim: int = 20,
        hidden_dim: int = 200,
        num_epochs: int = 50,
        num_layers: int = 3,
        batch_size: int = 100,
        learning_rate: float = 1e-3,
        learning_rate_decay: float = 0.1,
        num_samples: int = 1000,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ):
        self.outcome_dist = outcome_dist
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.num_samples = num_samples
        self.weight_decay = weight_decay
        self.seed = seed
        self._net = None
        self._feature_dim = None

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_net(self, feature_dim: int) -> _CEVAENet:
        rngs = nnx.Rngs(self.seed)
        return _CEVAENet(
            feature_dim=feature_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            outcome_dist=self.outcome_dist,
            rngs=rngs,
        )

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X, treatment, y, p=None):
        """Fits CEVAE on ``(X, treatment, y)``.

        Args:
            X: Feature matrix of shape ``(n, feature_dim)``.
            treatment: Binary treatment vector of shape ``(n,)``.
            y: Outcome vector of shape ``(n,)``.
            p: Ignored (kept for API compatibility with meta-learners).
        """
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        X = X.astype(np.float32)
        treatment = treatment.astype(np.float32).reshape(-1)
        y = y.astype(np.float32).reshape(-1)

        n, feature_dim = X.shape
        self._feature_dim = feature_dim
        self._net = self._build_net(feature_dim)

        # (Re)fit the whitener from data.
        self._net.whiten = PreWhitener(X)

        # The batch loop below runs ceil(n / batch_size) batches per epoch.
        n_batches = max(1, -(-n // self.batch_size))
        num_steps = max(1, self.num_epochs * n_batches)
        per_step_decay = self.learning_rate_decay ** (1.0 / num_steps)
        schedule = optax.exponential_decay(
            init_value=self.learning_rate,
            transition_steps=1,
            decay_rate=per_step_decay,
            staircase=False,
        )
        tx = optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.adamw(learning_rate=schedule, weight_decay=self.weight_decay),
        )
        optimizer = nnx.Optimizer(self._net, tx, wrt=nnx.Param)

        train_step = _make_train_step(cevae_loss)

        rng = np.random.default_rng(self.seed)
        key = jax.random.PRNGKey(self.seed)

        losses = []
        for epoch in range(self.num_epochs):
            perm = rng.permutation(n)
            X_shuf, t_shuf, y_shuf = X[perm], treatment[perm], y[perm]
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                xb = jnp.asarray(X_shuf[start:end])
                tb = jnp.asarray(t_shuf[start:end])
                yb = jnp.asarray(y_shuf[start:end])
                xb_w = self._net.whiten(xb)
                key, subkey = jax.random.split(key)
                loss_val = train_step(self._net, optimizer, subkey, xb_w, tb, yb)
                if not np.isfinite(loss_val):
                    raise FloatingPointError(
                        f"CEVAE training diverged: non-finite loss at epoch {epoch + 1}."
                    )
                epoch_loss += float(loss_val)
            losses.append(epoch_loss / n)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("epoch %d loss=%.6g", epoch + 1, epoch_loss / n)
        self._losses = losses
        return self

    def _ite_batch(self, X_batch, num_samples, key):
        """Monte Carlo estimate of E[y1 - y0 | x] for one batch of ``X``."""
        x_w = self._net.whiten(jnp.asarray(X_batch))
        # Auxiliary samples t_aux ~ q(t|x), y_aux ~ q(y|t_aux,x).
        key_t, key_y, key_z = jax.random.split(key, 3)
        n = x_w.shape[0]
        # Replicate x across the particle axis to match pyro's
        # `pyro.plate("num_particles", num_samples, dim=-2)`.
        x_rep = jnp.broadcast_to(x_w, (num_samples, n, x_w.shape[-1]))

        # t_aux
        t_logits = self._net.guide.t_params(x_rep)[0]
        t_uniform = jax.random.uniform(key_t, t_logits.shape)
        t_aux = (t_uniform < jax.nn.sigmoid(t_logits)).astype(jnp.float32)

        # y_aux ~ q(y|t_aux,x)
        y_aux = self._sample_outcome_from_guide(x_rep, t_aux, key_y)

        # z ~ q(z|y_aux, t_aux, x)
        loc, scale = self._net.guide.z_params(y_aux, t_aux, x_rep)
        eps = jax.random.normal(key_z, loc.shape)
        z = loc + scale * eps

        # Counterfactual predictions.
        t0 = jnp.zeros_like(t_aux)
        t1 = jnp.ones_like(t_aux)
        y0 = self._net.model.y_mean(t0, z)
        y1 = self._net.model.y_mean(t1, z)
        ite = (y1 - y0).mean(axis=0)  # average over particles
        return ite

    def _sample_outcome_from_guide(self, x_rep, t_aux, key):
        """Samples ``y ~ q(y | t_aux, x)`` under the current guide."""
        outcome = self.outcome_dist
        params = self._net.guide.y_params(t_aux, x_rep)
        if outcome == "bernoulli":
            (logits,) = params
            u = jax.random.uniform(key, logits.shape)
            return (u < jax.nn.sigmoid(logits)).astype(jnp.float32)
        if outcome == "exponential":
            (rate,) = params
            return jax.random.exponential(key, rate.shape) / rate
        if outcome == "laplace":
            loc, scale = params
            return loc + scale * jax.random.laplace(key, loc.shape)
        if outcome == "normal":
            loc, scale = params
            return loc + scale * jax.random.normal(key, loc.shape)
        if outcome == "studentt":
            df, loc, scale = params
            return loc + scale * jax.random.t(key, df, loc.shape)
        raise ValueError(f"Unknown outcome_dist {outcome!r}")

    def predict(self, X, treatment=None, y=None, p=None):
        """Predicts individual treatment effect for each row of ``X``.

        Args:
            X: Feature matrix of shape ``(n, feature_dim)``.
            treatment: Ignored (API compatibility).
            y: Ignored (API compatibility).
            p: Ignored (API compatibility).

        Returns:
            ``np.ndarray`` of shape ``(n,)`` with ITE estimates.
        """
        if self._net is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=np.float32)
        if X.shape[-1] != self._feature_dim:
            raise ValueError(
                f"X has {X.shape[-1]} features, expected {self._feature_dim}."
            )
        n = X.shape[0]
        batch_size = self.batch_size if self.batch_size else n
        key = jax.random.PRNGKey(self.seed + 1)
        results = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            key, subkey = jax.random.split(key)
            ite = self._ite_batch(X[start:end], self.num_samples, subkey)
            results.append(np.asarray(ite))
        return np.concatenate(results, axis=0)

    def fit_predict(self, X, treatment, y, p=None):
        """Fits the model then returns ITE estimates on the same ``X``.

        Args:
            X: Feature matrix.
            treatment: Binary treatment vector.
            y: Outcome vector.
            p: Ignored (API compatibility).

        Returns:
            ``np.ndarray`` of shape ``(n,)`` with ITE estimates.
        """
        self.fit(X, treatment, y)
        return self.predict(X)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path):
        """Saves parameters to an orbax checkpoint directory."""
        import orbax.checkpoint as ocp

        if self._net is None:
            raise RuntimeError("Call fit() before save().")
        path = str(path)
        state = nnx.state(self._net)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(path, state)
        checkpointer.wait_until_finished()

    def load(self, path, feature_dim):
        """Restores parameters from an orbax checkpoint.

        Args:
            path: Directory of a previously saved checkpoint.
            feature_dim: Number of input features (needed to rebuild the net).
        """
        import orbax.checkpoint as ocp

        path = str(path)
        self._feature_dim = feature_dim
        self._net = self._build_net(feature_dim)
        state = nnx.state(self._net)
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(path, state)
        nnx.update(self._net, restored)
