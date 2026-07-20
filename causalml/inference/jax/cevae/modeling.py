"""flax.nnx modules for CEVAE.

Ports the pyro CEVAE architecture (:mod:`pyro.contrib.cevae`) to
`jax`/`flax.nnx`, keeping module/parameter names as close to the reference
as possible so the two implementations can be diffed line-by-line.

References:
    [1] C. Louizos, U. Shalit, J. Mooij, D. Sontag, R. Zemel, M. Welling
        (2017). Causal Effect Inference with Deep Latent-Variable Models.
        http://papers.nips.cc/paper/7223-causal-effect-inference-with-deep-latent-variable-models.pdf
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
from flax import nnx

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class FullyConnected(nnx.Module):
    """Multi-layer perceptron with ELU activations.

    Mirrors :class:`pyro.contrib.cevae.FullyConnected`. Each pair of consecutive
    ``sizes`` defines a :class:`flax.nnx.Linear` followed by ``elu``, except for
    the last linear layer whose activation is controlled by ``final_activation``.

    Args:
        sizes: Layer widths, including input and output dimensions. Must have
            at least two entries.
        final_activation: Optional callable applied after the last linear
            layer. Defaults to ``None`` (identity).
        rngs: :class:`flax.nnx.Rngs` container.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        final_activation=None,
        *,
        rngs: nnx.Rngs,
    ):
        if len(sizes) < 2:
            raise ValueError(f"FullyConnected needs at least 2 sizes, got {sizes}")
        self._sizes = tuple(sizes)
        self.final_activation = final_activation
        self.layers = nnx.List(
            [
                nnx.Linear(in_size, out_size, rngs=rngs)
                for in_size, out_size in zip(sizes[:-1], sizes[1:])
            ]
        )

    def __call__(self, x):
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < last:
                x = jax.nn.elu(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


# ---------------------------------------------------------------------------
# Distribution wrappers
#
# Each distribution "net" outputs the parameters of a conditional distribution.
# ``log_prob(params, value)`` and ``mean(params)`` are provided as class
# methods so that the same code path works for training (ELBO) and inference
# (``y_mean``). This mirrors the ``DistributionNet.make_dist`` pattern from
# pyro without pulling a full probabilistic-programming stack into the
# training loop.
# ---------------------------------------------------------------------------


class BernoulliNet(nnx.Module):
    """Bernoulli net outputting a single ``logits`` value."""

    def __init__(self, sizes: Sequence[int], *, rngs: nnx.Rngs):
        if len(sizes) < 1:
            raise ValueError("BernoulliNet needs at least 1 size")
        self.fc = FullyConnected(list(sizes) + [1], rngs=rngs)

    def __call__(self, x):
        logits = jnp.clip(self.fc(x).squeeze(-1), min=-10.0, max=10.0)
        return (logits,)

    @staticmethod
    def log_prob(params, value):
        (logits,) = params
        # log Bernoulli(value | logits) using the log-sigmoid trick.
        return -jax.nn.softplus(-logits) * value - jax.nn.softplus(logits) * (
            1.0 - value
        )

    @staticmethod
    def mean(params):
        (logits,) = params
        return jax.nn.sigmoid(logits)


class ExponentialNet(nnx.Module):
    """Exponential net outputting a positive ``rate``."""

    def __init__(self, sizes: Sequence[int], *, rngs: nnx.Rngs):
        self.fc = FullyConnected(list(sizes) + [1], rngs=rngs)

    def __call__(self, x):
        scale = jnp.clip(
            jax.nn.softplus(self.fc(x).squeeze(-1)),
            min=1e-3,
            max=1e6,
        )
        rate = jnp.reciprocal(scale)
        return (rate,)

    @staticmethod
    def log_prob(params, value):
        (rate,) = params
        # log Exp(value | rate) = log(rate) - rate * value  for value >= 0
        safe_value = jnp.maximum(value, 0.0)
        return jnp.log(rate) - rate * safe_value

    @staticmethod
    def mean(params):
        (rate,) = params
        return jnp.reciprocal(rate)


class LaplaceNet(nnx.Module):
    """Laplace net outputting ``(loc, scale)``."""

    def __init__(self, sizes: Sequence[int], *, rngs: nnx.Rngs):
        self.fc = FullyConnected(list(sizes) + [2], rngs=rngs)

    def __call__(self, x):
        loc_scale = self.fc(x)
        loc = jnp.clip(loc_scale[..., 0], min=-1e6, max=1e6)
        scale = jnp.clip(jax.nn.softplus(loc_scale[..., 1]), min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def log_prob(params, value):
        loc, scale = params
        return jss.laplace.logpdf(value, loc=loc, scale=scale)

    @staticmethod
    def mean(params):
        loc, _ = params
        return loc


class NormalNet(nnx.Module):
    """Normal net outputting ``(loc, scale)``."""

    def __init__(self, sizes: Sequence[int], *, rngs: nnx.Rngs):
        self.fc = FullyConnected(list(sizes) + [2], rngs=rngs)

    def __call__(self, x):
        loc_scale = self.fc(x)
        loc = jnp.clip(loc_scale[..., 0], min=-1e6, max=1e6)
        scale = jnp.clip(jax.nn.softplus(loc_scale[..., 1]), min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def log_prob(params, value):
        loc, scale = params
        return jss.norm.logpdf(value, loc=loc, scale=scale)

    @staticmethod
    def mean(params):
        loc, _ = params
        return loc


class StudentTNet(nnx.Module):
    """Student's T net outputting ``(df, loc, scale)`` with shared ``df > 1``."""

    def __init__(self, sizes: Sequence[int], *, rngs: nnx.Rngs):
        self.fc = FullyConnected(list(sizes) + [2], rngs=rngs)
        self.df_unconstrained = nnx.Param(jnp.zeros(()))

    def __call__(self, x):
        loc_scale = self.fc(x)
        loc = jnp.clip(loc_scale[..., 0], min=-1e6, max=1e6)
        scale = jnp.clip(jax.nn.softplus(loc_scale[..., 1]), min=1e-3, max=1e6)
        df = jax.nn.softplus(self.df_unconstrained[...]) + 1.0
        df = jnp.broadcast_to(df, loc.shape)
        return df, loc, scale

    @staticmethod
    def log_prob(params, value):
        df, loc, scale = params
        return jss.t.logpdf(value, df=df, loc=loc, scale=scale)

    @staticmethod
    def mean(params):
        _, loc, _ = params
        return loc


_OUTCOME_NETS = {
    "bernoulli": BernoulliNet,
    "exponential": ExponentialNet,
    "laplace": LaplaceNet,
    "normal": NormalNet,
    "studentt": StudentTNet,
}


def get_outcome_net(name: str):
    """Returns the outcome-distribution net class for ``name``.

    Args:
        name: One of ``"bernoulli"``, ``"exponential"``, ``"laplace"``,
            ``"normal"``, ``"studentt"``.

    Returns:
        The corresponding :class:`nnx.Module` subclass.
    """
    try:
        return _OUTCOME_NETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown outcome_dist {name!r}; expected one of {sorted(_OUTCOME_NETS)}"
        ) from exc


class DiagNormalNet(nnx.Module):
    """Diagonal Normal net outputting ``(loc, scale)`` for a ``sizes[-1]`` vec.

    Used for the latent ``z`` and for the observation ``x`` conditional in the
    generative model. Applies the same conservative clipping as the pyro
    implementation.
    """

    def __init__(self, sizes: Sequence[int], *, rngs: nnx.Rngs):
        if len(sizes) < 2:
            raise ValueError(f"DiagNormalNet needs at least 2 sizes, got {sizes}")
        self.dim = sizes[-1]
        self.fc = FullyConnected(list(sizes[:-1]) + [self.dim * 2], rngs=rngs)

    def __call__(self, x):
        loc_scale = self.fc(x)
        loc = jnp.clip(loc_scale[..., : self.dim], min=-1e2, max=1e2)
        scale = jnp.clip(
            jax.nn.softplus(loc_scale[..., self.dim :]) + 1e-3,
            min=None,
            max=1e2,
        )
        return loc, scale


# ---------------------------------------------------------------------------
# Pre-whitener
# ---------------------------------------------------------------------------


class PreWhitener(nnx.Module):
    """Standardises inputs using stats fitted on the training data.

    Stats are stored as :class:`nnx.Variable` (not :class:`nnx.Param`) so they
    are excluded from gradient updates but included in ``nnx.state`` for
    checkpointing.
    """

    def __init__(self, data):
        loc = jnp.asarray(data).mean(axis=0)
        scale = jnp.asarray(data).std(axis=0)
        scale = jnp.where(scale > 0, scale, jnp.ones_like(scale))
        self.loc = nnx.Variable(loc)
        self.inv_scale = nnx.Variable(jnp.reciprocal(scale))

    def __call__(self, data):
        return (data - self.loc[...]) * self.inv_scale[...]


# ---------------------------------------------------------------------------
# CEVAE generative model and inference model (guide)
# ---------------------------------------------------------------------------


class Model(nnx.Module):
    """Generative model ``p(z) p(x|z) p(t|z) p(y|t,z)`` from the CEVAE paper."""

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
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.outcome_dist = outcome_dist

        self.x_nn = DiagNormalNet(
            [latent_dim] + [hidden_dim] * num_layers + [feature_dim], rngs=rngs
        )
        outcome_cls = get_outcome_net(outcome_dist)
        self.y0_nn = outcome_cls([latent_dim] + [hidden_dim] * num_layers, rngs=rngs)
        self.y1_nn = outcome_cls([latent_dim] + [hidden_dim] * num_layers, rngs=rngs)
        self.t_nn = BernoulliNet([latent_dim], rngs=rngs)

    # --- distributions -----------------------------------------------------

    def z_params(self):
        loc = jnp.zeros((self.latent_dim,))
        scale = jnp.ones((self.latent_dim,))
        return loc, scale

    def x_params(self, z):
        return self.x_nn(z)

    def t_params(self, z):
        return self.t_nn(z)

    def y_params(self, t, z):
        params0 = self.y0_nn(z)
        params1 = self.y1_nn(z)
        t_bool = t.astype(jnp.bool_)
        selected = []
        for p0, p1 in zip(params0, params1):
            # Broadcast t across the parameter shape (works for scalar per-row
            # params such as loc/scale, and preserves diag-normal semantics).
            t_expanded = t_bool
            while t_expanded.ndim < p0.ndim:
                t_expanded = t_expanded[..., None]
            selected.append(jnp.where(t_expanded, p1, p0))
        return tuple(selected)

    # --- log-probs ---------------------------------------------------------

    def log_prob_z(self, z):
        loc, scale = self.z_params()
        return jss.norm.logpdf(z, loc=loc, scale=scale).sum(axis=-1)

    def log_prob_x(self, x, z):
        loc, scale = self.x_params(z)
        return jss.norm.logpdf(x, loc=loc, scale=scale).sum(axis=-1)

    def log_prob_t(self, t, z):
        params = self.t_params(z)
        return BernoulliNet.log_prob(params, t)

    def log_prob_y(self, y, t, z):
        params = self.y_params(t, z)
        outcome_cls = get_outcome_net(self.outcome_dist)
        return outcome_cls.log_prob(params, y)

    def y_mean(self, t, z):
        params = self.y_params(t, z)
        outcome_cls = get_outcome_net(self.outcome_dist)
        return outcome_cls.mean(params)


class Guide(nnx.Module):
    """Inference network ``q(t|x) q(y|t,x) q(z|y,t,x)`` from the CEVAE paper."""

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
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.outcome_dist = outcome_dist

        self.t_nn = BernoulliNet([feature_dim], rngs=rngs)

        # y network: shared trunk + per-treatment heads.
        self.y_nn = FullyConnected(
            [feature_dim] + [hidden_dim] * (num_layers - 1),
            final_activation=jax.nn.elu,
            rngs=rngs,
        )
        outcome_cls = get_outcome_net(outcome_dist)
        self.y0_nn = outcome_cls([hidden_dim], rngs=rngs)
        self.y1_nn = outcome_cls([hidden_dim], rngs=rngs)

        # z network: shared trunk on (y, x) + per-treatment DiagNormal heads.
        self.z_nn = FullyConnected(
            [1 + feature_dim] + [hidden_dim] * (num_layers - 1),
            final_activation=jax.nn.elu,
            rngs=rngs,
        )
        self.z0_nn = DiagNormalNet([hidden_dim, latent_dim], rngs=rngs)
        self.z1_nn = DiagNormalNet([hidden_dim, latent_dim], rngs=rngs)

    # --- parameters --------------------------------------------------------

    def t_params(self, x):
        return self.t_nn(x)

    def y_params(self, t, x):
        hidden = self.y_nn(x)
        params0 = self.y0_nn(hidden)
        params1 = self.y1_nn(hidden)
        t_bool = t.astype(jnp.bool_)
        selected = []
        for p0, p1 in zip(params0, params1):
            t_expanded = t_bool
            while t_expanded.ndim < p0.ndim:
                t_expanded = t_expanded[..., None]
            selected.append(jnp.where(t_expanded, p1, p0))
        return tuple(selected)

    def z_params(self, y, t, x):
        y_x = jnp.concatenate([y[..., None], x], axis=-1)
        hidden = self.z_nn(y_x)
        loc0, scale0 = self.z0_nn(hidden)
        loc1, scale1 = self.z1_nn(hidden)
        t_expanded = t.astype(jnp.bool_)[..., None]
        loc = jnp.where(t_expanded, loc1, loc0)
        scale = jnp.where(t_expanded, scale1, scale0)
        return loc, scale

    # --- log-probs ---------------------------------------------------------

    def log_prob_t(self, t, x):
        params = self.t_params(x)
        return BernoulliNet.log_prob(params, t)

    def log_prob_y(self, y, t, x):
        params = self.y_params(t, x)
        outcome_cls = get_outcome_net(self.outcome_dist)
        return outcome_cls.log_prob(params, y)

    def sample_z(self, y, t, x, key):
        loc, scale = self.z_params(y, t, x)
        eps = jax.random.normal(key, loc.shape)
        z = loc + scale * eps
        log_q = jss.norm.logpdf(z, loc=loc, scale=scale).sum(axis=-1)
        return z, log_q

    def t_mean(self, x):
        return BernoulliNet.mean(self.t_params(x))

    def y_mean(self, t, x):
        params = self.y_params(t, x)
        outcome_cls = get_outcome_net(self.outcome_dist)
        return outcome_cls.mean(params)
