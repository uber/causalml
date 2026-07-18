"""Loss functions for the CEVAE JAX implementation.

Implements the ``TraceCausalEffect_ELBO`` objective from
:mod:`pyro.contrib.cevae`::

    -loss = ELBO + log q(t|x) + log q(y|t,x)

with

    ELBO = E_q[ log p(x|z) + log p(t|z) + log p(y|t,z) + log p(z) - log q(z|y,t,x) ]

Everything is computed per row and then summed to a scalar. Sign conventions
match the pyro implementation: the returned value is the loss to **minimize**.
"""

from __future__ import annotations

import jax


@jax.jit
def cevae_loss(model, guide, key, x, t, y):
    """CEVAE training loss (to minimize).

    Args:
        model: :class:`~causalml.inference.jax.cevae.modeling.Model` instance.
        guide: :class:`~causalml.inference.jax.cevae.modeling.Guide` instance.
        key: PRNG key used to sample ``z ~ q(z | y, t, x)``.
        x: Whitened features, shape ``(batch, feature_dim)``.
        t: Binary treatments, shape ``(batch,)``.
        y: Observed outcomes, shape ``(batch,)``.

    Returns:
        Scalar loss (sum over the batch), suitable for :func:`jax.grad`.
    """
    # Sample z from q(z | y, t, x) using the reparametrization trick.
    z, log_q_z = guide.sample_z(y, t, x, key)

    # Generative log-probs under the model.
    log_p_z = model.log_prob_z(z)
    log_p_x = model.log_prob_x(x, z)
    log_p_t = model.log_prob_t(t, z)
    log_p_y = model.log_prob_y(y, t, z)

    elbo = log_p_x + log_p_t + log_p_y + log_p_z - log_q_z

    # Auxiliary CEVAE terms: teach the guide to predict t and y from x.
    log_q_t = guide.log_prob_t(t, x)
    log_q_y = guide.log_prob_y(y, t, x)

    per_sample = elbo + log_q_t + log_q_y
    return -per_sample.sum()


# Public alias mirroring the pyro class name for discoverability.
trace_causal_effect_elbo = cevae_loss


__all__ = ["cevae_loss", "trace_causal_effect_elbo"]
