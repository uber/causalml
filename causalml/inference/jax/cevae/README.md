# CEVAE (Causal Effect Variational Autoencoder) — JAX

Pure JAX / [Flax NNX](https://flax.readthedocs.io/en/latest/nnx/index.html)
implementation of the Causal Effect Variational Autoencoder [1], porting the
`pyro.contrib.cevae` reference used by `causalml.inference.torch.CEVAE`.

Generative model with latent confounder `z` and binary treatment `t`:

```
z ~ p(z)      # latent confounder
x ~ p(x|z)    # partial noisy observation of z
t ~ p(t|z)    # treatment, whose application is biased by z
y ~ p(y|t,z)  # outcome
```

The `y` distribution is defined by a disjoint pair of neural networks defining
`p(y|t=0,z)` and `p(y|t=1,z)`, allowing highly imbalanced treatment.

## Files

| file | contents |
| --- | --- |
| `modeling.py` | `Model`, `Guide`, distribution nets (`BernoulliNet`, `NormalNet`, `StudentTNet`, ...), `PreWhitener` — all `flax.nnx` modules |
| `losses.py`   | `cevae_loss` — CEVAE ELBO objective (`ELBO + log q(t|x) + log q(y|t,x)`) |
| `cevae.py`    | scikit-learn style `CEVAE` wrapper (`fit`, `predict`, `fit_predict`, `save`, `load`) |

## Usage

```python
from causalml.inference.jax.cevae import CEVAE

cevae = CEVAE(outcome_dist="normal", num_epochs=50)
ite = cevae.fit_predict(X, treatment, y)
ate = ite.mean()
```

The API matches `causalml.inference.torch.CEVAE`; see
`docs/examples/cevae_example.ipynb` for a benchmark against the pyro
implementation on IHDP + synthetic data. For a side-by-side benchmark of the
JAX and PyTorch backends, see `docs/examples/cevae_jax_vs_torch.ipynb`.

## References

1. C. Louizos, U. Shalit, J. Mooij, D. Sontag, R. Zemel, M. Welling (2017).
   *Causal Effect Inference with Deep Latent-Variable Models.*
   [pdf](http://papers.nips.cc/paper/7223-causal-effect-inference-with-deep-latent-variable-models.pdf)
