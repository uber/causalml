"""Loaders for the standard causal-inference benchmark datasets.

Each dataset is downloaded from its original source at first use and cached; none
is redistributed with CausalML. ``docs/datasets.rst`` records where each comes
from, on what terms, and which of the benchmarks are not shipped here.

Which metrics apply depends on how the data was made:

- ``fetch_ihdp`` and ``fetch_twins`` carry per-unit ground truth, so
  :func:`causalml.metrics.pehe` and :func:`causalml.metrics.ate_error` are defined.
- ``fetch_lalonde`` is a randomized experiment with no per-unit ground truth; its
  experimental ATE is the yardstick, so only :func:`causalml.metrics.ate_error`
  against that number applies.
"""

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from ._base import fetch_remote

# Pinned to a commit rather than a branch, so the digest below stays valid.
LALONDE_URL = (
    "https://raw.githubusercontent.com/NickCH-K/causaldata/"
    "94e0c05b394597b65fd7b3722b3fbd62f7a6c113/Python/causaldata/nsw_mixtape/"
    "nsw_mixtape.dta"
)
LALONDE_SHA256 = "e4a64e4436c2c178f47d6c82a371d20f1596b82b44862ce24bf13c71ac797339"

IHDP_URLS = {
    "train": "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz",
    "test": "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz",
}
IHDP_SHA256 = {
    "train": "750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240",
    "test": "a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c",
}
IHDP_N_REPLICATIONS = 100

TWINS_URL = (
    "https://bitbucket.org/mvdschaar/mlforhealthlabpub/raw/"
    "0b0190bcd38a76c405c805f1ca774971fcd85233/data/twins/Twin_Data.csv.gz"
)
TWINS_SHA256 = "0128e5dfdeb468e85fd2818330beff49d4d433b7f7d56d3eb678f87ab15b00a9"
# The outcome columns hold days survived, with 9999 standing for "did not die
# within the year". Read as a number the column means nothing; read as this
# indicator it reproduces the 17.7% mortality Louizos et al. (2017) report.
TWINS_SURVIVED = 9999


def _return(bunch: Bunch, return_X_y_t: bool):
    """Return the Bunch, or the (X, y, treatment) triple callers usually want."""
    if return_X_y_t:
        return bunch.data, bunch.target, bunch.treatment
    return bunch


def fetch_lalonde(data_home=None, download_if_missing=True, return_X_y_t=False):
    """Load the LaLonde National Supported Work experiment.

    A randomized job-training experiment: 445 rows, 185 treated, outcome ``re78``
    (1978 earnings in dollars). Its experimental estimate is the yardstick
    observational estimators are judged against (LaLonde 1986; Dehejia and Wahba
    1999). The sample here is the Dehejia-Wahba one, taken from the MIT-licensed
    ``causaldata`` package at a pinned revision.

    There is no per-unit ground truth, so PEHE is not defined on it. The
    experimental difference in means, about 1794 dollars, is what an estimate is
    compared against.

    Args:
        data_home (str or Path, optional): cache directory
        download_if_missing (bool): if False, raise rather than download
        return_X_y_t (bool): return ``(X, y, treatment)`` instead of a Bunch

    Returns:
        sklearn.utils.Bunch with ``data``, ``target``, ``treatment``,
        ``feature_names`` and ``DESCR``; or the triple if ``return_X_y_t``
    """
    path = fetch_remote(
        url=LALONDE_URL,
        filename="lalonde_nsw_mixtape.dta",
        sha256=LALONDE_SHA256,
        data_home=data_home,
        download_if_missing=download_if_missing,
        dataset_name="LaLonde NSW",
    )
    df = pd.read_stata(path)
    feature_names = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]

    return _return(
        Bunch(
            data=df[feature_names].to_numpy(dtype=float),
            target=df["re78"].to_numpy(dtype=float),
            treatment=df["treat"].to_numpy(dtype=int),
            feature_names=feature_names,
            frame=df,
            DESCR=fetch_lalonde.__doc__,
        ),
        return_X_y_t,
    )


def fetch_ihdp(
    replication=0,
    split="train",
    data_home=None,
    download_if_missing=True,
    return_X_y_t=False,
):
    """Load one replication of the IHDP benchmark.

    Covariates from the Infant Health and Development Program randomized trial,
    with outcomes simulated on response surface B (Hill 2011). The file holds 100
    replications of the same 747-unit sample: each one simulates its own outcomes
    **and draws its own 672 / 75 train-test split**, so row ``i`` is a different
    unit in each replication and the number of treated rows varies with it. Only
    the pooled 747 covariate rows are common to all of them.

    Results on IHDP are reported as a mean and standard error **across**
    replications, so ``replication`` selects one and the caller loops. A single
    replication is not comparable to a published IHDP number.

    Both potential outcome surfaces are returned, so the individual effect
    ``tau = mu1 - mu0`` is known and PEHE is defined.

    Args:
        replication (int): which replication to load, 0 to 99
        split (str): ``"train"`` (672 rows) or ``"test"`` (75 rows)
        data_home (str or Path, optional): cache directory
        download_if_missing (bool): if False, raise rather than download
        return_X_y_t (bool): return ``(X, y, treatment)`` instead of a Bunch

    Returns:
        sklearn.utils.Bunch with ``data``, ``target`` (the factual outcome),
        ``treatment``, ``tau``, ``mu0``, ``mu1``, ``y_cf``, ``feature_names`` and
        ``DESCR``; or the triple if ``return_X_y_t``

    Raises:
        ValueError: if ``split`` is not train/test or ``replication`` is out of range
    """
    if split not in IHDP_URLS:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if not 0 <= replication < IHDP_N_REPLICATIONS:
        raise ValueError(
            f"replication must be in [0, {IHDP_N_REPLICATIONS}), got {replication!r}"
        )

    path = fetch_remote(
        url=IHDP_URLS[split],
        filename=f"ihdp_npci_1-100.{split}.npz",
        sha256=IHDP_SHA256[split],
        data_home=data_home,
        download_if_missing=download_if_missing,
        dataset_name=f"IHDP ({split})",
    )
    with np.load(path) as npz:
        # every array indexes the replication on its last axis
        x = npz["x"][:, :, replication]
        t = npz["t"][:, replication]
        yf = npz["yf"][:, replication]
        ycf = npz["ycf"][:, replication]
        mu0 = npz["mu0"][:, replication]
        mu1 = npz["mu1"][:, replication]

    return _return(
        Bunch(
            data=x.astype(float),
            target=yf.astype(float),
            treatment=t.astype(int),
            tau=(mu1 - mu0).astype(float),
            mu0=mu0.astype(float),
            mu1=mu1.astype(float),
            y_cf=ycf.astype(float),
            feature_names=[f"x{i}" for i in range(x.shape[1])],
            replication=replication,
            DESCR=fetch_ihdp.__doc__,
        ),
        return_X_y_t,
    )


def fetch_twins(
    data_home=None,
    download_if_missing=True,
    return_X_y_t=False,
    random_state=None,
):
    """Load the Twins benchmark.

    Same-sex twin births from the NBER linked birth / infant death records, 11400
    pairs and 30 covariates. Treatment is being the heavier twin and the outcome
    is one-year mortality. Because both twins are observed, **both potential
    outcomes are measured rather than simulated** — the only dataset here where
    the ground truth is not a modelling assumption. Introduced as a causal
    benchmark by Louizos et al. (2017).

    The raw outcome columns hold days survived with 9999 standing for "survived
    the year", so mortality is ``outcome < 9999``. That reproduces the 17.7%
    mortality for the lighter twin reported in the paper; reading the column as a
    number instead gives a mean near 8000 and no meaning.

    One twin per pair is revealed, which makes this an observational dataset. The
    assignment here is **randomized** (a fair coin per pair), so the design is an
    RCT with known counterfactuals. The confounded variants in the literature
    assign treatment from a covariate, and they differ between papers; both
    potential outcomes are returned so a caller can build their own and say which.

    Args:
        data_home (str or Path, optional): cache directory
        download_if_missing (bool): if False, raise rather than download
        return_X_y_t (bool): return ``(X, y, treatment)`` instead of a Bunch
        random_state (int or np.random.RandomState, optional): seeds which twin
            of each pair is revealed

    Returns:
        sklearn.utils.Bunch with ``data``, ``target`` (the revealed twin's
        mortality), ``treatment``, ``tau``, ``y0``, ``y1``, ``feature_names`` and
        ``DESCR``; or the triple if ``return_X_y_t``
    """
    path = fetch_remote(
        url=TWINS_URL,
        filename="twins.csv.gz",
        sha256=TWINS_SHA256,
        data_home=data_home,
        download_if_missing=download_if_missing,
        dataset_name="Twins",
    )
    df = pd.read_csv(path)
    # the shipped header quotes every name, and the outcome columns close with a
    # typographic apostrophe rather than the one they open with
    df.columns = [c.strip().strip("'’") for c in df.columns]

    y0 = (df["outcome(t=0)"].to_numpy() < TWINS_SURVIVED).astype(int)
    y1 = (df["outcome(t=1)"].to_numpy() < TWINS_SURVIVED).astype(int)
    feature_names = [c for c in df.columns if not c.startswith("outcome(")]

    rng = np.random.RandomState(random_state) if random_state is not None else np.random
    treatment = rng.binomial(1, 0.5, size=len(df)).astype(int)

    return _return(
        Bunch(
            data=df[feature_names].to_numpy(dtype=float),
            target=np.where(treatment == 1, y1, y0),
            treatment=treatment,
            tau=(y1 - y0).astype(float),
            y0=y0,
            y1=y1,
            feature_names=feature_names,
            DESCR=fetch_twins.__doc__,
        ),
        return_X_y_t,
    )
