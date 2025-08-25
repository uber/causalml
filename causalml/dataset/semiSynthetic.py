# Synthetic Validation Dataset Generator according to the paper: "Synth-Validation: Selecting the Best Causal Inference Method for a Given Dataset"
# https://arxiv.org/pdf/1711.00083

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Callable, List, Optional, Union
from numpy.typing import ArrayLike
import multiprocessing as mp
from functools import partial
from sklearn.linear_model import LinearRegression
from causalml.inference.meta import BaseXRegressor, BaseTRegressor
from scipy.special import expit


class SemiSynthDataGenerator:
    def __init__(
        self,
        Q: int = 5,
        gamma: float = 2.0,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        B: int = 5,
        maxdepths: List[int] = [1, 2, 3],
        lambdas: List[float] = np.logspace(-5, 1, num=5).tolist(),
        M: int = 30,
        early_stopping_rounds: int = 3,
        verbose: bool = False,
        **kwargs,
    ):
        self.Q = Q
        self.gamma = gamma
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.B = B
        self.maxdepths = maxdepths
        self.lambdas = lambdas
        self.M = M
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.kwargs = kwargs

    def fit(
        self,
        X: pd.DataFrame,
        w: pd.Series,
        y: pd.Series,
        initial_taus: Optional[List[float]] = None,
    ):
        self.X = X
        self.y = y
        self.w = w
        np.random.seed(42)
        if initial_taus is None:
            # raw_tau
            raw_tau = y[w == 1].mean() - y[w == 0].mean()
            # lm_tau
            X_lm = pd.concat([w, X], axis=1)
            lm = LinearRegression().fit(X_lm, y)
            lm_tau = lm.coef_[0]
            # x_learner_tau
            x_learner = BaseXRegressor(DecisionTreeRegressor())
            x_learner_tau = x_learner.estimate_ate(X=X, treatment=w, y=y)[0]
            # t_learner_tau
            t_learner = BaseTRegressor(RandomForestRegressor())
            t_learner_tau = t_learner.estimate_ate(X=X, treatment=w, y=y)[0]
            initial_taus = [
                float(raw_tau),
                float(lm_tau),
                float(x_learner_tau),
                float(t_learner_tau),
            ]
        else:
            initial_taus = [float(t) for t in initial_taus]
        initial_taus_arr = np.array(initial_taus, dtype=float)
        initial_taus_range = initial_taus_arr.max() - initial_taus_arr.min()
        initial_taus_median = np.median(initial_taus_arr)
        taus = np.linspace(
            initial_taus_median - self.gamma * initial_taus_range,
            initial_taus_median + self.gamma * initial_taus_range,
            self.Q,
        )
        self.taus = taus
        self.dgps = []
        for real_tau in taus:
            self.dgps.append(
                miu_cv(
                    y=np.asarray(self.y),
                    w=np.asarray(self.w),
                    X=self.X,
                    real_tau=real_tau,
                    train_frac=self.train_frac,
                    val_frac=self.val_frac,
                    B=self.B,
                    max_depths=self.maxdepths,
                    lambdas=self.lambdas,
                    M=self.M,
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=self.verbose,
                    **self.kwargs,
                )
            )

    def generate(self, K: int = 10, n=None) -> List[List[pd.DataFrame]]:
        if n is None:
            n = len(self.X)
        if all((self.y == 0) | (self.y == 1)):
            binary_y = True
        else:
            binary_y = False
        ctrl_idx = np.where(self.w == 0)[0]
        trt_idx = np.where(self.w == 1)[0]
        ctrl_n = int(n * (len(ctrl_idx) / len(self.X)))
        trt_n = int(n * (len(trt_idx) / len(self.X)))
        ans = []
        for q in range(len(self.dgps)):
            datasets = []
            dgp_q = self.dgps[q]["final_model"]
            data_tau = self.X.copy()
            y0 = dgp_q[0](data_tau)
            y1 = dgp_q[1](data_tau)
            if binary_y:
                y0 = logistic(y0)
                y1 = logistic(y1)
            data_tau["w"] = self.w
            data_tau["tau_i"] = y1 - y0
            data_tau["y_w"] = np.where(self.w == 1, y1, y0)
            resid = self.y - data_tau["y_w"]
            for k in range(K):
                rng = np.random.default_rng(seed=k)
                ctrl_idx_qk = rng.choice(ctrl_idx, size=ctrl_n, replace=True)
                trt_idx_qk = rng.choice(trt_idx, size=trt_n, replace=True)
                idx = np.concatenate([ctrl_idx_qk, trt_idx_qk])
                data_qk = data_tau.iloc[idx].copy()
                if not binary_y:
                    data_qk["y"] = data_qk["y_w"] + rng.choice(
                        resid, size=len(data_qk), replace=True
                    )  # aka observed y
                else:
                    data_qk["y"] = data_qk["y_w"].apply(lambda x: rng.binomial(1, x))
                data_qk = data_qk[["y", "w", "tau_i"] + list(self.X)]
                datasets.append(data_qk)
            ans.append(datasets)
        return ans


def continuous_objective(x, Q, a, d):
    """
    Compute the continuous objective function for quadratic optimization.

    Parameters:
    -----------
    x : np.ndarray
        The variable vector to optimize over.
    Q : np.ndarray
        The quadratic coefficient matrix.
    a : np.ndarray
        The linear coefficient vector.
    d : float
        The constant term.

    Returns:
    --------
    float
        The value of the objective function: x^T Q x + a^T x + d
    """
    return np.dot(x, Q @ x) + np.dot(a, x) + d


def deviance(y, pred):
    """
    Compute the binomial deviance loss function.

    Parameters:
    -----------
    y : np.ndarray
        True binary outcomes (0 or 1).
    pred : np.ndarray
        Predicted logits.

    Returns:
    --------
    float
        The binomial deviance loss: -2 * mean(y * pred - log(1 + exp(pred)))
    """
    return -2.0 * np.mean((y * pred) - np.logaddexp(0.0, pred))


def logit(x):
    """
    Compute the logit (log-odds) transformation.

    Parameters:
    -----------
    x : np.ndarray
        Input values between 0 and 1.

    Returns:
    --------
    np.ndarray
        Logit-transformed values: log(x / (1 - x))
    """
    return np.log(x / (1 - x))


def logistic(x):
    """
    Compute the logistic (sigmoid) transformation.

    Parameters:
    -----------
    x : np.ndarray
        Input values (can be any real number).

    Returns:
    --------
    np.ndarray
        Logistic-transformed values: 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-x))


def binary_objective(x, w, y):
    """
    Compute the binary objective function for treatment effect estimation.

    Parameters:
    -----------
    x : np.ndarray
        Parameter vector [x0, x1] where x0 is for control group, x1 for treatment group.
    w : np.ndarray
        Treatment assignment vector (0 for control, 1 for treatment).
    y : np.ndarray
        Binary outcome vector.

    Returns:
    --------
    float
        The binary deviance loss for the given parameters.
    """
    pred = np.where(w == 0, x[0], x[1])
    return deviance(y, pred)


def negative_gradient(y, pred):
    """
    Compute the negative gradient for binary outcomes.

    Parameters:
    -----------
    y : np.ndarray
        True binary outcomes (0 or 1).
    pred : np.ndarray
        Predicted logits.

    Returns:
    --------
    np.ndarray
        The negative gradient: y - losgistic_sigmoid(pred)
    """
    return y - expit(pred.ravel())


def miu_m(
    y: ArrayLike,
    w: ArrayLike,
    X: Union[pd.DataFrame, ArrayLike],
    real_tau: Optional[float] = None,
    miu_m_minus_1: Optional[List[Callable]] = None,
    val_y: Optional[ArrayLike] = None,
    val_w: Optional[ArrayLike] = None,
    val_X: Optional[Union[pd.DataFrame, ArrayLike]] = None,
    max_depth: Union[int, float] = 3,
    lambda_: float = 0.0,
    **tree_args,
) -> List[Callable]:
    """
    Build the m-th iteration of the MIU (Model-based Imputation with Uncertainty) ensemble.

    This function implements a single iteration of the MIU algorithm, which builds
    treatment-specific models while maintaining a constraint on the treatment effect.

    Parameters:
    -----------
    y : ArrayLike
        Outcome array. Can be continuous or binary (0/1). Will be converted to np.ndarray.
    w : ArrayLike
        Treatment assignment array (0 for control, 1 for treatment). Will be converted to np.ndarray.
    X : Union[pd.DataFrame, ArrayLike]
        Covariate matrix for training the models. Will be converted to pd.DataFrame.
    real_tau : Optional[float], default=None
        The true treatment effect to constrain the model. Required for m=1.
    miu_m_minus_1 : Optional[List[Callable]], default=None
        List of two functions [miu_0, miu_1] from the previous iteration.
        If None, this is the first iteration (m=1).
    val_y : Optional[ArrayLike], default=None
        Validation outcome array. Used for constraint calculation if provided.
    val_w : Optional[ArrayLike], default=None
        Validation treatment assignment array.
    val_X : Optional[Union[pd.DataFrame, ArrayLike]], default=None
        Validation covariate matrix. Used for constraint calculation if provided.
    max_depth : Union[int, float], default=3
        Maximum depth of the decision trees used in this iteration.
    lambda_ : float, default=0.0
        L2 regularization parameter for the leaf values.
    **tree_args
        Additional arguments passed to DecisionTreeRegressor.

    Returns:
    --------
    List[Callable]
        List containing two functions [miu_0_m, miu_1_m]:
        - miu_0_m: Function that predicts outcomes for control group (w=0)
        - miu_1_m: Function that predicts outcomes for treatment group (w=1)

    Notes:
    ------
    - For m=1, the function fits simple constant models with treatment effect constraint
    - For m>1, the function fits regression trees to residuals from previous iteration
    - The treatment effect constraint ensures honest estimation of treatment effects
    - Binary outcomes use logistic regression, continuous outcomes use linear regression
    """
    # Convert inputs to appropriate types
    y = np.asarray(y)
    w = np.asarray(w)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if val_y is not None:
        val_y = np.asarray(val_y)
    if val_w is not None:
        val_w = np.asarray(val_w)
    if val_X is not None and not isinstance(val_X, pd.DataFrame):
        val_X = pd.DataFrame(val_X)

    if all((y == 0) | (y == 1)):
        binary_y = True
    else:
        binary_y = False
    if miu_m_minus_1 is None:
        # m == 1
        if real_tau is None:
            raise ValueError("For m=1 (first call to miu_m) real_tau must be supplied")
        x0 = np.zeros(2)
        if not binary_y:
            n0 = (w == 0).sum()
            n1 = (w == 1).sum()
            Q = np.array([[n0, 0], [0, n1]])
            a = np.array(
                [
                    -2 * y[w == 0].sum(),
                    -2 * y[w == 1].sum(),
                ]
            )
            d = (y**2).sum()

            constraints = {"type": "eq", "fun": lambda x: x[1] - x[0] - real_tau}
            res = minimize(
                fun=continuous_objective,
                x0=x0,
                args=(Q, a, d),
                constraints=constraints,
                method="SLSQP",
            )
        else:
            constraints = {
                "type": "eq",
                "fun": lambda x: logistic(x[1]) - logistic(x[0]) - real_tau,
            }
            res = minimize(
                fun=binary_objective,
                x0=x0,
                args=(w, y),
                constraints=constraints,
                method="SLSQP",
            )

        res01, res11 = res.x[0], res.x[1]

        def miu_01(x):
            return np.repeat(res01, len(x))

        def miu_11(x):
            return np.repeat(res11, len(x))

        return [miu_01, miu_11]
    else:
        # m > 1
        miu_0_m_minus_1, miu_1_m_minus_1 = miu_m_minus_1
        # Predict y_hat using previous miu functions
        y_1 = miu_1_m_minus_1(X)
        y_0 = miu_0_m_minus_1(X)
        y_hat = np.where(w == 1, y_1, y_0)
        if not binary_y:
            resid = y - y_hat
        else:
            resid = negative_gradient(y, y_hat)
        treat = w == 1
        # Fit regression trees to residuals
        b_0m = DecisionTreeRegressor(max_depth=max_depth, **tree_args, random_state=42)
        b_0m.fit(X.loc[~treat], resid[~treat])
        b_1m = DecisionTreeRegressor(max_depth=max_depth, **tree_args, random_state=42)
        b_1m.fit(X.loc[treat], resid[treat])
        # Predict leaf node for each sample
        R0 = b_0m.apply(X)
        R1 = b_1m.apply(X)
        if not binary_y:
            # Group sizes and residuals
            resid0 = (
                pd.Series(resid)
                .groupby(R0)
                .agg(["count", "sum"])
                .reset_index()
                .rename(columns={"index": "leaf_node"})
            )
            resid1 = (
                pd.Series(resid)
                .groupby(R1)
                .agg(["count", "sum"])
                .reset_index()
                .rename(columns={"index": "leaf_node"})
            )
            num_params = len(resid0) + len(resid1)
            Q = np.diag(
                np.concatenate([resid0["count"].to_numpy(), resid1["count"].to_numpy()])
                * lambda_
            )
            a = -2 * np.concatenate(
                [resid0["sum"].to_numpy(), resid1["sum"].to_numpy()]
            )
            d = (resid**2).sum()
            if val_X is not None and val_y is not None and val_w is not None:
                # Optionally add validation data
                X_full = pd.concat([X, val_X], ignore_index=True, axis=0)
                R0 = b_0m.apply(X_full)
                R1 = b_1m.apply(X_full)
                # Making the constraint apply over the entire dataset - this is still honest
                resid0 = (
                    pd.Series(R0)
                    .groupby(R0)
                    .agg(["count"])
                    .reset_index()
                    .rename(columns={"index": "leaf_node"})
                )
                resid1 = (
                    pd.Series(R1)
                    .groupby(R1)
                    .agg(["count"])
                    .reset_index()
                    .rename(columns={"index": "leaf_node"})
                )

            constraints = {
                "type": "eq",
                "fun": lambda x: np.dot(resid1["count"].to_numpy(), x[len(resid0) :])
                - np.dot(resid0["count"].to_numpy(), x[: len(resid0)]),
            }
            x0 = np.zeros(num_params)
            res = minimize(
                continuous_objective,
                x0,
                args=(Q, a, d),
                constraints=constraints,
                method="SLSQP",
            )
        else:
            resid0 = pd.DataFrame({"leaf_node": np.unique(R0)})
            resid1 = pd.DataFrame({"leaf_node": np.unique(R1)})
            x0 = np.zeros(len(resid0) + len(resid1))

            def binary_objective_m(x, y, w, R0, R1, lambda_):
                loss = []
                r0 = np.unique(R0)
                r1 = np.unique(R1)
                ctrl_nodes = len(r0)
                for i in range(len(x)):
                    if i < ctrl_nodes - 1:
                        idx = (R0 == r0[i]) & (w == 0)
                    else:
                        idx = (R1 == r1[i - ctrl_nodes]) & (w == 1)
                    pred = np.full(sum(idx), x[i])
                    loss.append(deviance(y[idx], pred) + sum(idx) * lambda_ * x[i] ** 2)
                return np.array(loss).sum()

            if val_X is not None and val_y is not None and val_w is not None:
                # Optionally add validation data
                X_full = pd.concat([X, val_X], ignore_index=True, axis=0)
                R0_constraint = b_0m.apply(X_full)
                R1_constraint = b_1m.apply(X_full)
                prev0 = miu_0_m_minus_1(X_full)
                prev1 = miu_1_m_minus_1(X_full)
                # Making the constraint apply over the entire dataset - this is still honest
            else:
                R0_constraint = R0
                R1_constraint = R1
                prev0 = y_0
                prev1 = y_1

            real_tau = (logistic(prev1) - logistic(prev0)).mean()

            def con_m(x, R0_constraint, R1_constraint, prev0, prev1):
                group_sum = []
                r0 = np.unique(R0_constraint)
                r1 = np.unique(R1_constraint)
                ctrl_nodes = len(r0)
                for i in range(len(x)):
                    if i < ctrl_nodes:
                        idx = R0_constraint == r0[i]
                        group_sum.append(
                            (logistic(prev0[idx] + x[i])).sum()
                            / len(R0_constraint)
                            * -1
                        )
                    else:
                        idx = R1_constraint == r1[i - ctrl_nodes]
                        group_sum.append(
                            (logistic(prev1[idx] + x[i])).sum() / len(R0_constraint)
                        )
                return np.array(group_sum).sum() - real_tau

            constraints = {
                "type": "eq",
                "fun": lambda x: con_m(x, R0_constraint, R1_constraint, prev0, prev1),
            }

            res = minimize(
                fun=binary_objective_m,
                x0=x0,
                args=(y, w, R0, R1, lambda_),
                constraints=constraints,
                method="SLSQP",
            )

        # Assign fitted values to leaves
        resid0["leaf_value"] = res.x[: len(resid0)]
        resid1["leaf_value"] = res.x[len(resid0) :]

        def miu_0m(x):
            prev = miu_0_m_minus_1(x)
            leaves = pd.DataFrame({"leaf_node": b_0m.apply(x)})
            return (
                prev
                + leaves.merge(resid0, on="leaf_node", how="left")[
                    "leaf_value"
                ].to_numpy()
            )

        def miu_1m(x):
            prev = miu_1_m_minus_1(x)
            leaves = pd.DataFrame({"leaf_node": b_1m.apply(x)})
            return (
                prev
                + leaves.merge(resid1, on="leaf_node", how="left")[
                    "leaf_value"
                ].to_numpy()
            )

        return [miu_0m, miu_1m]


def miu(
    y: ArrayLike,
    w: ArrayLike,
    X: Union[pd.DataFrame, ArrayLike],
    real_tau: float,
    val_y: Optional[ArrayLike] = None,
    val_w: Optional[ArrayLike] = None,
    val_X: Optional[Union[pd.DataFrame, ArrayLike]] = None,
    max_depth: Union[int, float] = 3,
    lambda_: float = 0.0,
    M: int = 10,
    early_stopping_rounds: Union[int, float] = float("inf"),
    verbose: bool = False,
    **tree_args,
) -> dict:
    """
    Train an ensemble of M MIU models and return the best one.

    This function implements the complete MIU (Model-based Imputation with Uncertainty)
    algorithm, which builds an ensemble of treatment-specific models while maintaining
    constraints on the treatment effect for honest estimation.

    Parameters:
    -----------
    y : ArrayLike
        Outcome array. Can be continuous or binary (0/1). Will be converted to np.ndarray.
    w : ArrayLike
        Treatment assignment array (0 for control, 1 for treatment). Will be converted to np.ndarray.
    X : Union[pd.DataFrame, ArrayLike]
        Covariate matrix for training the models. Will be converted to pd.DataFrame.
    real_tau : float
        The true treatment effect to constrain the model. This is used to ensure
        honest estimation of treatment effects.
    val_y : Optional[ArrayLike], default=None
        Validation outcome array. Used for early stopping and model selection.
    val_w : Optional[ArrayLike], default=None
        Validation treatment assignment array.
    val_X : Optional[Union[pd.DataFrame, ArrayLike]], default=None
        Validation covariate matrix. Used for early stopping and model selection.
    max_depth : Union[int, float], default=3
        Maximum depth of the decision trees used in each iteration.
    lambda_ : float, default=0.0
        L2 regularization parameter for the leaf values in each iteration.
    M : int, default=10
        Maximum number of ensemble iterations to perform.
    early_stopping_rounds : Union[int, float], default=float('inf')
        Number of rounds without improvement before stopping early.
        If val_X is None, this must be float('inf').
    verbose : bool, default=False
        Whether to print progress information during training.
    **tree_args
        Additional arguments passed to DecisionTreeRegressor in each iteration.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'best_model': List[Callable] - The best ensemble model [miu_0, miu_1]
        - 'loss': np.ndarray - Array of validation losses for each iteration
        - 'best_model_m': int - The iteration number of the best model

    Notes:
    ------
    - The algorithm builds an ensemble by iteratively fitting models to residuals
    - Each iteration maintains the treatment effect constraint using real_tau
    - Early stopping is based on validation loss if validation data is provided
    - The best model is selected based on validation loss or training loss
    - Binary outcomes use logistic regression, continuous outcomes use linear regression
    """
    if val_X is None and not np.isinf(early_stopping_rounds):
        raise ValueError("If val_X is None then early_stopping_rounds must be Inf")

    # Convert inputs to appropriate types
    y = np.asarray(y)
    w = np.asarray(w)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if val_y is not None:
        val_y = np.asarray(val_y)
    if val_w is not None:
        val_w = np.asarray(val_w)
    if val_X is not None and not isinstance(val_X, pd.DataFrame):
        val_X = pd.DataFrame(val_X)

    if all((y == 0) | (y == 1)):
        binary_y = True
    else:
        binary_y = False

    loss = np.full(M, np.nan)
    best_model_ind = 0
    best_model = None

    for i in range(M):
        if i == 0:
            ans = miu_m(
                y=y,
                w=w,
                X=X,
                real_tau=real_tau,
                val_X=val_X,
                val_y=val_y,
                val_w=val_w,
                max_depth=max_depth,
                lambda_=lambda_,
                **tree_args,
            )
            best_model = ans
        else:
            ans = miu_m(
                y=y,
                w=w,
                X=X,
                miu_m_minus_1=ans,
                val_X=val_X,
                val_y=val_y,
                val_w=val_w,
                max_depth=max_depth,
                lambda_=lambda_,
                **tree_args,
            )

        if val_X is None:
            # Use training data for loss calculation
            pred = np.where(w == 1, ans[1](X), ans[0](X))
            if not binary_y:
                loss[i] = np.mean((y - pred) ** 2)
            else:
                loss[i] = deviance(y, pred)
        else:
            # Use validation data for loss calculation
            pred = np.where(val_w == 1, ans[1](val_X), ans[0](val_X))
            if not binary_y:
                loss[i] = np.mean((val_y - pred) ** 2)
            else:
                loss[i] = deviance(val_y, pred)

        if np.nanargmin(loss) != best_model_ind:
            best_model_ind = np.nanargmin(loss)
            best_model = ans
        elif i - np.nanargmin(loss) > early_stopping_rounds:
            if verbose:
                print(
                    f"Best tree: {best_model_ind + 1}, best tree loss: {loss[best_model_ind]}"
                )
            return {
                "best_model": best_model,
                "loss": loss,
                "best_model_m": best_model_ind + 1,
            }

    if verbose:
        print(
            f"Best tree: {best_model_ind + 1}, best tree loss: {loss[best_model_ind]}"
        )
    return {"best_model": best_model, "loss": loss, "best_model_m": best_model_ind + 1}


def miu_cv(
    y: ArrayLike,
    w: ArrayLike,
    X: Union[pd.DataFrame, ArrayLike],
    real_tau: float,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    B: int = 5,
    max_depths: List[int] = [1, 3, 5],
    lambdas: List[float] = np.logspace(
        -5, 1, num=5
    ).tolist(),  # range of lambdas is like in glmnet
    M: int = 30,
    early_stopping_rounds: Union[int, float] = float("inf"),
    verbose: bool = False,
    n_jobs: int = -1,
    **tree_args,
) -> dict:
    """
    Perform cross-validation to find optimal hyperparameters for the MIU model.

    This function performs bootstrap-based cross-validation to tune the hyperparameters
    of the MIU algorithm, including max_depth and lambda regularization parameter.

    Parameters:
    -----------
    y : ArrayLike
        Outcome array. Can be continuous or binary (0/1). Will be converted to np.ndarray.
    w : ArrayLike
        Treatment assignment array (0 for control, 1 for treatment). Will be converted to np.ndarray.
    X : Union[pd.DataFrame, ArrayLike]
        Covariate matrix for training the models. Will be converted to pd.DataFrame.
    real_tau : float
        The true treatment effect to constrain the model.
    train_frac : float, default=0.8
        Fraction of data to use for training in each bootstrap iteration.
    val_frac : float, default=0.1
        Fraction of training data to use for validation. If 0, no validation is performed.
    B : int, default=5
        Number of bootstrap iterations for cross-validation.
    max_depths : List[int], default=[1, 3, 5]
        List of maximum tree depths to try during hyperparameter tuning.
    lambdas : List[float], default=np.logspace(-5, 1, 5).tolist()
        List of L2 regularization parameters to try during hyperparameter tuning.
        Range is similar to glmnet: from 1e-5 to 10.
    M : int, default=30
        Maximum number of ensemble iterations for each model.
    early_stopping_rounds : Union[int, float], default=float('inf')
        Number of rounds without improvement before stopping early.
        If val_frac is 0, this must be float('inf').
    verbose : bool, default=False
        Whether to print progress information during cross-validation.
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 means using all processors - 1.
    **tree_args
        Additional arguments passed to DecisionTreeRegressor.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'final_model': List[Callable] - The best ensemble model trained on full data
        - 'params_loss': pd.DataFrame - Cross-validation results for all parameter combinations

    Notes:
    ------
    - Uses stratified bootstrap sampling to maintain treatment group proportions
    - Performs parallel processing across parameter combinations and bootstrap iterations
    - Selects best parameters based on mean test loss across bootstrap iterations
    - Final model is trained on the full dataset using the best parameters
    - The params_loss DataFrame contains loss, r_sq, and m for each parameter combination
    """
    # Convert inputs to appropriate types
    y = np.asarray(y)
    w = np.asarray(w)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    # Create parameter grid
    param_combinations = []
    for max_depth in max_depths:
        for lambda_ in lambdas:
            param_combinations.append({"max_depth": max_depth, "lambda_": lambda_})

    params_loss = pd.DataFrame(param_combinations)
    params_loss["loss"] = np.nan
    params_loss["r_sq"] = np.nan
    params_loss["m"] = np.nan
    params_loss = params_loss.merge(pd.DataFrame({"b": range(B)}), how="cross")

    # Set number of jobs
    if n_jobs == -1:
        n_jobs = mp.cpu_count() - 1  # don't freeze the computer

    # Run rows in parallel
    if n_jobs > 1:
        with mp.Pool(processes=n_jobs) as pool:
            params_loss = pool.map(
                partial(
                    miu_row,
                    y=y,
                    w=w,
                    X=X,
                    real_tau=real_tau,
                    train_frac=train_frac,
                    val_frac=val_frac,
                    M=M,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                    **tree_args,
                ),
                [row for _, row in params_loss.iterrows()],
            )
    else:
        params_loss = [
            miu_row(
                row,
                y=y,
                w=w,
                X=X,
                real_tau=real_tau,
                train_frac=train_frac,
                val_frac=val_frac,
                M=M,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
                **tree_args,
            )
            for _, row in params_loss.iterrows()
        ]

    # Aggregate results back into params_loss DataFrame
    params_loss = pd.concat(params_loss, axis=0)
    params_loss = (
        params_loss.groupby(["max_depth", "lambda_"])
        .agg({"loss": "mean", "r_sq": "mean", "m": "mean"})
        .reset_index()
        .assign(m=lambda x: x["m"].astype(int))
    )

    # Find best parameters
    best_idx = np.argmin(params_loss["loss"])
    params_loss["best_params"] = params_loss.index == best_idx
    best_params = params_loss.iloc[best_idx]

    if verbose:
        print(
            f"Best params: max_depth - {best_params['max_depth']}, "
            f"lambda - {best_params['lambda_']}, m - {best_params['m']}"
        )

    # Train final model with best parameters
    final_model = miu(
        y=y,
        w=w,
        X=X,
        real_tau=real_tau,
        val_y=None,
        val_w=None,
        val_X=None,
        max_depth=int(best_params["max_depth"]),
        lambda_=best_params["lambda_"],
        M=int(best_params["m"]),
        early_stopping_rounds=float("inf"),
        verbose=False,
        **tree_args,
    )

    return {
        "final_model": final_model["best_model"],
        "params_loss": params_loss,
    }


def miu_row(
    row: pd.Series,
    y: ArrayLike,
    w: ArrayLike,
    X: Union[pd.DataFrame, ArrayLike],
    real_tau: float,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    M: int = 30,
    early_stopping_rounds: Union[int, float] = float("inf"),
    verbose: bool = False,
    **tree_args,
) -> pd.Series:
    """
    Train a single MIU model for a specific parameter combination and bootstrap iteration.

    This function is designed to be used in parallel processing for cross-validation.
    It trains a MIU model with specific hyperparameters on a bootstrap sample and
    evaluates it on the out-of-bag test set.

    Parameters:
    -----------
    row : pd.Series
        A pandas Series containing the parameter combination to evaluate.
        Must contain 'max_depth' and 'lambda_' keys.
    y : ArrayLike
        Outcome array. Can be continuous or binary (0/1). Will be converted to np.ndarray.
    w : ArrayLike
        Treatment assignment array (0 for control, 1 for treatment). Will be converted to np.ndarray.
    X : Union[pd.DataFrame, ArrayLike]
        Covariate matrix for training the models. Will be converted to pd.DataFrame.
    real_tau : float
        The true treatment effect to constrain the model.
    train_frac : float, default=0.8
        Fraction of data to use for training.
    val_frac : float, default=0.1
        Fraction of training data to use for validation. If 0, no validation is performed.
    M : int, default=30
        Maximum number of ensemble iterations for the model.
    early_stopping_rounds : Union[int, float], default=float('inf')
        Number of rounds without improvement before stopping early.
        If val_frac is 0, this must be float('inf').
    verbose : bool, default=False
        Whether to print progress information during training.
    **tree_args
        Additional arguments passed to DecisionTreeRegressor.

    Returns:
    --------
    pd.Series
        A pandas Series containing the original parameters plus:
        - 'm': int - The number of iterations in the best model
        - 'loss': float - The test loss (MSE for continuous, deviance for binary)
        - 'r_sq': float - The R-squared value on the test set

    Notes:
    ------
    - Performs stratified bootstrap sampling to maintain treatment group proportions
    - Uses the parameters from 'row' to train the MIU model
    - Evaluates the model on the out-of-bag test set
    - Returns results as a pandas Series for easy aggregation
    - Designed for parallel processing in cross-validation
    """
    # Convert inputs to appropriate types
    y = np.asarray(y)
    w = np.asarray(w)

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if all((y == 0) | (y == 1)):
        binary_y = True
    else:
        binary_y = False

    # Stratified split data into train/test based on treatment w
    n_samples = len(y)
    w_0_indices = np.where(w == 0)[0]
    w_1_indices = np.where(w == 1)[0]

    # Calculate split sizes for each treatment group
    train_size_0 = int(train_frac * len(w_0_indices))
    train_size_1 = int(train_frac * len(w_1_indices))

    # Randomly select train indices for each treatment group
    train_indices_0 = np.random.choice(w_0_indices, size=train_size_0, replace=False)
    train_indices_1 = np.random.choice(w_1_indices, size=train_size_1, replace=False)
    train_indices = np.concatenate([train_indices_0, train_indices_1])

    # Remaining indices go to test
    test_indices = np.setdiff1d(np.arange(n_samples), train_indices)

    if val_frac > 0:
        # Further stratified split train into train/validation
        val_size_0 = int(val_frac * len(train_indices_0))
        val_size_1 = int(val_frac * len(train_indices_1))

        val_indices_0 = np.random.choice(
            train_indices_0, size=val_size_0, replace=False
        )
        val_indices_1 = np.random.choice(
            train_indices_1, size=val_size_1, replace=False
        )
        val_indices = np.concatenate([val_indices_0, val_indices_1])

        # Remove validation indices from train
        train_indices = np.setdiff1d(train_indices, val_indices)

        val_X = X.iloc[val_indices]
        val_y = y[val_indices]
        val_w = w[val_indices]
    else:
        val_X = None
        val_y = None
        val_w = None

    train_X = X.iloc[train_indices]
    train_y = y[train_indices]
    train_w = w[train_indices]
    test_X = X.iloc[test_indices]
    test_y = y[test_indices]
    test_w = w[test_indices]
    miu_row = miu(
        y=train_y,
        w=train_w,
        X=train_X,
        real_tau=real_tau,
        val_y=val_y,
        val_w=val_w,
        val_X=val_X,
        max_depth=int(row["max_depth"]),
        lambda_=float(row["lambda_"]),
        M=M,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
        **tree_args,
    )

    # Make predictions on test set
    pred = np.where(
        test_w == 1, miu_row["best_model"][1](test_X), miu_row["best_model"][0](test_X)
    )

    # Create result dict
    row["m"] = miu_row["best_model_m"]

    if not binary_y:
        row["loss"] = np.mean((test_y - pred) ** 2)
        row["r_sq"] = 1 - row["loss"] / np.mean((test_y - np.mean(test_y)) ** 2)
    else:
        row["loss"] = deviance(test_y, pred)
        baseline_pred = np.where(
            test_w == 1,
            logit(np.mean(test_y[test_w == 1])),
            logit(np.mean(test_y[test_w == 0])),
        )
        row["r_sq"] = 1 - row["loss"] / deviance(test_y, baseline_pred)

    return row.to_frame().T
