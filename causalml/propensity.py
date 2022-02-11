from abc import ABCMeta, abstractmethod
import logging
import numpy as np
from pygam import LogisticGAM, s
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb


logger = logging.getLogger("causalml")


class PropensityModel(metaclass=ABCMeta):
    def __init__(self, clip_bounds=(1e-3, 1 - 1e-3), **model_kwargs):
        """
        Args:
            clip_bounds (tuple): lower and upper bounds for clipping propensity scores. Bounds should be implemented
                    such that: 0 < lower < upper < 1, to avoid division by zero in BaseRLearner.fit_predict() step.
            model_kwargs: Keyword arguments to be passed to the underlying classification model.
        """
        self.clip_bounds = clip_bounds
        self.model_kwargs = model_kwargs
        self.model = self._model

    @property
    @abstractmethod
    def _model(self):
        pass

    def __repr__(self):
        return self.model.__repr__()

    def fit(self, X, y):
        """
        Fit a propensity model.

        Args:
            X (numpy.ndarray): a feature matrix
            y (numpy.ndarray): a binary target vector
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict propensity scores.

        Args:
            X (numpy.ndarray): a feature matrix

        Returns:
            (numpy.ndarray): Propensity scores between 0 and 1.
        """
        return np.clip(self.model.predict_proba(X)[:, 1], *self.clip_bounds)

    def fit_predict(self, X, y):
        """
        Fit a propensity model and predict propensity scores.

        Args:
            X (numpy.ndarray): a feature matrix
            y (numpy.ndarray): a binary target vector

        Returns:
            (numpy.ndarray): Propensity scores between 0 and 1.
        """
        self.fit(X, y)
        propensity_scores = self.predict(X)
        logger.info("AUC score: {:.6f}".format(auc(y, propensity_scores)))
        return propensity_scores


class LogisticRegressionPropensityModel(PropensityModel):
    """
    Propensity regression model based on the LogisticRegression algorithm.
    """

    @property
    def _model(self):
        kwargs = {
            "penalty": "elasticnet",
            "solver": "saga",
            "Cs": np.logspace(1e-3, 1 - 1e-3, 4),
            "l1_ratios": np.linspace(1e-3, 1 - 1e-3, 4),
            "cv": StratifiedKFold(
                n_splits=self.model_kwargs.pop("n_fold")
                if "n_fold" in self.model_kwargs
                else 4,
                shuffle=True,
                random_state=self.model_kwargs.get("random_state", 42),
            ),
            "random_state": 42,
        }
        kwargs.update(self.model_kwargs)

        return LogisticRegressionCV(**kwargs)


class ElasticNetPropensityModel(LogisticRegressionPropensityModel):
    pass


class GradientBoostedPropensityModel(PropensityModel):
    """
    Gradient boosted propensity score model with optional early stopping.

    Notes
    -----
    Please see the xgboost documentation for more information on gradient boosting tuning parameters:
    https://xgboost.readthedocs.io/en/latest/python/python_api.html
    """

    def __init__(self, early_stop=False, clip_bounds=(1e-3, 1 - 1e-3), **model_kwargs):
        super(GradientBoostedPropensityModel, self).__init__(
            clip_bounds, **model_kwargs
        )
        self.early_stop = early_stop

    @property
    def _model(self):
        kwargs = {
            "max_depth": 8,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "objective": "binary:logistic",
            "nthread": -1,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        kwargs.update(self.model_kwargs)

        return xgb.XGBClassifier(**kwargs)

    def fit(self, X, y, early_stopping_rounds=10, stop_val_size=0.2):
        """
        Fit a propensity model.

        Args:
            X (numpy.ndarray): a feature matrix
            y (numpy.ndarray): a binary target vector
        """

        if self.early_stop:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=stop_val_size
            )

            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
            )
        else:
            super(GradientBoostedPropensityModel, self).fit(X, y)

    def predict(self, X):
        """
        Predict propensity scores.

        Args:
            X (numpy.ndarray): a feature matrix

        Returns:
            (numpy.ndarray): Propensity scores between 0 and 1.
        """
        if self.early_stop:
            return np.clip(
                self.model.predict_proba(X, ntree_limit=self.model.best_ntree_limit)[
                    :, 1
                ],
                *self.clip_bounds
            )
        else:
            return super(GradientBoostedPropensityModel, self).predict(X)


def calibrate(ps, treatment):
    """Calibrate propensity scores with logistic GAM.

    Ref: https://pygam.readthedocs.io/en/latest/api/logisticgam.html

    Args:
        ps (numpy.array): a propensity score vector
        treatment (numpy.array): a binary treatment vector (0: control, 1: treated)

    Returns:
        (numpy.array): a calibrated propensity score vector
    """

    gam = LogisticGAM(s(0)).fit(ps, treatment)

    return gam.predict_proba(ps)


def compute_propensity_score(
    X, treatment, p_model=None, X_pred=None, treatment_pred=None, calibrate_p=True
):
    """Generate propensity score if user didn't provide

    Args:
        X (np.matrix): features for training
        treatment (np.array or pd.Series): a treatment vector for training
        p_model (propensity model object, optional):
            ElasticNetPropensityModel (default) / GradientBoostedPropensityModel
        X_pred (np.matrix, optional): features for prediction
        treatment_pred (np.array or pd.Series, optional): a treatment vector for prediciton
        calibrate_p (bool, optional): whether calibrate the propensity score

    Returns:
        (tuple)
            - p (numpy.ndarray): propensity score
            - p_model (PropensityModel): a trained PropensityModel object
    """
    if treatment_pred is None:
        treatment_pred = treatment.copy()
    if p_model is None:
        p_model = ElasticNetPropensityModel()

    p_model.fit(X, treatment)

    if X_pred is None:
        p = p_model.predict(X)
    else:
        p = p_model.predict(X_pred)

    if calibrate_p:
        logger.info("Calibrating propensity scores.")
        p = calibrate(p, treatment_pred)

    # force the p values within the range
    eps = np.finfo(float).eps
    p = np.where(p < 0 + eps, 0 + eps * 1.001, p)
    p = np.where(p > 1 - eps, 1 - eps * 1.001, p)

    return p, p_model
