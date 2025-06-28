from abc import ABCMeta, abstractmethod
import logging
import numpy as np
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb


logger = logging.getLogger("causalml")


class PropensityModel(metaclass=ABCMeta):
    def __init__(self, clip_bounds=(1e-3, 1 - 1e-3), calibrate=True, **model_kwargs):
        """
        Args:
            clip_bounds (tuple): lower and upper bounds for clipping propensity scores. Bounds should be implemented
                    such that: 0 < lower < upper < 1, to avoid division by zero in BaseRLearner.fit_predict() step.
            calibrate (bool): whether calibrate the propensity score
            model_kwargs: Keyword arguments to be passed to the underlying classification model.
        """
        self.clip_bounds = clip_bounds
        self.calibrate = calibrate
        self.model_kwargs = model_kwargs
        self.model = self._model
        self.calibrator = None

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
        if self.calibrate:
            # Fit a calibrator to the propensity scores with IsotonicRegression.
            # Ref: https://scikit-learn.org/stable/modules/isotonic.html
            self.calibrator = IsotonicRegression(
                out_of_bounds="clip",
                y_min=self.clip_bounds[0],
                y_max=self.clip_bounds[1],
            )
            self.calibrator.fit(self.model.predict_proba(X)[:, 1], y)

    def predict(self, X):
        """
        Predict propensity scores.

        Args:
            X (numpy.ndarray): a feature matrix

        Returns:
            (numpy.ndarray): Propensity scores between 0 and 1.
        """
        p = self.model.predict_proba(X)[:, 1]
        if self.calibrate:
            p = self.calibrator.transform(p)

        return np.clip(p, *self.clip_bounds)

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
                n_splits=(
                    self.model_kwargs.pop("n_fold")
                    if "n_fold" in self.model_kwargs
                    else 4
                ),
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

    def __init__(
        self,
        early_stop=False,
        clip_bounds=(1e-3, 1 - 1e-3),
        calibrate=True,
        **model_kwargs,
    ):
        self.early_stop = early_stop
        super().__init__(clip_bounds, calibrate, **model_kwargs)

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

        if self.early_stop:
            kwargs.update({"early_stopping_rounds": 10})

        return xgb.XGBClassifier(**kwargs)

    def fit(self, X, y, stop_val_size=0.2):
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
            )
            if self.calibrate:
                self.calibrator = IsotonicRegression(
                    out_of_bounds="clip",
                    y_min=self.clip_bounds[0],
                    y_max=self.clip_bounds[1],
                )
                self.calibrator.fit(self.model.predict_proba(X)[:, 1], y)
        else:
            super().fit(X, y)


def compute_propensity_score(
    X,
    treatment,
    p_model=None,
    X_pred=None,
    treatment_pred=None,
    calibrate_p=True,
    clip_bounds=(1e-3, 1 - 1e-3),
):
    """Generate propensity score if user didn't provide and optionally calibrate.

    Args:
        X (np.matrix): features for training
        treatment (np.array or pd.Series): a treatment vector for training
        p_model (model object, optional): a binary classifier with either a predict_proba or predict method
        X_pred (np.matrix, optional): features for prediction
        treatment_pred (np.array or pd.Series, optional): a treatment vector for prediciton
        calibrate_p (bool, optional): whether calibrate the propensity score
        clip_bounds (tuple, optional): lower and upper bounds for clipping propensity scores. Bounds should be implemented
                    such that: 0 < lower < upper < 1, to avoid division by zero in BaseRLearner.fit_predict() step.

    Returns:
        (tuple)
            - p (numpy.ndarray): propensity score
            - p_model (PropensityModel): either the original p_model or a trained ElasticNetPropensityModel
    """
    if treatment_pred is None:
        treatment_pred = treatment.copy()
    if p_model is None:
        p_model = ElasticNetPropensityModel(
            clip_bounds=clip_bounds, calibrate=calibrate_p
        )

    p_model.fit(X, treatment)

    X_pred = X if X_pred is None else X_pred

    try:
        p = p_model.predict_proba(X_pred)[:, 1]
    except AttributeError:
        logger.info("predict_proba not available, using predict instead")
        p = p_model.predict(X_pred)

    return p, p_model
