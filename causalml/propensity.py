import logging
import numpy as np
from pygam import LogisticGAM, s
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb
import multiprocessing

logger = logging.getLogger('causalml')


class ElasticNetPropensityModel(object):
    """Propensity regression model based on the ElasticNet algorithm.

    Attributes:
        model (sklearn.linear_model.ElasticNetCV): a propensity model object
    """

    def __init__(self, n_fold=4, Cs=np.logspace(1e-3, 1 - 1e-3, 4), l1_ratios=np.linspace(1e-3, 1 - 1e-3, 4),
                 clip_bounds=(1e-3, 1 - 1e-3), cv=None, random_state=None):
        """Initialize a propensity model object.
        Args:
            n_fold (int): the number of cross-validation fold
            Cs (int or array-like): Each of the values in Cs describes the inverse of regularization strength.
                If Cs is as an int, then a grid of Cs values are chosen in a logarithmic scale between 1e-4 and 1e4.
            l1_ratios (array-like): array of l1 ratios (0 <= value <= 1) to iterate through in CV estimator
            clip_bounds (tuple): lower and upper bounds for clipping propensity scores. Bounds should be implemented
                such that: 0 < lower < upper < 1, to avoid division by zero in BaseRLearner.fit_predict() step.
            cv (int or cross-validation generator): The default cross-validation generator used is Stratified K-Folds.
                If an integer is provided, then it is the number of folds used.
            random_state (numpy.random.RandomState or int): RandomState or an int seed

        Returns:
            None
        """
        if cv is None:
            self.cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_state)
        else:
            self.cv = cv
        self.model = LogisticRegressionCV(penalty='elasticnet', solver='saga', Cs=Cs, l1_ratios=l1_ratios,
                                          cv=self.cv, random_state=random_state)
        self.clip_bounds = clip_bounds

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
        ps = np.clip(self.model.predict_proba(X)[:, 1], *self.clip_bounds)
        return ps

    def fit_predict(self, X, y):
        """Fit a propensity model and predict propensity scores.

        Args:
            X (numpy.ndarray): a feature matrix
            y (numpy.ndarray): a binary target vector

        Returns:
            (numpy.ndarray): Propensity scores between 0 and 1.
        """
        self.fit(X, y)
        ps = self.predict(X)
        logger.info('AUC score: {:.6f}'.format(auc(y, ps)))
        return ps


class GradientBoostedPropensityModel(object):
    """
    Fits a simple gradient boosted propensity score model with optional early stopping.

    Notes
    -----
    Please see the xgboost documentation for more information on gradient boosting tuning parameters:
    https://xgboost.readthedocs.io/en/latest/python/python_api.html
    """

    cpu_count = multiprocessing.cpu_count()

    def __init__(self, max_depth=8, learning_rate=0.1, n_estimators=100, objective='binary:logistic',
                 n_thread=cpu_count, colsample_bytree=0.8, early_stop=False, stop_val_size=0.2, n_stop_rounds=10,
                 clip_bounds=(1e-3, 1 - 1e-3), random_state=None):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.objective = objective
        self.n_thread = n_thread
        self.colsample_bytree = colsample_bytree
        self.early_stop = early_stop
        self.stop_val_size = stop_val_size
        self.n_stop_rounds=n_stop_rounds
        self.clip_bounds = clip_bounds
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit a propensity model.

        Args:
            X (numpy.ndarray): a feature matrix
            y (numpy.ndarray): a binary target vector
        """
        gbmc = xgb.XGBClassifier(max_depth=self.max_depth,
                                 learning_rate=self.learning_rate,
                                 n_estimators=self.n_estimators,
                                 objective=self.objective,
                                 nthread=self.n_thread,
                                 colsample_bytree = self.colsample_bytree,
                                 random_state=self.random_state)

        if self.early_stop:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.stop_val_size)
            self.gbmc_fit = gbmc.fit(X_train,
                                     y_train,
                                     eval_set=[(X_val, y_val)],
                                     early_stopping_rounds=self.n_stop_rounds)
        else:
            self.gbmc_fit = gbmc.fit(X, y)

    def predict(self, X):
        """
        Predict propensity scores.

        Args:
            X (numpy.ndarray): a feature matrix

        Returns:
            (numpy.ndarray): Propensity scores between 0 and 1.
        """
        if self.early_stop:
            ps = self.gbmc_fit.predict_proba(X, ntree_limit=self.gbmc_fit.best_ntree_limit)[:, 1]
        else:
            ps = self.gbmc_fit.predict_proba(X)[:, 1]

        ps = np.clip(ps, *self.clip_bounds)

        return ps

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
        ps = self.predict(X)
        logger.info('AUC score: {:.6f}'.format(auc(y, ps)))
        return ps


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


def compute_propensity_score(X, treatment, p_model=None, X_pred=None, treatment_pred=None, calibrate_p=True):
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
            - p_model_dict (dict): dictionary of propensity model
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
        logger.info('Calibrating propensity scores.')
        p = calibrate(p, treatment_pred)

    # force the p values within the range
    eps = np.finfo(float).eps
    p = np.where(p < 0 + eps, 0 + eps*1.001, p)
    p = np.where(p > 1 - eps, 1 - eps*1.001, p)

    return p, p_model
