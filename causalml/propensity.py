import logging
import numpy as np
from pygam import LogisticGAM, s
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import ElasticNetCV


logger = logging.getLogger('causalml')


class ElasticNetPropensityModel(object):
    """Propensity regression model based on the ElasticNet algorithm.

    Attributes:
        model (sklearn.linear_model.ElasticNetCV): a propensity model object
    """

    def __init__(self, n_fold=5, clip_bounds=(1e-3, 1 - 1e-3), random_state=None):
        """Initialize a propensity model object.

        Args:
            n_fold (int): the number of cross-validation fold
            clip_bounds (tuple): lower and upper bounds for clipping propensity scores. Bounds should be implemented
                such that: 0 < lower < upper < 1, to avoid division by zero in BaseRLearner.fit_predict() step.
            random_state (numpy.random.RandomState or int): RandomState or an int seed

        Returns:
            None
        """
        self.model = ElasticNetCV(cv=n_fold, random_state=random_state)
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
        ps = np.clip(self.model.predict(X), *self.clip_bounds)
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


def compute_propensity_score(X, treatment, X_pred=None, treatment_pred=None, cv=None, calibrate_p=True):
    """Generate propensity score if user didn't provide

    Args:
        X (np.matrix): features for training
        treatment (np.array or pd.Series): a treatment vector for training
        X_pred (np.matrix, optional): features for prediction
        treatment_pred (np.array or pd.Series, optional): a treatment vector for prediciton
        cv (sklearn.model_selection._BaseKFold, optional): sklearn CV object
        calibrate_p (bool, optional): whether calibrate the propensity score

    Returns:
        (tuple)
            - p (numpy.ndarray): propensity score
            - p_model_dict (dict): dictionary of propensity model
    """
    if treatment_pred is None:
        treatment_pred = treatment.copy()

    p = np.zeros_like(treatment_pred, dtype=float)
    p_fold = np.zeros_like(treatment_pred, dtype=float)
    p_model = ElasticNetPropensityModel()
    p_model_dict = dict()

    if cv:
        for i_fold, (i_trn, i_val) in enumerate(cv.split(X, treatment), 1):
            logger.info('Training a propensity model for CV #{}'.format(i_fold))
            p_model.fit(X[i_trn], treatment[i_trn])
            p_model_dict[i_fold] = p_model

            if X_pred is None:
                p[i_val] = p_model.predict(X[i_val])
            else:
                i_rest = ~np.isin(X_pred, X[i_trn])
                i_rest = i_rest[:, 0]
                X_rest = X_pred[i_rest]
                p_fold[i_rest] += p_model.predict(X_rest)

        if X_pred is not None:
            p = p_fold/(cv.get_n_splits() - 1)
    else:
        p_model.fit(X, treatment)
        p_model_dict['all training'] = p_model
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

    return p, p_model_dict
