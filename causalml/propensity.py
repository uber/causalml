from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import ElasticNetCV


logger = logging.getLogger('causalml')


class ElasticNetPropensityModel(object):
    """
    Propensity regression model based on the ElasticNet algorithm.

    Attributes:
        model (sklearn.linear_model.ElasticNetCV): a propensity model object
    """

    def __init__(self, n_fold=5, clip_bounds=(1e-3, 1 - 1e-3), random_state=None):
        """Initialize a propensity model object.

        Args:
            n_fold (int): the number of cross-validation fold
            clip_bounds (tuple): lower and upper bounds for clipping propensity scores. Bounds should be implemented such that:
                0 < lower < upper < 1, to avoid division by zero in BaseRLearner.fit_predict() step.
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
