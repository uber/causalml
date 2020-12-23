import logging

import numpy as np
from causalml.propensity import compute_propensity_score
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier


logger = logging.getLogger("causalml")


class PolicyLearner(object):
    """
    A Learner that learns a treatment assignment policy with observational data using doubly robust estimator of causal
    effect for binary treatment.

    Details of the policy learner are available at Athey and Wager (2018) (https://arxiv.org/abs/1702.02896).

    """

    def __init__(
        self,
        outcome_learner=GradientBoostingRegressor(),
        treatment_learner=GradientBoostingClassifier(),
        policy_learner=DecisionTreeClassifier(),
        clip_bounds=(1e-3, 1 - 1e-3),
        n_fold=5,
        random_state=None,
        calibration=False,
    ):
        """Initialize a treatment assignment policy learner.

        Args:
            outcome_learner (optional): a regression model to estimate outcomes
            policy_learner (optional): a classification model to estimate treatment assignment. It needs to take
                `sample_weight` as an input argument for `fit()`
            clip_bounds (tuple, optional): lower and upper bounds for clipping propensity scores to avoid division by
                zero in PolicyLearner.fit()
            n_fold (int, optional): the number of cross validation folds for outcome_learner
            random_state (int or RandomState, optional): a seed (int) or random number generator (RandomState)
        """
        self.model_mu = outcome_learner
        self.model_w = treatment_learner
        self.model_pi = policy_learner
        self.clip_bounds = clip_bounds
        self.cv = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)
        self.calibration = calibration

        self._y_pred, self._tau_pred, self._w_pred, self._dr_score = (
            None,
            None,
            None,
            None,
        )

    def __repr__(self):
        return (
            "{}(model_mu={},\n"
            "\tmodel_w={},\n"
            "\model_pi={})".format(
                self.__class__.__name__,
                self.model_mu.__repr__(),
                self.model_w.__repr__(),
                self.model_pi.__repr__(),
            )
        )

    def _outcome_estimate(self, X, w, y):
        self._y_pred = np.zeros(len(y))
        self._tau_pred = np.zeros(len(y))

        for train_index, test_index in self.cv.split(y):
            X_train, X_test = X[train_index], X[test_index]
            w_train, w_test = w[train_index], w[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.model_mu.fit(
                np.concatenate([X_train, w_train.reshape(-1, 1)], axis=1), y_train
            )
            self._y_pred[test_index] = self.model_mu.predict(
                np.concatenate([X_test, w_test.reshape(-1, 1)], axis=1)
            )
            self._tau_pred[test_index] = self.model_mu.predict(
                np.concatenate([X_test, np.ones((len(w_test), 1))], axis=1)
            ) - self.model_mu.predict(
                np.concatenate([X_test, np.zeros((len(w_test), 1))], axis=1)
            )

    def _treatment_estimate(self, X, w):
        self._w_pred = np.zeros(len(w))

        for train_index, test_index in self.cv.split(w):
            X_train, X_test = X[train_index], X[test_index]
            w_train, w_test = w[train_index], w[test_index]

            self._w_pred[test_index], _ = compute_propensity_score(
                X=X_train,
                treatment=w_train,
                X_pred=X_test,
                treatment_pred=w_test,
                calibrate_p=self.calibration,
            )

        self._w_pred = np.clip(
            self._w_pred, a_min=self.clip_bounds[0], a_max=self.clip_bounds[1]
        )

    def fit(self, X, treatment, y, p=None, dhat=None):
        """Fit the treatment assignment policy learner.

        Args:
            X (np.matrix): a feature matrix
            treatment (np.array): a treatment vector (1 if treated, otherwise 0)
            y (np.array): an outcome vector
            p (optional, np.array): user provided propensity score vector between 0 and 1
            dhat (optinal, np.array): user provided predicted treatment effect vector

        Returns:
            self: returns an instance of self.
        """

        logger.info(
            "generating out-of-fold CV outcome estimates with {}".format(self.model_mu)
        )
        self._outcome_estimate(X, treatment, y)

        if dhat is not None:
            self._tau_pred = dhat

        if p is None:
            self._treatment_estimate(X, treatment)
        else:
            self._w_pred = np.clip(p, self.clip_bounds[0], self.clip_bounds[1])

        # Doubly Robust Modification
        self._dr_score = self._tau_pred + (treatment - self._w_pred) / self._w_pred / (
            1 - self._w_pred
        ) * (y - self._y_pred)

        target = self._dr_score.copy()
        target = np.sign(target)

        logger.info("training the treatment assignment model, {}".format(self.model_pi))
        self.model_pi.fit(X, target, sample_weight=abs(self._dr_score))

        return self

    def predict(self, X):
        """Predict treatment assignment that optimizes the outcome.

        Args:
            X (np.matrix): a feature matrix

        Returns:
            (numpy.ndarray): predictions of treatment assignment.
        """

        return self.model_pi.predict(X)

    def predict_proba(self, X):
        """Predict treatment assignment score that optimizes the outcome.

        Args:
            X (np.matrix): a feature matrix

        Returns:
            (numpy.ndarray): predictions of treatment assignment score.
        """

        pi_hat = self.model_pi.predict_proba(X)[:, 1]

        return pi_hat
