import logging
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


logger = logging.getLogger('causalml')


class PolicyLearner(object):
    """
    A Learner that learns a treatment assignment policy with observational data using doubly robust estimator of causal
    effect for binary treatment.

    Details of the policy learner are available at Athey and Wager (2018) (https://arxiv.org/abs/1702.02896).

    """

    def __init__(self,
                 outcome_learner=GradientBoostingRegressor(),
                 policy_learner=GradientBoostingClassifier(),
                 clip_bounds=(1e-3, 1 - 1e-3),
                 n_fold=5,
                 random_state=None):
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
        self.model_pi = policy_learner
        self.clip_bounds = clip_bounds
        self.cv = KFold(n_splits=n_fold, shuffle=True, random_state=random_state)

    def __repr__(self):
        return ('{}(model_mu={},\n'
                '\tmodel_pi={})'.format(self.__class__.__name__,
                                        self.model_mu.__repr__(),
                                        self.model_pi.__repr__()))

    def fit(self, X, p, treatment, y, dhat):
        """Fit the treatment assignment policy learner.

        Args:
            X (np.matrix): a feature matrix
            p (np.array): a propensity score vector between 0 and 1
            treatment (np.array): a treatment vector (1 if treated, otherwise 0)
            y (np.array): an outcome vector
            dhat (np.array): a predicted treatment effect vector

        Returns:
            self: returns an instance of self.
        """

        logger.info('generating out-of-fold CV outcome estimates with {}'.format(self.model_mu))
        yhat = cross_val_predict(self.model_mu, X, y, cv=self.cv)

        ps = np.clip(p, self.clip_bounds[0], self.clip_bounds[1])

        # Doubly Robust Modification
        g = (treatment-ps)/(ps*(1-ps))
        gamma = dhat + g * (y - yhat)

        target = gamma.copy()
        target[target < 0] = 0
        target[target > 0] = 1

        logger.info('training the treatment assignment model, {}'.format(self.model_pi))
        self.model_pi.fit(X, target, sample_weight=abs(gamma))

        return self

    def predict(self, X):
        """Predict treatment assignment that optimizes the outcome.

        Args:
            X (np.matrix): a feature matrix

        Returns:
            (numpy.ndarray): predictions of treatment assignment.
        """

        pi_hat = self.model_pi.predict_proba(X)[:, 1]

        return pi_hat
