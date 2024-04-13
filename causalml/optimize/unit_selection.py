import numpy as np

from sklearn.base import clone

import warnings


class CounterfactualUnitSelector:
    """
    A highly experimental implementation of the counterfactual unit selection
    model proposed by Li and Pearl (2019).

    Parameters
    ----------
    learner : object
        The base learner used to estimate the segment probabilities.

    nevertaker_payoff : float
        The payoff from targeting a never-taker

    alwaystaker_payoff : float
        The payoff from targeting an always-taker

    complier_payoff : float
        The payoff from targeting a complier

    defier_payoff : float
        The payoff from targeting a defier

    organic_conversion : float, optional (default=None)
        The organic conversion rate in the population without an intervention.
        If None, the organic conversion rate is obtained from tne control group.

        NB: The organic conversion in the control group is not always the same
        as the organic conversion rate without treatment.

    data : DataFrame
        A pandas DataFrame containing the features, treatment assignment
        indicator and the outcome of interest.

    treatment : string
        A string corresponding to the name of the treatment column. The
        assumed coding in the column is 1 for treatment and 0 for control.

    outcome : string
        A string corresponding to the name of the outcome column. The assumed
        coding in the column is 1 for conversion and 0 for no conversion.

    References
    ----------
    Li, Ang, and Judea Pearl. 2019. “Unit Selection Based on Counterfactual
    Logic.” https://ftp.cs.ucla.edu/pub/stat_ser/r488.pdf.
    """

    def __init__(
        self,
        learner,
        nevertaker_payoff,
        alwaystaker_payoff,
        complier_payoff,
        defier_payoff,
        organic_conversion=None,
    ):
        self.learner = learner
        self.nevertaker_payoff = nevertaker_payoff
        self.alwaystaker_payoff = alwaystaker_payoff
        self.complier_payoff = complier_payoff
        self.defier_payoff = defier_payoff
        self.organic_conversion = organic_conversion

    def fit(self, data, treatment, outcome):
        """
        Fits the class.
        """

        if self._gain_equality_check():
            self._fit_segment_model(data, treatment, outcome)

        else:
            self._fit_segment_model(data, treatment, outcome)
            self._fit_condprob_models(data, treatment, outcome)

    def predict(self, data, treatment, outcome):
        """
        Predicts an individual-level payoff. If gain equality is satisfied, uses
        the exact function; if not, uses the midpoint between bounds.
        """

        if self._gain_equality_check():
            est_payoff = self._get_exact_benefit(data, treatment, outcome)

        else:
            est_payoff = self._obj_func_midp(data, treatment, outcome)

        return est_payoff

    def _gain_equality_check(self):
        """
        Checks if gain equality is satisfied. If so, the optimization task can
        be simplified.
        """

        return (
            self.complier_payoff + self.defier_payoff
            == self.alwaystaker_payoff + self.nevertaker_payoff
        )

    @staticmethod
    def _make_segments(data, treatment, outcome):
        """
        Constructs the following segments:

        * AC = Pr(Y = 1, W = 1 /mid X)
        * AD = Pr(Y = 1, W = 0 /mid X)
        * ND = Pr(Y = 0, W = 1 /mid X)
        * ND = Pr(Y = 0, W = 0 /mid X)

        where the names of the outcomes correspond the combinations of
        the relevant segments, eg AC = Always-taker or Complier.
        """

        segments = np.empty(data.shape[0], dtype="object")

        segments[(data[treatment] == 1) & (data[outcome] == 1)] = "AC"
        segments[(data[treatment] == 0) & (data[outcome] == 1)] = "AD"
        segments[(data[treatment] == 1) & (data[outcome] == 0)] = "ND"
        segments[(data[treatment] == 0) & (data[outcome] == 0)] = "NC"

        return segments

    def _fit_segment_model(self, data, treatment, outcome):
        """
        Fits a classifier for estimating the probabilities for the unit
        segment combinations.
        """

        model = clone(self.learner)

        X = data.drop([treatment, outcome], axis=1)
        y = self._make_segments(data, treatment, outcome)

        self.segment_model = model.fit(X, y)

    def _fit_condprob_models(self, data, treatment, outcome):
        """
        Fits two classifiers to estimate conversion probabilities conditional
        on the treatment.
        """

        trt_learner = clone(self.learner)
        ctr_learner = clone(self.learner)

        treated = data[treatment] == 1

        X = data.drop([treatment, outcome], axis=1)
        y = data[outcome]

        self.trt_model = trt_learner.fit(X[treated], y[treated])
        self.ctr_model = ctr_learner.fit(X[~treated], y[~treated])

    def _get_exact_benefit(self, data, treatment, outcome):
        """
        Calculates the exact benefit function of Theorem 4 in Li and Pearl (2019).
        Returns the exact benefit.
        """
        beta = self.complier_payoff
        gamma = self.alwaystaker_payoff
        theta = self.nevertaker_payoff

        X = data.drop([treatment, outcome], axis=1)

        segment_prob = self.segment_model.predict_proba(X)
        segment_name = self.segment_model.classes_

        benefit = (
            (beta - theta) * segment_prob[:, segment_name == "AC"]
            + (gamma - beta) * segment_prob[:, segment_name == "AD"]
            + theta
        )

        return benefit

    def _obj_func_midp(self, data, treatment, outcome):
        """
        Calculates bounds for the objective function. Returns the midpoint
        between bounds.

        Parameters
        ----------
        pr_y1_w1 : float
            The probability of conversion given treatment assignment.

        pr_y1_w0 : float
            The probability of conversion given control assignment.

        pr_y0_w1 : float
            The probability of no conversion given treatment assignment
            (1 - pr_y1_w1).

        pr_y0_w0 : float
            The probability of no conversion given control assignment
            (1 - pr_1y_w0)

        pr_y1w1_x : float
            Probability of complier or always-taker given X.

        pr_y0w0_x : float
            Probability of complier or never-taker given X.

        pr_y1w0_x : float
            Probability of defier or always-taker given X.

        pr_y0w1_x : float
            Probability of never-taker or defier given X.

        pr_y_x : float
            Organic probability of conversion.
        """

        X = data.drop([treatment, outcome], axis=1)

        beta = self.complier_payoff
        gamma = self.alwaystaker_payoff
        theta = self.nevertaker_payoff
        delta = self.defier_payoff

        pr_y0_w1, pr_y1_w1 = np.split(
            self.trt_model.predict_proba(X), indices_or_sections=2, axis=1
        )
        pr_y0_w0, pr_y1_w0 = np.split(
            self.ctr_model.predict_proba(X), indices_or_sections=2, axis=1
        )

        segment_prob = self.segment_model.predict_proba(X)
        segment_name = self.segment_model.classes_

        pr_y1w1_x = segment_prob[:, segment_name == "AC"]
        pr_y0w0_x = segment_prob[:, segment_name == "NC"]
        pr_y1w0_x = segment_prob[:, segment_name == "AD"]
        pr_y0w1_x = segment_prob[:, segment_name == "ND"]

        if self.organic_conversion is not None:
            pr_y_x = self.organic_conversion

        else:
            pr_y_x = pr_y1_w0
            warnings.warn(
                "Probability of organic conversion estimated from control observations."
            )

        p1 = (beta - theta) * pr_y1_w1 + delta * pr_y1_w0 + theta * pr_y0_w0
        p2 = gamma * pr_y1_w1 + delta * pr_y0_w1 + (beta - gamma) * pr_y0_w0
        p3 = (
            (gamma - delta) * pr_y1_w1
            + delta * pr_y1_w0
            + theta * pr_y0_w0
            + (beta - gamma - theta + delta) * (pr_y1w1_x + pr_y0w0_x)
        )
        p4 = (
            (beta - theta) * pr_y1_w1
            - (beta - gamma - theta) * pr_y1_w0
            + theta * pr_y0_w0
            + (beta - gamma - theta + delta) * (pr_y1w0_x + pr_y0w1_x)
        )
        p5 = (gamma - delta) * pr_y1_w1 + delta * pr_y1_w0 + theta * pr_y0_w0
        p6 = (
            (beta - theta) * pr_y1_w1
            - (beta - gamma - theta) * pr_y1_w0
            + theta * pr_y0_w0
        )
        p7 = (
            (gamma - delta) * pr_y1_w1
            - (beta - gamma - theta) * pr_y1_w0
            + theta * pr_y0_w0
            + (beta - gamma - theta + delta) * pr_y_x
        )
        p8 = (
            (beta - theta) * pr_y1_w1
            + delta * pr_y1_w0
            + theta * pr_y0_w0
            - (beta - gamma - theta + delta) * pr_y_x
        )

        params_1 = np.concatenate((p1, p2, p3, p4), axis=1)
        params_2 = np.concatenate((p5, p6, p7, p8), axis=1)

        sigma = beta - gamma - theta + delta

        if sigma < 0:
            lower_bound = np.max(params_1, axis=1)
            upper_bound = np.min(params_2, axis=1)

        elif sigma > 0:
            lower_bound = np.max(params_2, axis=1)
            upper_bound = np.min(params_1, axis=1)

        return (lower_bound + upper_bound) / 2
