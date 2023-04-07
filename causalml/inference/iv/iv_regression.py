import numpy as np
import pandas as pd

from causalml.inference.meta.utils import convert_pd_to_np
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS


class IVRegressor(object):
    """A wrapper class that uses IV2SLS from statsmodel

    A linear 2SLS model that estimates the average treatment effect with endogenous treatment variable.
    """

    def __init__(self):
        """
        Initializes the class.
        """

        self.method = "2SLS"

    def fit(self, X, treatment, y, w):
        """Fits the 2SLS model.

        Args:
            X (np.matrix or np.array or pd.Dataframe): a feature matrix
            treatment (np.array or pd.Series): a treatment vector
            y (np.array or pd.Series): an outcome vector
            w (np.array or pd.Series): an instrument vector
        """

        X, treatment, y, w = convert_pd_to_np(X, treatment, y, w)

        exog = sm.add_constant(np.c_[X, treatment])
        endog = y
        instrument = sm.add_constant(np.c_[X, w])

        self.iv_model = IV2SLS(endog=endog, exog=exog, instrument=instrument)
        self.iv_fit = self.iv_model.fit()

    def predict(self):
        """Returns the average treatment effect and its estimated standard error

        Returns:
            (float): average treatment effect
            (float): standard error of the estimation
        """

        return self.iv_fit.params[-1], self.iv_fit.bse[-1]
