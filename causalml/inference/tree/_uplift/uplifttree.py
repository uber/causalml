"""Experimental kernel-backed uplift tree classifier.

Not part of the public API -- this exists to prove numerical parity of the
kernel-backed KL / ED / Chi criteria against the legacy ``UpliftTreeClassifier``
before the public classes are switched over. It supports the Rzepakowski
``n_reg`` / ``min_samples_treatment`` regularization; normalization, honesty,
pruning, and the forest are handled in later issues of the epic.
"""

from typing import Union

import numpy as np
from numpy import float32 as DTYPE
from sklearn.utils import check_array

from causalml.inference.meta.utils import check_treatment_vector

from ._tree import BaseUpliftDecisionTree


class _KernelUpliftTreeClassifier(BaseUpliftDecisionTree):
    """A single uplift tree grown on the shared ``_tree`` Cython kernel."""

    def __init__(
        self,
        *,
        criterion: str = "KL",
        control_name: Union[int, str] = "control",
        max_depth: int = 3,
        min_samples_leaf: int = 100,
        min_samples_split: Union[int, float] = 2,
        min_samples_treatment: int = 10,
        n_reg: int = 100,
        max_features: Union[int, float, str, None] = None,
        min_weight_fraction_leaf: float = 0.0,
        random_state: int = None,
    ):
        self.control_name = control_name
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        super().__init__(
            criterion=criterion,
            splitter="best",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=None,
            random_state=random_state,
            min_impurity_decrease=0.0,
            ccp_alpha=0.0,
        )

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
        sample_weight: Union[np.ndarray, None] = None,
        check_input: bool = True,
    ):
        """Fit the uplift tree.

        Args:
            X (np.ndarray): feature matrix
            treatment (np.ndarray): treatment vector, includes the control group
            y (np.ndarray): binary outcome vector (coerced to {0, 1})
            sample_weight (np.ndarray): optional sample weights
            check_input (bool): default True
        Returns:
            self
        """
        X, y = self._prepare_data(X=X, treatment=treatment, y=y)
        super().fit(X=X, y=y, sample_weight=sample_weight, check_input=check_input)
        return self

    def predict_proba_by_group(
        self, X: np.ndarray, check_input: bool = True
    ) -> np.ndarray:
        """Per-group leaf probabilities P(Y=1 | T=g).

        Returns:
            np.ndarray, shape (n_samples, n_groups); column 0 is the control
            group, columns 1..k the treatment groups in sorted order.
        """
        return BaseUpliftDecisionTree.predict(self, X, check_input=check_input)

    def predict(self, X: np.ndarray, check_input: bool = True) -> np.ndarray:
        """Per-treatment individual treatment effect P(Y=1|T=t) - P(Y=1|control).

        Returns:
            np.ndarray, shape (n_samples, n_treatments).
        """
        proba = self.predict_proba_by_group(X, check_input=check_input)
        return proba[:, 1:] - proba[:, [0]]

    def _prepare_data(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        y: np.ndarray,
    ) -> tuple:
        """Encode (X, treatment, y) as (X, group-matrix y).

        The outcome is coerced to binary and reshaped to a
        ``(n_samples, n_groups)`` NaN-masked matrix with the control group in
        column 0 -- the same encoding the causal trees use.
        """
        if y.shape[0] != treatment.shape[0]:
            raise ValueError(
                f"The number of `treatment` and `y` rows are not equal: "
                f"{y.shape[0]} {treatment.shape[0]}"
            )
        check_treatment_vector(treatment, self.control_name)
        self.unique_groups = list(set(treatment))
        self.unique_treatments = sorted(
            [x for x in self.unique_groups if x != self.control_name]
        )
        self._group2index = {
            self.control_name: 0,
            **{treatment: i + 1 for i, treatment in enumerate(self.unique_treatments)},
        }
        self.classes_ = [self.control_name] + self.unique_treatments

        X = check_array(X, dtype=DTYPE, accept_sparse="csc")
        y = check_array(y, ensure_2d=False, dtype=None)
        y = (y > 0).astype(np.float64)
        self.n_samples, self.n_features = X.shape

        y_2dim = np.zeros((self.n_samples, len(self.unique_treatments) + 1))
        for group, group_index in self._group2index.items():
            y_2dim[:, group_index] = np.where(treatment == group, y, np.nan)

        return X, y_2dim
