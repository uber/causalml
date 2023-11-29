import pandas as pd
import numpy as np
import pytest
from causalml.metrics.visualize import get_cumlift


def test_visualize_get_cumlift_errors_on_nan():
    df = pd.DataFrame(
        [[0, np.nan, 0.5], [1, np.nan, 0.1], [1, 1, 0.4], [0, 1, 0.3], [1, 1, 0.2]],
        columns=["w", "y", "pred"],
    )

    with pytest.raises(Exception):
        get_cumlift(df)
