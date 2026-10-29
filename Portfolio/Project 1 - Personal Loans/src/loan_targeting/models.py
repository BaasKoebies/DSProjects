"""The three ways of choosing who to call: logistic regression, LightGBM and an income cutoff.

Everything that learns from data (scaling, one-hot encoding, the model, the
calibration, the income cutoff) is fitted on the training fold only. The
pipelines make that automatic: cross-validation calls fit on the training
fold, and the test fold is only transformed with what was learned there.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_targeting.data import BINARY_INPUTS, CATEGORICAL_INPUTS, CONTINUOUS_INPUTS
from loan_targeting.economics import CONTACT_COST, MARGIN, campaign_profit

RANDOM_STATE = 20261018


def logistic_pipeline(C: float = 1.0, without: list[str] | None = None) -> Pipeline:
    """Logistic regression with standardised continuous inputs and one-hot categories (Q16, Q17).

    C is the inverse of the penalty strength: smaller C shrinks coefficients more.
    without: inputs to leave out, for the without-CD refit (Q13).
    """
    skip = set(without or [])
    preprocess = ColumnTransformer(
        [
            ("scale", StandardScaler(), [c for c in CONTINUOUS_INPUTS if c not in skip]),
            ("one_hot", OneHotEncoder(drop="first"), [c for c in CATEGORICAL_INPUTS if c not in skip]),
            ("flags", "passthrough", [c for c in BINARY_INPUTS if c not in skip]),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(C=C, max_iter=1000))])


def lightgbm_model(**params: Any) -> LGBMClassifier:
    """LightGBM on raw inputs: trees need neither scaling nor one-hot encoding.

    deterministic, force_row_wise and a single thread make reruns give identical numbers.
    """
    return LGBMClassifier(
        random_state=RANDOM_STATE,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
        **params,
    )


def calibrated(estimator: Any) -> CalibratedClassifierCV:
    """Platt scaling (Q29): fit the model and a sigmoid calibrator on 5 inner splits, then average.

    The inner splits are shuffled so the row order of the file cannot shape them.
    """
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=inner)


class IncomeRule:
    """Baseline (Q11): call everyone at or above an income cutoff.

    The cutoff is the one that earns the most profit on the training fold, so the
    rule is chosen and scored the same way as the models.
    """

    def __init__(self, contact_cost: float = CONTACT_COST, margin: float = MARGIN, borrow_anyway: float = 0.0) -> None:
        self.contact_cost = contact_cost
        self.margin = margin
        self.borrow_anyway = borrow_anyway

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "IncomeRule":
        income = X["Income"].to_numpy()
        accepted = y.to_numpy()
        candidates = np.unique(income)
        profits = [
            campaign_profit(income >= c, accepted, self.contact_cost, self.margin, self.borrow_anyway)
            for c in candidates
        ]
        self.cutoff_ = float(candidates[int(np.argmax(profits))])
        return self

    def decide(self, X: pd.DataFrame) -> npt.NDArray[np.bool_]:
        """True for each customer the rule calls."""
        return X["Income"].to_numpy() >= self.cutoff_

    @staticmethod
    def score(X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Income as a ranking score, so the rule also gets a precision-recall curve."""
        return X["Income"].to_numpy(dtype=float)
