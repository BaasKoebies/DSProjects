"""Repeated cross-validation of every calling rule. Run: uv run python -m loan_targeting.evaluate

5-fold stratified CV, repeated 10 times on the evaluation seed. In every fold,
each model is fitted fresh on the training part with the frozen settings from
tune.py, then scored on the held-out part. Writes two files to results/ (Q31):

- fold_metrics.csv: one row per repeat, fold, scenario and rule (profit, PR-AUC).
- oof_predictions.csv: each customer's out-of-fold probability, per repeat.

The memo's numbers and charts are read from these files, never typed in.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

from loan_targeting.data import PROJECT_ROOT, load_raw, model_inputs
from loan_targeting.economics import CONTACT_COST, MARGIN, break_even_threshold, campaign_profit, per_thousand, should_contact
from loan_targeting.models import RANDOM_STATE, IncomeRule, calibrated, lightgbm_model, logistic_pipeline
from loan_targeting.tune import RESULTS_DIR, TUNED_PARAMS_PATH

N_SPLITS = 5
N_REPEATS = 10
FOLD_METRICS_PATH = RESULTS_DIR / "fold_metrics.csv"
OOF_PATH = RESULTS_DIR / "oof_predictions.csv"


@dataclass(frozen=True)
class Scenario:
    """One set of economic assumptions. The base case is R500 per call, R5,000 margin, s = 0."""

    name: str
    contact_cost: float = CONTACT_COST
    borrow_anyway: float = 0.0

    @property
    def threshold(self) -> float:
        return break_even_threshold(self.contact_cost, MARGIN, self.borrow_anyway)


SCENARIOS = [
    Scenario("base"),
    Scenario("contact_cost_1000", contact_cost=1000.0),
    Scenario("borrow_anyway_25", borrow_anyway=0.25),
    Scenario("borrow_anyway_50", borrow_anyway=0.50),
]


def fit_models(X_train: pd.DataFrame, y_train: pd.Series, tuned: dict) -> dict:
    """Fit the four calibrated models on one training fold: both models, with and without CD Account (Q13)."""
    C = tuned["logistic"]["params"]["C"]
    gbm = tuned["lightgbm"]["params"]
    no_cd = X_train.drop(columns="CD Account")
    return {
        "logistic": calibrated(logistic_pipeline(C)).fit(X_train, y_train),
        "lightgbm": calibrated(lightgbm_model(**gbm)).fit(X_train, y_train),
        "logistic_no_cd": calibrated(logistic_pipeline(C, without=["CD Account"])).fit(no_cd, y_train),
        "lightgbm_no_cd": calibrated(lightgbm_model(**gbm)).fit(no_cd, y_train),
    }


def predict(models: dict, X_test: pd.DataFrame) -> dict[str, np.ndarray]:
    no_cd = X_test.drop(columns="CD Account")
    return {
        name: model.predict_proba(no_cd if name.endswith("_no_cd") else X_test)[:, 1]
        for name, model in models.items()
    }


def score_fold(probabilities: dict, X_train, y_train, X_test, y_test) -> list[dict]:
    """Profit of every rule under every scenario, plus PR-AUC, on one held-out fold."""
    accepted = y_test.to_numpy()
    n = len(accepted)
    rows = []
    for scenario in SCENARIOS:
        cost, s = scenario.contact_cost, scenario.borrow_anyway
        income_rule = IncomeRule(cost, MARGIN, s).fit(X_train, y_train)
        decisions = {name: should_contact(p, scenario.threshold) for name, p in probabilities.items()}
        decisions["income_rule"] = income_rule.decide(X_test)
        decisions["call_everyone"] = np.ones(n, dtype=bool)
        decisions["call_nobody"] = np.zeros(n, dtype=bool)
        decisions["perfect"] = accepted == 1
        for rule, called in decisions.items():
            profit = campaign_profit(called, accepted, cost, MARGIN, s)
            rows.append(
                {
                    "scenario": scenario.name,
                    "threshold": scenario.threshold,
                    "rule": rule,
                    "customers": n,
                    "called": int(called.sum()),
                    "accepted_and_called": int(accepted[called].sum()),
                    "profit_per_1000": per_thousand(profit, n),
                    "income_cutoff": income_rule.cutoff_ if rule == "income_rule" else np.nan,
                }
            )
    pr_auc = {name: average_precision_score(accepted, p) for name, p in probabilities.items()}
    pr_auc["income_rule"] = average_precision_score(accepted, IncomeRule.score(X_test))
    for row in rows:
        row["pr_auc"] = pr_auc.get(row["rule"], np.nan)
    return rows


def main() -> None:
    X, y = model_inputs(load_raw())
    tuned = json.loads(TUNED_PARAMS_PATH.read_text())
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    metric_rows, oof_frames = [], []
    for i, (train, test) in enumerate(cv.split(X, y)):
        repeat, fold = divmod(i, N_SPLITS)
        X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
        probabilities = predict(fit_models(X_train, y_train, tuned), X_test)
        for row in score_fold(probabilities, X_train, y_train, X_test, y_test):
            metric_rows.append({"repeat": repeat, "fold": fold, **row})
        oof_frames.append(
            pd.DataFrame({"repeat": repeat, "row": X_test.index, "accepted": y_test.to_numpy(),
                          "income": X_test["Income"].to_numpy(), **{f"p_{k}": v for k, v in probabilities.items()}})
        )
        print(f"repeat {repeat + 1:>2}/{N_REPEATS}  fold {fold + 1}/{N_SPLITS}")

    metrics = pd.DataFrame(metric_rows)
    oof = pd.concat(oof_frames).sort_values(["repeat", "row"])
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics.round(6).to_csv(FOLD_METRICS_PATH, index=False)
    oof.round(6).to_csv(OOF_PATH, index=False)

    base = metrics[metrics["scenario"] == "base"]
    summary = base.groupby("rule").agg(
        profit_mean=("profit_per_1000", "mean"),
        profit_sd=("profit_per_1000", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        called_per_1000=("called", lambda c: c.sum() / base.loc[c.index, "customers"].sum() * 1000),
    )
    ceiling = summary.loc["perfect", "profit_mean"]
    summary["share_of_ceiling"] = summary["profit_mean"] / ceiling
    print("\nBase case (R500 per call, R5,000 margin), 50 held-out folds:")
    print(summary.sort_values("profit_mean", ascending=False).round(3).to_string())
    print(f"\nSaved {FOLD_METRICS_PATH.relative_to(PROJECT_ROOT)} and {OOF_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
