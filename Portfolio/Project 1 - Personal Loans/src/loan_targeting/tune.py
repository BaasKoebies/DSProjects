"""One tuning run, then frozen. Run: uv run python -m loan_targeting.tune

Scores each candidate setting on mean log loss of the calibrated model (Q33),
using one 5-fold split on its own seed (Q34), so tuning never sees the
evaluation's splits. Writes the winners to results/tuned_params.json, which
evaluate.py reads. The ranges below were fixed before the run (Q32).
"""

import json
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from loan_targeting.data import PROJECT_ROOT, load_raw, model_inputs
from loan_targeting.models import RANDOM_STATE, calibrated, lightgbm_model, logistic_pipeline

RESULTS_DIR = PROJECT_ROOT / "results"
TUNED_PARAMS_PATH = RESULTS_DIR / "tuned_params.json"

TUNING_SEED = 20261019  # differs from the evaluation seed (RANDOM_STATE)
LOGISTIC_C = [0.01, 0.1, 1.0, 10.0, 100.0]
LIGHTGBM_DRAWS = 20


def lightgbm_candidates(n: int = LIGHTGBM_DRAWS, seed: int = RANDOM_STATE) -> list[dict[str, Any]]:
    """Draw n random LightGBM settings from the ranges approved in Q32."""
    rng = np.random.default_rng(seed)
    return [
        {
            "num_leaves": int(rng.integers(4, 32)),  # 4 to 31
            "min_child_samples": int(rng.integers(10, 101)),  # 10 to 100
            "learning_rate": float(10 ** rng.uniform(np.log10(0.01), np.log10(0.2))),
            "n_estimators": int(rng.integers(100, 801)),  # 100 to 800
            "reg_lambda": float(10 ** rng.uniform(np.log10(0.001), np.log10(10))),
        }
        for _ in range(n)
    ]


def mean_log_loss(estimator: Any, X, y, cv: StratifiedKFold) -> float:
    """Mean out-of-fold log loss of the calibrated estimator (lower is better)."""
    scores = cross_val_score(calibrated(estimator), X, y, cv=cv, scoring="neg_log_loss")
    return float(-scores.mean())


def main() -> None:
    X, y = model_inputs(load_raw())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=TUNING_SEED)

    logistic_trials = []
    for C in LOGISTIC_C:
        loss = mean_log_loss(logistic_pipeline(C), X, y, cv)
        logistic_trials.append({"params": {"C": C}, "log_loss": round(loss, 5)})
        print(f"LR  C={C:<6} log loss {loss:.4f}")

    lightgbm_trials = []
    for i, params in enumerate(lightgbm_candidates(), 1):
        loss = mean_log_loss(lightgbm_model(**params), X, y, cv)
        lightgbm_trials.append({"params": params, "log_loss": round(loss, 5)})
        print(f"GBM {i:>2}/{LIGHTGBM_DRAWS} log loss {loss:.4f}  {params}")

    best_logistic = min(logistic_trials, key=lambda t: t["log_loss"])
    best_lightgbm = min(lightgbm_trials, key=lambda t: t["log_loss"])
    output = {
        "tuning_seed": TUNING_SEED,
        "objective": "mean 5-fold log loss of the Platt-calibrated model",
        "logistic": best_logistic,
        "lightgbm": best_lightgbm,
        "all_trials": {"logistic": logistic_trials, "lightgbm": lightgbm_trials},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    TUNED_PARAMS_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nBest LR:  {best_logistic}\nBest GBM: {best_lightgbm}")
    print(f"Saved {TUNED_PARAMS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
