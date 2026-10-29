"""The memo's charts and summary tables. Run: uv run python -m loan_targeting.report

Reads only the saved CV results (results/fold_metrics.csv and
results/oof_predictions.csv), so every number and chart in the memo traces to
those files. Writes three charts to figures/ and two summary tables to results/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from loan_targeting.data import PROJECT_ROOT
from loan_targeting.economics import CONTACT_COST, MARGIN, break_even_threshold
from loan_targeting.eda import TEXT_PRIMARY, TEXT_SECONDARY, _legend, _new_figure, _save, _style_axes
from loan_targeting.evaluate import FOLD_METRICS_PATH, OOF_PATH
from loan_targeting.tune import RESULTS_DIR

SUMMARY_PATH = RESULTS_DIR / "summary.csv"
OPERATING_POINTS_PATH = RESULTS_DIR / "operating_points.csv"

# One colour per rule across the report charts (first three validated palette slots).
RULE_COLOURS = {"lightgbm": "#2a78d6", "logistic": "#eb6834", "income_rule": "#1baf7a"}
RULE_NAMES = {"lightgbm": "LightGBM", "logistic": "Logistic regression", "income_rule": "Income rule"}
COST_COLOURS = {CONTACT_COST: "#2a78d6", 1000.0: "#eb6834"}


def _legend_above(ax, ncols: int) -> None:
    """Legend in a row between title and plot, for charts where every corner holds data."""
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=ncols, fontsize=8, frameon=False)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of profit per 1,000 for every scenario and rule, with share of the ceiling."""
    grouped = metrics.groupby(["scenario", "rule"])
    table = grouped.agg(
        profit_mean=("profit_per_1000", "mean"),
        profit_sd=("profit_per_1000", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        called=("called", "sum"),
        customers=("customers", "sum"),
    )
    table["called_per_1000"] = table.pop("called") / table.pop("customers") * 1000
    ceiling = table.xs("perfect", level="rule")["profit_mean"]
    table["share_of_ceiling"] = table["profit_mean"] / ceiling.reindex(table.index.get_level_values("scenario")).to_numpy()

    # Paired gap to LightGBM within each fold: the fairer comparison than two separate means.
    wide = metrics.pivot_table(index=["scenario", "repeat", "fold"], columns="rule", values="profit_per_1000")
    gaps = wide.rsub(wide["lightgbm"], axis=0)
    table["gap_to_lightgbm_mean"] = gaps.groupby(level="scenario").mean().stack().reindex(table.index)
    table["lightgbm_wins_folds"] = (gaps > 0).groupby(level="scenario").sum().stack().reindex(table.index)
    return table.reset_index()


def operating_points(oof: pd.DataFrame) -> pd.DataFrame:
    """Precision and recall of each model at its contact threshold, averaged over the 10 repeats."""
    rows = []
    for cost in (CONTACT_COST, 1000.0):
        threshold = break_even_threshold(cost, MARGIN)
        for model in ("lightgbm", "logistic"):
            per_repeat = []
            for _, rep in oof.groupby("repeat"):
                called = rep[f"p_{model}"] >= threshold
                hits = (called & (rep["accepted"] == 1)).sum()
                per_repeat.append((hits / called.sum(), hits / rep["accepted"].sum(), called.mean()))
            precision, recall, share = np.mean(per_repeat, axis=0)
            rows.append({"contact_cost": cost, "threshold": threshold, "model": model,
                         "precision": precision, "recall": recall, "share_called": share})
    return pd.DataFrame(rows)


def plot_pr_curves(oof: pd.DataFrame, points: pd.DataFrame) -> Path:
    """Results chart: precision-recall curves, one thin line per repeat, with the 9.6% baseline."""
    base_rate = oof["accepted"].mean()
    fig, (ax,) = _new_figure(7, 4.8)
    for rule in ("lightgbm", "logistic", "income_rule"):
        score = "income" if rule == "income_rule" else f"p_{rule}"
        for i, (_, rep) in enumerate(oof.groupby("repeat")):
            precision, recall, _ = precision_recall_curve(rep["accepted"], rep[score])
            ax.plot(recall, precision, color=RULE_COLOURS[rule], linewidth=1, alpha=0.35,
                    label=RULE_NAMES[rule] if i == 0 else None)
    base = points[points["contact_cost"] == CONTACT_COST]
    for _, p in base.iterrows():
        ax.scatter(p["recall"], p["precision"], s=40, color=RULE_COLOURS[p["model"]], edgecolors="white",
                   linewidths=1.5, zorder=4)
        ax.annotate(f"{RULE_NAMES[p['model']]} at 10%: precision {p['precision']:.0%}, recall {p['recall']:.0%}",
                    (p["recall"], p["precision"]), xytext=(-8, -16 if p["model"] == "lightgbm" else 10),
                    textcoords="offset points", ha="right", fontsize=8, color=TEXT_PRIMARY)
    ax.axhline(base_rate, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    ax.text(0.01, base_rate + 0.015, f"Calling at random: precision {base_rate:.1%}", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _style_axes(ax, "", "Recall: share of acceptors called", "Precision: share of calls that accept", grid="both")
    ax.set_title("Precision against recall, 10 repeats of 5-fold CV", loc="left", fontsize=10, color=TEXT_PRIMARY, pad=24)
    _legend_above(ax, ncols=3)
    return _save(fig, "results_pr_curve.png")


def plot_calibration(oof: pd.DataFrame) -> Path:
    """Method chart: predicted probability against observed acceptance, pooled over all repeats."""
    threshold = break_even_threshold()
    fig, (ax,) = _new_figure(6, 5)
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linestyle="--", linewidth=1, label="Perfect calibration")
    for model in ("lightgbm", "logistic"):
        observed, predicted = calibration_curve(oof["accepted"], oof[f"p_{model}"], n_bins=10, strategy="uniform")
        ax.plot(predicted, observed, color=RULE_COLOURS[model], linewidth=2, marker="o", markersize=5,
                label=RULE_NAMES[model])
    ax.axvline(threshold, color=TEXT_SECONDARY, linewidth=0.8)
    ax.text(threshold + 0.01, 0.92, "10% contact threshold", fontsize=8, color=TEXT_SECONDARY)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _style_axes(ax, "Calibration: predicted against observed acceptance", "Predicted probability (10 bins)",
                "Observed acceptance rate", grid="both")
    _legend(ax, loc="lower right")
    return _save(fig, "method_calibration.png")


def plot_profit_curve(oof: pd.DataFrame) -> Path:
    """Sensitivity chart: LightGBM profit per 1,000 as more customers are called, at R500 and R1,000."""
    fig, (ax,) = _new_figure(7, 4.6)
    visible_low = 0.0
    for cost, colour in COST_COLOURS.items():
        curves, chosen = [], []
        for _, rep in oof.groupby("repeat"):
            p = rep["p_lightgbm"].to_numpy()
            accepted = rep["accepted"].to_numpy()[np.argsort(-p, kind="stable")]
            n = len(accepted)
            calls = np.arange(1, n + 1)
            curves.append((np.cumsum(accepted) * MARGIN - calls * cost) / n * 1000)
            chosen.append((p >= break_even_threshold(cost, MARGIN)).sum())
        curve = np.mean(curves, axis=0)
        share = np.arange(1, len(curve) + 1) / len(curve)
        k = int(round(np.mean(chosen))) - 1
        peak = int(np.argmax(curve))
        print(f"R{cost:,.0f} per call: threshold rule calls {share[k]:.1%} for R{curve[k]:,.0f}; "
              f"best possible cut calls {share[peak]:.1%} for R{curve[peak]:,.0f}")
        ax.plot(share, curve, color=colour, linewidth=2, label=f"R{cost:,.0f} per call")
        ax.scatter(share[k], curve[k], s=40, color=colour, edgecolors="white", linewidths=1.5, zorder=4)
        above = cost == CONTACT_COST
        ax.annotate(f"Threshold rule: call {share[k]:.0%}, R{curve[k]:,.0f}", (share[k], curve[k]),
                    xytext=(10, 6 if above else -14), textcoords="offset points", fontsize=8, color=TEXT_PRIMARY)
        visible_low = min(visible_low, curve[share <= 0.5].min())
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(visible_low * 1.2 - 20_000, None)  # scale to the plotted range, not the full curve
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"R{v / 1000:,.0f}k")
    _style_axes(ax, "LightGBM: profit per 1,000 customers as more are called",
                "Share of customers called, highest probability first", "Profit per 1,000 customers", grid="both")
    _legend(ax, loc="upper right")
    return _save(fig, "sensitivity_profit_curve.png")


def main() -> None:
    metrics = pd.read_csv(FOLD_METRICS_PATH)
    oof = pd.read_csv(OOF_PATH)

    summary = summary_table(metrics)
    points = operating_points(oof)
    summary.round(4).to_csv(SUMMARY_PATH, index=False)
    points.round(4).to_csv(OPERATING_POINTS_PATH, index=False)
    print(f"Saved {SUMMARY_PATH.relative_to(PROJECT_ROOT)} and {OPERATING_POINTS_PATH.relative_to(PROJECT_ROOT)}")
    print(points.round(3).to_string(index=False))

    plot_pr_curves(oof, points)
    plot_calibration(oof)
    plot_profit_curve(oof)


if __name__ == "__main__":
    main()
