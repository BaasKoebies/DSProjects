"""Exploratory profile of the loan data. Run: uv run python -m loan_targeting.eda

Prints the evidence behind each cleaning decision and saves the EDA charts. It
changes nothing: drops are decided by reading this output.

Rule (decided 21 Oct): features are dropped only for label-free reasons
(redundancy, identifiers, broken data). The acceptance rates below describe
the data; they are not used to choose features.
"""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from scipy.stats import chi2_contingency

from loan_targeting.data import PROJECT_ROOT, TARGET, load_raw

FIGURES_DIR = PROJECT_ROOT / "figures"

# Inputs to the model, excluding the identifier and the outcome.
FEATURES = [
    "Age",
    "Experience",
    "Income",
    "ZIP Code",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
    "Securities Account",
    "CD Account",
    "Online",
    "CreditCard",
]
CATEGORICAL = ["Family", "Education", "Securities Account", "CD Account", "Online", "CreditCard"]
CONTINUOUS = ["Age", "Experience", "Income", "CCAvg", "Mortgage"]

# The lowest-income acceptor earns exactly 60, so the first band ends at 59.
INCOME_BANDS = [0, 59, 100, 150, 250]
EDUCATION_LABELS = {1: "Undergraduate", 2: "Graduate", 3: "Advanced"}

# Chart tokens: white background so the PNGs read in GitHub dark mode.
# Colours checked with the dataviz palette validator (colourblind separation, contrast).
SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
ACCEPTED = SERIES  # blue: accepted the offer, on every chart
NOT_ACCEPTED = "#eb6834"  # orange: did not accept, on every chart
HIGHLIGHT = "#1baf7a"  # aqua: broken values; below 3:1 contrast, so always direct-labelled
NEUTRAL_MID = "#f0efec"  # grey midpoint of the diverging correlation scale


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --- Label-free checks: these may justify a drop -------------------------


def overview(df: pd.DataFrame) -> None:
    section("1. Overview")
    print(f"Rows: {len(df):,}   Accepted: {int(df[TARGET].sum())}   Base rate: {df[TARGET].mean():.1%}")
    print(f"Missing values: {int(df.isna().sum().sum())}   Duplicate rows (ignoring ID): {int(df.drop(columns='ID').duplicated().sum())}")
    print("\nRange of each input:")
    print(df[FEATURES].describe().T[["min", "50%", "max"]].rename(columns={"50%": "median"}).to_string())


def broken_values(df: pd.DataFrame) -> None:
    section("2. Values that cannot be right")
    neg = df.loc[df["Experience"] < 0, "Experience"]
    print(f"Negative Experience: {len(neg)} rows ({len(neg) / len(df):.1%})")
    print(neg.value_counts().sort_index().to_string())
    short_zip = df.loc[df["ZIP Code"] < 10000, "ZIP Code"]
    print(f"\nZIP codes with fewer than 5 digits: {len(short_zip)} {short_zip.tolist()}")
    print(f"Distinct ZIP codes: {df['ZIP Code'].nunique()}")


def redundancy(df: pd.DataFrame, redundant: float = 0.9, related: float = 0.5) -> None:
    """List input pairs in two tiers (decided 26 Oct).

    Redundant (|r| >= 0.9): one input restates the other, a label-free reason to drop one.
    Related (0.5 <= |r| < 0.9): shared variation worth knowing, not a reason to drop.
    r squared is the share of one input's variation the other explains.
    """
    section("3. Redundant and related inputs (feature-to-feature correlation, no outcome used)")
    corr = df[FEATURES].corr()
    pairs = [
        (a, b, corr.loc[a, b])
        for i, a in enumerate(FEATURES)
        for b in FEATURES[i + 1 :]
        if abs(corr.loc[a, b]) >= related
    ]
    tiers = [
        (f"Redundant, |r| >= {redundant} (may justify a drop):", [p for p in pairs if abs(p[2]) >= redundant]),
        (f"Related, {related} <= |r| < {redundant} (shared variation, not a drop reason):", [p for p in pairs if abs(p[2]) < redundant]),
    ]
    for heading, tier in tiers:
        print(heading)
        for a, b, r in tier:
            print(f"  {a} ~ {b}: r = {r:.3f}, r squared = {r * r:.2f}")
        if not tier:
            print("  none")
    gap = df["Age"] - df["Experience"]
    print(f"\nAge minus Experience: min {gap.min()}, median {gap.median():.0f}, max {gap.max()}")
    print("A narrow range means Experience is Age shifted by a near-constant.")


# --- Descriptive only: these do not choose features ----------------------


def acceptance_by_feature(df: pd.DataFrame) -> None:
    section("4. Acceptance rate by input (descriptive only)")
    for col in CATEGORICAL:
        table = df.groupby(col)[TARGET].agg(Rate="mean", Customers="size")
        print(f"\n{col}:\n{table.round(3).to_string()}")
    for col in CONTINUOUS:
        bands = pd.qcut(df[col], 5, duplicates="drop")
        table = df.groupby(bands, observed=True)[TARGET].agg(Rate="mean", Customers="size")
        print(f"\n{col} (quintile bands):\n{table.round(3).to_string()}")
    print(f"\nLowest income among acceptors: {df.loc[df[TARGET] == 1, 'Income'].min()}")


def cd_account_evidence(df: pd.DataFrame) -> None:
    section("5. CD Account: real signal or leakage?")
    table = pd.crosstab(df["CD Account"], df[TARGET])
    chi2, p, _, _ = chi2_contingency(table)
    print(f"Customers by CD Account (rows) and acceptance (columns):\n{table.to_string()}")
    print(f"Chi-square test of independence: chi2 = {chi2:.1f}, p = {p:.1e}")

    holders = table.loc[1]
    print(f"\nCD holders who did not accept: {holders[0]} of {holders.sum()}")
    print("If CDs were only opened with the loan, this would be near zero.")

    bands = pd.cut(df["Income"], INCOME_BANDS)
    within = df.groupby([bands, TARGET], observed=True)["CD Account"].agg(cd_rate="mean", customers="size")
    print(f"\nCD rate by income band and acceptance:\n{within.round(3).to_string()}")
    print("A gap within every band that has acceptors means income does not explain the association.")
    print("(The lowest band has no acceptors, so it cannot be compared.)")

    products = df.groupby("CD Account")[["Securities Account", "Online", "CreditCard"]].mean()
    print(f"\nOther products held, by CD Account:\n{products.round(3).to_string()}")
    print("\nNot answerable from this data: whether some CDs were opened after the loan.")
    print("Handled by refitting the chosen model without CD Account (Sensitivity).")


# --- Chart helpers ---------------------------------------------------------


def _new_figure(width: float = 7, height: float = 4.2, ncols: int = 1, sharey: bool = False) -> tuple[Figure, list[Axes]]:
    fig = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE)
    axes = list(fig.subplots(1, ncols, squeeze=False, sharey=sharey)[0])
    for ax in axes:
        ax.set_facecolor(SURFACE)
    return fig, axes


def _style_axes(ax: Axes, title: str = "", xlabel: str = "", ylabel: str = "", grid: str = "y") -> None:
    """Recessive axes: light grid, no box, secondary-ink labels."""
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=TEXT_PRIMARY, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)


def _legend(ax: Axes, loc: str = "upper left") -> None:
    legend = ax.legend(loc=loc, fontsize=8, frameon=False, markerscale=2)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)


def _save(fig: Figure, name: str) -> Path:
    fig.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, facecolor=SURFACE)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")
    return path


def _band_labels(intervals: pd.Index, lowest: float, decimals: int = 0) -> list[str]:
    """Readable labels for qcut bands; the first band starts at the data minimum."""
    if decimals == 0:
        labels = [f"{int(b.left) + 1}–{int(b.right)}" for b in intervals]
        labels[0] = f"{int(lowest)}–{int(intervals[0].right)}"
    else:
        labels = [f"{b.left:.{decimals}f}–{b.right:.{decimals}f}" for b in intervals]
        labels[0] = f"{lowest:.{decimals}f}–{intervals[0].right:.{decimals}f}"
    return labels


def _rate_bars(ax: Axes, labels: list[str], rates: pd.Series, base: float, label_base: bool = True) -> None:
    """Acceptance-rate bars with the base rate behind them and a value on each bar."""
    bars = ax.bar(labels, rates.values * 100, color=SERIES, width=0.6, zorder=2)
    for bar, rate in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{rate:.1%}",
                ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY, zorder=3,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.axhline(base * 100, color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=1)
    if label_base:
        ax.text(-0.3, base * 100 + 0.8, f"Base rate {base:.1%}", fontsize=8, color=TEXT_SECONDARY, va="bottom")


# --- Charts: label-free ------------------------------------------------------


def plot_correlation(df: pd.DataFrame) -> Path:
    """Heatmap of input-to-input correlations. ZIP Code is left out: it is a code, not a quantity."""
    cols = [c for c in FEATURES if c != "ZIP Code"]
    corr = df[cols].corr()
    # Lower triangle only: each pair appears once, and the empty top row and right column are trimmed.
    lower = corr.where(np.tril(np.ones(corr.shape, dtype=bool), k=-1)).iloc[1:, :-1]
    cmap = LinearSegmentedColormap.from_list("diverging", [NOT_ACCEPTED, NEUTRAL_MID, SERIES])

    fig, (ax,) = _new_figure(7.5, 6.2)
    image = ax.imshow(lower.values, cmap=cmap, vmin=-1, vmax=1)
    for i in range(lower.shape[0]):
        for j in range(lower.shape[1]):
            r = lower.iat[i, j]
            if not np.isnan(r):
                ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=7,
                        color=SURFACE if abs(r) > 0.6 else TEXT_PRIMARY)
    ax.set_xticks(range(lower.shape[1]), lower.columns, rotation=45, ha="right")
    ax.set_yticks(range(lower.shape[0]), lower.index)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Correlation between inputs (no outcome used)", loc="left", fontsize=11, color=TEXT_PRIMARY, pad=10)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.7)
    colorbar.set_label("Pearson r", fontsize=8, color=TEXT_SECONDARY)
    colorbar.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY, length=0)
    colorbar.outline.set_visible(False)
    return _save(fig, "eda_correlation.png")


def plot_age_experience(df: pd.DataFrame) -> Path:
    """Scatter showing Experience is Age shifted by a near-constant, with the negative values marked."""
    neg = df["Experience"] < 0
    r = df["Age"].corr(df["Experience"])
    gap = df["Age"] - df["Experience"]

    fig, (ax,) = _new_figure()
    ax.scatter(df.loc[~neg, "Age"], df.loc[~neg, "Experience"], s=6, color=SERIES, alpha=0.35,
               linewidths=0, label=f"Valid ({(~neg).sum():,})")
    ax.scatter(df.loc[neg, "Age"], df.loc[neg, "Experience"], s=18, color=HIGHLIGHT,
               linewidths=0, label=f"Negative Experience ({neg.sum()})", zorder=3)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8, zorder=1)
    ax.annotate(f"{neg.sum()} impossible values", xy=(df.loc[neg, "Age"].max(), -1),
                xytext=(32, -8), fontsize=8, color=TEXT_PRIMARY,
                arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "linewidth": 0.8})
    _style_axes(ax, f"Experience is Age minus {gap.min()} to {gap.max()} years (r = {r:.3f})",
                "Age (years)", "Experience (years)", grid="both")
    _legend(ax)
    return _save(fig, "eda_age_experience.png")


def plot_distributions(df: pd.DataFrame) -> Path:
    """Histograms of the three skewed inputs."""
    panels = [("Income", "Annual income ($000s)"), ("CCAvg", "Card spend per month ($000s)"), ("Mortgage", "Mortgage ($000s)")]
    fig, axes = _new_figure(10, 3.4, ncols=3)
    for ax, (col, xlabel) in zip(axes, panels):
        ax.hist(df[col], bins=30, color=SERIES, edgecolor=SURFACE, linewidth=0.6, zorder=2)
        _style_axes(ax, col, xlabel, "Customers" if col == "Income" else "")
    zero_share = (df["Mortgage"] == 0).mean()
    axes[2].text(0.97, 0.95, f"{zero_share:.0%} have no mortgage", transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=8, color=TEXT_PRIMARY)
    fig.suptitle("Distribution of the skewed inputs", x=0.01, ha="left", fontsize=11, color=TEXT_PRIMARY)
    return _save(fig, "eda_distributions.png")


# --- Charts: descriptive only ------------------------------------------------


def plot_income_acceptance(df: pd.DataFrame) -> Path:
    """Bar chart of acceptance rate by income quintile, with the base rate marked."""
    bands = pd.qcut(df["Income"], 5)
    rates = df.groupby(bands, observed=True)[TARGET].mean()
    labels = _band_labels(rates.index, df["Income"].min())
    base = df[TARGET].mean()

    fig, (ax,) = _new_figure()
    _rate_bars(ax, labels, rates, base)
    _style_axes(ax, xlabel="Annual income ($000s)", ylabel="Accepted the loan offer (%)")
    ax.set_title("Acceptance rate by income quintile", loc="left", fontsize=12, color=TEXT_PRIMARY, pad=12)
    ax.set_ylim(0, max(rates.values) * 100 * 1.15)
    return _save(fig, "eda_income_acceptance.png")


def plot_income_ccavg(df: pd.DataFrame) -> Path:
    """Scatter of the two strongest signals, coloured by whether the customer accepted."""
    accepted = df[TARGET] == 1
    r = df["Income"].corr(df["CCAvg"])
    floor = df.loc[accepted, "Income"].min()

    fig, (ax,) = _new_figure(7, 4.6)
    ax.scatter(df.loc[~accepted, "Income"], df.loc[~accepted, "CCAvg"], s=6, color=NOT_ACCEPTED, alpha=0.3,
               linewidths=0, label=f"Did not accept ({(~accepted).sum():,})")
    ax.scatter(df.loc[accepted, "Income"], df.loc[accepted, "CCAvg"], s=9, color=ACCEPTED, alpha=0.8,
               linewidths=0, label=f"Accepted ({accepted.sum()})", zorder=3)
    ax.axvline(floor, color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=1)
    ax.text(floor + 2, df["CCAvg"].max(), f"No acceptor below {floor}", fontsize=8, color=TEXT_SECONDARY, va="top")
    _style_axes(ax, "", "Annual income ($000s)", "Card spend per month ($000s)", grid="both")
    ax.set_title(f"Income against card spend, by offer outcome (r = {r:.2f})", loc="left",
                 fontsize=10, color=TEXT_PRIMARY, pad=24)
    # Legend in a row above the plot: every corner of the plot area holds data.
    legend = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, fontsize=8, frameon=False, markerscale=2)
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)
    return _save(fig, "eda_income_ccavg.png")


def plot_rates(df: pd.DataFrame) -> Path:
    """Acceptance rate by card-spend quintile, education and family size, on one shared scale."""
    base = df[TARGET].mean()
    ccavg = df.groupby(pd.qcut(df["CCAvg"], 5), observed=True)[TARGET].mean()
    education = df.groupby("Education")[TARGET].mean()
    family = df.groupby("Family")[TARGET].mean()
    panels = [
        (_band_labels(ccavg.index, df["CCAvg"].min(), decimals=1), ccavg, "Card spend quintile", "Card spend per month ($000s)"),
        ([EDUCATION_LABELS[k] for k in education.index], education, "Education", ""),
        ([str(k) for k in family.index], family, "Family size", "People in household"),
    ]

    fig, axes = _new_figure(10.5, 3.8, ncols=3, sharey=True)
    for i, (ax, (labels, rates, title, xlabel)) in enumerate(zip(axes, panels)):
        _rate_bars(ax, labels, rates, base, label_base=(i == 0))
        _style_axes(ax, title, xlabel, "Accepted the loan offer (%)" if i == 0 else "")
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylim(0, max(ccavg.max(), education.max(), family.max()) * 100 * 1.15)
    fig.suptitle("Acceptance rate by input, on a shared scale", x=0.01, ha="left", fontsize=11, color=TEXT_PRIMARY)
    return _save(fig, "eda_rates.png")


def plot_cd_by_income(df: pd.DataFrame) -> Path:
    """Share holding a CD Account, acceptors against non-acceptors, within income bands."""
    bands = pd.cut(df["Income"], INCOME_BANDS)
    rates = df.groupby([bands, TARGET], observed=False)["CD Account"].mean().unstack()
    edges = INCOME_BANDS.copy()
    edges[0], edges[-1] = int(df["Income"].min()) - 1, int(df["Income"].max())
    labels = [f"{lo + 1}–{hi}" for lo, hi in pairwise(edges)]

    x = np.arange(len(labels))
    width = 0.36
    fig, (ax,) = _new_figure()
    for offset, outcome, colour, name in [(-1, 0, NOT_ACCEPTED, "Did not accept"), (1, 1, ACCEPTED, "Accepted")]:
        positions = x + offset * (width / 2 + 0.01)  # small gap between the paired bars
        values = rates[outcome].values * 100
        ax.bar(positions, np.nan_to_num(values), width, color=colour, label=name, zorder=2)
        for pos, value in zip(positions, values):
            text = "none" if np.isnan(value) else f"{value:.0f}%"
            ax.text(pos, (0 if np.isnan(value) else value) + 0.8, text, ha="center", va="bottom",
                    fontsize=8, color=TEXT_PRIMARY if not np.isnan(value) else TEXT_SECONDARY)
    ax.set_xticks(x, labels)
    _style_axes(ax, "Share holding a CD Account, within income bands", "Annual income ($000s)", "Hold a CD Account (%)")
    ax.set_ylim(0, np.nanmax(rates.values) * 100 * 1.2)
    _legend(ax)
    return _save(fig, "eda_cd_by_income.png")


def main() -> None:
    df = load_raw()
    overview(df)
    broken_values(df)
    redundancy(df)
    acceptance_by_feature(df)
    cd_account_evidence(df)
    section("6. Charts")
    plot_correlation(df)
    plot_age_experience(df)
    plot_distributions(df)
    plot_income_acceptance(df)
    plot_income_ccavg(df)
    plot_rates(df)
    plot_cd_by_income(df)


if __name__ == "__main__":
    main()
