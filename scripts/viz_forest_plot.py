"""
Forest plot of the pooled mixed-effects tier coefficients.

Refits the same two models as scripts/mixed_effects_pressing.py
(log(ppda) ~ tier + (1 | team), def_line_x ~ tier + (1 | team)) and draws
each tier's coefficient (shift relative to Weak-tier opponents) with its
95% CI as a dot-and-whisker, one panel per outcome. The point of the picture
is the one the README's prose already makes: the Weak->Elite climb is a
real, monotonic, population-level shift, not an artifact of a few teams.

    python scripts/viz_forest_plot.py [--results results] [--out results/figures]
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

TIERS = ["Weak", "Moderate", "Strong", "Elite"]
NON_REF_TIERS = ["Moderate", "Strong", "Elite"]

# dataviz skill palette (references/palette.md): categorical slots 1 (blue) and
# 2 (orange) - adjacent pair, validated CVD-safe, used consistently for the
# PPDA vs. defensive-line-height metrics across every figure in this project.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def fit(df, outcome):
    formula = f"{outcome} ~ C(opponent_tier, Treatment('Weak'))"
    model = smf.mixedlm(formula, data=df, groups=df["team"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(reml=True)

    ci_lo, ci_hi = result.conf_int()[0], result.conf_int()[1]
    rows = []
    for name, coef, p, lo, hi in zip(result.params.index, result.params, result.pvalues, ci_lo, ci_hi):
        if name in ("Group Var", "Intercept"):
            continue
        tier = name.split("T.")[-1].rstrip("]")
        rows.append({"tier": tier, "coef": coef, "p": p, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).set_index("tier").loc[NON_REF_TIERS]


def sig_marker(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def draw_panel(ax, coefs, color, title, xlabel, value_fmt):
    y = np.arange(len(NON_REF_TIERS))[::-1]  # Moderate at top, Elite at bottom
    for yi, tier in zip(y, NON_REF_TIERS):
        row = coefs.loc[tier]
        ax.plot([row["ci_lo"], row["ci_hi"]], [yi, yi], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.plot(row["coef"], yi, "o", color=color, ms=9, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        label = f"{value_fmt(row['coef'])} {sig_marker(row['p'])}".strip()
        offset = (row["ci_hi"] - row["ci_lo"]) * 0.12 + 0.003
        ax.text(row["ci_hi"] + offset, yi, label, va="center", ha="left",
                 fontsize=10, color=INK)

    ax.axvline(0, color=BASELINE, lw=1.2, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t} vs. Weak" for t in NON_REF_TIERS], fontsize=11, color=INK)
    ax.set_xlabel(xlabel, fontsize=10, color=SECONDARY_INK)
    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=12, fontweight="bold")
    ax.set_ylim(-0.7, len(NON_REF_TIERS) - 0.3)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRIDLINE, lw=0.8, zorder=0)
    ax.set_facecolor(SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    results_dir, out_dir = Path(a.results), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tm = pd.read_csv(results_dir / "team_matches.csv").replace([np.inf, -np.inf], np.nan)
    tm = tm.dropna(subset=["ppda", "def_line_x", "opponent_tier"]).copy()
    tm["opponent_tier"] = pd.Categorical(tm["opponent_tier"], categories=TIERS, ordered=True)
    tm["log_ppda"] = np.log(tm["ppda"])

    ppda_coefs = fit(tm, "log_ppda")
    line_coefs = fit(tm, "def_line_x")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    draw_panel(axes[0], ppda_coefs, BLUE, "PPDA (log)", "Shift in log(PPDA) vs. Weak-tier opponents",
               lambda v: f"{v:+.2f}")
    draw_panel(axes[1], line_coefs, ORANGE, "Defensive line height", "Shift in line height (0-100 scale) vs. Weak",
               lambda v: f"{v:+.1f}")

    fig.suptitle("Pooled mixed-effects tier coefficients (n = 628 team-matches, 76 teams, 6 tournaments)",
                 fontsize=12, color=INK, y=1.03, x=0.01, ha="left")
    fig.text(0.01, -0.02,
              "Dot = coefficient, line = 95% CI, team fit as a random intercept. * p<.05  ** p<.01  *** p<.001",
              fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out_path = out_dir / "forest_plot_tier_effects.png"
    fig.savefig(out_path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")
    print("\nPPDA (log) coefficients:\n", ppda_coefs)
    print("\nDefensive line height coefficients:\n", line_coefs)


if __name__ == "__main__":
    main()
