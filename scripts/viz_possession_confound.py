"""
Before/after dot-and-whisker for the possession-volume confound.

Refits the same two models as scripts/possession_confound.py,
log(ppda) ~ tier + (1 | team) with and without + log(opp_passes), and draws
each tier's PPDA coefficient (shift relative to Weak-tier opponents) with its
95% CI, paired per tier: before controlling for opponent pass volume vs.
after. The point of the picture is the one the README's prose already makes:
the Moderate and Strong effects barely move, while the Elite-tier effect
collapses toward zero, so most of it was the opponent simply having the ball
more rather than the pressing team doing less.

Reads results/ppda_components.csv (written by possession_confound.py) and
results/team_matches.csv, joined the same way possession_confound.py does.

    python scripts/viz_possession_confound.py [--results results] [--out results/figures]
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
# 2 (orange) - adjacent pair, validated CVD-safe. Here blue = before the
# opponent-pass-volume control, orange = after.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def fit(df, extra_terms=""):
    formula = f"log_ppda ~ C(opponent_tier, Treatment('Weak'))" + extra_terms
    model = smf.mixedlm(formula, data=df, groups=df["team"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(reml=True)

    ci_lo, ci_hi = result.conf_int()[0], result.conf_int()[1]
    rows = []
    for name, coef, p, lo, hi in zip(result.params.index, result.params, result.pvalues, ci_lo, ci_hi):
        if "T." not in name:  # skip Intercept, Group Var, and the log_opp_passes covariate
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    results_dir, out_dir = Path(a.results), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    comp = pd.read_csv(results_dir / "ppda_components.csv")
    tm = pd.read_csv(results_dir / "team_matches.csv")[["match_id", "team", "opponent_tier"]]
    df = comp.merge(tm, on=["match_id", "team"], how="inner")
    df = df[df["press_actions"] > 0].dropna(subset=["opponent_tier"]).copy()
    df["opponent_tier"] = pd.Categorical(df["opponent_tier"], categories=TIERS, ordered=True)
    df["log_ppda"] = np.log(df["ppda"])
    df["log_opp_passes"] = np.log(df["opp_passes"].clip(lower=1))
    print(f"n = {len(df)} team-matches, {df['team'].nunique()} teams")

    before = fit(df)
    after = fit(df, " + log_opp_passes")

    fig, ax = plt.subplots(figsize=(9.5, 5), facecolor=SURFACE)
    y = np.arange(len(NON_REF_TIERS))[::-1]  # Moderate at top, Elite at bottom
    dy = 0.17
    series = [(before, BLUE, dy, "Before: no covariate"),
              (after, ORANGE, -dy, "After: controlling for opponent pass volume")]

    for coefs, color, offset, label in series:
        for i, (yi, tier) in enumerate(zip(y, NON_REF_TIERS)):
            row = coefs.loc[tier]
            ypos = yi + offset
            ax.plot([row["ci_lo"], row["ci_hi"]], [ypos, ypos], color=color, lw=2,
                    solid_capstyle="round", zorder=2)
            ax.plot(row["coef"], ypos, "o", color=color, ms=9, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=label if i == 0 else None)
            text = f"{row['coef']:+.3f} {sig_marker(row['p'])}".strip()
            ax.text(row["ci_hi"] + 0.012, ypos, text, va="center", ha="left",
                    fontsize=10, color=INK)

    ax.axvline(0, color=BASELINE, lw=1.2, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t} vs. Weak" for t in NON_REF_TIERS], fontsize=11, color=INK)
    ax.set_xlabel("Shift in log(PPDA) vs. Weak-tier opponents", fontsize=10, color=SECONDARY_INK)
    ax.set_title("Controlling for opponent pass volume collapses the Elite-tier PPDA effect",
                 fontsize=13, color=INK, loc="left", pad=14, fontweight="bold")
    ax.set_ylim(-0.7, len(NON_REF_TIERS) - 0.3)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.12)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRIDLINE, lw=0.8, zorder=0)
    ax.set_facecolor(SURFACE)
    ax.legend(loc="lower right", fontsize=9.5, frameon=False, labelcolor=INK)

    fig.text(0.01, -0.02,
              f"Dot = coefficient, line = 95% CI, team fit as a random intercept (n = {len(df)} team-matches). "
              "* p<.05  ** p<.01  *** p<.001",
              fontsize=9, color=MUTED, ha="left")
    fig.text(0.01, -0.07,
              "Opponent pass volume = the opponent's completed passes in their own defensive two-thirds, "
              "the numerator of PPDA; see scripts/possession_confound.py.",
              fontsize=9, color=MUTED, ha="left", style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path = out_dir / "possession_confound.png"
    fig.savefig(out_path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")
    print("\nlog(PPDA) ~ tier, no covariate:\n", before)
    print("\nlog(PPDA) ~ tier + log(opp_passes):\n", after)


if __name__ == "__main__":
    main()
