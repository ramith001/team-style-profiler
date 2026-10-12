"""
Pooled mixed-effects model: is there a real population-level pressing effect
by opponent strength, once repeated observations per team are accounted for?

The per-team typology in style_profiler.py treats each team as its own
20%-swing test on 3-13 matches, which scripts/typology_stability.py showed is
mostly indistinguishable from a shuffled-tier null. This script asks the
different, pooled question: across all 294 team-matches, does PPDA (and
defensive line height) shift with opponent tier at the population level,
with team fit as a random intercept so that teams with more matches don't
dominate and each team's own baseline is absorbed rather than re-estimated
by hand per team.

Model: log(ppda) ~ C(opponent_tier, Treatment('Weak')) + (1 | team)
       def_line_x ~ C(opponent_tier, Treatment('Weak')) + (1 | team)

    python scripts/mixed_effects_pressing.py [--results results]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

TIERS = ["Weak", "Moderate", "Strong", "Elite"]


def fit_and_report(df, outcome, label):
    formula = f"{outcome} ~ C(opponent_tier, Treatment('Weak'))"
    model = smf.mixedlm(formula, data=df, groups=df["team"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(reml=True)
        # ML fits (not REML) so the full vs. null log-likelihoods are comparable for the LRT
        result_ml = model.fit(reml=False)
        null_ml = smf.mixedlm(f"{outcome} ~ 1", data=df, groups=df["team"]).fit(reml=False)

    print(f"\n=== {label} ===")
    print(f"n = {len(df)} team-matches, {df['team'].nunique()} teams, converged={result.converged}")

    group_var = result.cov_re.iloc[0, 0]
    resid_var = result.scale
    icc = group_var / (group_var + resid_var)
    print(f"Between-team variance: {group_var:.4f}  |  Residual variance: {resid_var:.4f}  |  ICC: {icc:.2f}")
    print("(ICC = share of variance that is stable per-team identity rather than match-to-match noise)")

    lr_stat = 2 * (result_ml.llf - null_ml.llf)
    df_diff = 3  # Moderate, Strong, Elite vs. Weak
    p_lrt = chi2.sf(lr_stat, df_diff)
    print(f"Likelihood-ratio test (tier vs. team-only null): chi2={lr_stat:.2f}, df={df_diff}, p={p_lrt:.2e}")

    print(f"\n{'tier':<10} {'coef':>8} {'se':>7} {'z':>7} {'p':>8} {'95% CI':>20}")
    for name, coef, se, z, p, ci_lo, ci_hi in zip(
        result.params.index, result.params, result.bse, result.tvalues, result.pvalues,
        result.conf_int()[0], result.conf_int()[1],
    ):
        if name == "Group Var":
            continue
        tier = name.split("T.")[-1].rstrip("]") if "T." in name else name
        print(f"{tier:<10} {coef:>8.3f} {se:>7.3f} {z:>7.2f} {p:>8.4f} [{ci_lo:>7.3f}, {ci_hi:>7.3f}]")

    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()

    tm = pd.read_csv(Path(a.results) / "team_matches.csv").replace([np.inf, -np.inf], np.nan)
    tm = tm.dropna(subset=["ppda", "def_line_x", "opponent_tier"]).copy()
    tm["opponent_tier"] = pd.Categorical(tm["opponent_tier"], categories=TIERS, ordered=True)
    tm["log_ppda"] = np.log(tm["ppda"])

    fit_and_report(tm, "log_ppda", "PPDA (log), tier fixed effect, team random intercept")
    fit_and_report(tm, "def_line_x", "Defensive line height, tier fixed effect, team random intercept")

    print(
        "\nReading: a coefficient here is the average, population-level shift relative to Weak-tier "
        "opponents, after a per-team baseline (the random intercept) soaks up each team's own general "
        "level. This is a different test from the per-team 20%-swing typology: it asks whether pressing "
        "or line height moves with opponent strength on average across all 52 teams, not whether any "
        "single team's handful of matches cleared a threshold."
    )


if __name__ == "__main__":
    main()
