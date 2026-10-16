"""
Empirical-Bayes (shrunken) per-team Elite-vs-Weak effects, instead of the raw
20%-swing typology.

The typology in style_profiler.py compares each team's Elite-tier mean with its
Weak-tier mean on whatever handful of matches it has. That is unstable on 3-5
matches, and it cannot say anything at all about teams that never met both an
Elite and a Weak opponent (22 of 76 are labelled "Insufficient data").

This extends scripts/mixed_effects_pressing.py from a random intercept to a
random intercept + random slope on opponent tier, per team:

    outcome ~ tier_score + (1 + tier_score | team)

where tier_score is Weak=0, Moderate=1, Strong=2, Elite=3 and the outcome is
log(ppda) or def_line_x. Using one numeric slope (not three tier dummies) keeps
the random-effects covariance at 2x2, which is about all that 1-13 matches per
team can support. The cost is a linearity assumption: the effect of moving up
one tier is the same at each step. The pooled coefficients in the README
(0.149, 0.300, 0.404 for log PPDA) are close to linear, so this is reasonable
but is an assumption, not a result.

A team's Elite-minus-Weak effect is 3 * (fixed slope + team's random slope).
The random slope is the model's posterior mean (the empirical-Bayes estimate),
which pulls teams with little or noisy data toward the population slope. Its
uncertainty is the conditional (posterior) variance of the random slope plus
the sampling variance of the fixed slope; the covariance between the two is
ignored, so the intervals are approximate.

Teams that never played an Elite or a Weak opponent still get a shrunken
estimate (from whatever tiers they did play, plus the population slope); that
is the point of the comparison. Their raw swing is simply undefined.

Writes results/team_shrinkage.csv and prints a comparison against the raw
per-team swing used by the existing typology.

    python scripts/team_shrinkage.py [--results results]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2

TIER_SCORE = {"Weak": 0, "Moderate": 1, "Strong": 2, "Elite": 3}
SPAN = 3  # Elite minus Weak is three tier steps


def fit_random_slope(df, outcome):
    model = smf.mixedlm(f"{outcome} ~ tier_score", data=df, groups=df["team"],
                        re_formula="~tier_score")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(reml=True, method=["lbfgs"])
    return result


def shrunken_effects(result, teams):
    """Per-team Elite-minus-Weak effect: 3 * (fixed slope + posterior-mean random slope)."""
    beta = result.fe_params["tier_score"]
    var_beta = result.cov_params().loc["tier_score", "tier_score"]
    re = result.random_effects
    re_cov = result.random_effects_cov
    rows = []
    for team in teams:
        b = re[team]["tier_score"]
        v_b = re_cov[team].loc["tier_score", "tier_score"]
        est = SPAN * (beta + b)
        se = SPAN * np.sqrt(v_b + var_beta)
        rows.append({"team": team, "shrunk": est, "shrunk_se": se,
                     "shrunk_lo": est - 1.96 * se, "shrunk_hi": est + 1.96 * se})
    return pd.DataFrame(rows).set_index("team"), SPAN * beta, SPAN * np.sqrt(var_beta)


def raw_swing(df, outcome):
    g = df.groupby(["team", "opponent_tier"], observed=True)[outcome].mean().unstack()
    swing = g["Elite"] - g["Weak"]  # NaN when a team has no Elite or no Weak match
    n_el = df[df["opponent_tier"] == "Elite"].groupby("team").size()
    n_wk = df[df["opponent_tier"] == "Weak"].groupby("team").size()
    return pd.DataFrame({"raw": swing, "n_elite": n_el, "n_weak": n_wk})


def summarise(name, table, typed, unit):
    both = table[table["raw"].notna()]
    ins = table[~table.index.isin(typed)]
    print(f"\n--- {name} ({unit}) ---")
    print(f"Teams with a raw Elite-vs-Weak swing: {len(both)}; without: {table['raw'].isna().sum()}")
    r = np.corrcoef(both["raw"], both["shrunk"])[0, 1]
    print(f"Raw vs shrunken, teams with a raw swing: correlation {r:.2f}; "
          f"SD raw {both['raw'].std():.3f} -> SD shrunken {both['shrunk'].std():.3f} "
          f"(mean shrunken {both['shrunk'].mean():+.3f}, raw {both['raw'].mean():+.3f})")
    sign_flip = (np.sign(both["raw"]) != np.sign(both["shrunk"])).sum()
    print(f"Sign changes after shrinkage: {sign_flip} of {len(both)}")
    ex0 = ((table["shrunk_lo"] > 0) | (table["shrunk_hi"] < 0))
    print(f"Teams whose shrunken 95% interval excludes 0: {ex0.sum()} of {len(table)}")
    print(f"Median shrunken SE, all teams {table['shrunk_se'].median():.3f}")
    print(f"'Insufficient data' teams (no combined typology), n={len(ins)}: "
          f"mean shrunken {ins['shrunk'].mean():+.3f}, SD {ins['shrunk'].std():.3f}, "
          f"intervals excluding 0: {int(((ins['shrunk_lo'] > 0) | (ins['shrunk_hi'] < 0)).sum())}, "
          f"median SE {ins['shrunk_se'].median():.3f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    results = Path(a.results)

    tm = pd.read_csv(results / "team_matches.csv").replace([np.inf, -np.inf], np.nan)
    tm = tm.dropna(subset=["ppda", "def_line_x", "opponent_tier"]).copy()
    tm["tier_score"] = tm["opponent_tier"].map(TIER_SCORE)
    tm["log_ppda"] = np.log(tm["ppda"])
    teams = sorted(tm["team"].unique())
    prof = pd.read_csv(results / "combined_profiles.csv")
    typed = set(prof.loc[~prof["combined_typology"].str.startswith("Insufficient"), "team"])
    print(f"n = {len(tm)} team-matches, {len(teams)} teams; {len(typed)} have a raw combined typology, "
          f"{len(teams) - len(typed)} are 'Insufficient data'")

    out = pd.DataFrame(index=teams)
    out.index.name = "team"
    out["n_matches"] = tm.groupby("team").size()
    out["has_raw_typology"] = [t in typed for t in teams]
    tables = {}
    for outcome, label, unit in [("log_ppda", "ppda", "log PPDA units; positive = presses less vs Elite"),
                                 ("def_line_x", "line", "pitch units 0-100; negative = drops deeper vs Elite")]:
        res = fit_random_slope(tm, outcome)
        print(f"\n=== {outcome} ~ tier_score + (1 + tier_score | team): converged={res.converged} ===")
        print(f"Fixed slope per tier step {res.fe_params['tier_score']:.3f} "
              f"(Elite-minus-Weak {SPAN * res.fe_params['tier_score']:+.3f}); "
              f"random-slope SD {np.sqrt(res.cov_re.loc['tier_score', 'tier_score']):.3f}; "
              f"random-intercept SD {np.sqrt(res.cov_re.loc['Group', 'Group']):.3f}; "
              f"residual SD {np.sqrt(res.scale):.3f}")
        # Is there real between-team variation in the slope at all? Compare against a
        # random-intercept-only model (ML fits). The variance sits on a boundary under the
        # null, so the chi2(2) p-value is conservative.
        m_full = smf.mixedlm(f"{outcome} ~ tier_score", data=tm, groups=tm["team"],
                             re_formula="~tier_score")
        m_null = smf.mixedlm(f"{outcome} ~ tier_score", data=tm, groups=tm["team"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ll_full = m_full.fit(reml=False, method=["lbfgs"]).llf
            ll_null = m_null.fit(reml=False).llf
        lr = 2 * (ll_full - ll_null)
        print(f"LRT, random slope vs intercept-only: chi2={lr:.2f}, df=2, p={chi2.sf(lr, 2):.3f} "
              f"(conservative: variance is on a boundary under the null)")
        shr, pop, pop_se = shrunken_effects(res, teams)
        raw = raw_swing(tm, outcome)
        table = shr.join(raw)
        tables[label] = table
        for c in ["raw", "n_elite", "n_weak", "shrunk", "shrunk_se", "shrunk_lo", "shrunk_hi"]:
            out[f"{label}_{c}"] = table[c]
        summarise(outcome, table, typed, unit)

    out.round(4).to_csv(results / "team_shrinkage.csv")
    print(f"\nSaved {results / 'team_shrinkage.csv'}")
    print("\nReading: a shrunken estimate sits between the team's own raw swing and the population "
          "Elite-minus-Weak effect, closer to the population value the less data the team has. "
          "Teams without both an Elite and a Weak match get a model-based estimate with a wide "
          "interval; it should be read as 'consistent with the population effect' rather than as a "
          "team-specific finding.")


if __name__ == "__main__":
    main()
