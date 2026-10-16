"""
Host-nation sensitivity check for the pooled tier effects.

A host plays on home soil with home support, which could move both how it
presses and how high its line sits, and hosts are not randomly spread across
opponent tiers. This refits the models from scripts/mixed_effects_pressing.py
three ways and compares the tier coefficients:

  1. original: log(ppda) or def_line_x ~ tier + (1 | team)
  2. with a host_match covariate (1 when the team is that tournament's host)
  3. excluding every host match

Host mapping (one host team per tournament):
  wc2022 -> Qatar, wc2018 -> Russia, euro2024 -> Germany,
  afcon2023 -> Cote d'Ivoire, copa2024 -> United States.

Copa America 2024 was hosted by the United States, which also took part as a
participating team (it is in team_matches.csv), so it is included.

Euro 2020 is excluded from the host flag entirely: its matches were spread
across 11 countries, so there is no single host nation. England played the
semi-finals and final at Wembley, a partial home effect, but identifying those
matches needs venue data that team_matches.csv does not carry and the cached
event data does not guarantee, so rather than guess it is left unflagged (and
therefore stays in all three models as a non-host tournament).

The host sample is small, roughly one team's worth of matches per included
tournament, so this can rule out a large host confound but cannot rule out a
subtle one. It is a sensitivity check, not a definitive test.

    python scripts/host_sensitivity.py [--results results]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

TIERS = ["Weak", "Moderate", "Strong", "Elite"]
HOSTS = {"wc2022": "Qatar", "wc2018": "Russia", "euro2024": "Germany",
         "afcon2023": "Ivoire", "copa2024": "United States"}  # "Ivoire": matched by substring (Cote d'Ivoire)


def flag_hosts(tm):
    tm = tm.copy()
    tm["host_match"] = 0
    for comp, host in HOSTS.items():
        mask = (tm["competition"] == comp) & tm["team"].str.contains(host, regex=False)
        if mask.sum() == 0:
            raise ValueError(f"host {host!r} not found in {comp}")
        tm.loc[mask, "host_match"] = 1
    return tm


def fit(df, outcome, extra=""):
    formula = f"{outcome} ~ C(opponent_tier, Treatment('Weak'))" + extra
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(formula, data=df, groups=df["team"]).fit(reml=True)


def coef_rows(res):
    out = {}
    for name in res.params.index:
        if "T." in name:
            tier = name.split("T.")[-1].rstrip("]")
            out[tier] = (res.params[name], res.pvalues[name])
        elif name == "host_match":
            out["host_match"] = (res.params[name], res.pvalues[name])
    return out


def fmt_p(p):
    return "p<0.0001" if p < 0.0001 else f"p={p:.4f}"


def report(df, outcome, label):
    models = {
        "original": fit(df, outcome),
        "+ host covariate": fit(df, outcome, " + host_match"),
        "excluding hosts": fit(df[df["host_match"] == 0], outcome),
    }
    print(f"\n=== {label} ===")
    print(f"{'':<8}" + "".join(f"{k:>30}" for k in models))
    rows = {k: coef_rows(v) for k, v in models.items()}
    for term in ["Moderate", "Strong", "Elite", "host_match"]:
        cells = []
        for k in models:
            c = rows[k].get(term)
            cells.append(f"{c[0]:+.3f} ({fmt_p(c[1])})".rjust(30) if c else "-".rjust(30))
        print(f"{term:<8}" + "".join(cells))
    print(f"n: original {len(df)}, excluding hosts {int((df['host_match'] == 0).sum())}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    tm = pd.read_csv(Path(a.results) / "team_matches.csv").replace([np.inf, -np.inf], np.nan)
    tm = tm.dropna(subset=["ppda", "def_line_x", "opponent_tier"]).copy()
    tm["opponent_tier"] = pd.Categorical(tm["opponent_tier"], categories=TIERS, ordered=True)
    tm["log_ppda"] = np.log(tm["ppda"])
    tm = flag_hosts(tm)

    print("Host matches flagged (Euro 2020 has no single host and is left unflagged):")
    print(tm[tm["host_match"] == 1].groupby(["competition", "team"]).size().to_string())
    print(f"Total host team-matches: {int(tm['host_match'].sum())} of {len(tm)}")
    print("Host matches by opponent tier:", tm[tm["host_match"] == 1]["opponent_tier"].value_counts()
          .reindex(TIERS).to_dict())
    print("WARNING: the host sample is small (about one team per included tournament). This check can "
          "rule out a large host confound but not a subtle one; it is not a definitive test.")

    report(tm, "log_ppda", "log(PPDA) ~ tier + (1 | team)")
    report(tm, "def_line_x", "Defensive line height ~ tier + (1 | team)")


if __name__ == "__main__":
    main()
