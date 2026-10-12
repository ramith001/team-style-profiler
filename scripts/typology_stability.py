"""
How much of the combined typology is signal? Compares the real Elite-vs-Weak
direction labels against a null in which each team's opponent-tier labels are
shuffled across its own matches (same 20% swing rule, same tier counts).

    python scripts/typology_stability.py [--results results] [--draws 2000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style_profiler import build_team_profile, _direction


def labels(df, team, metric):
    p = build_team_profile(df, team, metric)
    return _direction(p)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    tm = pd.read_csv(Path(a.results) / "team_matches.csv").replace([np.inf, -np.inf], np.nan)
    rng = np.random.default_rng(0)

    for metric in ["ppda", "def_line_x"]:
        teams = [t for t in sorted(tm["team"].unique())
                 if {"Elite", "Weak"} <= set(tm[tm.team == t].dropna(subset=[metric])["opponent_tier"])]
        real = pd.Series({t: labels(tm, t, metric) for t in teams})
        real_share = (real != "consistent").mean()
        null_shares = []
        for _ in range(a.draws):
            sh = tm.copy()
            sh["opponent_tier"] = sh.groupby("team")["opponent_tier"].transform(lambda s: rng.permutation(s.values))
            null_shares.append(np.mean([labels(sh, t, metric) != "consistent" for t in teams]))
        null_shares = np.array(null_shares)
        print(f"{metric}: {len(teams)} teams with Elite and Weak matches; "
              f"{real_share:.0%} labelled a shift (real) vs {null_shares.mean():.0%} expected from "
              f"shuffled tiers (95th pct {np.percentile(null_shares, 95):.0%})")


if __name__ == "__main__":
    main()
