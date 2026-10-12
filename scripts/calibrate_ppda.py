"""
Calibrate and validate the PPDA classification thresholds in tactical_engine.py.

    python scripts/calibrate_ppda.py [--cache DIR]

Over every team-match in the three tournaments (294):

1. CALIBRATION. The original cut-offs (2.5 Aggressive, 4.4 Passive) were tuned to a
   PPDA that mostly counted Pressure events and applied x <= 50 in both teams'
   mirrored frames. The script reconstructs that legacy metric (legacy_ppda, kept
   only for this comparison), finds the percentile each old cut-off sat at, and
   reports the value at the same percentile of the corrected metric.
2. CURRENT SPLIT. Label shares tactical_engine.classify_ppda gives today.
3. VALIDATION. Spearman correlation between tactical_engine.compute_ppda and an
   independently written standard-definition PPDA (Opta style: opponent passes in
   their own 60% of the pitch over tackles, interceptions, fouls and challenges in
   the opposition's 60%).

Thresholds are percentile-based on these three tournaments, not an absolute
standard.
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from statsbombpy import sb

import tactical_engine as te
from run_full_stratification import COMPETITIONS, load_events
from schema_mapper import map_statsbomb_to_opta

OLD = {"ppda_aggressive": 2.5, "ppda_passive": 4.4}


def _x(df):
    return df["location"].apply(lambda l: float(l[0]) if isinstance(l, (list, tuple)) else np.nan)


def legacy_ppda(ev, team, opp):
    """The pre-fix computation, reconstructed from raw StatsBomb events: ALL opponent
    passes with x <= 60 over the team's Foul/Interception/Clearance/Block/Pressure
    events with x <= 60 (Opta x <= 50), in each team's own frame. 'Tackle' never
    matched, and Duel (including tackle duels) was not in the list."""
    x, play = _x(ev), ev["period"] != 5
    passes = ((ev["team"] == opp) & (ev["type"] == "Pass") & play & (x <= 60)).sum()
    acts = ((ev["team"] == team) & play & (x <= 60) &
            ev["type"].isin(["Foul Committed", "Interception", "Clearance", "Block", "Pressure"])).sum()
    return passes / acts if acts else np.nan


def standard_ppda(ev, team, opp):
    x, play = _x(ev), ev["period"] != 5
    passes = ((ev["team"] == opp) & (ev["type"] == "Pass") & play & (x <= 72)).sum()
    tackle = (ev["type"] == "Duel") & (ev["duel_type"] == "Tackle")
    action = tackle | ev["type"].isin(["Interception", "Foul Committed", "Dribbled Past"])
    acts = ((ev["team"] == team) & action & play & (x >= 48)).sum()
    return passes / acts if acts else np.nan


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)

    rows = []
    for comp, cfg in COMPETITIONS.items():
        for mid in sb.matches(competition_id=cfg["competition_id"], season_id=cfg["season_id"])["match_id"]:
            ev = load_events(mid, cache)
            opta = map_statsbomb_to_opta(ev)
            teams = te.get_teams(opta)
            for team in teams:
                opp = next(t for t in teams if t != team)
                p = te.compute_ppda(opta, team)
                rows.append(dict(comp=comp, match=mid, team=team, ppda=p,
                                 label=te.classify_ppda(p),
                                 legacy=legacy_ppda(ev, team, opp),
                                 standard=standard_ppda(ev, team, opp)))
    R = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    print(f"{R['match'].nunique()} matches, {len(R)} team-matches\n")

    print("== 1. CALIBRATION (old cut-offs -> same percentile of the corrected metric) ==")
    legacy, new = R["legacy"].dropna(), R["ppda"].dropna()
    print(f"legacy PPDA p5/25/50/75/95 = {np.round(np.percentile(legacy, [5, 25, 50, 75, 95]), 2)}")
    print(f"corrected   p5/25/50/75/95 = {np.round(np.percentile(new, [5, 25, 50, 75, 95]), 1)}")
    for name, thr in OLD.items():
        pct = 100 * (legacy < thr).mean()
        print(f"  old {name:<16}{thr:>4}: percentile {pct:5.1f} -> corrected value {np.percentile(new, pct):5.1f}")
    print(f"current constants: aggressive < {te.THRESHOLDS['ppda_aggressive']:g}, "
          f"passive >= {te.THRESHOLDS['ppda_passive']:g}")

    print("\n== 2. CURRENT LABEL SPLIT ==")
    print("overall:", " | ".join(f"{k} {v}%" for k, v in (100 * R["label"].value_counts(normalize=True)).round(0).astype(int).items()))
    print((100 * pd.crosstab(R["comp"], R["label"], normalize="index")).round(0).astype(int).to_string())

    print("\n== 3. VALIDATION vs independent standard-definition PPDA ==")
    both = R[["ppda", "standard"]].dropna()
    print(f"Spearman, compute_ppda vs standard: {both['ppda'].corr(both['standard'], method='spearman'):.2f} (n={len(both)})")
    print(f"Spearman, legacy vs standard:       {R[['legacy','standard']].dropna().corr(method='spearman').iloc[0,1]:.2f}")


if __name__ == "__main__":
    main()
