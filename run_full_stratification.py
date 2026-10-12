"""
Run the full three-tournament stratification.

    python run_full_stratification.py [--cache DIR] [--out DIR]

For WC 2022, Euro 2024 and Copa America 2024: map every match to Opta format,
compute PPDA and defensive line per team-match, tag opponent strength tier from
pre-tournament Elo (tiers are relative to each tournament's own field), pool the
team-matches across tournaments, and build the combined typology per team.

Writes team_matches.csv and combined_profiles.csv to --out (default: results/).
Raw StatsBomb events are cached as pickles in --cache so reruns are offline.
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from statsbombpy import sb

from elo_loader import load_elo_data
from schema_mapper import map_statsbomb_to_opta
from style_profiler import (build_all_combined_profiles, extract_match_metrics,
                            tag_opponent_strength)

COMPETITIONS = {
    "wc2022":    {"competition_id": 43,   "season_id": 106, "cutoff_date": "2022-11-20"},
    "euro2024":  {"competition_id": 55,   "season_id": 282, "cutoff_date": "2024-06-14"},
    "copa2024":  {"competition_id": 223,  "season_id": 282, "cutoff_date": "2024-06-20"},
    "wc2018":    {"competition_id": 43,   "season_id": 3,   "cutoff_date": "2018-06-14"},
    "euro2020":  {"competition_id": 55,   "season_id": 43,  "cutoff_date": "2021-06-11"},
    "afcon2023": {"competition_id": 1267, "season_id": 107, "cutoff_date": "2024-01-13"},
}


def load_events(match_id: int, cache: Path) -> pd.DataFrame:
    path = cache / f"{match_id}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    df = sb.events(match_id=match_id)
    path.write_bytes(pickle.dumps(df))
    return df


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cache, out = Path(a.cache), Path(a.out)
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    elo_df = load_elo_data("eloratings.csv")
    parts = []
    for name, cfg in COMPETITIONS.items():
        matches = sb.matches(competition_id=cfg["competition_id"], season_id=cfg["season_id"])
        rows = []
        for mid in matches["match_id"]:
            opta = map_statsbomb_to_opta(load_events(mid, cache))
            rows.extend(extract_match_metrics(mid, opta_df=opta))
        comp = tag_opponent_strength(pd.DataFrame(rows), elo_df, cfg["cutoff_date"])
        comp.insert(0, "competition", name)
        print(f"{name}: {len(matches)} matches, {len(comp)} team-matches")
        parts.append(comp)

    tm = pd.concat(parts, ignore_index=True)
    tm.to_csv(out / "team_matches.csv", index=False)
    profiles = build_all_combined_profiles(tm)
    profiles.to_csv(out / "combined_profiles.csv", index=False)
    print(f"\n{len(tm)} team-matches, {len(profiles)} teams with a combined typology")
    print(profiles[["team", "n_matches_ppda", "ppda_overall", "line_overall",
                    "combined_typology"]].to_string(index=False))


if __name__ == "__main__":
    main()
