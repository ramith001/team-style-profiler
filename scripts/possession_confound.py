"""
Does PPDA still shift with opponent tier once the opponent's own pass volume is
controlled for? The README caveats that PPDA = opponent passes / defensive
actions rises mechanically whenever the opponent simply passes more, which
elite sides do, so part of the earlier mixed-effects PPDA result could be that
mechanical channel rather than the pressing team doing anything differently.

This reuses the cached raw StatsBomb events from run_full_stratification.py
(no re-fetch) and, for every team-match, pulls both halves of the PPDA ratio
separately: opp_passes (the opponent's completed passes in their own defensive
two-thirds, the possession-volume side) and press_actions (the pressing team's
tackles/interceptions/fouls in the opposition's two-thirds, the behavioural
side), via tactical_engine.compute_ppda_components.

Then it fits log(ppda) ~ tier + log(opp_passes) + (1 | team) and compares the
tier coefficients against the no-covariate model: if they shrink toward zero,
the tier effect was mostly mechanical; if they survive, it's a real pressing
effect net of how much the opponent had the ball there. It also fits
log(opp_passes) ~ tier and log(press_actions) ~ tier separately, as a direct
check of which half of the ratio opponent tier actually moves.

    python scripts/possession_confound.py [--cache data_cache] [--results results]
"""
import argparse
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema_mapper import map_statsbomb_to_opta
from tactical_engine import compute_ppda_components, get_teams

COMPETITIONS = {
    "wc2022":    {"competition_id": 43,   "season_id": 106},
    "euro2024":  {"competition_id": 55,   "season_id": 282},
    "copa2024":  {"competition_id": 223,  "season_id": 282},
    "wc2018":    {"competition_id": 43,   "season_id": 3},
    "euro2020":  {"competition_id": 55,   "season_id": 43},
    "afcon2023": {"competition_id": 1267, "season_id": 107},
}
TIERS = ["Weak", "Moderate", "Strong", "Elite"]


def load_cached_opta(match_id: int, cache: Path) -> pd.DataFrame:
    path = cache / f"{match_id}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not cached, run run_full_stratification.py first")
    return map_statsbomb_to_opta(pickle.loads(path.read_bytes()))


def build_components(cache: Path) -> pd.DataFrame:
    from statsbombpy import sb
    rows = []
    for comp_name, cfg in COMPETITIONS.items():
        matches = sb.matches(competition_id=cfg["competition_id"], season_id=cfg["season_id"])
        for mid in matches["match_id"]:
            opta = load_cached_opta(mid, cache)
            teams = get_teams(opta)
            if len(teams) < 2:
                continue
            for team in teams:
                c = compute_ppda_components(opta, team)
                rows.append({"competition": comp_name, "match_id": mid, "team": team,
                             "opp_passes": c["opp_passes"], "press_actions": c["press_actions"],
                             "ppda": c["ppda"]})
    return pd.DataFrame(rows)


def fit(df, outcome, extra_terms, label):
    formula = f"{outcome} ~ C(opponent_tier, Treatment('Weak'))" + extra_terms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = smf.mixedlm(formula, data=df, groups=df["team"]).fit(reml=True)
    print(f"\n=== {label} ===")
    print(f"{'term':<40} {'coef':>8} {'se':>7} {'p':>8}")
    for name, coef, se, p in zip(result.params.index, result.params, result.bse, result.pvalues):
        if name == "Group Var":
            continue
        print(f"{name:<40} {coef:>8.3f} {se:>7.3f} {p:>8.4f}")
    return result


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    cache, results = Path(a.cache), Path(a.results)

    print("Extracting opp_passes / press_actions from cached matches...")
    comp = build_components(cache)
    comp.to_csv(results / "ppda_components.csv", index=False)

    tm = pd.read_csv(results / "team_matches.csv")[["match_id", "team", "opponent_tier"]]
    df = comp.merge(tm, on=["match_id", "team"], how="inner")
    df = df[df["press_actions"] > 0].dropna(subset=["opponent_tier"]).copy()
    df["opponent_tier"] = pd.Categorical(df["opponent_tier"], categories=TIERS, ordered=True)
    df["log_ppda"] = np.log(df["ppda"])
    df["log_opp_passes"] = np.log(df["opp_passes"].clip(lower=1))
    df["log_press_actions"] = np.log(df["press_actions"].clip(lower=1))

    print(f"\nn = {len(df)} team-matches (joined against team_matches.csv), "
          f"{df['team'].nunique()} teams")

    fit(df, "log_ppda", "", "1. log(PPDA) ~ tier, no covariate (replicates the earlier result)")
    fit(df, "log_ppda", " + log_opp_passes",
        "2. log(PPDA) ~ tier + log(opp_passes)  <- does tier survive controlling for possession volume?")
    fit(df, "log_opp_passes", "", "3. log(opp_passes) ~ tier  <- does the opponent really pass more when they're stronger?")
    fit(df, "log_press_actions", "", "4. log(press_actions) ~ tier  <- do raw defensive actions in the opp third fall with tier too?")

    print(
        "\nReading: if the tier coefficients in (2) are close to the ones in (1), the pressing effect "
        "is independent of how much the opponent passed, a real behavioural shift. If they shrink "
        "toward zero in (2) while (3) shows a strong positive tier effect on opp_passes, the original "
        "PPDA result was substantially mechanical: PPDA moved mainly because elite sides keep the ball "
        "in that zone more, not because the pressing team worked less."
    )


if __name__ == "__main__":
    main()
