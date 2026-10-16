"""
Does the tier effect on defensive line height survive controlling for game state?

A team can sit deep against a stronger opponent out of respect for it, or it
can sit deep because it is protecting a lead, whoever the opponent is. This
mirrors scripts/possession_confound.py, but for the defensive-line finding
instead of PPDA.

Covariate: leading_time_share, the share of a team's played match time spent
ahead on the scoreboard (the rest is level or trailing). The choice is
deliberate and has costs, so it is worth spelling out:

  * Time-weighted, not final score or goal difference. A team that scores in
    the 89th minute led for almost none of the match; one that scores in the
    10th and holds on led for most of it. Time spent leading is the thing that
    would plausibly drive a deeper line, and it is what a reader means by
    "protecting a lead".
  * Leading only, not leading and trailing. A trailing team is also expected
    to push up, so trailing share is a second, related covariate; it is
    reported as a robustness variant (leading + trailing share) rather than as
    the headline.
  * Match-level, not matched to the line-height measurement window. def_line_x
    in team_matches.csv is one number for the whole match, so the covariate is
    match-level too. A coarse proxy: it cannot say a team dropped after taking
    the lead.
  * Game state is partly an outcome of how the team defended, so it is a
    mediator as much as a confounder. A deep line may help a team protect a
    lead, which puts some of the tier effect on the "game state" path. Reading
    a shrunken tier coefficient as "the part that was just game state" is
    therefore an upper bound on that channel, not a clean decomposition.

Goals come from the cached StatsBomb events (no re-fetch). After
schema_mapper.map_statsbomb_to_opta, goals are Opta type 16 with a minute and
second. Two corrections are needed on top of that, both checked against the
cached data:
  * Penalty shoot-out kicks (period 5) are also type 16 and are excluded; they
    are not part of the match score or of match time.
  * Own goals are not type 16. They are read from the raw events ("Own Goal
    For", logged under the team that benefits).
Time is played time: each period's length is taken from its last event, so
first-half stoppage time and extra time (periods 3-4) are counted once and in
order.

It also fits leading_time_share ~ tier + (1 | team): if stronger opponents
simply put the team behind less (or more) often, tier would predict game state
on its own, which is a confound in its own right.

    python scripts/game_state_confound.py [--cache data_cache] [--results results]
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

TIERS = ["Weak", "Moderate", "Strong", "Elite"]
PERIOD_START = {1: 0, 2: 2700, 3: 5400, 4: 6300}  # StatsBomb clock seconds at the start of each period


def played_seconds(period, minute, second, period_len):
    """Seconds of played time at a clock reading, counting earlier periods at their full length."""
    before = sum(period_len.get(p, 0) for p in range(1, period))
    return before + (minute * 60 + second - PERIOD_START[period])


def match_state(raw):
    """Return (goals, total_time): goals = sorted [(played_seconds, scoring_team)] over periods 1-4."""
    opta = map_statsbomb_to_opta(raw)
    opta = opta[opta["period_id"].between(1, 4)]
    period_len = {}
    for p in range(1, 5):
        sub = raw[raw["period"] == p]
        if len(sub):
            period_len[p] = max(0, int((sub["minute"] * 60 + sub["second"]).max()) - PERIOD_START[p])
    goals = []
    for _, r in opta[opta["opta_event_type_id"] == 16].iterrows():
        goals.append((played_seconds(int(r["period_id"]), r["minute"], r["second"], period_len), r["team"]))
    own = raw[(raw["type"] == "Own Goal For") & raw["period"].between(1, 4)]
    for _, r in own.iterrows():
        goals.append((played_seconds(int(r["period"]), r["minute"], r["second"], period_len), r["team"]))
    goals.sort()
    return goals, sum(period_len.values())


def time_shares(goals, total, team, opponent):
    lead = trail = 0.0
    score = 0  # this team's goals minus the opponent's
    last = 0.0
    for t, scorer in goals + [(total, None)]:
        dt = max(0.0, t - last)
        if score > 0:
            lead += dt
        elif score < 0:
            trail += dt
        last = max(last, t)
        if scorer == team:
            score += 1
        elif scorer == opponent:
            score -= 1
    return lead / total, trail / total, score


def build_game_state(tm, cache):
    rows = []
    for mid, sub in tm.groupby("match_id"):
        path = cache / f"{mid}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not cached, run run_full_stratification.py first")
        goals, total = match_state(pickle.loads(path.read_bytes()))
        for _, r in sub.iterrows():
            lead, trail, final = time_shares(goals, total, r["team"], r["opponent"])
            rows.append({"match_id": mid, "team": r["team"], "leading_time_share": lead,
                         "trailing_time_share": trail, "final_diff": final})
    return pd.DataFrame(rows)


def fit(df, outcome, extra, label):
    formula = f"{outcome} ~ C(opponent_tier, Treatment('Weak'))" + extra
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = smf.mixedlm(formula, data=df, groups=df["team"]).fit(reml=True)
    print(f"\n=== {label} ===")
    print(f"{'term':<44} {'coef':>8} {'se':>7} {'p':>9}")
    for name, coef, se, p in zip(res.params.index, res.params, res.bse, res.pvalues):
        if name == "Group Var":
            continue
        short = name.replace("C(opponent_tier, Treatment('Weak'))[T.", "tier ").rstrip("]")
        ps = "<0.0001" if p < 0.0001 else f"{p:.4f}"
        print(f"{short:<44} {coef:>8.3f} {se:>7.3f} {ps:>9}")
    return res


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    cache, results = Path(a.cache), Path(a.results)

    tm = pd.read_csv(results / "team_matches.csv").replace([np.inf, -np.inf], np.nan)
    tm = tm.dropna(subset=["def_line_x", "opponent_tier"]).copy()
    print("Reconstructing running score from cached events...")
    gs = build_game_state(tm, cache)
    gs.to_csv(results / "game_state.csv", index=False)

    df = tm.merge(gs, on=["match_id", "team"], how="inner")
    df["opponent_tier"] = pd.Categorical(df["opponent_tier"], categories=TIERS, ordered=True)
    # Sanity check: the two teams in a match must have mirrored final goal differences.
    mirror = df.groupby("match_id")["final_diff"].sum()
    print(f"n = {len(df)} team-matches, {df['team'].nunique()} teams. "
          f"Matches whose two final differences do not cancel: {int((mirror != 0).sum())}")
    lead_m, trail_m = df["leading_time_share"].mean(), df["trailing_time_share"].mean()
    print(f"Mean share of match time leading {lead_m:.3f}, trailing {trail_m:.3f}, level {1 - lead_m - trail_m:.3f}")
    print("Mean leading share by opponent tier:",
          df.groupby("opponent_tier", observed=True)["leading_time_share"].mean().round(3).to_dict())

    fit(df, "def_line_x", "", "1. def_line_x ~ tier (replicates the earlier result)")
    fit(df, "def_line_x", " + leading_time_share",
        "2. def_line_x ~ tier + leading_time_share  <- does tier survive controlling for time spent leading?")
    fit(df, "def_line_x", " + leading_time_share + trailing_time_share",
        "3. robustness: def_line_x ~ tier + leading + trailing time share")
    fit(df, "leading_time_share", "",
        "4. leading_time_share ~ tier  <- does opponent tier itself predict game state?")

    print(
        "\nReading: compare the tier coefficients in (2) with (1). If they are close, the line-height "
        "effect is not a protecting-a-lead artefact. If they shrink toward zero while (4) shows that tier "
        "predicts leading time, a share of the effect runs through game state. Game state is partly caused "
        "by how a team defends, so a shrunken coefficient is an upper bound on the game-state channel."
    )


if __name__ == "__main__":
    main()
