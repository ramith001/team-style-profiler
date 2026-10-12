"""
style_profiler.py
==================
Team Style and Tendency Profiling
Ramith Vaid | NTU Singapore

Built on a suggestion from Will Thomson (Principal Data Scientist, Leeds United;
ex-StatsBomb Head of Soccer Research): rather than producing a single static
tactical profile per team, stratify tactical metrics by opponent strength
(and eventually venue context) and compare each split back to the team's own
overall average. This reveals typologies that a single match snapshot can't:

    - "Consistently aggressive": presses hard regardless of opponent
    - "Conditionally pragmatic": presses hard vs weak teams, sits off vs elite teams
    - "Home specific presser": more aggressive at home than away (future extension)

Pipeline:
    StatsBomb events -> schema_mapper (Opta format) -> tactical_engine (PPDA,
    defensive line) -> elo_loader (opponent strength tier) -> this module
    (stratification + typology classification)

Validation method (also Will's suggestion, noted for future work):
    Backtest real player transfers between tactically "similar" vs "different"
    clubs per this system's classification, and check whether transfer success
    correlates with similarity, as an external check that the style
    classification captures something real rather than just noise.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsbombpy import sb
from schema_mapper import load_match_as_opta
from tactical_engine import compute_ppda, get_defensive_events, get_teams
from elo_loader import load_elo_data, get_pre_tournament_ratings, classify_strength_tier

# ── Competition configs ──────────────────────────────────────────────────────────

COMPETITIONS = {
    "wc2022": {
        "competition_id": 43, "season_id": 106,
        "cutoff_date": "2022-11-20",
        "teams": None,  # filled dynamically from match list
    },
}

# ── Per-match metric extraction ──────────────────────────────────────────────────

def extract_match_metrics(match_id: int, opta_df: pd.DataFrame = None) -> list:
    """
    Run a single match through the schema mapper and tactical engine,
    returning per-team metrics for that match. Pass `opta_df` (already mapped)
    to avoid a network fetch.
    """
    try:
        df = opta_df if opta_df is not None else load_match_as_opta(match_id)
    except Exception as e:
        print(f"  Skipping match {match_id}: {e}")
        return []

    teams = get_teams(df)
    if len(teams) < 2:
        return []

    records = []
    for team in teams:
        opponent = next(t for t in teams if t != team)

        ppda = compute_ppda(df, team)

        def_events = get_defensive_events(df, team, exclude_gk=True)
        avg_def_line = float(def_events["x"].mean()) if not def_events.empty else None

        records.append({
            "match_id": match_id,
            "team": team,
            "opponent": opponent,
            "ppda": ppda,
            "def_line_x": avg_def_line,
        })

    return records


def build_competition_dataset(competition_id: int, season_id: int, max_matches: int = None) -> pd.DataFrame:
    """
    Run extract_match_metrics across every match in a competition.

    Args:
        competition_id, season_id: StatsBomb identifiers
        max_matches: optional cap for testing (e.g. 5), None = run all matches

    Returns:
        Long DataFrame: one row per (team, match)
    """
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    match_ids = matches["match_id"].tolist()
    if max_matches:
        match_ids = match_ids[:max_matches]

    all_records = []
    for i, mid in enumerate(match_ids):
        print(f"Processing match {i+1}/{len(match_ids)} (id={mid})...")
        all_records.extend(extract_match_metrics(mid))

    return pd.DataFrame(all_records)


# ── Strength tier tagging ────────────────────────────────────────────────────────

def tag_opponent_strength(df: pd.DataFrame, elo_df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """
    Add opponent_rating and opponent_tier columns based on pre-tournament Elo.
    """
    from elo_loader import normalise_team_name

    all_teams_raw = sorted(set(df["team"].tolist() + df["opponent"].tolist()))
    all_teams = [normalise_team_name(t) for t in all_teams_raw]
    ratings = get_pre_tournament_ratings(elo_df, cutoff_date, all_teams)
    ratings = ratings[ratings["team"].isin(all_teams)]

    rating_map_normalised = dict(zip(ratings["team"], ratings["rating"]))
    # Map back through normalisation so original StatsBomb names work as lookup keys
    rating_map = {raw: rating_map_normalised.get(normalise_team_name(raw)) for raw in all_teams_raw}
    field_ratings = ratings["rating"]

    df = df.copy()
    df["opponent_rating"] = df["opponent"].map(rating_map)
    df["opponent_tier"] = df["opponent_rating"].apply(
        lambda r: classify_strength_tier(r, field_ratings) if pd.notna(r) else None
    )

    return df


# ── Stratification and typology ──────────────────────────────────────────────────

def build_team_profile(df: pd.DataFrame, team: str, metric: str = "ppda") -> dict:
    """
    Build a stratified profile for one team across opponent strength tiers.

    Returns dict with overall average, per-tier averages, and a typology label.
    """
    team_df = df[df["team"] == team].dropna(subset=[metric, "opponent_tier"])
    if team_df.empty:
        return {"team": team, "metric": metric, "n_matches": 0, "typology": "Insufficient data"}

    overall_avg = team_df[metric].mean()

    tier_avgs = team_df.groupby("opponent_tier")[metric].mean().to_dict()
    tier_counts = team_df.groupby("opponent_tier")[metric].count().to_dict()

    # Typology classification
    # For PPDA: lower = more aggressive pressing
    elite_avg = tier_avgs.get("Elite", None)
    weak_avg = tier_avgs.get("Weak", None)

    typology = "Insufficient spread"
    if elite_avg is not None and weak_avg is not None:
        diff = elite_avg - weak_avg
        # For PPDA specifically: positive diff means LESS aggressive (higher PPDA) vs elite
        threshold = overall_avg * 0.20  # 20% swing threshold

        if metric == "ppda":
            if diff > threshold:
                typology = "Conditionally pragmatic (presses harder vs weaker teams, sits off vs elite)"
            elif diff < -threshold:
                typology = "Front-foot vs elite (presses harder vs strong teams specifically)"
            else:
                typology = "Consistently aggressive (similar press intensity regardless of opponent)"
        elif metric == "def_line_x":
            if diff < -threshold:
                typology = "Drops deeper vs elite opposition"
            elif diff > threshold:
                typology = "Holds line higher vs elite opposition (unusual)"
            else:
                typology = "Consistent defensive line regardless of opponent"

    return {
        "team": team,
        "metric": metric,
        "n_matches": len(team_df),
        "overall_avg": round(overall_avg, 2),
        "tier_averages": {k: round(v, 2) for k, v in tier_avgs.items()},
        "tier_counts": tier_counts,
        "typology": typology,
    }


def build_all_profiles(df: pd.DataFrame, metric: str = "ppda") -> pd.DataFrame:
    """Build stratified profiles for every team in the dataset."""
    teams = sorted(df["team"].unique())
    profiles = [build_team_profile(df, t, metric) for t in teams]
    return pd.DataFrame(profiles)


# ── Combined PPDA + Defensive Line typology ──────────────────────────────────────

def _direction(profile: dict) -> str:
    """
    Classify the Elite-vs-Weak direction for a single metric profile.
    Returns 'aggressive_vs_elite', 'passive_vs_elite', 'consistent', or 'insufficient'.
    """
    tiers = profile.get("tier_averages", {})
    elite = tiers.get("Elite")
    weak = tiers.get("Weak")
    overall = profile.get("overall_avg")

    if elite is None or weak is None or overall is None:
        return "insufficient"

    diff = elite - weak
    threshold = overall * 0.20

    if profile["metric"] == "ppda":
        # lower PPDA vs elite = more aggressive vs elite
        if diff < -threshold:
            return "aggressive_vs_elite"
        elif diff > threshold:
            return "passive_vs_elite"
        else:
            return "consistent"
    else:  # def_line_x
        # lower x vs elite = deeper line vs elite
        if diff < -threshold:
            return "deeper_vs_elite"
        elif diff > threshold:
            return "higher_vs_elite"
        else:
            return "consistent"


# Combination matrix: (ppda_direction, line_direction) -> (label, explanation)
COMBINED_TYPOLOGY = {
    ("aggressive_vs_elite", "deeper_vs_elite"): (
        "Compact aggressive block vs elite",
        "Drops into a deeper defensive line against elite opposition, but presses "
        "harder once the opponent enters that zone, rather than pressing high up "
        "the pitch. A deliberately compact, ambush-style approach against the best teams."
    ),
    ("aggressive_vs_elite", "higher_vs_elite"): (
        "All-out high press vs elite",
        "Holds a higher defensive line AND presses harder against elite opposition. "
        "A bold, high-risk approach committing fully to disrupting the best teams early."
    ),
    ("aggressive_vs_elite", "consistent"): (
        "Same shape, harder work-rate vs elite",
        "Defensive line height doesn't change by opponent, but pressing intensity "
        "increases against elite teams. Effort scales with opponent quality, shape doesn't."
    ),
    ("passive_vs_elite", "higher_vs_elite"): (
        "False High Line risk vs elite",
        "Holds a HIGHER defensive line against elite opposition while pressing LESS "
        "intensely, the exact combination that creates False High Line vulnerability "
        "(see Tactical Verdict Engine). A genuine structural risk against the best teams."
    ),
    ("passive_vs_elite", "deeper_vs_elite"): (
        "Fully conservative vs elite",
        "Drops deeper AND presses less against elite opposition, a fully defensive, "
        "risk-averse approach that cedes territory and initiative against the best teams."
    ),
    ("passive_vs_elite", "consistent"): (
        "Same shape, lower work-rate vs elite",
        "Defensive line height is stable regardless of opponent, but pressing intensity "
        "drops off against elite teams. Conserves effort rather than changing structure."
    ),
    ("consistent", "deeper_vs_elite"): (
        "Shape adapts, effort doesn't",
        "Drops the defensive line deeper against elite opposition without actually "
        "changing how hard the team presses."
    ),
    ("consistent", "higher_vs_elite"): (
        "Bolder shape, same work-rate vs elite",
        "Holds a higher line against elite opposition without a matching increase in "
        "pressing intensity, worth checking for False High Line risk specifically."
    ),
    ("consistent", "consistent"): (
        "Tactically stable regardless of opponent",
        "Neither pressing intensity nor defensive line height changes meaningfully "
        "based on opponent strength. One settled identity, applied uniformly."
    ),
}


def build_combined_profile(df: pd.DataFrame, team: str) -> dict:
    """
    Build a combined PPDA + defensive line typology for one team,
    crossing both dimensions into a single descriptive pattern.
    """
    ppda_profile = build_team_profile(df, team, metric="ppda")
    line_profile = build_team_profile(df, team, metric="def_line_x")

    ppda_dir = _direction(ppda_profile)
    line_dir = _direction(line_profile)

    if ppda_dir == "insufficient" or line_dir == "insufficient":
        label, explanation = "Insufficient data for combined typology", ""
    else:
        label, explanation = COMBINED_TYPOLOGY.get(
            (ppda_dir, line_dir),
            ("Unclassified combination", "")
        )

    return {
        "team": team,
        "n_matches_ppda": ppda_profile.get("n_matches", 0),
        "ppda_overall": ppda_profile.get("overall_avg"),
        "ppda_tiers": ppda_profile.get("tier_averages"),
        "line_overall": line_profile.get("overall_avg"),
        "line_tiers": line_profile.get("tier_averages"),
        "combined_typology": label,
        "explanation": explanation,
    }


def build_all_combined_profiles(df: pd.DataFrame, min_matches: int = 3) -> pd.DataFrame:
    """Build combined typology profiles for every team with enough matches."""
    teams = sorted(df["team"].unique())
    profiles = []
    for t in teams:
        p = build_combined_profile(df, t)
        if p["n_matches_ppda"] >= min_matches:
            profiles.append(p)
    return pd.DataFrame(profiles)


# ── Quick test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building WC2022 dataset (limited to 10 matches for initial test)...")
    df = build_competition_dataset(competition_id=43, season_id=106, max_matches=10)
    print(f"\nRaw dataset: {len(df)} team-match rows")
    print(df.head(10).to_string())

    print("\nLoading Elo ratings and tagging opponent strength...")
    elo_df = load_elo_data("eloratings.csv")
    df = tag_opponent_strength(df, elo_df, cutoff_date="2022-11-20")
    print(df[["team", "opponent", "ppda", "def_line_x", "opponent_rating", "opponent_tier"]].to_string())

    print("\n── Sample team profiles (PPDA) ──")
    profiles = build_all_profiles(df, metric="ppda")
    print(profiles.to_string())
