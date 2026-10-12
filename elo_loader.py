"""
elo_loader.py
=============
Team Strength Profiling — Elo Rating Loader
Ramith Vaid | NTU Singapore

Loads historical World Football Elo Ratings (eloratings.net, via Kaggle)
and provides pre-tournament strength ratings for any team at any cutoff date.

Why Elo instead of fitting our own model:
A Dixon-Coles style model fitted on a single tournament (64 matches, 32 teams)
is badly underdetermined, there's roughly one free parameter per match played,
producing unstable, unrealistic ratings (e.g. Serbia and Iran ranking above
Argentina and France in initial testing). Elo solves this properly since it's
built from decades of accumulated match history, not just one tournament.

Data source: eloratings.net, packaged via Kaggle
(https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings)

Known data quirk: team names in the raw CSV contain non-breaking space
characters (\\xa0) instead of regular spaces, e.g. "South\\xa0Korea".
This module normalises that automatically.
"""

import pandas as pd

# ── Loader ──────────────────────────────────────────────────────────────────────

def load_elo_data(csv_path: str = "eloratings.csv") -> pd.DataFrame:
    """
    Load and clean the raw Elo ratings CSV.

    Returns a DataFrame with columns: date, team, rating, change
    Team names are normalised (non-breaking spaces replaced with regular spaces).
    """
    df = pd.read_csv(csv_path)
    df["team"] = df["team"].str.replace("\xa0", " ", regex=False).str.strip()
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df


# ── Known naming mismatches (StatsBomb name -> eloratings.net name) ──────────────
# eloratings.net sometimes uses different conventions than StatsBomb.
# Add to this dict whenever a new mismatch is found.
NAME_MAPPING = {
    "Czech Republic": "Czechia",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "Democratic Republic of Congo",
    "Cape Verde Islands": "Cape Verde",
}


def normalise_team_name(name: str) -> str:
    """Map a StatsBomb team name to its eloratings.net equivalent if needed."""
    return NAME_MAPPING.get(name, name)


def get_pre_tournament_ratings(df: pd.DataFrame, cutoff_date: str,
                                 teams_needed: list = None) -> pd.DataFrame:
    """
    Get each team's most recent Elo rating strictly before a cutoff date.

    Args:
        df: cleaned Elo DataFrame (from load_elo_data)
        cutoff_date: tournament start date, e.g. "2022-11-20"
        teams_needed: optional list of team names to verify coverage for

    Returns:
        DataFrame with columns: team, rating, date (date = when that rating was set)
    """
    cutoff = pd.to_datetime(cutoff_date)
    sub = df[df["date"] < cutoff].sort_values("date")
    latest = sub.groupby("team").last().reset_index()[["team", "rating", "date"]]

    if teams_needed:
        missing = set(teams_needed) - set(latest["team"])
        if missing:
            print(f"WARNING: no Elo rating found for: {missing}")
            print("Check team name spelling matches eloratings.net convention.")

    return latest


def get_team_rating(ratings_df: pd.DataFrame, team: str) -> float:
    """Quick lookup of a single team's pre-tournament rating."""
    row = ratings_df[ratings_df["team"] == team]
    if row.empty:
        return None
    return float(row["rating"].iloc[0])


def classify_strength_tier(rating: float, all_ratings: pd.Series) -> str:
    """
    Classify a team's strength into a tier relative to the tournament field.
    Uses quartiles of the full set of participating teams' ratings.
    """
    q1, q2, q3 = all_ratings.quantile([0.25, 0.5, 0.75])
    if rating >= q3:
        return "Elite"
    elif rating >= q2:
        return "Strong"
    elif rating >= q1:
        return "Moderate"
    else:
        return "Weak"


# ── Tournament team lists (for verification) ─────────────────────────────────────

WC2022_TEAMS = [
    "Argentina", "France", "Brazil", "Germany", "Spain", "England", "Portugal",
    "Netherlands", "Belgium", "Croatia", "Morocco", "Japan", "South Korea",
    "Saudi Arabia", "Tunisia", "Senegal", "Ghana", "Cameroon", "Iran", "Qatar",
    "Ecuador", "Uruguay", "Mexico", "United States", "Canada", "Wales",
    "Switzerland", "Serbia", "Poland", "Denmark", "Australia", "Costa Rica"
]

EURO2024_TEAMS = [
    "Germany", "Scotland", "Hungary", "Switzerland", "Spain", "Croatia", "Italy",
    "Albania", "Slovenia", "Denmark", "Serbia", "England", "Poland",
    "Netherlands", "Austria", "France", "Belgium", "Slovakia", "Romania",
    "Ukraine", "Turkey", "Georgia", "Portugal", "Czech Republic"
]

COPA2024_TEAMS = [
    "Argentina", "Canada", "Chile", "Peru", "Ecuador", "Mexico", "Venezuela",
    "Jamaica", "United States", "Bolivia", "Panama", "Uruguay", "Brazil",
    "Colombia", "Paraguay", "Costa Rica"
]


# ── Quick test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_elo_data("eloratings.csv")
    print(f"Loaded {len(df)} rows, {df['team'].nunique()} unique teams.")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print()

    print("── 2022 World Cup pre-tournament ratings ──")
    wc22 = get_pre_tournament_ratings(df, "2022-11-20", WC2022_TEAMS)
    print(wc22[wc22["team"].isin(WC2022_TEAMS)].sort_values("rating", ascending=False).to_string())
    print()

    print("── Strength tier classification example ──")
    wc22_field = wc22[wc22["team"].isin(WC2022_TEAMS)]
    for team in ["Argentina", "Saudi Arabia", "Poland", "Denmark"]:
        r = get_team_rating(wc22_field, team)
        tier = classify_strength_tier(r, wc22_field["rating"])
        print(f"  {team}: {r:.0f} -> {tier}")
