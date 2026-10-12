"""
schema_mapper.py
================
Tactical Verdict Engine — Opta Schema Mapper
Ramith Vaid | NTU Singapore

Converts StatsBomb open data into Stats Perform (Opta) MA3 feed format.
This allows the Tactical Verdict Engine to be built entirely to Opta's spec,
ready to accept a live Opta feed with zero pipeline changes.

Opta MA3 Event Type IDs used:
    1  = Pass
    4  = Foul committed
    7  = Tackle (StatsBomb Duel with duel_type == "Tackle")
    8  = Interception
    12 = Clearance / Block
    13 = Miss (shot off target)
    15 = Attempt Saved (shot on target)
    16 = Post (shot hits post)
    44 = Other duels (lost aerials; StatsBomb "Duel" that is not a tackle)
    49 = Pressure (no Opta equivalent; NOT a Ball Recovery)
    61 = Keeper Pick-Up

Opta coordinate system:
    - Pitch dimensions: 100 x 100 (percentage-based)
    - Origin: bottom-left
    - Attacking direction: left to right (x increases toward goal)

StatsBomb coordinate system:
    - Pitch dimensions: 120 x 80
    - Origin: bottom-left
    - Attacking direction: left to right, FOR EACH TEAM. StatsBomb records every
      team's events in that team's own attacking frame (its opponent's goal is
      always x = 120), so the two teams' coordinates are mirror images of each
      other, not one shared physical frame.

Conversion:
    opta_x = (statsbomb_x / 120) * 100
    opta_y = (statsbomb_y / 80) * 100

The conversion is a pure rescale and deliberately keeps each team in its own
attacking frame: every detector that reads "x" (defensive line, progressive
passes, danger corridor, distance to goal) assumes the team attacks toward
x = 100. Anything that compares the two teams' positions (PPDA) must account for
the frames explicitly; see tactical_engine.compute_ppda.
"""

import pandas as pd
import math
from statsbombpy import sb

# ── Constants ──────────────────────────────────────────────────────────────────

# Opta pitch is 100x100
OPTA_PITCH_LENGTH = 100.0
OPTA_PITCH_WIDTH  = 100.0

# StatsBomb pitch is 120x80
SB_PITCH_LENGTH = 120.0
SB_PITCH_WIDTH  = 80.0

# Opta goal centre (attacking end)
OPTA_GOAL = (100.0, 50.0)

# StatsBomb to Opta event type mapping
SB_TO_OPTA_EVENT_TYPE = {
    "Pass":          1,
    "Pressure":      49,   # Opta has no pressure event; kept as its own type
    "Interception":  8,
    "Clearance":     12,
    "Block":         12,   # mapped to clearance as closest
    "Foul Committed": 4,
    "Duel":          44,   # refined below: duel_type == "Tackle" -> 7
    "Shot":          15,   # default; refined below by outcome
    "Carry":         None, # Opta does not have carry events
    "Ball Receipt*": None, # Opta does not have ball receipt
    "Goal Keeper":   61,
}

# StatsBomb shot outcomes to Opta event type refinement
SB_SHOT_OUTCOME_TO_OPTA = {
    "Goal":           16,  # Opta type 16 = goal (we use 16 for goal)
    "Saved":          15,  # Attempt Saved
    "Off T":          13,  # Miss
    "Blocked":        15,  # treated as saved for our purposes
    "Post":           14,  # Post (type 14 in Opta)
    "Wayward":        13,  # Miss
}

# ── Coordinate conversion ──────────────────────────────────────────────────────

def sb_to_opta_coords(x: float, y: float) -> tuple:
    """Convert StatsBomb XY coordinates to Opta 100x100 coordinate system."""
    opta_x = round((x / SB_PITCH_LENGTH) * OPTA_PITCH_LENGTH, 2)
    opta_y = round((y / SB_PITCH_WIDTH)  * OPTA_PITCH_WIDTH,  2)
    return opta_x, opta_y

def opta_distance_to_goal(x: float, y: float) -> float:
    """Euclidean distance to goal in Opta coordinate space."""
    return math.sqrt((OPTA_GOAL[0] - x) ** 2 + (OPTA_GOAL[1] - y) ** 2)

# ── Period mapping ──────────────────────────────────────────────────────────────

def map_period(sb_period: int) -> str:
    """Map StatsBomb period integer to Opta period label."""
    mapping = {1: "FirstHalf", 2: "SecondHalf", 3: "ExtraFirstHalf",
               4: "ExtraSecondHalf", 5: "PenaltyShootout"}
    return mapping.get(sb_period, "Unknown")

# ── Under pressure qualifier ────────────────────────────────────────────────────

def get_pressure_qualifier(row) -> bool:
    """
    In StatsBomb, under_pressure is a boolean column on pass events.
    In Opta, this is qualifier ID 72 on pass events.
    Returns True if the player was under pressure when the event occurred.
    """
    val = row.get("under_pressure", None)
    if val is True:
        return True
    return False

# ── Main mapper ─────────────────────────────────────────────────────────────────

def map_statsbomb_to_opta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a StatsBomb events DataFrame into Opta MA3 feed format.

    Input:  StatsBomb events DataFrame (from sb.events())
    Output: DataFrame with Opta-compatible column names and values
    """
    records = []

    for _, row in df.iterrows():
        event_type_sb = row.get("type", None)
        if isinstance(event_type_sb, dict):
            event_type_sb = event_type_sb.get("name", None)

        # Skip events we don't map
        opta_type = SB_TO_OPTA_EVENT_TYPE.get(event_type_sb, None)
        if opta_type is None:
            continue

        # StatsBomb has no "Tackle" event type: tackles are Duel rows with
        # duel_type == "Tackle". Everything else (lost aerials) stays 44.
        duel_type = None
        if event_type_sb == "Duel":
            duel_type = row.get("duel_type", None)
            if isinstance(duel_type, dict):
                duel_type = duel_type.get("name", None)
            opta_type = 7 if duel_type == "Tackle" else 44

        # Refine shot type by outcome
        if event_type_sb == "Shot":
            outcome = row.get("shot_outcome", None)
            if isinstance(outcome, dict):
                outcome = outcome.get("name", None)
            opta_type = SB_SHOT_OUTCOME_TO_OPTA.get(outcome, 15)

        # Get coordinates
        loc = row.get("location", None)
        x_sb, y_sb = (float(loc[0]), float(loc[1])) if isinstance(loc, (list, tuple)) else (None, None)

        if x_sb is not None:
            x_opta, y_opta = sb_to_opta_coords(x_sb, y_sb)
        else:
            x_opta, y_opta = None, None

        # End location (for passes)
        end_loc = row.get("pass_end_location", None)
        if isinstance(end_loc, (list, tuple)):
            x_end_opta, y_end_opta = sb_to_opta_coords(float(end_loc[0]), float(end_loc[1]))
        else:
            x_end_opta, y_end_opta = None, None

        # Pass outcome
        pass_outcome = row.get("pass_outcome", None)
        if isinstance(pass_outcome, dict):
            pass_outcome = pass_outcome.get("name", None)
        # No pass_outcome = completed in StatsBomb. sb.events() gives NaN (not None)
        # for those rows, so test with pd.isna or every pass reads as incomplete.
        pass_success = 1 if pd.isna(pass_outcome) else 0

        # Team
        team = row.get("team", None)
        if isinstance(team, dict):
            team = team.get("name", None)

        # Player
        player = row.get("player", None)
        if isinstance(player, dict):
            player = player.get("name", None)

        # Position (for goalkeeper detection)
        position = row.get("position", None)
        if isinstance(position, dict):
            position = position.get("name", None)

        # Period
        period_sb  = row.get("period", 1)
        period_opta = map_period(int(period_sb))

        # Under pressure qualifier
        under_pressure = get_pressure_qualifier(row)

        # Distance to goal (Opta space)
        dist_to_goal = opta_distance_to_goal(x_opta, y_opta) if x_opta is not None else None

        records.append({
            # ── Opta MA3 standard fields ──
            "opta_event_type_id":   opta_type,
            "opta_event_type_name": event_type_sb,
            "duel_type":            duel_type,
            "period":               period_opta,
            "period_id":            int(period_sb),
            "minute":               row.get("minute", None),
            "second":               row.get("second", None),
            "team":                 team,
            "player":               player,
            "position":             position,
            "x":                    x_opta,
            "y":                    y_opta,
            "end_x":                x_end_opta,
            "end_y":                y_end_opta,
            "outcome":              pass_success if event_type_sb == "Pass" else None,
            # ── Opta qualifier equivalents ──
            "qualifier_72_under_pressure": under_pressure,  # Qualifier 72 = Under Pressure
            "qualifier_pass_success":      pass_success if event_type_sb == "Pass" else None,
            # ── Derived metrics (Opta-space) ──
            "distance_to_goal_opta": round(dist_to_goal, 2) if dist_to_goal else None,
            # ── Original StatsBomb fields retained for validation ──
            "_sb_event_type":       event_type_sb,
            "_sb_x":                x_sb,
            "_sb_y":                y_sb,
            "_sb_period":           int(period_sb),
            "_sb_match_id":         row.get("match_id", None),
        })

    opta_df = pd.DataFrame(records)
    return opta_df


# ── Loader ──────────────────────────────────────────────────────────────────────

def load_match_as_opta(match_id: int) -> pd.DataFrame:
    """
    Load a StatsBomb match and return it in Opta MA3 format.

    Args:
        match_id: StatsBomb match ID

    Returns:
        DataFrame in Opta MA3 format
    """
    print(f"Loading StatsBomb match {match_id}...")
    df = sb.events(match_id=match_id)
    print(f"  {len(df)} raw events loaded.")

    opta_df = map_statsbomb_to_opta(df)
    print(f"  {len(opta_df)} events mapped to Opta MA3 format.")

    return opta_df


# ── Quick test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with Argentina vs France World Cup Final
    MATCH_ID = 3869685

    opta_df = load_match_as_opta(MATCH_ID)

    print("\n── Sample output (first 5 rows) ──")
    print(opta_df[["opta_event_type_id", "opta_event_type_name", "team", "player",
                   "x", "y", "period", "qualifier_72_under_pressure"]].head(10).to_string())

    print("\n── Event type distribution ──")
    print(opta_df["opta_event_type_name"].value_counts())

    print("\n── Teams in match ──")
    print(opta_df["team"].unique())

    print("\n── Coordinate range check (should be 0-100) ──")
    print(f"X range: {opta_df['x'].min():.1f} – {opta_df['x'].max():.1f}")
    print(f"Y range: {opta_df['y'].min():.1f} – {opta_df['y'].max():.1f}")

    print("\nSchema mapper working correctly.")
