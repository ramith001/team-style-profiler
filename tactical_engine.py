"""
tactical_engine.py
==================
Tactical Verdict Engine — Core Analytical Layer
Ramith Vaid | NTU Singapore

Built entirely to Stats Perform (Opta) MA3 feed specification.
Accepts data in Opta format (output of schema_mapper.py) and produces
five tactical detectors with dynamic two-team verdicts.

Five Detectors:
    1. False High Line       — High defensive line + passive press (PPDA > threshold)
    2. Momentum Shift        — Progressive pass rate change between halves
    3. Press Bypass          — Press being beaten with significant forward gain
    4. Clinical Efficiency   — Danger corridor arrival rate vs total progressions
    5. Progress Trap         — Where forward passes are stalling and why

All thresholds calibrated against 2022 FIFA World Cup data.
Coordinate system: Opta 100x100 pitch.

Opta Event Type IDs used:
    1  = Pass
    4  = Foul committed
    7  = Tackle (StatsBomb Duel with duel_type == "Tackle")
    8  = Interception
    12 = Clearance / Block
    44 = Other duels (lost aerials)
    49 = Pressure (not a Ball Recovery)
"""

import math
import pandas as pd
import numpy as np
from schema_mapper import load_match_as_opta, opta_distance_to_goal, OPTA_GOAL

# ── Calibrated thresholds (2022 World Cup, Opta 100x100 coordinate space) ─────
#
# Note: StatsBomb thresholds are converted to Opta coordinate space
# StatsBomb 120x80 → Opta 100x100
# x: divide by 1.2  |  y: divide by 0.8
#
# Defensive line thresholds (Opta x-axis, 0-100):
#   Low Block  : avg_x < 37.5u  (was 45u in StatsBomb / 1.2)
#   High Line  : avg_x > 50.0u  (was 60u in StatsBomb / 1.2)
#
# Danger corridor (Opta space):
#   x: 73.3 – 100  (was 88-120 / 1.2)
#   y: 22.5 – 77.5 (was 18-62 / 0.8)
#
# Progressive pass gates (Opta space, Euclidean gain):
#   Own half → own half  : gain > 25u  (was 30 / 1.2)
#   Own half → opp half  : gain > 12.5u (was 15 / 1.2)
#   Opp half             : gain > 8.3u  (was 10 / 1.2)
#
# Press break threshold:
#   Euclidean gain > 12.5u (was 15u / 1.2)

THRESHOLDS = {
    # Defensive line (Opta x, 0-100)
    "low_block_max":     37.5,
    "high_line_min":     50.0,

    # PPDA (opponent passes per defensive action), corrected definition (see
    # compute_ppda). Percentile-based on the three tournaments (294 team-matches):
    # Aggressive < 9 is roughly the lowest fifth, Passive >= 20 the highest fifth.
    # These are the same cut-offs as sideline-engine, whose calibration is on the
    # same matches and the same definition; scripts/calibrate_ppda.py shows the
    # derivation. The old 2.5 / 4.4 belonged to the pre-fix metric (~1-4 range) and
    # would label every team-match "Passive / Low Press".
    "ppda_aggressive":    9.0,
    "ppda_passive":      20.0,

    # Compactness (IQR of defensive event x-coordinates, Opta space)
    "compact_max":       30.8,   # was 37u / 1.2
    "gapped_min":        35.0,   # was 42u / 1.2

    # Danger corridor (Opta 100x100)
    "danger_x_min":      73.3,
    "danger_x_max":     100.0,
    "danger_y_min":      22.5,
    "danger_y_max":      77.5,

    # Progressive pass gates (Euclidean gain, Opta space)
    "prog_own_to_own":   25.0,
    "prog_own_to_opp":   12.5,
    "prog_opp":           8.3,

    # Press break (Euclidean gain under pressure, Opta space)
    "press_break_gain":  12.5,

    # Momentum trigger
    "momentum_surge":     4.0,   # progressive pass delta between halves

    # Shape collapse
    "shape_collapse_min": 8.3,   # was 10u / 1.2

    # Drama thresholds
    "clinical_gap_large":   15,  # % efficiency gap
    "clinical_gap_moderate": 7,
}

# Opta event type IDs that locate a team's defending (defensive line, compactness).
# 44 (non-tackle duels, i.e. lost aerials) is included, as in the Tactical Verdict
# Engine; 7 is tackle duels.
OPTA_DEFENSIVE_TYPES = [4, 7, 8, 12, 44, 49]
OPTA_PASS_TYPE       = 1

# PPDA denominator: tackles (7), interceptions (8) and fouls committed (4).
# Pressure (49) is NOT counted: it is an off-the-ball event that happens many
# times per possession, so counting it measures how often a team is recorded
# pressing, not how much it disrupts passing.
PPDA_ACTION_TYPES = [4, 7, 8]

# PPDA zones. Each team's events are in its own attacking frame (x = 100 is the
# opponent's goal), so one zone per team, defined from that team's own frame:
#   opponent passes counted where the OPPONENT is in its own 66.7% of the pitch
#   (x <= 66.7 in the opponent's frame), pressing actions where the PRESSING team
#   is in the opposition's 66.7% (x >= 33.3 in its own frame). Both select the
#   same physical region: the area up to two thirds of the way from the
#   opponent's goal line.
PPDA_PASS_MAX_X  = 100 * 80 / 120   # StatsBomb x <= 80 of 120
PPDA_PRESS_MIN_X = 100 * 40 / 120   # StatsBomb x >= 40 of 120

# ── Helper functions ───────────────────────────────────────────────────────────

def get_teams(df: pd.DataFrame) -> list:
    return sorted(df["team"].dropna().unique().tolist())

def is_goalkeeper(row) -> bool:
    pos = row.get("position", None)
    if isinstance(pos, str):
        return "Goalkeeper" in pos or "GK" in pos
    return False

def compute_ppda(df: pd.DataFrame, pressing_team: str) -> float:
    """
    PPDA = opponent completed passes in their own 66.7% of the pitch /
           pressing team's tackles + interceptions + fouls in the opposition's 66.7%.
    Each team's x is in its own attacking frame (see PPDA_* constants above).
    Penalty-shootout events (period 5) are excluded.
    """
    teams = get_teams(df)
    opponent = next((t for t in teams if t != pressing_team), None)
    if not opponent:
        return float("inf")

    # Opponent completed passes while the opponent is in its own defensive two thirds
    opp_passes = df[
        (df["team"] == opponent) &
        (df["opta_event_type_id"] == OPTA_PASS_TYPE) &
        (df["outcome"] == 1) &
        (~df["period_id"].isin([5])) &
        (df["x"] <= PPDA_PASS_MAX_X)
    ]

    # Pressing team's defensive actions while it is in the opposition's two thirds
    press_actions = df[
        (df["team"] == pressing_team) &
        (df["opta_event_type_id"].isin(PPDA_ACTION_TYPES)) &
        (~df["period_id"].isin([5])) &
        (df["x"] >= PPDA_PRESS_MIN_X)
    ]

    if len(press_actions) == 0:
        return float("inf")

    return round(len(opp_passes) / len(press_actions), 2)

def classify_ppda(ppda: float) -> str:
    if ppda < THRESHOLDS["ppda_aggressive"]:
        return "Aggressive Press"
    if ppda < THRESHOLDS["ppda_passive"]:
        return "Moderate Press"
    return "Passive / Low Press"

def get_defensive_events(df: pd.DataFrame, team: str, exclude_gk: bool = True) -> pd.DataFrame:
    """Get all defensive events for a team, optionally excluding goalkeeper."""
    events = df[
        (df["team"] == team) &
        (df["opta_event_type_id"].isin(OPTA_DEFENSIVE_TYPES)) &
        (~df["period_id"].isin([5]))
    ].copy()

    events = events.dropna(subset=["x"])

    if exclude_gk:
        events = events[~events.apply(is_goalkeeper, axis=1)]

    return events

def is_progressive_pass(row) -> bool:
    """
    Progressive pass definition using Opta coordinate space.
    Uses Euclidean distance gain toward goal.
    """
    x1, y1 = row.get("x", None), row.get("y", None)
    x2, y2 = row.get("end_x", None), row.get("end_y", None)

    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [x1, y1, x2, y2]):
        return False

    d_start = opta_distance_to_goal(x1, y1)
    d_end   = opta_distance_to_goal(x2, y2)
    gain    = d_start - d_end

    # Zone-based gates (Opta space, half = x < 50)
    if x1 <= 50 and x2 <= 50:
        return gain > THRESHOLDS["prog_own_to_own"]
    if x1 <= 50 and x2 > 50:
        return gain > THRESHOLDS["prog_own_to_opp"]
    if x1 > 50:
        return gain > THRESHOLDS["prog_opp"]
    return False

def is_danger_corridor(x2: float, y2: float) -> bool:
    """Check if a coordinate falls in the danger corridor (Opta 100x100 space)."""
    return (
        THRESHOLDS["danger_x_min"] <= x2 <= THRESHOLDS["danger_x_max"] and
        THRESHOLDS["danger_y_min"] <= y2 <= THRESHOLDS["danger_y_max"]
    )

def get_progressive_passes(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Get all completed progressive passes for a team."""
    passes = df[
        (df["team"] == team) &
        (df["opta_event_type_id"] == OPTA_PASS_TYPE) &
        (~df["period_id"].isin([5])) &
        (df["outcome"] == 1)  # 1 = completed in our mapper
    ].copy()

    passes = passes.dropna(subset=["x", "y", "end_x", "end_y"])
    passes["is_progressive"] = passes.apply(is_progressive_pass, axis=1)
    passes["is_danger"]      = passes.apply(
        lambda r: is_danger_corridor(r["end_x"], r["end_y"])
        if pd.notna(r["end_x"]) and pd.notna(r["end_y"]) else False, axis=1
    )

    return passes[passes["is_progressive"]].copy()

# ── Detector 1: False High Line ────────────────────────────────────────────────

def detect_false_high_line(df: pd.DataFrame, team: str) -> dict:
    """
    Detects False High Line: high defensive line + passive/moderate press.
    In Opta space: avg defensive x > 50.0u AND press class not Aggressive
    (PPDA >= THRESHOLDS["ppda_aggressive"]).
    """
    def_events = get_defensive_events(df, team, exclude_gk=True)
    if def_events.empty:
        return {"team": team, "false_high_line": False, "def_line_x": None,
                "ppda": None, "press_class": None, "verdict": "Insufficient data"}

    # P15 of defensive events = defensive line (Opta x)
    def_line_x = float(def_events["x"].quantile(0.15))
    mid_line_x = float(def_events["x"].quantile(0.60))
    avg_x      = float(def_events["x"].mean())
    gap_u      = mid_line_x - def_line_x

    # IQR compactness
    iqr = float(def_events["x"].quantile(0.75) - def_events["x"].quantile(0.25))

    ppda        = compute_ppda(df, team)
    press_class = classify_ppda(ppda)

    v_state = "High Line" if avg_x > THRESHOLDS["high_line_min"] else \
              "Low Block" if avg_x < THRESHOLDS["low_block_max"] else "Mid Block"

    false_hl = (v_state == "High Line" and
                press_class in ["Passive / Low Press", "Moderate Press"])

    comp_class = "Compact" if iqr < THRESHOLDS["compact_max"] else \
                 "Gapped"  if iqr > THRESHOLDS["gapped_min"]  else "Standard"

    return {
        "team":           team,
        "false_high_line": false_hl,
        "def_line_x":     round(def_line_x, 1),
        "mid_line_x":     round(mid_line_x, 1),
        "avg_x":          round(avg_x, 1),
        "gap_u":          round(gap_u, 1),
        "iqr":            round(iqr, 1),
        "ppda":           ppda,
        "press_class":    press_class,
        "v_state":        v_state,
        "comp_class":     comp_class,
    }

# ── Detector 2: Momentum Shift ─────────────────────────────────────────────────

def detect_momentum_shift(df: pd.DataFrame, team: str) -> dict:
    """
    Detects momentum shift between first and second half.
    Measures change in progressive pass volume per half.
    """
    def get_prog_count(period_id: int) -> int:
        passes = df[
            (df["team"] == team) &
            (df["opta_event_type_id"] == OPTA_PASS_TYPE) &
            (df["period_id"] == period_id) &
            (df["outcome"] == 1)
        ].copy()
        passes = passes.dropna(subset=["x", "y", "end_x", "end_y"])
        passes["prog"] = passes.apply(is_progressive_pass, axis=1)
        return int(passes["prog"].sum())

    p1 = get_prog_count(1)
    p2 = get_prog_count(2)
    delta = p2 - p1
    drama = min(round(abs(delta) * 0.25, 1), 10.0)

    # Defensive line shift between halves
    def_events = get_defensive_events(df, team, exclude_gk=True)
    p1_def = def_events[def_events["period_id"] == 1]
    p2_def = def_events[def_events["period_id"] == 2]

    line_shift = None
    if not p1_def.empty and not p2_def.empty:
        x1_mean = float(p1_def["x"].mean())
        x2_mean = float(p2_def["x"].mean())
        line_shift = round(x2_mean - x1_mean, 1)

    return {
        "team":        team,
        "prog_p1":     p1,
        "prog_p2":     p2,
        "delta":       delta,
        "drama_score": drama,
        "line_shift":  line_shift,
        "direction":   "surged" if delta > 0 else "dropped",
    }

# ── Detector 3: Press Bypass ───────────────────────────────────────────────────

def detect_press_bypass(df: pd.DataFrame, team: str) -> dict:
    """
    Detects press-breaking passes: completed passes under pressure (Opta qualifier 72)
    where Euclidean gain toward goal exceeds threshold.
    """
    # All passes under pressure
    under_pressure = df[
        (df["team"] == team) &
        (df["opta_event_type_id"] == OPTA_PASS_TYPE) &
        (~df["period_id"].isin([5])) &
        (df["qualifier_72_under_pressure"] == True)
    ].copy()

    total_under = len(under_pressure)

    # Completed passes under pressure
    completed = under_pressure[under_pressure["outcome"] == 1].copy()
    completed = completed.dropna(subset=["x", "y", "end_x", "end_y"])

    # Calculate gain for each
    def calc_gain(row):
        d_start = opta_distance_to_goal(row["x"], row["y"])
        d_end   = opta_distance_to_goal(row["end_x"], row["end_y"])
        return d_start - d_end

    if not completed.empty:
        completed["gain"] = completed.apply(calc_gain, axis=1)
        press_breaks = completed[completed["gain"] >= THRESHOLDS["press_break_gain"]]
        avg_gain = round(float(press_breaks["gain"].mean()), 1) if not press_breaks.empty else 0.0
    else:
        press_breaks = pd.DataFrame()
        avg_gain = 0.0

    exit_rate = round((len(press_breaks) / total_under) * 100) if total_under > 0 else 0
    ppda      = compute_ppda(df, team)

    return {
        "team":              team,
        "total_under":       total_under,
        "press_break_count": len(press_breaks),
        "exit_rate_pct":     exit_rate,
        "avg_gain":          avg_gain,
        "ppda":              ppda,
        "press_class":       classify_ppda(ppda),
    }

# ── Detector 4: Clinical Efficiency ───────────────────────────────────────────

def detect_clinical_efficiency(df: pd.DataFrame, team: str) -> dict:
    """
    Measures how many progressive passes arrive in the danger corridor.
    Clinical efficiency = danger arrivals / total progressive passes.
    """
    prog = get_progressive_passes(df, team)

    if prog.empty:
        return {"team": team, "total_prog": 0, "danger_count": 0,
                "danger_pct": 0, "avg_gain": 0.0, "efficiency_class": "No data"}

    total_prog   = len(prog)
    danger_count = int(prog["is_danger"].sum())
    danger_pct   = round((danger_count / total_prog) * 100) if total_prog > 0 else 0

    prog["gain"] = prog.apply(
        lambda r: opta_distance_to_goal(r["x"], r["y"]) - opta_distance_to_goal(r["end_x"], r["end_y"]),
        axis=1
    )
    avg_gain = round(float(prog["gain"].mean()), 1)

    if danger_pct >= 35:
        eff_class = "High Clinical Efficiency"
    elif danger_pct >= 20:
        eff_class = "Moderate Clinical Efficiency"
    else:
        eff_class = "Low Clinical Efficiency"

    return {
        "team":             team,
        "total_prog":       total_prog,
        "danger_count":     danger_count,
        "danger_pct":       danger_pct,
        "avg_gain":         avg_gain,
        "efficiency_class": eff_class,
    }

# ── Detector 5: Progress Trap ──────────────────────────────────────────────────

def detect_progress_trap(df: pd.DataFrame, team: str, opponent: str) -> dict:
    """
    Identifies where a team's forward passes are stalling.
    Zones: Defensive Third (x<33), Middle Third (33-67), Final Third (x>67)
    Uses Opta 100x100 coordinate space.
    """
    passes = df[
        (df["team"] == team) &
        (df["opta_event_type_id"] == OPTA_PASS_TYPE) &
        (~df["period_id"].isin([5]))
    ].copy()

    passes = passes.dropna(subset=["x", "end_x"])

    # Forward passes only (end_x > x + 4.2u, equivalent to 5u in StatsBomb / 1.2)
    forward = passes[passes["end_x"] > passes["x"] + 4.2].copy()

    if forward.empty:
        return {"team": team, "total_attempts": 0, "stall_rate": 0,
                "primary_stall_zone": "No data", "forced_wide_pct": 0,
                "block_depth": None, "block_label": "—"}

    total_att  = len(forward)
    completed  = forward[forward["outcome"] == 1]
    stalled    = forward[forward["outcome"] == 0]
    stall_rate = round((len(stalled) / total_att) * 100) if total_att > 0 else 0

    # Zone classification (Opta 100x100)
    def zone(x):
        if x < 33:   return "Defensive Third"
        if x < 67:   return "Middle Third"
        return "Final Third"

    if not stalled.empty:
        stalled = stalled.copy()
        stalled["zone"] = stalled["x"].apply(zone)
        primary_zone = stalled["zone"].value_counts().index[0]
        primary_n    = int(stalled["zone"].value_counts().iloc[0])
    else:
        primary_zone = "None"
        primary_n    = 0

    # Forced wide (|y - 50| > 16.7u in Opta, equivalent to |y-40|>20 in StatsBomb / 0.8 * 0.667)
    wide_threshold = 16.7
    forced_wide = 0
    if not completed.empty:
        forced_wide = int((abs(completed["y"] - 50) > wide_threshold).sum())
        forced_wide_pct = round((forced_wide / len(completed)) * 100) if len(completed) > 0 else 0
    else:
        forced_wide_pct = 0

    # Opponent block depth (P15 of opponent defensive events, GK excluded)
    opp_def = get_defensive_events(df, opponent, exclude_gk=True)
    if not opp_def.empty:
        block_x = float(opp_def["x"].quantile(0.15))
        if block_x < 29.2:   block_label = "Extreme Low Block"
        elif block_x < 37.5: block_label = "Deep Block"
        elif block_x < 45.8: block_label = "Mid Block"
        else:                 block_label = "High Block / Pressing"
    else:
        block_x     = None
        block_label = "—"

    return {
        "team":              team,
        "total_attempts":    total_att,
        "completed_count":   len(completed),
        "stalled_count":     len(stalled),
        "stall_rate":        stall_rate,
        "primary_stall_zone": primary_zone,
        "primary_stall_n":   primary_n,
        "forced_wide_pct":   forced_wide_pct,
        "block_x":           round(block_x, 1) if block_x else None,
        "block_label":       block_label,
    }

# ── Dynamic Two-Team Verdicts ──────────────────────────────────────────────────

def verdict_false_high_line(t0: dict, t1: dict) -> str:
    n0, n1 = t0["team"], t1["team"]
    fhl0, fhl1 = t0["false_high_line"], t1["false_high_line"]
    ppda0, ppda1 = t0["ppda"], t1["ppda"]
    line0, line1 = t0["def_line_x"], t1["def_line_x"]
    gap0, gap1   = t0["gap_u"], t1["gap_u"]
    comp0, comp1 = t0["comp_class"], t1["comp_class"]

    if fhl0 and fhl1:
        return (f"Both {n0} and {n1} showed False High Line structures — "
                f"high defensive lines ({line0}u and {line1}u) with passive pressing "
                f"(PPDA {ppda0} and {ppda1}). Either side was vulnerable to balls in behind.")
    if fhl0:
        return (f"{n0}'s False High Line ({line0}u, PPDA {ppda0}) was the key structural finding. "
                f"{n1} were more coherent — {comp1} shape ({gap1}u between lines), PPDA {ppda1}. "
                f"{n0}'s exposure in behind was the defining tactical risk.")
    if fhl1:
        return (f"{n1}'s False High Line ({line1}u, PPDA {ppda1}) was the key structural finding. "
                f"{n0} were more coherent — {comp0} shape ({gap0}u between lines), PPDA {ppda0}. "
                f"{n1}'s exposure in behind was the defining tactical risk.")
    if comp0 == "Gapped" and comp1 == "Gapped":
        return (f"Both teams left space between their lines — {n0} {gap0}u, {n1} {gap1}u. "
                f"Creative midfielders on either side had room to operate.")
    if comp0 == "Gapped":
        return (f"Clear structural contrast: {n0} were Gapped ({gap0}u) while {n1} were {comp1} ({gap1}u). "
                f"{n0} carried the greater structural risk.")
    if comp1 == "Gapped":
        return (f"Clear structural contrast: {n1} were Gapped ({gap1}u) while {n0} were {comp0} ({gap0}u). "
                f"{n1} carried the greater structural risk.")
    return (f"Both {n0} and {n1} maintained compact defensive shapes — {gap0}u and {gap1}u between lines. "
            f"Structural vulnerability was low on both sides.")

def verdict_momentum(t0: dict, t1: dict) -> str:
    n0, n1 = t0["team"], t1["team"]
    d0, d1 = t0["delta"], t1["delta"]
    p1_0, p2_0 = t0["prog_p1"], t0["prog_p2"]
    p1_1, p2_1 = t1["prog_p1"], t1["prog_p2"]

    biggest = t0 if abs(d0) >= abs(d1) else t1
    other   = t1 if biggest == t0 else t0

    if abs(biggest["delta"]) >= 8:
        direction = "surged" if biggest["delta"] > 0 else "dropped sharply"
        return (f"The momentum story is clear. {biggest['team']}'s progressive passing "
                f"{direction} by {abs(biggest['delta'])} between halves — "
                f"from {biggest['prog_p1']} in the first half to {biggest['prog_p2']} in the second. "
                f"{other['team']} shifted by {'+' if other['delta']>0 else ''}{other['delta']} in comparison. "
                f"The half-time tactical adjustment was the defining moment.")
    if abs(d0) >= 4 or abs(d1) >= 4:
        return (f"{n0}: {p1_0} progressive passes in the first half, {p2_0} in the second "
                f"({'+' if d0>0 else ''}{d0}). "
                f"{n1}: {p1_1} in the first half, {p2_1} in the second "
                f"({'+' if d1>0 else ''}{d1}). "
                f"The brighter second-half picture tells the momentum story.")
    return (f"Both teams maintained consistent progression across both halves. "
            f"{n0}: {p1_0} → {p2_0}. {n1}: {p1_1} → {p2_1}. "
            f"No major tactical reset detected — this was decided by execution, not adjustment.")

def verdict_press_bypass(t0: dict, t1: dict) -> str:
    n0, n1 = t0["team"], t1["team"]
    ppda0, ppda1 = t0["ppda"], t1["ppda"]
    exit0, exit1 = t0["exit_rate_pct"], t1["exit_rate_pct"]

    harder = t0 if ppda0 <= ppda1 else t1
    softer = t1 if harder == t0 else t0

    if harder["ppda"] < THRESHOLDS["ppda_aggressive"] and softer["exit_rate_pct"] >= 35:
        return (f"{harder['team']} pressed hard (PPDA {harder['ppda']}) but {softer['team']} "
                f"found a way through — {softer['exit_rate_pct']}% of passes under pressure escaped. "
                f"A high press that wasn't suffocating enough.")
    if abs(exit0 - exit1) >= 15:
        better = t0 if exit0 >= exit1 else t1
        worse  = t1 if better == t0 else t0
        return (f"{better['team']} were far more effective under pressure — {better['exit_rate_pct']}% exit rate "
                f"vs {worse['exit_rate_pct']}% for {worse['team']}. "
                f"PPDA: {n0} {ppda0} vs {n1} {ppda1}. "
                f"The difference in press resistance was the clearest tactical gap.")
    return (f"Both teams pressed with similar intent — {n0} PPDA {ppda0}, {n1} PPDA {ppda1}. "
            f"Press-breaking exit rates were comparable: {exit0}% vs {exit1}%. "
            f"Neither press dominated decisively.")

def verdict_clinical(t0: dict, t1: dict) -> str:
    n0, n1 = t0["team"], t1["team"]
    pct0, pct1 = t0["danger_pct"], t1["danger_pct"]
    tot0, tot1 = t0["total_prog"], t1["total_prog"]
    dc0, dc1   = t0["danger_count"], t1["danger_count"]
    gap = abs(pct0 - pct1)
    more = t0 if pct0 >= pct1 else t1
    less = t1 if more == t0 else t0

    if gap >= THRESHOLDS["clinical_gap_large"]:
        return (f"{more['team']} were significantly more clinical — {more['danger_pct']}% of progressive passes "
                f"reached the danger corridor ({more['danger_count']}/{more['total_prog']}), "
                f"compared to {less['danger_pct']}% for {less['team']} ({less['danger_count']}/{less['total_prog']}). "
                f"The {gap}% efficiency gap was the clearest tactical difference.")
    if gap >= THRESHOLDS["clinical_gap_moderate"]:
        vol_leader = t0 if tot0 >= tot1 else t1
        return (f"{more['team']} edged clinical efficiency ({more['danger_pct']}% vs {less['danger_pct']}%). "
                f"{vol_leader['team']} dominated in volume with {max(tot0,tot1)} progressive passes.")
    return (f"Both teams were closely matched in danger corridor efficiency — "
            f"{n0} {pct0}% ({dc0}/{tot0}) vs {n1} {pct1}% ({dc1}/{tot1}). "
            f"Neither side found a clear edge in turning possession into genuine threat.")

def verdict_progress_trap(t0: dict, t1: dict) -> str:
    n0, n1     = t0["team"], t1["team"]
    sr0, sr1   = t0["stall_rate"], t1["stall_rate"]
    z0, z1     = t0["primary_stall_zone"], t1["primary_stall_zone"]
    bl0, bl1   = t0["block_label"], t1["block_label"]
    w0, w1     = t0["forced_wide_pct"], t1["forced_wide_pct"]
    gap = abs(sr0 - sr1)

    if gap >= 20:
        higher = t0 if sr0 >= sr1 else t1
        lower  = t1 if higher == t0 else t0
        return (f"A sharp contrast in progression efficiency. {higher['team']} stalled on {higher['stall_rate']}% "
                f"of forward passes — primarily in the {higher['primary_stall_zone']}. "
                f"{lower['team']}, by contrast, stalled on only {lower['stall_rate']}%. "
                f"The difference in build-up quality was decisive.")
    if sr0 >= 40 and sr1 >= 40:
        return (f"Both teams struggled to progress — {n0} {sr0}% stall rate, {n1} {sr1}%. "
                f"Forced wide: {n0} {w0}%, {n1} {w1}%. "
                f"The central corridor was largely shut down for both sides.")
    return (f"{n0}: {sr0}% stall rate (primary: {z0}), {bl1} from opponent, {w0}% forced wide. "
            f"{n1}: {sr1}% stall rate (primary: {z1}), {bl0} from opponent, {w1}% forced wide.")

# ── Full Match Analysis ────────────────────────────────────────────────────────

def analyse_match(match_id: int) -> dict:
    """
    Run all five detectors on a match and return full results.

    Args:
        match_id: StatsBomb match ID (will be converted to Opta format)

    Returns:
        dict with all detector outputs and verdicts
    """
    df     = load_match_as_opta(match_id)
    teams  = get_teams(df)

    if len(teams) < 2:
        return {"error": "Could not identify two teams"}

    t0, t1 = teams[0], teams[1]

    print(f"\nAnalysing: {t0} vs {t1}")
    print("=" * 60)

    # Run all five detectors
    fhl_t0 = detect_false_high_line(df, t0)
    fhl_t1 = detect_false_high_line(df, t1)

    mom_t0 = detect_momentum_shift(df, t0)
    mom_t1 = detect_momentum_shift(df, t1)

    press_t0 = detect_press_bypass(df, t0)
    press_t1 = detect_press_bypass(df, t1)

    clin_t0 = detect_clinical_efficiency(df, t0)
    clin_t1 = detect_clinical_efficiency(df, t1)

    trap_t0 = detect_progress_trap(df, t0, t1)
    trap_t1 = detect_progress_trap(df, t1, t0)

    return {
        "match_id": match_id,
        "teams":    [t0, t1],
        "false_high_line": {
            "teams":   [fhl_t0, fhl_t1],
            "verdict": verdict_false_high_line(fhl_t0, fhl_t1)
        },
        "momentum_shift": {
            "teams":   [mom_t0, mom_t1],
            "verdict": verdict_momentum(mom_t0, mom_t1)
        },
        "press_bypass": {
            "teams":   [press_t0, press_t1],
            "verdict": verdict_press_bypass(press_t0, press_t1)
        },
        "clinical_efficiency": {
            "teams":   [clin_t0, clin_t1],
            "verdict": verdict_clinical(clin_t0, clin_t1)
        },
        "progress_trap": {
            "teams":   [trap_t0, trap_t1],
            "verdict": verdict_progress_trap(trap_t0, trap_t1)
        },
    }


# ── Quick test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with Argentina vs France World Cup Final
    MATCH_ID = 3869685

    results = analyse_match(MATCH_ID)

    print("\n\n══════════════════════════════════════════════════════════")
    print("  TACTICAL VERDICT ENGINE — MATCH REPORT")
    print(f"  Match ID: {MATCH_ID}")
    print(f"  {results['teams'][0]} vs {results['teams'][1]}")
    print("══════════════════════════════════════════════════════════\n")

    detectors = [
        ("1. FALSE HIGH LINE",       "false_high_line"),
        ("2. MOMENTUM SHIFT",        "momentum_shift"),
        ("3. PRESS BYPASS",          "press_bypass"),
        ("4. CLINICAL EFFICIENCY",   "clinical_efficiency"),
        ("5. PROGRESS TRAP",         "progress_trap"),
    ]

    for label, key in detectors:
        data = results[key]
        print(f"── {label} ──")
        for td in data["teams"]:
            team = td["team"]
            # Print key metrics per detector
            if key == "false_high_line":
                print(f"  {team}: Line {td['def_line_x']}u | PPDA {td['ppda']} ({td['press_class']}) | "
                      f"{td['comp_class']} | FHL: {'YES ⚠' if td['false_high_line'] else 'No'}")
            elif key == "momentum_shift":
                print(f"  {team}: H1={td['prog_p1']} → H2={td['prog_p2']} "
                      f"({'+' if td['delta']>=0 else ''}{td['delta']}) | "
                      f"Line shift: {td['line_shift']}u")
            elif key == "press_bypass":
                print(f"  {team}: PPDA {td['ppda']} | {td['press_break_count']} exits / "
                      f"{td['total_under']} under pressure = {td['exit_rate_pct']}% | "
                      f"Avg gain {td['avg_gain']}u")
            elif key == "clinical_efficiency":
                print(f"  {team}: {td['danger_count']}/{td['total_prog']} danger arrivals = "
                      f"{td['danger_pct']}% | {td['efficiency_class']}")
            elif key == "progress_trap":
                print(f"  {team}: {td['stall_rate']}% stall rate | "
                      f"Primary zone: {td['primary_stall_zone']} | "
                      f"vs {td['block_label']} | {td['forced_wide_pct']}% forced wide")

        print(f"\n  VERDICT: {data['verdict']}\n")
