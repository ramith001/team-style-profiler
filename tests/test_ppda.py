"""Regression tests for the PPDA / event-mapping fixes (synthetic StatsBomb frames)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema_mapper import map_statsbomb_to_opta
from tactical_engine import classify_ppda, compute_ppda


def sb_row(team, type_, x, *, duel_type=None, pass_outcome=np.nan, period=1):
    return {"type": type_, "team": team, "location": [x, 40.0], "period": period,
            "duel_type": duel_type, "pass_outcome": pass_outcome,
            "pass_end_location": [x + 5, 40.0] if type_ == "Pass" else np.nan,
            "minute": 1, "second": 0, "player": "p", "position": "Center Back"}


def mapped(rows):
    return map_statsbomb_to_opta(pd.DataFrame(rows))


def test_tackle_duel_maps_to_type_7_and_aerial_does_not():
    out = mapped([sb_row("A", "Duel", 60, duel_type="Tackle"),
                  sb_row("A", "Duel", 60, duel_type="Aerial Lost")])
    assert list(out["opta_event_type_id"]) == [7, 44]


def test_completed_pass_flag_survives_nan_outcome():
    out = mapped([sb_row("A", "Pass", 20), sb_row("A", "Pass", 20, pass_outcome="Incomplete")])
    assert list(out["outcome"]) == [1, 0]


def test_ppda_known_value_with_tackles_counted():
    # B makes 12 completed passes in its own third; A makes 1 tackle, 1 interception
    # and 1 foul in the opposition's half -> PPDA = 12 / 3 = 4.0
    rows = [sb_row("B", "Pass", 30) for _ in range(12)]
    rows += [sb_row("A", "Duel", 70, duel_type="Tackle"),
             sb_row("A", "Interception", 70), sb_row("A", "Foul Committed", 70)]
    assert compute_ppda(mapped(rows), "A") == 4.0


def test_pressure_is_not_a_defensive_action():
    rows = [sb_row("B", "Pass", 30) for _ in range(10)]
    rows += [sb_row("A", "Interception", 70)] + [sb_row("A", "Pressure", 70) for _ in range(20)]
    assert compute_ppda(mapped(rows), "A") == 10.0


def test_each_team_is_filtered_in_its_own_frame():
    # B's pass at x=30 is in B's own third; A's tackle at x=70 is in the opposition's
    # half. Reading both in one frame (x <= 50 for both) would drop A's tackle.
    rows = [sb_row("B", "Pass", 30), sb_row("A", "Duel", 70, duel_type="Tackle")]
    assert compute_ppda(mapped(rows), "A") == 1.0


def test_zone_filters_and_incomplete_passes():
    rows = [sb_row("B", "Pass", 100),                                    # B's attacking third: ignored
            sb_row("B", "Pass", 30, pass_outcome="Incomplete"),          # incomplete: ignored
            sb_row("B", "Pass", 30),
            sb_row("A", "Interception", 10),                             # A's own third: ignored
            sb_row("A", "Interception", 70),
            sb_row("A", "Interception", 70, period=5)]                   # shootout: ignored
    assert compute_ppda(mapped(rows), "A") == 1.0


def test_classifier_boundaries_are_ordered():
    assert classify_ppda(1.0) == "Aggressive Press"
    assert classify_ppda(1000.0) == "Passive / Low Press"


def test_classifier_cutoffs_9_and_20():
    assert classify_ppda(8.99) == "Aggressive Press"
    assert classify_ppda(9.0) == "Moderate Press"
    assert classify_ppda(19.99) == "Moderate Press"
    assert classify_ppda(20.0) == "Passive / Low Press"
