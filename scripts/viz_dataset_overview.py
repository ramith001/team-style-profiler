"""
Dataset overview: what the rest of the analysis is built on.

Two panels from results/team_matches.csv: team-matches per tournament (in
chronological order, each bar labelled with how many distinct teams played in
it) and team-matches per opponent-strength tier (an ordinal Weak -> Elite
ramp). It introduces the dataset before the findings are read.

    python scripts/viz_dataset_overview.py [--results results] [--out results/figures]
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Chronological by tournament start, not alphabetical.
TOURNAMENTS = [
    ("wc2018", "World Cup\n2018"),
    ("euro2020", "Euro\n2020"),
    ("afcon2023", "AFCON\n2023"),
    ("wc2022", "World Cup\n2022"),
    ("euro2024", "Euro\n2024"),
    ("copa2024", "Copa América\n2024"),
]
TIERS = ["Weak", "Moderate", "Strong", "Elite"]

# dataviz skill palette (references/palette.md). Six tournaments exceed the
# validated 3-slot categorical set (see viz_typology_scatter.py), so panel 1
# uses one hue (BLUE) for every bar and the labels carry identity. Panel 2 is
# ordinal, so it uses the palette's sequential blue ramp, steps 250/400/550/700
# (the lightest ordinal step must clear 2:1 against the surface: 250 does).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TIER_RAMP = {"Weak": "#86b6ef", "Moderate": "#3987e5", "Strong": "#1c5cab", "Elite": "#0d366b"}
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def style_axes(ax):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", length=0, labelsize=10, colors=INK)
    ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRIDLINE, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    results_dir, out_dir = Path(a.results), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "team_matches.csv")
    n_total, n_teams, n_comp = len(df), df["team"].nunique(), df["competition"].nunique()

    by_comp = df.groupby("competition").agg(matches=("match_id", "size"), teams=("team", "nunique"))
    by_comp = by_comp.reindex([c for c, _ in TOURNAMENTS])
    by_tier = df["opponent_tier"].value_counts().reindex(TIERS).fillna(0).astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [6, 4], "wspace": 0.18})

    x = range(len(TOURNAMENTS))
    ax1.bar(x, by_comp["matches"], color=BLUE, width=0.62, zorder=2)
    for xi, (_, row) in zip(x, by_comp.iterrows()):
        ax1.text(xi, row["matches"] + 2, f"{int(row['matches'])}", ha="center", va="bottom",
                 fontsize=11, color=INK, fontweight="bold")
        ax1.text(xi, row["matches"] + 12, f"{int(row['teams'])} teams", ha="center", va="bottom",
                 fontsize=9, color=SECONDARY_INK)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([label for _, label in TOURNAMENTS])
    ax1.set_ylim(0, by_comp["matches"].max() * 1.25)
    ax1.set_title("By tournament (chronological)", fontsize=11, color=INK, loc="left", pad=10)
    ax1.set_ylabel("Team-matches", fontsize=10, color=SECONDARY_INK)
    style_axes(ax1)

    ax2.bar(range(len(TIERS)), by_tier.values, color=[TIER_RAMP[t] for t in TIERS], width=0.62, zorder=2)
    for xi, v in enumerate(by_tier.values):
        ax2.text(xi, v + 2, str(v), ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    ax2.set_xticks(range(len(TIERS)))
    ax2.set_xticklabels(TIERS)
    ax2.set_ylim(0, by_tier.max() * 1.25)
    ax2.set_title("By opponent strength tier", fontsize=11, color=INK, loc="left", pad=10)
    style_axes(ax2)

    fig.suptitle(f"{n_comp} tournaments, {n_teams} teams, {n_total} team-matches",
                 fontsize=14, color=INK, x=0.01, ha="left", fontweight="bold", y=1.0)

    fig.text(0.01, -0.02,
             "Included: FIFA World Cup 2018 and 2022, UEFA Euro 2020 and 2024, Africa Cup of Nations 2023, "
             "Copa América 2024. Each match contributes one row per team.",
             fontsize=9, color=MUTED, ha="left")
    fig.text(0.01, -0.07,
             "Tier = quartile of pre-tournament Elo within that tournament's field; "
             "teams appearing in several tournaments are counted once in the headline total.",
             fontsize=9, color=MUTED, ha="left", style="italic")

    out_path = out_dir / "dataset_overview.png"
    fig.savefig(out_path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"\n{n_comp} tournaments, {n_teams} teams, {n_total} team-matches")
    print("\nBy tournament:\n", by_comp)
    print("\nBy opponent tier:\n", by_tier)


if __name__ == "__main__":
    main()
