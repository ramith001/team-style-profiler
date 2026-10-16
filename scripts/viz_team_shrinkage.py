"""
Shrinkage plot: each team's raw Elite-minus-Weak swing against its
empirical-Bayes shrunken estimate, joined by a thin line.

Reads results/team_shrinkage.csv (written by scripts/team_shrinkage.py). Only
teams with a raw swing (an Elite and a Weak match) are drawn; the 22 teams
without one have a shrunken estimate only, so there is nothing to connect.
Marker size encodes how many Elite plus Weak matches the raw swing rests on.

    python scripts/viz_team_shrinkage.py [--results results] [--out results/figures]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# dataviz skill palette (references/palette.md): categorical slots 1 (blue) and
# 2 (orange), the validated adjacent pair. Orange = raw, blue = shrunken.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def panel(ax, d, prefix, title, xlabel, pop):
    d = d.dropna(subset=[f"{prefix}_raw"]).sort_values(f"{prefix}_raw").reset_index(drop=True)
    y = np.arange(len(d))
    n = d[f"{prefix}_n_elite"] + d[f"{prefix}_n_weak"]
    size = 14 + 10 * n
    for yi, raw, shr in zip(y, d[f"{prefix}_raw"], d[f"{prefix}_shrunk"]):
        ax.plot([raw, shr], [yi, yi], color=BASELINE, lw=0.9, zorder=1)
    ax.scatter(d[f"{prefix}_raw"], y, s=size, color=ORANGE, edgecolor=SURFACE, lw=0.8, zorder=2,
               label="Raw swing (the typology's basis)")
    ax.scatter(d[f"{prefix}_shrunk"], y, s=size, color=BLUE, edgecolor=SURFACE, lw=0.8, zorder=3,
               label="Shrunken estimate")
    ax.axvline(0, color=BASELINE, lw=1.2, ls="--", zorder=0)
    ax.axvline(pop, color=MUTED, lw=1, ls=":", zorder=0)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=10, color=SECONDARY_INK)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRIDLINE, lw=0.8, zorder=0)
    ax.set_facecolor(SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(Path(a.results) / "team_shrinkage.csv")
    n_drawn = int(d["ppda_raw"].notna().sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7), facecolor=SURFACE,
                                   gridspec_kw={"wspace": 0.08})
    panel(ax1, d, "ppda", "PPDA", "Elite minus Weak shift in log(PPDA) (higher = presses less)",
          d["ppda_shrunk"].mean())
    panel(ax2, d, "line", "Defensive line", "Elite minus Weak shift in line height (lower = drops deeper)",
          d["line_shrunk"].mean())
    ax1.legend(loc="upper left", fontsize=9, frameon=False, labelcolor=INK)
    fig.suptitle("Shrinkage pulls nearly every team toward the pooled effect",
                 fontsize=13, color=INK, x=0.01, ha="left", fontweight="bold", y=0.98)
    fig.text(0.01, -0.01,
             f"One row per team ({n_drawn} teams with both an Elite and a Weak match), sorted by raw swing in each panel. "
             "Marker size = Elite + Weak matches behind the raw swing. Dotted line = mean shrunken estimate.",
             fontsize=9, color=MUTED, ha="left")
    fig.text(0.01, -0.045,
             "Random intercept + random slope on tier per team (scripts/team_shrinkage.py). "
             "The slope variance is not distinguishable from zero, which is why the shrunken points barely spread.",
             fontsize=9, color=MUTED, ha="left", style="italic")
    out_path = out_dir / "team_shrinkage.png"
    fig.savefig(out_path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
