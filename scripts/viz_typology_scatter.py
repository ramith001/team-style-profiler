"""
Per-team typology scatter: where does each team sit in the two-dimensional
space the combined typology (style_profiler.py) actually classifies on?

X axis: shift in PPDA from Weak-tier to Elite-tier opponents (positive = the
team presses less hard against stronger opposition).
Y axis: shift in defensive line height from Weak-tier to Elite-tier opponents
(negative = the team drops its line deeper against stronger opposition).

Every team with at least one Weak-tier and one Elite-tier match (the same 54
teams results/combined_profiles.csv assigns a combined_typology to) gets a
point, colored by that typology. The two smallest categories (3 and 1 teams)
would need a 4th/5th hue that fails the palette's all-pairs CVD floor at this
series count, so they're kept in muted ink with distinct marker shapes and
direct team labels instead of a new color, per the skill's accessibility
guidance (secondary encoding, never a hue nobody can tell apart).

    python scripts/viz_typology_scatter.py [--results results] [--out results/figures]
"""
import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# dataviz skill palette (references/palette.md). Slots 1-3 (blue/orange/aqua)
# are the only ones that clear the all-pairs CVD floor at this series count,
# so the two smallest categories use muted ink + shape instead of slots 4/5.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Fixed order: the three biggest categories keep the validated hue slots;
# the two smallest (3 teams, 1 team) get muted ink + a distinct marker shape.
CATEGORY_STYLE = {
    "Same shape, lower work-rate vs elite":   {"color": BLUE,   "marker": "o"},
    "Tactically stable regardless of opponent": {"color": ORANGE, "marker": "o"},
    "Fully conservative vs elite":            {"color": AQUA,   "marker": "o"},
    "Same shape, harder work-rate vs elite":  {"color": MUTED,  "marker": "^"},
    "Shape adapts, effort doesn't":           {"color": MUTED,  "marker": "D"},
}
LEGEND_ORDER = list(CATEGORY_STYLE.keys())


def parse_tier_dict(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    results_dir, out_dir = Path(a.results), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "combined_profiles.csv")
    df["ppda_tiers"] = df["ppda_tiers"].apply(parse_tier_dict)
    df["line_tiers"] = df["line_tiers"].apply(parse_tier_dict)

    rows = []
    for _, row in df.iterrows():
        pt, lt = row["ppda_tiers"], row["line_tiers"]
        if {"Elite", "Weak"} <= pt.keys() and {"Elite", "Weak"} <= lt.keys():
            rows.append({
                "team": row["team"],
                "typology": row["combined_typology"],
                "ppda_shift": pt["Elite"] - pt["Weak"],
                "line_shift": lt["Elite"] - lt["Weak"],
            })
    pts = pd.DataFrame(rows)
    print(f"{len(pts)} teams with both Weak and Elite tiers (matches the README's 54-team figure)")
    print(pts["typology"].value_counts())

    fig, ax = plt.subplots(figsize=(9, 7.5), facecolor=SURFACE)
    ax.axvline(0, color=BASELINE, lw=1.2, ls="--", zorder=1)
    ax.axhline(0, color=BASELINE, lw=1.2, ls="--", zorder=1)

    small_categories = {"Same shape, harder work-rate vs elite", "Shape adapts, effort doesn't"}
    for typology in LEGEND_ORDER:
        sub = pts[pts["typology"] == typology]
        if sub.empty:
            continue
        style = CATEGORY_STYLE[typology]
        ax.scatter(sub["ppda_shift"], sub["line_shift"], s=70, color=style["color"],
                   marker=style["marker"], alpha=0.85, edgecolor=SURFACE, linewidth=0.8,
                   zorder=3, label=f"{typology} (n={len(sub)})")
        if typology in small_categories:
            # Côte d'Ivoire and Switzerland sit within ~0.6 units of each other, so a
            # fixed offset collides; give the four labeled points distinct offsets.
            label_offsets = {
                "Côte d'Ivoire": (10, 10),
                "Switzerland": (10, -12),
            }
            for _, r in sub.iterrows():
                xytext = label_offsets.get(r["team"], (7, 5))
                ax.annotate(r["team"], (r["ppda_shift"], r["line_shift"]),
                            textcoords="offset points", xytext=xytext, fontsize=9, color=INK)

    ax.set_xlabel("Shift in PPDA, Elite minus Weak opponents (higher = presses less)",
                  fontsize=10, color=SECONDARY_INK)
    ax.set_ylabel("Shift in defensive line height, Elite minus Weak opponents (lower = drops deeper)",
                  fontsize=10, color=SECONDARY_INK)
    ax.set_title("Where each team sits in the typology's own two dimensions (n=54)",
                 fontsize=13, color=INK, loc="left", pad=14, fontweight="bold")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRIDLINE, lw=0.8, zorder=0)
    ax.set_facecolor(SURFACE)

    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9.5,
                       frameon=False, labelcolor=INK, handletextpad=0.6)
    fig.text(0.01, -0.02,
              "Each point is a team. Color/shape = the combined_typology label from style_profiler.py. "
              "The two smallest categories are labeled directly rather than given a 4th/5th hue, "
              "which would fail the palette's colorblind-safety check at this series count.",
              fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out_path = out_dir / "typology_scatter.png"
    fig.savefig(out_path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
