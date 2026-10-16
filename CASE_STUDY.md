# Does a team's style change with the opponent? A case study

**Ramith Vaid | NTU Singapore | Data Science & Artificial Intelligence**

Most tactical analysis reduces a team's playing style to one number: average PPDA across a tournament, average defensive line height across a season. The idea behind this project, suggested by Will Thomson (Principal Data Scientist, Leeds United), is that a team's tactical identity often changes depending on who it is playing, and that the variation is itself the signal.

The question: **across six international tournaments, do teams press less and defend deeper against stronger opponents, and how much of that is real behaviour rather than an artefact of how the metrics are built?**

This is the narrative version. The engineering log (corrections, reproduction commands, the full list of limitations) lives in [README.md](README.md).

---

## 1. The data

![Dataset overview: team-matches per tournament and per opponent-strength tier](results/figures/dataset_overview.png)

Six tournaments, processed end to end from StatsBomb open event data: the 2022 and 2018 FIFA World Cups (64 matches each), UEFA Euro 2024 and 2020 (51 matches each), Copa América 2024 (32 matches) and the 2023 Africa Cup of Nations (52 matches). That is 628 team-matches across 76 teams.

Opponent strength comes from each team's World Football Elo rating immediately before the tournament, classified into quartile tiers (Elite, Strong, Moderate, Weak) relative to that tournament's own field. Every team-match is then reduced to two numbers: how aggressively the team pressed (PPDA) and how high its defensive line sat.

---

## 2. The pooled result: the effect is real

![Forest plot of pooled mixed-effects tier coefficients](results/figures/forest_plot_tier_effects.png)

Pooling all 628 team-matches and giving each team its own random intercept, both outcomes move monotonically with opponent tier (Weak is the reference).

- **PPDA** rises through Moderate, Strong and Elite: coefficients 0.149, 0.300 and 0.404 in log units (p = 0.0027, p < 0.0001, p < 0.0001). That is PPDA about 50% higher against Elite opponents than Weak ones on average (exp(0.404) = 1.50).
- **Defensive line height** falls by 2.8, 3.7 and 4.9 units against Moderate, Strong and Elite opponents (0-100 pitch scale, roughly 3-5 metres; all p < 0.0001).
- A likelihood-ratio test against a team-only null confirms tier improves the fit for both outcomes (PPDA: chi2 = 84.0, df = 3, p = 4.2e-18; line height: chi2 = 81.6, df = 3, p = 1.4e-17).

The effect is not subtle at the population level. It is just hard to see one team at a time: the intraclass correlation is around 0.22-0.24, meaning most variance is match-to-match, which survives pooling but swamps any single team's handful of matches.

The defensive line result is a direct physical measurement and stands on its own. The PPDA result needs one more test, because PPDA is a ratio.

---

## 3. Stress-testing the headline: is it pressing, or is it possession?

PPDA is the opponent's completed passes (in their own defensive two-thirds) divided by the pressing team's tackles, interceptions and fouls (in the opposition's two-thirds). It can rise because the pressing team does less, or because the opponent simply does more of the numerator. Elite sides are expected to have the ball more regardless of how hard they are pressed.

So the project's own biggest finding was put through a decomposition (`scripts/possession_confound.py`). Opponent tier does move opponent pass volume in that zone (log(opp_passes): Strong +0.231, Elite +0.380, both p < 0.0001; Moderate +0.028, not significant). Adding that volume to the PPDA model as a covariate splits the tiers apart:

![Before/after PPDA tier coefficients, with and without controlling for opponent pass volume](results/figures/possession_confound.png)

- **Moderate survives almost unchanged:** 0.135, p = 0.0002.
- **Strong shrinks but holds:** 0.088, p = 0.0145.
- **Elite collapses:** from 0.404 to 0.048, and loses significance entirely (p = 0.19).

These are different claims. Against Moderate and Strong opponents, teams commit fewer raw defensive actions in the opposition's third (log(press_actions): Moderate -0.134 and Strong -0.082, both significant), a real if modest drop in work-rate. Against Elite opponents, the large PPDA rise is mostly mechanical: the opponent keeps the ball in that zone more, and the pressing team's own defensive activity there barely changes (Elite -0.039, p = 0.24, not significant). "Teams press less against the best teams" is the wrong headline for the Elite tier. "Teams have less of the ball to press against the best teams, and otherwise keep roughly their usual work-rate" is closer to what the data shows.

The defensive line result does not have this problem, since line height is not built from the opponent's output.

---

## 4. Individual teams around the pooled pattern

![Per-team typology scatter: PPDA shift vs. defensive line shift, Elite minus Weak opponents](results/figures/typology_scatter.png)

Each team with both an Elite and a Weak opponent (54 of 76) is placed by how its PPDA and defensive line shift between those two tiers, then labelled by a combined typology. The distribution is 30 "Same shape, lower work-rate vs elite", 13 "Tactically stable", 7 "Fully conservative", 3 "Same shape, harder work-rate" and 1 "Shape adapts, effort doesn't". The dominant pattern matches the pooled result: higher PPDA against elite opponents.

The minority patterns are visible in the labelled points. Italy, Côte d'Ivoire and Switzerland are the three "harder work-rate" teams, sitting left of the zero line on the PPDA axis (lower PPDA against Elite opposition, so pressing more rather than less). Sweden is the single "Shape adapts, effort doesn't" team.

Individual labels should be read cautiously. A shuffle test (`scripts/typology_stability.py`) re-applies the same 20% rule to shuffled tier labels: 74% of teams are labelled as shifting their press between Elite and Weak, against 61% expected from shuffled tiers (95th percentile 70%), so the PPDA typology now clears the noise floor. The defensive-line typology does not: 15% against 10% expected (95th percentile 17%). Most teams still have one to three matches per tier within any tournament, so a single team's label can rest on one game.

---

## 5. What is established, and what is not

**Established**
- At the population level, defensive line height falls with opponent strength, and PPDA rises with it, across 628 team-matches and 76 teams.
- For PPDA, the Moderate and Strong effects survive controlling for opponent pass volume. The Elite effect does not.
- The per-team PPDA typology is distinguishable from shuffled noise.

**Open**
- **Sample size varies by team.** A team appearing in one tournament is still on three to five matches, and teams without both an Elite and a Weak opponent are labelled "Insufficient data" rather than forced into a guess.
- **The per-team defensive-line typology is not distinguishable from noise**, even though the pooled line-height effect is clear.
- **Home advantage is not modelled.** Host-versus-away is not meaningful in most international tournaments, and a proxy based on crowd composition was investigated and found not to exist in usable form.

The full list, including the transfer-backtest validation that club-level data would be needed for, is under "Known Limitations" in [README.md](README.md).
