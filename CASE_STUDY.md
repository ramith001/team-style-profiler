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

## 4. Stress-testing the line height: is it respect, or protecting a lead?

A team might drop deep against a strong opponent out of respect for it, or simply because it is winning and sitting back, which would happen whoever it was playing. `scripts/game_state_confound.py` rebuilds the running score for every match and uses each team-match's share of time spent leading as a covariate, the same before/after structure as the possession check.

Game state does depend on the opponent. Against Moderate, Strong and Elite opponents a team spends 0.077, 0.162 and 0.242 less of the match leading than against Weak ones (raw means 0.375 Weak, 0.295 Moderate, 0.210 Strong, 0.129 Elite), and leading is itself associated with a lower line (-2.6 units per full match spent leading, p < 0.0001). Controlling for it does not shrink the tier effect, it enlarges it slightly: Moderate, Strong and Elite go from -2.8, -3.7 and -4.9 to -3.0, -4.1 and -5.5 units (all p < 0.0001). Stronger opponents put a team behind more often, and trailing teams push their line up, which works against the drop. Holding game state fixed removes that offset, so the line-height drop is not a protect-the-lead artefact. If anything, game state was masking part of it.

Two limits apply. Game state is measured per match while line height is a whole-match average, so this cannot show when in a match a team dropped. And game state is partly an effect of how a team defends as well as a cause, so the controlled coefficients are a sensitivity check and not a clean causal decomposition.

---

## 5. Individual teams around the pooled pattern

![Per-team typology scatter: PPDA shift vs. defensive line shift, Elite minus Weak opponents](results/figures/typology_scatter.png)

Each team with both an Elite and a Weak opponent (54 of 76) is placed by how its PPDA and defensive line shift between those two tiers, then labelled by a combined typology. The distribution is 30 "Same shape, lower work-rate vs elite", 13 "Tactically stable", 7 "Fully conservative", 3 "Same shape, harder work-rate" and 1 "Shape adapts, effort doesn't". The dominant pattern matches the pooled result: higher PPDA against elite opponents.

The minority patterns are visible in the labelled points. Italy, Côte d'Ivoire and Switzerland are the three "harder work-rate" teams, sitting left of the zero line on the PPDA axis (lower PPDA against Elite opposition, so pressing more rather than less). Sweden is the single "Shape adapts, effort doesn't" team.

Individual labels should be read cautiously. A shuffle test (`scripts/typology_stability.py`) re-applies the same 20% rule to shuffled tier labels: 74% of teams are labelled as shifting their press between Elite and Weak, against 61% expected from shuffled tiers (95th percentile 70%), so the PPDA typology now clears the noise floor. The defensive-line typology does not: 15% against 10% expected (95th percentile 17%). Most teams still have one to three matches per tier within any tournament, so a single team's label can rest on one game.

The scatter invites reading each point as a team's identity. A second test says to be careful. `scripts/team_shrinkage.py` refits the pooled models with a random slope on opponent tier per team and takes each team's shrunken Elite-minus-Weak effect, which covers all 76 teams, including the 22 with too little data for a typology. The between-team variation in the slope is small and not statistically distinguishable from zero (PPDA chi2 = 3.16, p = 0.21; line height chi2 = 0.17, p = 0.92). Shrinkage therefore pulls almost every team to the pooled value: the spread of per-team PPDA swings falls from an SD of 0.373 raw to 0.078 shrunken, and for line height from 4.97 to 0.31. The shrunken ranking only weakly resembles the raw one for PPDA (correlation 0.14, against 0.83 for line height).

![Raw vs. shrunken per-team Elite-minus-Weak effects](results/figures/team_shrinkage.png)

This agrees with the shuffle test rather than overturning it. The typology is better read as a population effect that most teams share than as a set of distinct team identities, and the 22 teams without a typology come out at essentially the pooled effect, which says the data cannot separate them from the average team, not that they behave like it.

---

## 6. Could host nations be driving it?

Hosts may play differently at home, and they are not spread evenly across tiers. `scripts/host_sensitivity.py` flags the 23 of 628 team-matches played by tournament hosts (Qatar 2022, Russia 2018, Germany at Euro 2024, Cote d'Ivoire at AFCON 2023 and the United States at Copa America 2024) and refits the pooled models with a host covariate and with host matches dropped. The covariate is not significant for either outcome (PPDA +0.077, p = 0.50; line height -0.065, p = 0.96), and the tier coefficients barely move. PPDA Moderate, Strong and Elite go from 0.149, 0.300 and 0.404 to 0.161, 0.302 and 0.417 with hosts excluded, and line height from -2.8, -3.7 and -4.9 to -2.9, -3.7 and -4.9.

This is a weak test. 23 matches is about one team's worth per tournament, and Euro 2020, which had no single host, is not flagged. It can rule out a large host confound but not a subtle one.

---

## 7. What is established, and what is not

**Established**
- At the population level, defensive line height falls with opponent strength, and PPDA rises with it, across 628 team-matches and 76 teams.
- The defensive line drop is not a protect-the-lead artefact: it survives controlling for time spent leading, and gets slightly larger.
- For PPDA, the Moderate and Strong effects survive controlling for opponent pass volume. The Elite effect does not.
- The tier effects are essentially unchanged with host-nation matches excluded (a weak test, 23 matches).
- The per-team PPDA typology is distinguishable from shuffled noise.

**Open**
- **Sample size varies by team.** A team appearing in one tournament is still on three to five matches, and teams without both an Elite and a Weak opponent are labelled "Insufficient data" rather than forced into a guess.
- **The per-team defensive-line typology is not distinguishable from noise**, even though the pooled line-height effect is clear. Shrinkage also finds little evidence that teams differ in how they respond to opponent strength at all, so per-team labels are best read as descriptions of a few matches, not stable identities.
- **Game state is measured per match.** The lead-protection check cannot show when in a match a team dropped its line.
- **Home advantage is not modelled.** Host-versus-away is not meaningful in most international tournaments, and a proxy based on crowd composition was investigated and found not to exist in usable form. The host-nation check above is a weak substitute.

The full list, including the transfer-backtest validation that club-level data would be needed for, is under "Known Limitations" in [README.md](README.md).
