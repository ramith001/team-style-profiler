# Team Style Profiler
### Stratified Tactical Tendency Profiling Across International Tournaments
**Ramith Vaid | NTU Singapore | Data Science & Artificial Intelligence**

---

## Overview

Most tactical analysis treats a team's playing style as a single, fixed number, average PPDA across a tournament, average defensive line height across a season. This project starts from a different premise, suggested directly by Will Thomson (Principal Data Scientist, Leeds United, formerly Head of Soccer Research at StatsBomb): a team's tactical identity often changes depending on the opponent, and that variation is itself the interesting signal.

This project stratifies pressing intensity and defensive line height by opponent strength across three full international tournaments, then classifies each team into a typology describing how its approach shifts, or doesn't, against elite opposition versus weaker sides.

---

## A Methodological Detour Worth Documenting

The original plan was to fit a Dixon-Coles style model to estimate team strength directly from each tournament's results. This failed in an instructive way: with only 64 matches and 32 teams in a single World Cup, the model has roughly one free parameter for every match played. The result was statistically valid but footballingly meaningless, Serbia, Iran, Canada and Costa Rica ranked above Argentina and France, the tournament's eventual finalists.

Rather than force a broken model to look reasonable through more regularisation, the model was abandoned in favour of an established, validated alternative: the World Football Elo Ratings (eloratings.net), built from over 150 years of accumulated match history rather than 64 games in isolation. This produced an immediately sensible ranking, Brazil, Argentina and France at the top of the 2022 World Cup field, Saudi Arabia, Ghana and Cameroon at the bottom, exactly as expected.

This is included here deliberately. Knowing when a custom model is underdetermined, and choosing an established alternative rather than forcing a plausible looking but unstable result, is itself a real piece of analytical judgement.

---

## Pipeline

```
StatsBomb event data  --->  schema_mapper.py (Opta format)  --->  tactical_engine.py
                                                                   (PPDA, defensive line)
                                                                          |
                                                                          v
World Football Elo Ratings  --->  elo_loader.py (pre-tournament strength, by tier)
                                                                          |
                                                                          v
                                                          style_profiler.py
                                              (stratification + combined typology)
```

Three tournaments processed end to end: 2022 FIFA World Cup (64 matches), UEFA Euro 2024 (51 matches), Copa América 2024 (32 matches). 294 team-match observations in total. 52 teams have three or more matches; 35 of those have both an Elite and a Weak opponent and so get a combined typology, the rest are labelled "Insufficient data".

---

## Methodology

**Opponent strength tiering.** Each team's Elo rating immediately before a tournament's start is pulled from the historical Elo dataset, then classified into a quartile based tier (Elite, Strong, Moderate, Weak) relative to that specific tournament's field. This means strength is always relative to the actual competition a team is in, not a global, context-free number.

**Per-match metrics.** Reusing the PPDA and defensive line height calculations from the Tactical Verdict Engine (with the corrections described below), every team's performance in every match is reduced to two numbers: how aggressively they pressed, and how high their defensive line sat.

**Stratification.** For each team, matches are grouped by opponent strength tier, and each tier's average is compared back to that team's own overall tournament average, not to a generic league average. A 20% swing threshold (relative to the team's own mean) separates a meaningful tactical shift from normal match to match noise.

**Combined typology.** Rather than reporting PPDA and defensive line shifts separately, the two dimensions are crossed into a single descriptive pattern. This is where the more interesting findings emerged.

---

## Key Findings

These replace an earlier set of findings that were computed with a broken PPDA (see "Corrections" below). Re-running the full stratification with the corrected PPDA, and checking each earlier finding against it:

| Earlier finding | With corrected PPDA | Verdict |
|---|---|---|
| Japan, Senegal: compact aggressive block vs elite | Japan "Fully conservative vs elite", Senegal "Shape adapts, effort doesn't". Both have one Elite match, so the Elite figure is a single game (Japan PPDA 29.0 vs elite, 9.0 vs weak) | Disappears |
| Spain, Portugal: compact aggressive block vs elite, at scale | Spain "Same shape, lower work-rate vs elite", Portugal "Fully conservative vs elite". The press direction reversed (PPDA higher vs elite, so less pressing, not more) | Disappears, reversed |
| Uruguay: False High Line risk vs elite | "Tactically stable". The committed code never produced the False High Line label for Uruguay, even before the fix | Disappears (did not reproduce even before the fix) |
| Argentina: tactically stable | "Same shape, lower work-rate vs elite" (PPDA 18.6 vs elite against 12.3 vs weak, defensive line unchanged) | Shifts |
| France, England: same shape, harder work-rate vs elite | Both "Same shape, lower work-rate vs elite". The committed code gave "Tactically stable" for both | Shifts, and reversed |

Across all 35 teams with both Elite and Weak matches, the corrected PPDA gives 19 "Same shape, lower work-rate vs elite", 9 "Tactically stable", 3 "Fully conservative", 2 "Same shape, harder work-rate", and one each of "Compact aggressive block" and "Shape adapts, effort doesn't". The dominant pattern, higher PPDA against elite opponents, is plausible mechanically: PPDA counts opponent passes, and elite sides pass more, so it rises against them without the team doing anything differently. It should not be read as a deliberate "sit off the best teams" plan.

**These typologies are mostly not distinguishable from noise.** `scripts/typology_stability.py` shuffles each team's opponent-tier labels across its own matches and re-applies the same 20% rule. With the corrected PPDA, 71% of teams are labelled as shifting their press between Elite and Weak opponents, against 66% expected from shuffled tiers (95th percentile 77%); for defensive line, 14% against 10% (95th percentile 17%). Most teams have one to three matches per tier, so a 20% swing is easily a single-match effect. The honest summary is that this dataset does not support strong claims about how individual teams change their press by opponent strength. The method is a framework for when more matches per tier exist (for example club seasons), not a set of established team findings.

---

## Does Opponent Strength Have a Real Effect? A Pooled Mixed-Effects Model

The per-team typology test above is underpowered by design, most teams have three to five matches split across up to four tiers, so a single team's 20% swing is close to indistinguishable from noise. That's the wrong question for this dataset. The right question is whether, pooling all 294 team-matches and letting each team keep its own baseline, pressing intensity and defensive line height move with opponent strength at the population level.

`scripts/mixed_effects_pressing.py` fits two linear mixed-effects models: log(PPDA) and defensive line height, each regressed on opponent tier (Weak as reference) with team fit as a random intercept, so a team with nine matches doesn't outweigh one with three and each team's own general level is absorbed rather than competing with the tier effect. Both effects are real and monotonic. PPDA rises from Weak through Moderate, Strong and Elite (coefficients 0.224, 0.354, 0.420 in log units; p = 0.0017, p < 0.0001, p < 0.0001), meaning PPDA is about 52% higher against Elite opponents than Weak ones on average (exp(0.420) = 1.52), consistent with markedly less intense pressing against the strongest teams. Defensive line height falls the same way, by 3.5, 3.7 and 4.4 units (on the 0-100 pitch scale, roughly 4 metres at the largest) against Moderate, Strong and Elite opponents respectively (all p < 0.0001). A likelihood-ratio test against a team-only null model confirms opponent tier meaningfully improves the fit for both outcomes (PPDA: chi2 = 42.4, df = 3, p = 3.2e-9; line height: chi2 = 34.7, df = 3, p = 1.4e-7).

So the population-level effect is real and not subtle: international teams genuinely press less and sit deeper against stronger opposition, on average. It just isn't reliably detectable one team at a time with three to five matches each. The two results aren't in tension. A population-level mean shift with an intraclass correlation of 0.24-0.33 (most of the variance is match-to-match, not team identity) is exactly the kind of signal that survives pooling but gets swamped by noise in any single team's handful of matches. The per-team typology table is still a useful descriptive summary of each team's own matches, but it should be read that way, not as 35 independently validated findings.

One caveat on the PPDA half of this: PPDA is opponent passes divided by defensive actions, so it also rises when the opponent simply passes more, which elite sides do. Part of the PPDA effect may therefore be possession volume rather than pressing intent; the defensive line result does not have that problem.

---

## Corrections

An earlier version of this project reused pre-fix code from the Tactical Verdict Engine. Verified against real data (the 2022 final, match 3869685) and fixed:

- **Tackles were never counted.** StatsBomb has no "Tackle" event type; tackles are `Duel` events with `duel_type == "Tackle"`. The mapping for `"Tackle"` never matched, and Duel was not in the defensive-action list. Tackle duels now map to Opta type 7.
- **Pressure dominated the PPDA denominator.** Pressure (type 49, previously mislabelled "Ball Recovery") is no longer counted; PPDA uses tackles, interceptions and fouls.
- **Two teams' coordinate frames were mixed.** StatsBomb records each team's events in its own attacking frame, so `x <= 50` applied to both teams selected different physical areas. PPDA now takes opponent completed passes in the opponent's own two thirds over the pressing team's defensive actions in the opposition's two thirds, each in the team's own frame. Other detectors keep the attacking-frame coordinates they were written for.
- **Every pass was recorded as incomplete.** The completed flag tested `is None` against the NaN that `sb.events()` returns.
- **PPDA thresholds recalibrated.** The old 2.5 / 4.4 cut-offs belong to the old metric and would label every team-match "Passive". The corrected metric has median 13.9 (5th to 95th percentile 6.8 to 29.8). Cut-offs are now Aggressive < 9 and Passive >= 20, giving 19% / 60% / 21% over the 294 team-matches (Aggressive: Copa 39%, World Cup 16%, Euro 11%). They are percentile-based on these three tournaments, not an absolute standard, and are the same as in the Tactical Verdict Engine. The corrected PPDA has a Spearman rank correlation of 0.95 with an independently written standard-definition PPDA. (`scripts/calibrate_ppda.py`.) Note that the stratification itself compares each team with its own average and does not depend on the absolute cut-offs.
- 2022 final, Argentina / France: PPDA 1.95 / 2.57 before, 7.89 / 10.45 after.

---

## Known Limitations

**Sample size varies considerably by team.** Teams eliminated early in their tournament, or who only faced a narrow band of opponent strength, do not have enough data for a reliable typology and are explicitly labelled "Insufficient data" rather than forced into a guess.

**Small samples and PPDA's dependence on the opponent.** Most teams have one to three matches in each tier, and PPDA rises mechanically against teams that pass more. See the stability check under Key Findings.

**Home advantage is not yet incorporated.** A genuine home and away split is not meaningful for most international tournaments since every team except the host is technically "away." A more accurate proxy, crowd composition by team allegiance, was investigated and found not to exist in any published or scrapeable form, not even for completed tournaments, partly because FIFA itself has acknowledged it does not reliably track which fans are seated where, particularly at the 2026 World Cup, which has no fan segregation. A travel distance and historical fanbase size proxy was considered as a fallback but not yet implemented.

**The transfer backtest validation method (also Will's suggestion) is not applicable here.** Validating a style classification system by checking whether real player transfers between tactically similar versus dissimilar teams correlate with transfer success is a genuinely strong idea, but it requires club level data, and current club season data for the specific clubs and players involved is not available through StatsBomb's free open data, that tier is their core paid commercial product. This remains a natural extension if club level data becomes accessible.

---

## Reproducing

```
python run_full_stratification.py        # team_matches.csv, combined_profiles.csv in results/
python scripts/calibrate_ppda.py         # PPDA calibration and validation
python scripts/typology_stability.py     # real labels vs shuffled-tier null
python scripts/mixed_effects_pressing.py # pooled population-level tier effect
python -m pytest tests                   # mapping and PPDA regression tests
```

---

## Stack

- **Language**: Python 3.13
- **Libraries**: pandas, numpy, statsbombpy, statsmodels, scipy
- **Data sources**: StatsBomb open data (event coordinates), World Football Elo Ratings via eloratings.net (team strength)
- **Built on**: schema_mapper.py and tactical_engine.py from the Tactical Verdict Engine project

---

## Acknowledgement

The core idea behind this project, stratifying tactical metrics by opponent context rather than reporting a single static average, came directly from a conversation with Will Thomson (Principal Data Scientist, Leeds United). The combined typology framework, the cross referencing of pressing intensity and defensive line height into a single descriptive pattern, was developed from that starting point.

---

*Built by Ramith Vaid | NTU Singapore | ramith001@e.ntu.edu.sg*
