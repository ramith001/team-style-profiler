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

Three tournaments processed end to end: 2022 FIFA World Cup (64 matches), UEFA Euro 2024 (51 matches), Copa América 2024 (32 matches). 294 team-match observations in total, with 52 teams carrying enough matches across enough strength tiers to support a full combined typology.

---

## Methodology

**Opponent strength tiering.** Each team's Elo rating immediately before a tournament's start is pulled from the historical Elo dataset, then classified into a quartile based tier (Elite, Strong, Moderate, Weak) relative to that specific tournament's field. This means strength is always relative to the actual competition a team is in, not a global, context-free number.

**Per-match metrics.** Reusing the PPDA and defensive line height calculations already built for the Tactical Verdict Engine, every team's performance in every match is reduced to two numbers: how aggressively they pressed, and how high their defensive line sat.

**Stratification.** For each team, matches are grouped by opponent strength tier, and each tier's average is compared back to that team's own overall tournament average, not to a generic league average. A 20% swing threshold (relative to the team's own mean) separates a meaningful tactical shift from normal match to match noise.

**Combined typology.** Rather than reporting PPDA and defensive line shifts separately, the two dimensions are crossed into a single descriptive pattern. This is where the more interesting findings emerged.

---

## Key Findings

**Japan and Senegal: Compact aggressive block vs elite.** Both nations dropped into a notably deeper defensive line against Elite opposition while simultaneously pressing harder once the ball entered their own third. This is not a passive defensive identity, it is a deliberately compact, ambush style approach, and it matches the widely remembered story of Japan's wins over Germany and Spain in the 2022 World Cup: sit deep, then press intensely and counter.

**Uruguay: False High Line risk vs elite.** Uruguay's defensive line was unusually higher against Elite opposition specifically, while their pressing intensity simultaneously dropped. A high defensive line combined with passive pressing is precisely the structural vulnerability the Tactical Verdict Engine's False High Line detector was built to flag at the individual match level. This stratified team level profiling surfaced the same risk pattern independently, across a whole tournament's worth of matches, providing a direct link between the two projects.

**Spain and Portugal: Compact aggressive block vs elite, at scale.** With the largest sample sizes in the dataset (11 and 10 matches respectively across two tournaments), both nations showed the same compact-but-intense pattern as Japan and Senegal, suggesting this may be a more common approach against elite opposition than the high press, possession dominant identity often assumed of teams at this level.

**Argentina, France, England, Netherlands: Tactically stable or work-rate scaling.** Argentina remained "Tactically stable" across 13 matches spanning two tournaments, the same conclusion held with one tournament's data alone, a good sign of a genuinely repeatable signal rather than noise. France and England both showed "Same shape, harder work-rate vs elite," holding the same defensive line height regardless of opponent, but pressing visibly harder against the best teams specifically.

---

## Known Limitations

**Sample size varies considerably by team.** Teams eliminated early in their tournament, or who only faced a narrow band of opponent strength, do not have enough data for a reliable typology and are explicitly labelled "Insufficient data" rather than forced into a guess.

**Home advantage is not yet incorporated.** A genuine home and away split is not meaningful for most international tournaments since every team except the host is technically "away." A more accurate proxy, crowd composition by team allegiance, was investigated and found not to exist in any published or scrapeable form, not even for completed tournaments, partly because FIFA itself has acknowledged it does not reliably track which fans are seated where, particularly at the 2026 World Cup, which has no fan segregation. A travel distance and historical fanbase size proxy was considered as a fallback but not yet implemented.

**The transfer backtest validation method (also Will's suggestion) is not applicable here.** Validating a style classification system by checking whether real player transfers between tactically similar versus dissimilar teams correlate with transfer success is a genuinely strong idea, but it requires club level data, and current club season data for the specific clubs and players involved is not available through StatsBomb's free open data, that tier is their core paid commercial product. This remains a natural extension if club level data becomes accessible.

---

## Stack

- **Language**: Python 3.13
- **Libraries**: pandas, numpy, statsbombpy
- **Data sources**: StatsBomb open data (event coordinates), World Football Elo Ratings via eloratings.net (team strength)
- **Built on**: schema_mapper.py and tactical_engine.py from the Tactical Verdict Engine project

---

## Acknowledgement

The core idea behind this project, stratifying tactical metrics by opponent context rather than reporting a single static average, came directly from a conversation with Will Thomson (Principal Data Scientist, Leeds United). The combined typology framework, the cross referencing of pressing intensity and defensive line height into a single descriptive pattern, was developed from that starting point.

---

*Built by Ramith Vaid | NTU Singapore | ramith001@e.ntu.edu.sg*
