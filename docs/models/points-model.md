# Model Card: Points

Code: `src/models/prop_model.py points`, `src/models/prop_market_blend.py points`.
Results: `docs/models/points_results.json`, `docs/models/points_blend_results.json`.
Same method as the shots model ([sog-model.md](sog-model.md)); only the
target and features differ.

## Setup

Target: points (goals + assists) per skater-game. SportsGameOdds labels this
market `goals+assists`; its `points` market is goals (see sgo-props audit).

Features (30): recent, season, career and previous-season points, goals,
assists and power-play points; shots; points per 60; ice time and
power-play time; defenseman, home, back-to-back and playoff flags; own
team's goals and shots for, opponent's goals and shots allowed (last 20
games); the opposing starter's save percentage over his previous 40
appearances, shrunk toward .905 (leakage-tested). The opposing-goalie
feature uses the goalie who actually started (gap G-E) and is missing for
2026-11-03 to 2026-11-06, where the goalie logs end; those rows use .905.

Fitted dispersion alpha goes to its lower bound (0.0001): points are not
over-dispersed, so the distribution is effectively Poisson.

## Results

### All player-games (log-likelihood per game, higher is better)

| Test season | Player-games | baseline | glm | gbm | Actual zero-point rate | glm predicted |
|-------------|-------------:|---------:|----:|----:|-----:|-----:|
| 2021-22 | 49,645 | -0.8561 | -0.8419 | -0.8522 | 64.9% | 65.7% |
| 2022-23 | 49,860 | -0.8595 | -0.8440 | -0.8444 | 64.6% | 64.7% |
| 2023-24 | 49,889 | -0.8480 | -0.8331 | -0.8339 | 65.5% | 65.2% |
| 2024-25 | 49,800 | -0.8401 | -0.8268 | -0.8270 | 65.7% | 67.7% |
| 2025-26 | 49,627 | -0.8503 | -0.8368 | -0.8375 | 65.2% | 66.4% |
| 2026-27 (to 11-06) | 9,323 | -0.8690 | -0.8540 | -0.8561 | 64.5% | 66.2% |

Both fitted models beat the baseline every season; the GLM is as good as or
better than boosting here.

### Against closing prop prices

18,851 graded lines in 910 games (0.5 points: 17,921; 1.5: 930). Overs hit
42.8%; median margin 7.0%.

| | 2025-26 log loss | 2026-27 log loss |
|---|---:|---:|
| Market (de-vigged close) | 0.6621 | 0.6340 |
| glm | 0.6642 | 0.6405 |
| gbm | 0.6653 | 0.6411 |

Market + model blend fitted on 2025-26 lines and tested on 2026-27: the
model's weight in training is 0.43 (SE 0.11) for glm, but on 2026-27 the blend
scores 0.6341 against the market's 0.6340. **No out-of-sample gain.** Betting
the blend loses at every threshold (4-point edge: glm -2.8%, SE 2.5%, 867
bets; gbm -4.4%, SE 2.7%, 723 bets).

### Why

The model under-predicts the players books post lines on, by about 5-10% in
every scoring tier (top fifth: 0.92 predicted vs 1.01 actual points per
game). League scoring rose in 2025-26 and 2026-27, and a model fitted on
earlier seasons lags it. The blend corrects the overall level and still
gains nothing, so the gap is information, not calibration: points depend on
linemates, power-play unit and deployment, which the closing price knows on
game day and these features do not.

## Conclusion

The points model is a sound projection (it beats the baseline every season)
but has no demonstrated edge over the closing price. It is not used for
betting. Possible next steps, each tested only on games after 2026-11-06:
line combinations and power-play unit, an as-of league scoring-level
feature, and separate goal and assist models.


## Version 2: lineup features (2026-11-07)

Added five features describing each player's role in that night's lineup,
built only from the dressed skaters' pregame form (assumption L-1: lineups
are announced before the game): ice-time rank at his position, power-play
rank, mean recent points of the five teammates nearest him in ice-time rank
(a linemate proxy), the team's summed recent points among dressed skaters,
and that sum's gap from the team's previous 10 games (scorers out injured).
Leakage-tested in `tests/test_skater_feature_leakage.py`. Version 1 results
are kept in `points_v1_results.json` and `points_v1_blend_results.json`.

| | v1 | v2 |
|---|---:|---:|
| GLM log-likelihood per game, 2025-26 (higher is better) | -0.83683 | -0.83587 |
| GLM log-likelihood per game, 2026-27 | -0.85369 | -0.85301 |
| GLM log loss on 18,902 prop lines (market 0.6515) | 0.6551 | 0.6542 |
| GLM log loss on 2026-27 lines (market 0.6333) | 0.6394 | 0.6382 |
| Market + GLM blend on 2026-27 | 0.6332 | 0.6332 |
| Blend at a 4-point edge, 2026-27 | -2.4% (824 bets) | -4.1% (1,042 bets) |

The lineup features improve the model in every season, but the gap to the
closing price stays, and the blend still gains nothing on held-out lines.
The proxies capture role; the market also knows actual line combinations
and same-day news. The next step would be linemates measured from shift
data (who was on the ice together in recent games), which the NHL publishes
per game and a Colab cell could collect.

## Variants with new data (2026-11-09)

Code: `src/models/prop_model.py points_lm`, `points_edge`,
`points_lm_edge`, `points_mp`, `points_all`. Features:
`src/features/build_linemates.py` (NHL shift charts,
`src/ingest/fetch_shiftcharts.py`), `build_edge_features.py`,
`build_mp_features.py`; each has a leakage test. Points v2 stays the
reference.

**Linemates from shift charts.** For each player-game, from his previous 5
games: his teammates' recent points per game weighted by seconds shared at
5-on-5 (`lm_ev_pts`) and on the power play (`lm_pp_pts`), the share of his
usual 5-on-5 time with teammates dressed tonight (`lm_ev_present`), his
share of the team's power-play time (`pp_unit_share`), and how many of the 5
games have shift data. 9,314 of 9,421 games have shift charts; the NHL
returns none for the last 57 games of the 2024-25 regular season and for
2026-27 games 154-203. Shift ice time matches the box score within 30
seconds for 99.9% of skater-games (power play 97.9%; even strength 75%,
because empty-net time is not counted as even strength here).

| GLM | v2 | + Edge | + linemates | + linemates + Edge | + MoneyPuck | + all three |
|---|---:|---:|---:|---:|---:|---:|
| Log-likelihood per game, 2025-26 | -0.83587 | -0.83556 | -0.83580 | -0.83547 | -0.83499 | **-0.83496** |
| Log-likelihood per game, 2026-27 | -0.85301 | -0.85213 | -0.85174 | -0.85102 | -0.85134 | **-0.85029** |
| Log loss, 2026-27 lines (market 0.6333) | 0.6382 | 0.6374 | 0.6372 | 0.6366 | 0.6368 | **0.6359** |
| Blend on 2026-27 | 0.6332 | 0.6329 | 0.6330 | **0.6327** | 0.6334 | 0.6329 |
| Blend at a 4-point edge | -4.1%, 1,042 (2.4) | -3.5%, 943 (2.5) | -0.9%, 1,142 (2.3) | -2.0%, 1,028 (2.5) | -3.5%, 1,687 (2.0) | -2.4%, 1,640 (1.9) |

Every source improves the points model, and together they close about half
of the gap to the closing price on 2026-27 (0.6382 to 0.6359, market
0.6333). The blend still gains only 0.0004-0.0006 over the market, and no
rule is profitable. Points remain a projection, not a betting model
(decision 35).
