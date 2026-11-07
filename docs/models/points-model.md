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
