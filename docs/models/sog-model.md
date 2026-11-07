# Model Card: Shots on Goal

Code: `src/models/prop_model.py shots`, `src/models/prop_market_blend.py sog`.
Results: `docs/models/sog_results.json`, `docs/models/sog_blend_results.json`.
Features: `src/features/build_skater_features.py` (leakage-tested).

## What it predicts

The number of shots on goal a skater records in a game, as a negative
binomial distribution (mean mu, Var = mu + alpha * mu^2). The probability of
going over a prop line L is P(X > L). Whole-number lines are graded with
pushes removed.

## Models

| Name | Method |
|------|--------|
| baseline | mu = the player's exponentially weighted shots per game (half-life 10 games), no fitting |
| glm | Poisson regression, log link, 20 standardized features |
| gbm | Histogram gradient boosting with Poisson loss (300 trees, learning rate 0.05, min 200 rows per leaf) |

Features: recent, season, career and previous-season shot rates; ice time and
power-play time; shots per 60; points form; games played this season;
defenseman, home, back-to-back and playoff flags; own team's shots for and
opponent's shots allowed over their last 20 games. Players need at least 5
earlier games in the data.

## Validation

Walk-forward by season: each test season is predicted by models trained only
on earlier seasons (training starts 2019-20). The negative binomial alpha is
fitted on the training seasons only.

### All player-games (negative binomial log-likelihood per game, higher is better)

| Test season | Player-games | baseline | glm | gbm | Actual zero-shot rate | gbm predicted |
|-------------|-------------:|---------:|----:|----:|-----:|-----:|
| 2021-22 | 49,645 | -1.6292 | -1.6233 | -1.6234 | 22.7% | |
| 2022-23 | 49,860 | -1.6142 | -1.6039 | -1.6039 | 23.2% | 22.0% |
| 2023-24 | 49,889 | -1.5877 | -1.5807 | -1.5784 | 24.7% | 23.3% |
| 2024-25 | 49,800 | -1.5563 | -1.5486 | -1.5470 | 26.5% | 25.9% |
| 2025-26 | 49,627 | -1.5465 | -1.5387 | -1.5368 | 27.1% | 26.6% |
| 2026-27 (to 11-06) | 9,323 | -1.5386 | -1.5285 | -1.5259 | 27.2% | 27.0% |

The fitted models beat the baseline in every season, and gbm is best from
2023-24 on. League shot volume per skater has fallen each season (1.69 to
1.53 per game); the models track it because recent-form features lead.

### Against closing prop prices (SportsGameOdds consensus close)

15,777 graded lines in 910 games, 2026-01-17 to 2026-11-06 (lines 1.5: 9,697;
2.5: 4,970; 3.5: 1,025). Median margin 7.0%; overs hit 47.0%.

| | Log loss | Note |
|---|---:|---|
| Market (de-vigged close) | 0.6815 | |
| Constant | 0.6914 | |
| baseline | 0.6874 | |
| glm | 0.6844 | |
| gbm | 0.6821 | beats the market in 2026-27 (0.6779 vs 0.6797), not in 2025-26 (0.6845 vs 0.6825) |

Betting the model alone at a 3-point edge loses (gbm -1.8%, SE 0.95%, on
9,641 bets), mostly unders. The model's probabilities are too extreme: when
gbm says under 30%, the over hits 31.7%; when it says 60-70%, it hits 59.2%.

### Does the model add information beyond the market?

Logistic regression y ~ logit(p_market) + logit(p_model), fitted on the
2025-26 lines (9,963) and tested on 2026-27 lines (5,814) that it never saw:

| | Market weight | Model weight (SE, game-clustered) | 2026-27 log loss |
|---|---:|---:|---:|
| Market alone | | | 0.6797 |
| Market + glm | 0.79 | 0.27 (0.09) | 0.6773 |
| Market + gbm | 0.70 | 0.35 (0.09) | 0.6766 |

The model's weight is about 3.9 standard errors above zero: it carries shot
information the closing price has not fully priced.

Betting the blend on 2026-27 at the closing price (rerun 2026-11-07 after games
from 2026-11-03 to 11-06 gained real team and goalie context; the first run,
with those filled by league averages, gave +9.2% on 824 bets at 4 points; a
later fix to player-name matching added 29 lines to 2025-26, which moved the
blend coefficients slightly and the 4-point result from 839 bets at +8.4% to
the figures below):

| Edge threshold | Bets | ROI | SE (game-clustered) |
|---:|---:|---:|---:|
| 2 points | 2,725 | +1.2% | 1.6% |
| 4 points | 907 | +8.3% | 2.7% |
| 6 points | 262 | +19.9% | 4.5% |

## Limits (read before quoting the ROI)

1. **One short test period.** About 250 games of one season. The thresholds
   were not fixed in advance and three are shown; the 2-point result is
   within noise. Treat the ROI as a hypothesis to confirm on the rest of
   2026-27, not as a result.
2. **The price may not be bettable.** The consensus closing price is a
   summary across books, not a quote from one book. Individual books may
   offer better or worse.
3. **The model is at an information disadvantage, not an advantage.**
   Features stop at the previous game; the closing price knows same-day
   lineups and news. A contaminated (in-game) market price would make the
   market look better, not worse, so it cannot explain the model's weight.
4. **7.5% of prop lines are unmatched** to an NHL player by name and are
   left out.
5. Bets in the same game are correlated; standard errors are clustered by
   game for that reason.

## Next

Live tracking on games after 2026-11-06 with the rule fixed in advance:
[docs/live/README.md](../live/README.md).


## Player matching (2026-11-07)

An alias list for 12 players the provider lists under another first name
(Nick/Nicholas Robertson, Freddy/Frederick Gaudreau and others) cut unmatched
shots and points lines from 3.4% to 3.1%. The rest are players with no
box-score row that night (scratched), whose bets are void. Current figures:
15,806 graded shots lines; model 0.6821 vs market 0.6815; blend on 2026-27
0.6764 vs 0.6796, model weight 0.36 (SE 0.09).
