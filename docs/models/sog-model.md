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

![Walk-forward gain over the baseline](../figures/sog_walk_forward.svg)

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

![Calibration of the model and the closing line](../figures/sog_calibration.svg)

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

## Closing-line value at the opening price (2026-11-07)

Code: `src/models/sog_clv.py`. Results: `docs/models/sog_clv_results.json`.

The question: are the model's picks, taken at the opening price, worth more
than the closing price says? Data: SportsGameOdds consensus opening and
closing prices on all 15,806 graded lines from 2026-01-16 to 2026-11-06
(1,076 games). 91.4% of lines close at the line they opened on. Median margin
is 7.1% at the open and 7.0% at the close. The model's opening probability
uses the gradient-boosted negative binomial at the opening line. That model
was trained on earlier seasons only, so every line is out of sample.

**1. The market moves toward the model.** On same-line lines, the move from
the opening to the closing fair probability (over side) is regressed on the
model's opening edge. Standard errors are clustered by game.

| Predictor | Slope | SE | R² |
|-----------|------:|---:|---:|
| Model | 0.207 | 0.007 | 0.181 |
| Model-free recent average (EWMA of shots) | 0.151 | 0.005 | 0.148 |
| Both: model / baseline | 0.148 / 0.069 | 0.007 / 0.005 | 0.198 |

Part of what the market corrects between open and close is plain staleness
on recent form, which even the model-free average catches. The model predicts
the move better and keeps most of its slope next to the baseline. The effect
is larger in 2026-27 (slope 0.27, SE 0.010) than in 2025-26 (0.14, SE 0.008).

**2. But the classic closing-line-value test fails on price.** For a bet at
the open, CLV is the closing fair probability of the side minus the opening
price's break-even probability.

| Rule (both seasons) | Bets | Mean CLV, same-line bets | Market move toward bet | Lines that moved: our way / against | ROI at open (SE) |
|------|-----:|------:|------:|------:|------:|
| Model, 2-point edge | 12,206 | -2.4 pts (0.05) | +1.1 pts | 1,137 / 94 | +3.2% (0.9) |
| Model, 4-point edge | 9,008 | -2.1 pts (0.06) | +1.5 pts | 1,057 / 56 | +5.5% (1.0) |
| Model, 6-point edge | 6,320 | -1.8 pts (0.07) | +1.8 pts | 987 / 32 | +9.0% (1.3) |
| Baseline, 4-point edge | 9,467 | -2.2 pts (0.06) | +1.4 pts | 1,042 / 66 | +1.6% (1.0) |
| Control: every line, both sides | 31,612 | -3.5 pts | 0 | | -6.0% |

The market moves our way by 1.1-1.8 points on average, but each side of an
opening price carries about 3.5 points of margin. So a bet at the open is,
on average, *worse* than the fair close: it beats the close on price only
29% of the time. Note that "same-line" and "line moved" are known only
after the close. The ROI over all bets is the ex-ante result.

**3. Where the opening-price profit comes from.** At a 4-point edge, model alone:

| Bets | Count | Win rate at open line | Close fair prob. | Break-even at open | ROI at open (SE) | ROI at close |
|------|------:|------:|------:|------:|------:|------:|
| Same line | 7,895 | 55.0% | 52.0% | 54.1% | +2.0% (1.1) | -1.3% |
| Line moved | 1,113 | 67.0% | 43.2% | 53.5% | +30.8% | -3.4% |
| Control: every under, line moved | 1,354 | 58.2% | 48.3% | 52.0% | +16.3% (3.0) | -2.4% |

Most of the profit at the open is the 7% of bets whose line later moved, 95%
of them in the bet's favor. Even betting every under on lines that later
moved earns +16%, so this is mostly the line move itself. It is real money
only if those opening lines could be bet in size. The SportsGameOdds open is
a consensus of each book's first posted line, with no timestamp or limits,
and that cannot be checked here.

**4. Walk-forward blend at the open.** Fitted on 2025-26 opening prices and
bet on 2026-27 only. Weights: market 0.37 (SE 0.11), model 0.64 (SE 0.08).
At the close, the weights were 0.58 and 0.42: the opening price is a weaker
market. At a 4-point edge: 2,574 bets, ROI at the open +9.7% (SE 1.9);
same-line bets +6.6% (SE 2.0, 2,253 bets) against +0.6% for the same picks
at the close; mean CLV -1.05 points.

**Verdict.** The model carries information that the opening market lacks
and the closing market partly absorbs, beyond what recent form alone
explains. That supports the earlier finding that the model adds information
to the price. But on price, bets at the open do not beat the fair close on
average, because the opening margin is larger than the move. The large
opening-price ROI rests mainly on lines that moved, whose opening prices may
not have been available to bet. Betting at the open is therefore not
adopted (decision 30). The frozen live test stays at the closing price. A
per-book, timestamped check of opening lines (PuckAPI `get_prop_movement`,
10 credits per player-game) could settle whether the opening prices were
bettable.

## Older seasons against PuckAPI closing prices (2026-11-08)

Code: `src/ingest/fetch_puckapi_sog_history.py` (purchase),
`src/models/sog_older_seasons.py` (test). Results:
`docs/models/sog_older_seasons_results.json`. The purchased rows stay in
`data/` and are not committed.

**Why.** All earlier market tests use SportsGameOdds prices from 2026-01-16
on. PuckAPI's archive holds about 51,700 shots-on-goal lines from 2023-24 to
2026-01-15. The model has never been tested against prices from that period,
and none of these lines were used to build or tune anything.

**What was bought (500 credits).** 10 credits per 25 games, so about 1,100
lines. The sample is random 25-game blocks for regular players drawn at
random (seeded; pool = 215 players with 150+ games in the window and 30+
lines in 2026), plus Connor McDavid's full window, which was bought first to
find the price. A restart bug gave 9 players a second block instead of
adding 9 new players. Total: 1,134 rows, 1,116 graded lines, 30 players,
graded with official box-score shots (3 disagreed with the provider's
result). Market: PuckAPI's median closing line and median prices across
books (median 6 books), de-vigged proportionally. That is a different
consensus than SportsGameOdds', and the two medians need not come from the
same book.

**Results**

| | Lines | Market log loss | Model | Frozen blend | Model weight in blend (SE) |
|---|---:|---:|---:|---:|---:|
| Older seasons, PuckAPI | 1,116 | 0.6747 | 0.6769 | 0.6739 | 0.28 (0.20 by player; 0.30 by game) |
| 2026, SportsGameOdds (reference) | 15,806 | 0.6815 | 0.6821 | | 0.42 (0.07) |

| 4-point rule, older seasons | Bets | Win rate | ROI | SE by player / by game |
|---|---:|---:|---:|---:|
| Model alone | 556 | 53.6% | 0.0% | 3.1 / 4.0 |
| Frozen live rule (blend) | 179 | 47.5% | **-12.8%** | 4.3 / 6.9 |
| Control: every under | 1,116 | 49.5% | -5.1% | |

By season, the frozen rule loses 13.9% (61 bets) in 2023-24 and 12.2% (118
bets) in 2024-25, and makes no bets in the 115 early 2025-26 lines.

**Reading it.** The older seasons do not repeat the shots edge. The model
alone breaks even, and the frozen rule loses: its under picks, priced at a
49.9% chance by the market and 55% by the blend, came in 47.2% of the time.
In 2026 the same rule's unders hit 58.3% against a market price of 52.3%.
The information test points the same way as before (model weight 0.28) but
is far from significant, and the blend improves on the market by only 0.0008
in log loss.

Caveats: 1,116 lines from 30 players is a small sample (the frozen rule's
ROI has a standard error of 4-7 points), the market is a different consensus
(PuckAPI medians, not SportsGameOdds), and the frozen blend was fitted on a
later period. Still, the older seasons and the 2026-27 held-out result
(+8.3%, SE 2.7) differ by about 21 points, roughly 2.5-4 standard errors
depending on clustering. The honest summary is that **the shots edge is not
established**: it holds in the 2026 data and fails in this older sample. The
pre-registered live test (decision 20) decides it; nothing in it changes.

### Second batch (2026-11-08, another 499 credits)

A second key bought 49 calls of 25 games for **49 new players**, none of them
in batch 1, drawn the same way. A connection reset after 30 calls; the
restart drew its 19 players afresh, still all new. That gives 1,158 rows and
1,144 graded lines from 48 players (one had no matched box scores). The
batch-1 results above are kept as reported. The results file now holds batch
1, batch 2 and both combined.

| 4-point rules | Lines | Model alone: bets, ROI (SE) | Frozen rule: bets, ROI (SE) | Model weight in blend (SE) | Log loss: market / model / frozen blend |
|---|---:|---|---|---|---|
| Batch 1 | 1,116 | 556, 0.0% (3.1-4.0) | 179, -12.8% (4.3-6.9) | 0.28 (0.20-0.30) | 0.6747 / 0.6769 / 0.6739 |
| Batch 2 | 1,144 | 589, -4.5% (4.0-4.3) | 170, -2.3% (7.4-9.0) | 0.08 (0.30-0.35) | 0.6869 / 0.6936 / 0.6882 |
| **Combined** | **2,260** | **1,145, -2.3% (2.7-2.8)** | **349, -7.7% (5.0)** | **0.19 (0.20-0.21)** | 0.6809 / 0.6854 / 0.6812 |
| 2026-27, SportsGameOdds | | | 907, +8.3% (2.7) | 0.42 (0.07) | |

By season, combined: 2023-24 frozen rule -16.5% (138 bets), 2024-25 -5.0%
(201 bets), early 2025-26 only 10 bets.

**Reading it.** The independent second batch agrees with the first. Over
2,260 lines from 78 players, the model alone loses 2.3%, a little better
than the roughly -3.4% that random bets lose to the margin, but not
profitable. The frozen rule loses 7.7%, and the blend adds nothing to the
market (log loss 0.6812 against 0.6809). The model's weight beside the price
(0.19, SE 0.20) is about half the 2026 estimate and not distinguishable
from zero. The gap to 2026-27 (+8.3% against -7.7%) is about 16 points,
roughly 2.8 standard errors. Either the 2026 result was partly luck, or the
model's information was specific to that market (a different consensus,
SportsGameOdds, in a different period). Prior to the live test, the shots
model should be described as having **no demonstrated edge over the
market** (decision 33).

### Third batch (2026-11-09, 490 credits)

A third key bought 49 calls of 25 games for **49 new players** (none in
batches 1-2 or McDavid; the earlier player lists were restored from the
owner's copy, checksums verified). 1,162 rows; Andrew Copp's block returned
one game with no line rows, so 1,139 graded lines from 48 players. Batches
1-2 reproduce exactly on the restored files; their two-batch results are
kept in `sog_older_seasons_results_batches1-2.json`.

| 4-point rules | Lines | Model alone: bets, ROI (SE) | Frozen rule: bets, ROI (SE) | Model weight in blend (SE) | Log loss: market / model / frozen blend |
|---|---:|---|---|---|---|
| Batch 1 | 1,116 | 556, 0.0% (3.1-4.0) | 179, -12.8% (4.3-6.9) | 0.28 (0.20-0.30) | 0.6747 / 0.6769 / 0.6739 |
| Batch 2 | 1,144 | 589, -4.5% (4.0-4.3) | 170, -2.3% (7.4-9.0) | 0.08 (0.30-0.35) | 0.6869 / 0.6936 / 0.6882 |
| Batch 3 | 1,139 | 603, +2.3% (3.7-4.0) | 189, -1.2% (6.8-8.4) | 0.69 (0.30-0.31) | 0.6846 / **0.6834** / **0.6823** |
| **All three** | **3,399** | **1,748, -0.7% (2.2-2.3)** | **538, -5.4% (4.0-4.5)** | **0.36 (0.17)** | 0.6821 / 0.6847 / **0.6815** |
| 2026-27, SportsGameOdds | | | 907, +8.3% (2.7) | 0.42 (0.07) | |

By season, all three: 2023-24 frozen rule -7.0% (248 bets), 2024-25 -2.9%
(262), early 2025-26 -14.8% (28).

**Reading it.** Batch 3 is the first older sample in which the model alone
beats the market (0.6834 against 0.6846) and the blend is ahead of it.
Over all 3,399 lines the frozen blend is now slightly better than the market
(0.6815 against 0.6821), and the model's weight beside the price is 0.36
(SE 0.17): equal to the 2025-26 fit (0.36), a little below the frozen
blend's 0.42, and about two standard errors from zero. So the model probably does carry a little
information the older prices lack. But the frozen rule still loses (-5.4%
on 538 bets, SE 4.0): the information is too small to clear the margin at
the 4-point threshold, and the gap to 2026-27 (+8.3%) is still about 2.8
standard errors. The description stays **no demonstrated edge**; the live
test decides (decision 42).

## Variants with new data (2026-11-09)

Code: `src/models/prop_model.py shots_edge` and `shots_mp`; features in
`src/features/build_edge_features.py` and `src/features/build_mp_features.py`,
each with a leakage test. The frozen model (`prop_model.py shots`) is
unchanged. Older-season comparison: `python3 src/models/sog_older_seasons.py
variants` -> `docs/models/sog_older_seasons_variants.json`.

- **Edge** (11 features): the player's previous-season NHL Edge profile
  (high-danger shot share and rate, offensive-zone time, hardest shot, top
  speed, bursts, distance). Previous season only, because Edge is published
  as season totals (decision 34).
- **MoneyPuck** (9 features): exponentially weighted means of earlier games:
  individual expected goals, shot attempts, expected shots on goal,
  high-danger shots, power-play expected goals, on-ice expected goals for,
  5-on-5 expected-goals share, zone-start share. MoneyPuck shots on goal
  equal the NHL box score on 99.8% of skater-games.

| | Frozen | + Edge | + MoneyPuck |
|---|---:|---:|---:|
| Log-likelihood per game, 2025-26 (gbm) | -1.53676 | -1.53685 | **-1.53484** |
| Log-likelihood per game, 2026-27 (gbm) | -1.52596 | -1.52589 | **-1.52367** |
| Log loss, 2026-27 lines (market 0.6796) | 0.6781 | 0.6782 | **0.6771** |
| Blend on 2026-27 | 0.6764 | 0.6764 | **0.6759** |
| Model weight in the blend (2025-26 fit) | 0.36 (0.09) | 0.37 (0.09) | 0.51 (0.09) |
| Blend at a 4-point edge, 2026-27 | +8.3%, 907 bets (2.7) | +10.6%, 928 (2.5) | +5.8%, 1,391 (2.3) |
| Older PuckAPI lines, model log loss (market 0.6809) | 0.6854 | 0.6857 | **0.6840** |
| Older lines, model alone at 4 points | -2.3% (2.7) | -4.5% (3.0) | -3.7% (3.1) |
| Older lines, frozen blend at 4 points | -7.7%, 349 (5.0) | -5.7%, 341 (5.3) | -4.9%, 229 (6.1) |
| Older lines, model weight beside the market | 0.19 (0.20) | 0.15 (0.21) | 0.18 (0.22) |

Edge adds nothing. MoneyPuck makes the model a better forecaster everywhere,
including the older lines it never saw, and doubles its weight beside the
2026 market, but on the older lines it still trails the closing price and
its rules lose. The higher 2026-27 ROI of the Edge variant comes from 125
bets it adds and 104 it drops (+23 units, SE 12.6) and does not repeat on
the older lines. No variant replaces the frozen model (decisions 35-36).

**Update with batch 3 (2026-11-09, decision 42).** The MoneyPuck and Edge
data were fetched again for this run; both sources had revised earlier
games, so the variant results moved slightly (the 2026-11-08 files are kept
as `sog_mp_results_2026-11-08.json`, `sog_edge_results_2026-11-08.json`
and the matching `*_blend_results_2026-11-08.json`). The frozen model is
unchanged.

| All 3,399 older lines (market 0.6821) | Frozen | + Edge | + MoneyPuck |
|---|---:|---:|---:|
| Model log loss | 0.6847 | 0.6851 | **0.6831** |
| Paired difference from frozen (SE, game-clustered) | | +0.0004 (0.0006) | **-0.0016 (0.0010)** |
| ... batch 1 / 2 / 3 | | -0.0002 / +0.0014 / -0.0002 | -0.0020 / -0.0008 / -0.0020 |
| Frozen blend log loss | 0.6815 | 0.6816 | **0.6812** |
| Model alone at 4 points | -0.7% (2.3) | -1.2% (2.3) | -1.1% (2.4) |
| Frozen blend at 4 points | -5.4%, 538 (4.0) | -6.4%, 554 (4.0) | -1.4%, 362 (4.9) |
| Model weight beside the market | 0.36 (0.17) | 0.33 (0.17) | 0.42 (0.18) |
| 2026-27 lines: model log loss (market 0.6796) | 0.6781 | 0.6778 | **0.6765** |
| 2026-27: blend at 4 points | +8.3%, 907 (2.7) | +10.5%, 833 (2.7) | +6.2%, 1,426 (2.2) |

MoneyPuck is better than the frozen model in every batch, by 0.0016 overall
(about 1.5 standard errors), and is the variant closest to the market on
the older lines (0.6831 against 0.6821). Edge again adds nothing. None of
the rules is profitable on the older lines. `shots_mp` stays the candidate
for a future frozen rule, to be decided on new games (next steps).
