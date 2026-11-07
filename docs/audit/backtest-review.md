# Backtest Review: What the Earlier Projects Actually Show

Date: 2026-11-07. Script: [`src/audit/review_backtests.py`](../../src/audit/review_backtests.py).

## Where the backtest results are

| Location | What it holds |
|----------|---------------|
| v1 `manifests/production_manifest.json` | Headline holdout numbers for the production models |
| v1 `reports/stage8_verified_pregame_ledger.parquet` | Every verified pregame closing quote for 728 games (2026-01-17 to 2026-10-13), with model probabilities and EV |
| v2 `reports/stage08_benchmark` | Moneyline model vs Pinnacle closing line |
| v2 `reports/stage20_four_venue_comparison`, `stage26_external_closing_comparison` | Model vs FanDuel, DraftKings, Pinnacle, Novig and the external closing CSV |
| v2 `reports/stage27_walk_forward` | Monthly-refit walk-forward test, 2025-26 and 2026-27 |
| v2 `reports/model_improvement_qualification` | Final challenger vs control (2026-10-31) |
| Drive `nhl_model/` | Not a backtest: a single snapshot of 6,968 upcoming player-prop prices taken 2026-10-10 |

No project contains a multi-season betting backtest. Every market comparison
covers at most the 2025-26 season and the start of 2026-27, because clean
closing prices exist only from January 2026 (SportsGameOdds) and October 2025
(external CSV); see data audit finding O-3.

## Moneyline: probability accuracy

Log loss on the official winner, lower is better. A model that always predicts
the average home-win rate scores about 0.693.

| Test | Games | Best model | Market close | Constant |
|------|------:|-----------:|-------------:|---------:|
| v1 production holdout (Pinnacle close) | 727 | 0.6735 | 0.6706 | |
| v2 walk-forward 2025-26 (external CSV close) | 1,292 | 0.6867 (best of 33 variants) | 0.6819 | 0.6927 |
| v2 walk-forward 2026-27 (external CSV close) | 165 | 0.6493 (best of 33 variants) | 0.6520 | 0.6837 |

**Reproduced.** The v1 figures (0.6735 vs 0.6706 on 727 games) are recomputed
exactly from the saved ledger.

**Reading.** In 2025-26 the models improve on a constant by about 0.006 and
trail the closing market by about 0.005. The 2026-27 result where a variant
beats the market is the best of 33 variants on 165 games, picked after seeing
the results; the pre-declared challenger scores 0.6655, worse than the market.
This is consistent with v2's own conclusion: no demonstrated improvement over
the market.

## Moneyline and totals: betting results on clean closing prices

All prices are verified pregame closing prices (`pregame_dec_odds` equals
`close_decimal_odds` on every row). Standard errors are clustered by game.

| Selection rule | Bets | ROI | SE |
|----------------|-----:|----:|---:|
| Every book, model EV > 0 | 10,797 | -2.8% | 2.9 |
| Every book, odds-informed EV > 0 | 4,666 | -2.8% | 3.9 |
| Best price across 9 books, one bet per game and market, EV > 0 | 1,333 | -1.5% | 2.9 |
| same, moneyline only | 616 | +0.5% | 4.5 |
| same, totals only | 717 | -3.3% | 3.7 |
| Best price, EV > 2% | 1,211 | -0.6% | 3.0 |

**Not reproduced.** The production manifest reports a challenger holdout ROI
of +4.4% (SE 4.5%). No selection rule applied to the saved ledger reproduces
it: every rule tried returns between -26% and +1%. The code that produced the
+4.4% is not in the project (inventory finding I-1), so the figure cannot be
traced. Even as reported, +4.4% with a standard error of 4.5% is not
distinguishable from zero.

**Reading.** On clean prices there is no evidence of a betting edge on
moneylines or totals. Moneyline results are close to break-even, which is what
a model slightly behind the market produces after line shopping. Totals lose.

## Player props

The production manifest reports well-calibrated shots-on-goal and points
distributions on 50,182 skater-games in 2025-26 (predicted vs actual zero
rates 27.0% vs 27.2% and 65.1% vs 65.3%). These are probability-fit results,
not betting results: there is no historical prop-price data in either project,
so no prop backtest exists. The `nhl_model` folder holds one day of prop
prices (2026-10-10), and v1 has one graded day of predictions (2026-11-03).
The prop models were also trained on incomplete 2021-22 to 2023-24 skater data
(data audit finding D-1).

## Conclusions for the rebuild

1. The honest headline for the earlier work is: probability models close to,
   but not better than, the closing market; no demonstrated betting edge.
2. The +4.4% ROI must not be quoted. It cannot be reproduced and is within
   one standard error of zero.
3. A credible portfolio backtest needs several seasons of clean closing
   prices. Within the current data only 2025-26 qualifies; PuckAPI archive
   lines (data audit O-5) are the candidate source for earlier seasons.
4. Player props are where the earlier work is strongest (calibrated
   distributions) and least tested (no price history).
