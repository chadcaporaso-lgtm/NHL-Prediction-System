# Player Prop Price History (PuckAPI)

Date: 2026-11-07. Fetch: [`src/ingest/fetch_prop_history.py`](../../src/ingest/fetch_prop_history.py).
Profile: [`src/audit/prop_history_profile.py`](../../src/audit/prop_history_profile.py).
Numbers: [`prop_history_summary.json`](prop_history_summary.json).

Neither earlier project had historical prop prices, so no prop model had
ever been tested against the market. This sample is the first.

## What was bought

`get_prop_history` returns one row per game: the closing line (median across
about 7 books), the over and under prices, the player's recorded stat and
the result. PuckAPI charges 10 credits per 25 games covered.

| Request | Games | Credits |
|---------|------:|--------:|
| McDavid shots on goal, full history (2024-10-09 to 2026-11-05) | 195 | 80 |
| 12 players, shots on goal, 25 games from 2025-12-01 | 300 | 120 |
| 10 players, points, 25 games from 2025-12-01 | 250 | 100 |
| **Total** | **745** | **300** |

Players: McDavid, Matthews, MacKinnon, Draisaitl, Kucherov, Pastrnak, Kaprizov,
Makar, Q. Hughes, Ovechkin, Werenski, Bedard, Caufield (shots); Matthews,
MacKinnon, Draisaitl, Kucherov, Pastrnak, Kaprizov, Panarin, Makar, Rantanen,
Eichel (points). With a fixed credit budget the number of player-games is
fixed, so the sample spreads across many players rather than a few full
histories.

**Limitation.** These are all star players, sampled mid-season. They are the
players with the deepest prop markets, but results may not carry over to
depth players, whose lines are thinner and may be softer.

## Checks

| Check | Result |
|-------|--------|
| Recorded result vs NHL box score (626 rows in 2025-26) | 624 agree, 2 differ |
| Two-sided margin | 5.5% to 8.9%, median 6.5% (moneyline closing lines: 4.6%) |
| Impossible prices | none |

Of the two differences, one changes the grade: McDavid on 2025-12-13 is
recorded with 3 shots (over 2.5), but the NHL box score shows 2 (under). The
rebuild grades every prop from the official NHL box score, not from the
provider's result field.

## How good the closing prop price is

De-vigged closing probability against the outcome, graded games only (no
pushes):

| Market | Games | Over rate | Market mean P(over) | Market log loss | Constant log loss |
|--------|------:|----------:|--------------------:|----------------:|------------------:|
| Shots on goal | 489 | 52.6% | 51.3% | 0.690 | 0.692 |
| Points | 243 | 51.9% | 46.5% | 0.657 | 0.693 |

- **Shots on goal.** Lines are set so each side is close to a coin flip, and
  the closing price adds little beyond that (0.690 vs 0.692). A model with
  genuine information on shot volume has room to add value, but it has to
  overcome a 6.5% margin, which needs roughly 53.5% accuracy at typical prices.
- **Points.** The closing price is informative (0.657 vs 0.693), mostly
  because 0.5 and 1.5 lines differ a lot. Overs hit 51.9% against a market
  estimate of 46.5%. On 243 games that gap is about 1.7 standard errors:
  worth testing, not yet evidence.

## Next

1. Score the prop models on exactly these 745 games, using only information
   available before each game, and compare with the closing price (log loss,
   then ROI at the closing price).
2. Treat this as a first, small backtest. More prop history (more players,
   depth players, 2024-25) can be added later at 10 credits per 25 games.
