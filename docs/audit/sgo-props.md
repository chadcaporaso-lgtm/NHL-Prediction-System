# Player Prop Closing Prices from SportsGameOdds

Date: 2026-11-07. Scripts:
[`fetch_sgo_events.py`](../../src/ingest/fetch_sgo_events.py),
[`extract_sgo_props.py`](../../src/ingest/extract_sgo_props.py),
[`match_sgo_props.py`](../../src/ingest/match_sgo_props.py).
Numbers: [`sgo_props_market_summary.json`](sgo_props_market_summary.json).

## Source and coverage

The SportsGameOdds Pro-tier API was called directly; the key is held in the
environment's network secrets and never appears in code or files.

- Events before 2026-01-16 carry no opening or closing prices, only the last
  price seen, which was captured during or after the game (data audit O-1).
  They are not used. Probes: 2024-01-16 returned no events; 2025-01-16
  returned 1,040 player-prop prices and none had a close.
- From 2026-01-16, 96% of player-prop lines have both closing prices
  (consensus across books).

Downloaded: 998 finished NHL games, 2026-01-16 to 2026-11-06, including the
2026 playoffs. Extracted: 119,948 player-prop lines (920 games, 941 players).

## Stat names: a trap

The provider's stat names do not mean what they say. Checked against NHL box
scores:

| Provider stat | Matches NHL | Agreement |
|---------------|-------------|----------:|
| `points` | goals | 99.9% |
| `goals+assists` | points | 99.3% |
| `assists` | assists | 99.3% |
| `shots_onGoal` | shots on goal | 97.2% |

Reading `points` as hockey points would have produced a backtest of the wrong
market. Every prop is graded from the NHL box score, not the provider's value.

## Matching to NHL ids

- 98.9% of events matched to an NHL game (same teams, start within 12 hours).
- 92% of prop rows matched to a player in the NHL box score by normalized
  name. The rest are mostly name spelling differences and players who did not
  dress; they are excluded and counted.
- Games from 2026-27 cannot be graded until the 2026-27 skater box scores are
  re-fetched (data audit D-1 follow-up).

## The market a model must beat (2025-26, graded on NHL box scores)

| Market | Lines | Games | Players | Over rate | Market P(over) | Market log loss | Constant log loss | Median margin |
|--------|------:|------:|--------:|----------:|---------------:|----------------:|------------------:|--------------:|
| Shots on goal | 9,964 | 646 | 430 | 48.5% | 50.1% | 0.683 | 0.693 | 6.9% |
| Points | 11,900 | 646 | 493 | 44.7% | 45.1% | 0.662 | 0.688 | 7.0% |
| Assists | 11,315 | 646 | 442 | 33.2% | 34.8% | 0.616 | 0.636 | 7.1% |
| Goals | 20,120 | 646 | 689 | 15.8% | 17.7% | 0.400 | 0.436 | 7.8% |

Dates: 2026-01-17 to 2026-06-15. Lines on whole numbers that pushed are
excluded.

**Readings**

- Closing prop prices are informative in every market, but only modestly for
  shots on goal (0.683 vs 0.693 for a constant).
- Across all players, points overs hit 44.7% against a market estimate of
  45.1%. The 5-point gap seen in the 745-game PuckAPI star sample does not
  hold up on 11,900 lines; it was a small-sample, star-only effect.
- Goal-scorer overs hit 15.8% against a market estimate of 17.7%. The market
  overprices "player scores" even after the margin is removed proportionally,
  the long-shot pattern seen in most betting markets.
- Margins are about 7%, against about 4.6% on moneylines. A model needs to be
  right about 53.5% of the time at typical prices to break even.

## What this enables

About 53,000 graded prop lines across 646 games and up to 689 players: a
large enough set for a real, out-of-sample prop backtest of shots on goal and
points, which neither earlier project could run.
