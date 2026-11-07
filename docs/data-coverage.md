# Data Coverage

What each model needs, what exists, and the gaps. Checked 2026-11-07 against
the downloaded files and the inventory. Seasons are labeled by start year
(2025-26 = 20252026).

## Training data (no odds needed)

| Data | Used by | Have | Status |
|------|---------|------|--------|
| Game schedule and results (real goals, OT/SO) | All game models, labels | 2019-20 to 2026-27 (through 2026-11-02), audited | Ready |
| Team game stats (shots, official) | Game models | 2019-20 to 2026-27 | Ready |
| MoneyPuck team xG and possession | Game models | 2018-19 to 2026-27 (through 2026-11-06, 264 games), refreshed 2026-11-07 | Ready (G-A closed) |
| Skater game box scores and ice time | Prop models | 2019-20 to 2025-26 complete (2021-24 re-fetched, D-1) | Ready |
| Skater box scores 2026-27 | Prop features, grading 2026-27 props | 256 games through 2026-11-06, 36 per game, re-fetched 2026-11-07 | Ready (G-B closed); re-fetch as the season goes on |
| Skater box scores 2019-20, 2020-21 | Prop models (two more training seasons) | 1,212 and 952 games, every scheduled game | Ready (G-C closed) |
| Goalie game logs (starter, saves, xGA) | Game and totals models | v2 `goalie_history_candidates`, not yet downloaded or audited | **Gap G-D:** audit |
| Announced starting goalie before the game | Game and totals models | No historical source | **Gap G-E:** use the actual starter and document it (starters are normally confirmed before the close) |

## Evaluation data (market prices)

| Market | Clean closing prices | Games or lines | Status |
|--------|----------------------|---------------:|--------|
| Moneyline | 2025-26 full and 2026-27 to date (external CSV, SportsGameOdds from 2026-01-16) | about 1,560 games | Enough for a one-season market benchmark |
| Puck line | SportsGameOdds from 2026-01-16 | 881 games | Partial 2025-26 plus playoffs and 2026-27 |
| Totals | External CSV 2025-26 and SportsGameOdds from 2026-01-16 | about 1,560 games | Same as moneyline |
| SOG, points props | SportsGameOdds from 2026-01-16, PuckAPI 745 player-games | about 22,000 graded lines | 2026-27 lines wait on G-B |
| Any market before 2025-26 | None clean (O-1, O-3, O-7) | 0 | **Gap G-F (optional):** PuckAPI archive or a paid source |

## Not needed

Play-by-play and shift data (MoneyPuck supplies xG), injury and lineup feeds
(a prop line exists only for a player expected to play, and books void the
bet if he does not), goalie saves data (out of scope).
