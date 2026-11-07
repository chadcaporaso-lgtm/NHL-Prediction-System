# Phase 1b: Data Audit

Findings are numbered D-1, D-2, ... and each records the evidence, the impact
and the remedy. Severity: **High** (changes model results), **Medium**
(affects some features or seasons), **Low** (housekeeping).

## Files examined

Every file examined is listed with its SHA-256 in `data/drive/checksums.csv`
(local; data is not committed). Files over 10 MB were split in Colab and
reassembled; all 10 matched the checksums computed from the originals:

| File | Size | Verified |
|------|-----:|:--------:|
| v1 `data/features/game_features_T24h.parquet` | 11.5 MB | yes |
| v1 `data/features/game_features_T60m.parquet` | 12.3 MB | yes |
| v1 `data/features/player_features_T60m.parquet` | 29.8 MB | yes |
| v1 `data/raw/moneypuck/moneypuck_all_teams_2018_plus.parquet` | 13.1 MB | yes |
| v2 `reconstructed_research_features.parquet` (2026-11-03 run) | 24.7 MB | yes |
| v2 `quote_observations.parquet` | 81.5 MB | yes |
| v2 `paired_market_observations.parquet` | 65.6 MB | yes |
| v2 `closing_benchmarks.parquet` | 12.3 MB | yes |
| v2 `fresh_team_games_reconciled.parquet` | 12.9 MB | yes |
| v2 `legacy_team_games_reconciled.parquet` | 14.4 MB | yes |

## D-1 (High, raw data fixed): skater data for 2021-22 to 2023-24 holds about 14% of player-games

**Evidence.** v1 `data/raw/nhl_api/skater_summary_<season>.parquet`:

| Season | Rows | Games | Rows per game |
|--------|-----:|------:|--------------:|
| 2021-22 | 7,283 | 1,397 | 5.2 |
| 2024-25 | 50,320 | 1,398 | 36.0 |

A complete season has 36 skater rows per game (18 per team). Every 2021-22
game is present, but only 2 to 9 skaters per game. The time-on-ice table
shows the same gap (6,923 rows in 2021-22). The rows kept look like a
haphazard subset rather than a filter: mean points per row is 0.46 against
0.45 in the complete 2024-25 table, defencemen are 34% in both, and no
per-date row cap is visible. The cause is not yet known.

**Impact.** The gap carries into `data/features/player_features_T60m.parquet`,
the feature table behind the shots-on-goal and points models:

| Season | Rows | Rows per game |
|--------|-----:|--------------:|
| 2021-22 | 7,283 | 5.2 |
| 2022-23 | 7,073 | 5.1 |
| 2023-24 | 6,893 | 4.9 |
| 2024-25 | 50,320 | 36.0 |
| 2025-26 | 50,182 | 36.0 |
| 2026-27 (to date) | 5,543 | 23.6 |

The prop models were therefore trained on about 21,000 player-games from
three seasons where about 150,000 exist, plus 2024-25. Rolling features
for a player in those seasons are built from scattered games, so form and
ice-time windows are wrong for them. The 2025-26 holdout (50,182 rows) is
complete, so the reported holdout metrics are measured on full data; the
training data is what is affected.

The 2026-27 rows per game (23.6) are also below 36 and need checking.

**Remedy, part 1: raw data re-fetched (2026-11-07).**
[`src/ingest/colab_refetch_skater_games.py`](../../src/ingest/colab_refetch_skater_games.py)
re-downloaded the skater summary and time-on-ice tables from the NHL stats API
in weekly windows, paging with a fixed sort (gameId, playerId). Output is in
Drive `NHL_Prediction_System_Claude/01_raw/nhl_api_refetch/`.

| Season | Rows | Games (regular + playoff) | Rows per game | Duplicates |
|--------|-----:|--------------------------:|--------------:|-----------:|
| 2021-22 | 50,412 | 1,401 (1,312 + 89) | 35.98 | 0 |
| 2022-23 | 50,376 | 1,400 (1,312 + 88) | 35.98 | 0 |
| 2023-24 | 50,389 | 1,400 (1,312 + 88) | 35.99 | 0 |

Checks on the new files:

- Every regular-season game is present (1,312 per season).
- Every game has 34 to 36 skater rows. The 50 games below 36 are teams that
  dressed 16 or 17 skaters, not missing data: every short game is short by
  whole team-dressed players (17 or 16 rows for one team).
- The summary and time-on-ice tables cover exactly the same (game, player)
  pairs.
- Every row in the old truncated 2021-22 table appears in the new table with
  identical goals, assists, points, shots and time on ice. The old data was
  correct but incomplete; nothing in it was wrong.

The cause of the original gap was not established. Paging without a fixed sort
order is the most likely explanation, and the new fetch avoids it.

**Remedy, part 2 (to do).** Rebuild the player feature table from the
complete data before any prop model is retrained or compared.

## D-2 (Low): `<table>_full` files are byte-identical duplicates

`skater_summary_20242025.parquet` and `skater_summary_20242025_full.parquet`
have the same MD5. The other six `_full` pairs have identical sizes and are
expected to match. Keep one copy of each in `01_raw`.

## Game results and labels

Script: [`src/audit/audit_game_results.py`](../../src/audit/audit_game_results.py).
Every check, with counts, is in [`game_results_summary.json`](game_results_summary.json).

### Sources compared

| Source | Games | Role |
|--------|------:|------|
| NHL schedule responses saved by v1 (2019-20 to 2026-27, retrieved 2026-10-14) | 9,242 completed | Reference |
| PuckAPI `get_games` (retrieved 2026-11-07) | 9,421 completed | Independent check |
| v1 `data/labels/game_labels.parquet` | 9,242 | Labels under audit |
| v2 `game_labels.parquet` (run 2026-11-03) | 9,392 | Labels under audit |

The NHL schedule is the same source both projects were built from, so it
cannot catch an error made at the source. PuckAPI is an independent copy and
closes that gap. It labels the Arizona franchise `UTA` in every season, so the
cross-check matches on game id and score, not team abbreviation.

### Result: labels are clean

All 25 checks on both label tables pass, apart from the three expected
differences explained below.

| Check | v1 | v2 |
|-------|:--:|:--:|
| One row per game, no duplicates | pass | pass |
| Every completed game present | pass (9,242 of 9,242) | pass (9,242 of 9,242) |
| No labels for unplayed or unknown games | pass | pass |
| Final score matches NHL reference | pass | pass |
| Final score matches PuckAPI | pass (9,242) | pass (9,392) |
| Decision (REG / OT / SO) matches reference | pass | pass |
| Final = regulation + overtime (+ shootout credit) | pass | pass |
| Margin and total derived correctly | pass | pass |
| OT and SO games tied after regulation; REG games not | pass | n/a |
| OT and SO games decided by one goal | pass | pass |
| Shootout credit only in SO games, exactly one per game | pass | pass |
| v1 and v2 agree on every shared game | pass (9,242 games) | |

### Expected differences (not errors)

- **G-1. Two playoff games decided by shootout** (2019030002, 2019030016).
  These are 2020 bubble round-robin seeding games, which did use shootouts.
  They are labelled game type 3 like the rest of the playoffs.
- **G-2. v2 has 136 games the v1 schedule lacks.** All are 2026-27 games
  played after 2026-10-14, when v1 last saved the schedule. v1 has not been
  updated since.
- **G-3. 29 PuckAPI games are missing from v2.** All were played 2026-11-03 to
  2026-11-06, after the last v2 run.

### Notes for modeling

- **Shootout scoring.** Both projects store the official final score, which
  credits the shootout winner with one goal. That is the right number for
  grading moneyline, puck line and totals, because sportsbooks count the
  shootout winner as one goal. It is the wrong number for modeling goal
  scoring: in the 645 shootout games, real goals average 5.25 per game against
  an official 6.25. v2 stores real goals and shootout credit separately; v1
  does not. The rebuild uses real goals as the model target and the official
  score for grading bets.
- **Unusual seasons.** 2019-20 was paused in March 2020 and finished with a
  qualifying round and round robin in two bubble cities (no home ice); 2020-21
  was a 56-game season in realigned divisions with few or no fans. These
  seasons need flags, or exclusion from home-advantage features.
- **Decision mix is stable.** Over 2019-20 to 2026-27, 75-79% of games end in
  regulation, 14-17% in overtime and 6-9% in a shootout.

## Odds

Script: [`src/audit/audit_odds.py`](../../src/audit/audit_odds.py).
Numbers: [`odds_summary.json`](odds_summary.json).

### Sources found

| Source | Coverage | Contents |
|--------|----------|----------|
| SportsGameOdds via v1 `normalized/odds.parquet` | Feb 2024 to Oct 2026 (3,421 games) | 9 books plus a consensus; moneyline, puck line, totals |
| SportsGameOdds via v2 `quote_observations` | Same events | 942,494 quotes with timing flags |
| SportsGameOdds dedicated closing lines (v2 `closing_benchmarks`) | 2026-01-16 to 2026-10-13 (729 games) | 56 books |
| User-supplied closing CSV (v2 `user_closing_csv`) | 2025-26 full season and 165 games of 2026-27 (1,559 games) | Consensus of 16 to 24 books: opening and closing win probability, puck line, total |

There are no bookmaker odds before February 2024 in either project.

### O-1 (Critical): v1 graded most bets at prices captured during or after the game

**Evidence.** v1 stores four prices per quote (open, close, current, fair).
`close` exists on only 26% of rows. Where it is missing, v1 grades the bet at
the `current` price, which was captured when the data was downloaded, after
the game had started:

- 86% of moneyline prices were last updated after puck drop, a median of 157
  minutes after the start, which is near the end of a game.
- Measured against the result, `current` prices are far too accurate to be
  pregame prices. The best pregame line, the close, scores a log loss of
  about 0.67 in the NHL.

| v1 price | Rows | Log loss |
|----------|-----:|---------:|
| close | 13,096 | 0.670 (normal for a closing line) |
| current, updated after puck drop | 43,352 | 0.456 |
| current, no timestamp | 6,904 | 0.286 |

- Lines drifted in game too: puck lines of -2.5 and -3.5 appear, which are
  in-game lines.
- v1's grading column (`eval_decimal_odds`) uses the close price on 26% of
  rows and the in-game `current` price on 74%.

**Impact.** Any v1 result that grades bets or measures edge against
`eval_decimal_odds` on rows without a close price is invalid: the "market" it
was compared with already knew the score. This includes v1's paper-trading
edge calculations. Which v1 reports rely on these rows is checked in the model
audit.

**What is correct in v1.** The grading logic itself: all 149,760 bets are
graded correctly (win, loss and push, with the shootout goal counted on totals
and puck lines), and profit arithmetic has no errors.

### O-2 (Pass): v2 detected the same problem and excluded the data

v2 flags 99.9% of the SportsGameOdds quotes as captured after the event
started, and marks every quote ineligible for as-of features and for ROI
(0 of 942,494). v2 uses only dedicated closing lines and the external CSV as
market benchmarks, and does not claim ROI from them.

### O-3 (Medium): the clean closing-line history is short

| Clean source | Games | Log loss of closing moneyline |
|--------------|------:|------------------------------:|
| v2 dedicated SportsGameOdds closing lines | 729 | 0.670 |
| External closing CSV | 1,559 | 0.678 (open: 0.680) |

Both behave like genuine pregame prices. Between them they cover the
2025-26 season and the start of 2026-27, about 1,560 games. That is enough
to benchmark a model against the market for one season, but not to run a
multi-season betting backtest.

The external CSV's methodology (which books, which moment counts as the
close) is not documented, so it is used as a consensus benchmark, not as a
price anyone could have bet.

### O-4 (Low): exchanges distort margins

In the dedicated closing lines the median two-way margin is 4.6% and there are
no duplicates or impossible prices. The outliers are exchanges and prediction
markets: Sporttrade quotes 79 pairs below 100% (mid prices), and Matchbook,
Kalshi, Polymarket and a few others have pairs above 115%. Exclude these when
building a consensus line.

### O-5 (Open): PuckAPI odds as a multi-season source, one-game probe

PuckAPI holds odds for every game from 2019-20. A probe of one game
(2021020500, ANA at CGY, start 02:30 UTC, CGY won 6-2) returned 18 books with
two kinds of line:

- `source: archive`: opening and closing lines with timestamps before puck
  drop (closing between 22:49 and 02:25 UTC). These agree with each other:
  CGY between -213 and -233.
- `source: final`: one line per book stamped 02:30 UTC, the scheduled start.
  Several do not agree with the archive lines or with each other (DraftKings
  archive close -222/+185, final -145/+300; bet365 final -135/+275, whose two
  sides imply only 84%).

If this pattern holds, the `archive` lines are a usable multi-season closing
history and the `final` lines should be excluded. One game is not enough to
decide; a sample of about 30 games per season would settle it.

### O-6 (Pass): SportsGameOdds game lines from 2026-01-16, all three markets

`src/ingest/extract_sgo_game_odds.py` pulls the full-game moneyline, puck
line and total from the SportsGameOdds files already downloaded for the prop
work, using only the provider's consensus opening and closing prices. The
per-book `odds` fields are last-seen prices and include in-game quotes
(DraftKings home moneyline -50000 after the result was known), the same
problem as O-1, so they are never read. `src/audit/profile_sgo_game_odds.py`
matches the events to the NHL schedule and grades them on the official score
(how sportsbooks settle). Summary: `docs/audit/sgo_game_odds_summary.json`.

881 completed games, 2026-01-16 to 2026-11-02 (646 in 2025-26, 235 in
2026-27; 82 playoff games). Every game has both closing prices in all three
markets.

| Market | Closing log loss | Opening log loss | Constant log loss | Median margin |
|--------|-----------------:|-----------------:|------------------:|--------------:|
| Moneyline | 0.667 | 0.675 | 0.692 | 4.2% |
| Puck line (±1.5) | 0.641 | | 0.686 | 4.3% |
| Total goals | 0.688 | | 0.693 | 4.7% |

No closing moneyline is beyond ±1000, both puck-line sides always carry
opposite lines, and closing beats opening, all signs of pregame prices.
Totals: 19 pushes on whole-number lines (6.0, 7.0) are excluded from log
loss; the over hit 51.0%.

This adds puck line and opening prices to the clean history, but not more
seasons: it overlaps the sources in O-3. Clean game-market prices for
2025-26 and the start of 2026-27 now cover all three markets; every earlier
season still depends on O-5.

### O-7 (Fail): SportsGameOdds has no usable game lines before 2026-01-16

Probe on 2026-11-07, one API page (up to 50 games) per window, full-game
moneyline:

| Window | Games | With a "closing" price | Implausible (beyond ±1000 or two-way total outside 100-110%) |
|--------|------:|-----------------------:|---------------------------------------:|
| 2023-11-15 | 0 | 0 | |
| 2024-10-08 to 2024-10-20 | 50 | 0 (only a last price, e.g. -28590) | |
| 2025-03-10 to 2025-03-17 | 50 | 50 | 22 |
| 2025-11-10 to 2025-11-17 | 49 | 49 | 28 |

Before 2026-01-16 no game has an opening price, and the "closing" price is
the last price seen, often in-game (+650, -2981). The clean-looking ones
cannot be told apart from in-game prices taken early in a game, so none are
used. SportsGameOdds is not a source for earlier seasons.
