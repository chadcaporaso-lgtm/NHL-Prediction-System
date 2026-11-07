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
