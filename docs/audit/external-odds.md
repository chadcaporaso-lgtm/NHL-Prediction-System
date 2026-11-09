# Owner-supplied odds files (checked 2026-11-09)

Every file the owner supplied on 2026-11-09 was checked before use: contents,
matching to official NHL games and scores, overlap with sources already in
use, and what it can and cannot support. Files that were used are kept in
`data/external/` (gitignored, never committed; the repository is public and
the sources have terms of use). SHA-256 prefixes identify the exact files.

| File | Content | Verdict |
|---|---|---|
| `nhl_2025_26_closing_odds.csv` (459588b3) | One row per game: opening and consensus closing moneyline, puck line, total (de-vigged), 2025-26 | Byte-identical to the file behind v2's external crosswalk, already used. No change |
| `nhl_2026_27_odds.csv` (f357e746) | Same format, 231 games to 2026-11-01 | Used first (decision 37); replaced by the next row. Results kept as `game_market_results*_xw2026-11-01.json` |
| `nhl_2026_27_odds.csv` (f324cc32), sent later the same day | Same format, 282 games to 2026-11-08 | **Used** (decision 40). 261 games map to official results; the 21 after 2026-11-05 wait for the next weekly refresh |
| `sportsbook-nhl-2021-2022.csv` (891229ab), `sportsbook-nhl-2022-2023.csv` (b164f907) | Per game: score by period, the favorite's moneyline only, total line without prices; timing unknown | **Used for log loss only** (decision 39). A second copy inside `archive_6.zip` is identical |
| `archive (8).zip` (761b3c56): games and closing odds for the NBA, college basketball and the NHL, scraped in October 2026 from public odds-comparison sites by the dataset's author | For the NHL: average closing three-way prices (home / tie / away after 60 minutes) across bookmakers, 1998-99 to 2026-10-19. Per-bookmaker rows, timestamps, spreads and totals exist only for basketball | **Used for log loss only** (decision 38) |
| `archive (5).zip` (Kaggle "NHL Game Data") | Games, plays, shifts, box scores 2000-01 to 2019-20; no odds | Not used: no prices, ends before our seasons, copies NHL feeds we read directly |
| `archive (7).zip` (ESPN-derived dataset, also in the `nhempstead/nhl-model` repository) | One-side moneyline, spread and total per game, 2003-04 to 2025-12; rolling team features | **Rejected**: placeholder odds (moneyline -105, total 5.5, spread 1.5) fill 100% of games in 2003-04 to 2018-19 and 2024-25 and 81% of 2025-26; the "favorite" price is often the underdog's; season features are summed across all teams (`season_wins` up to 28,799) |

## 2026-27 closing file: copies disagree (finding E-1)

Three copies of the 2026-27 file cover the same 184 games: v2's copy (about
2026-10-26), the 231-game file and the 282-game file. The 282-game file
agrees exactly with v2's copy on all 184. The 231-game file does not: it
lists one more bookmaker for 86 of them (25 instead of 24) and differs on 46
moneyline closes (by at most 0.0033), 58 puck-line cover probabilities (up
to 0.40, where one game's closing spread is -1.5 instead of +1.5) and 5
closing total lines (half a goal). Against the 282-game file it differs on
70 moneyline closes, 81 puck-line probabilities, 7 total lines and 130
bookmaker counts (`docs/audit/external_crosswalk_2026-11-09.json`,
`external_crosswalk_2026-11-08.json`). The consensus depends on which
bookmakers an export includes, so a "closing price" for a past game is not
fixed: two exports can disagree, and a later one can revert. Scores never
differ. For
benchmarks this is minor; it is one more reason ROI is never computed from
this source. All 231 games map to official games (230 by start time and
teams, 1 by date and teams: Utah vs Tampa Bay on 2026-10-24 is listed an hour
earlier) and all scores agree; official ids equal v2's crosswalk on all 184
shared games.

The 282-game file maps 261 games (260 by start time and teams, 1 by date
and teams); all scores agree. Its 21 games after 2026-11-05 are left out
until the NHL schedule refresh covers them.

## Three-way average closing prices (October 2026 scrape)

- Matching: 100% of model games 2021-22 to 2025-26 and 52% of 2026-27 (the
  scrape ends 2026-10-19). The source dates games in European time: 6,189
  of 7,129 matches are one day after the North American date. Final scores
  agree on 99.99%.
- Median three-way margin 6.0%, plausible for average closing prices.
- No placeholder or filled values were found; empty means unpublished.
- Limits: averages are not bettable prices (log loss only); no NHL
  timestamps, so "closing" is the publisher's label; the exact sites are not
  named, so reuse terms are unknown and the data stays out of the repository.

## Favorite-only moneyline files, 2021-22 and 2022-23

- All 1,311 and 1,308 games match official games; every final score agrees.
- 45 and 36 lines are "Even" (no favorite). Prices are the favorite's only,
  so the margin is removed approximately (4.5 points); results are
  insensitive to that choice (0.6400 vs 0.6422; 0.6533 vs 0.6525).
- Limits: no underdog price (no exact de-vig, no ROI), no total prices, and
  unknown timing (not called a closing line).
