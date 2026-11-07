# Decision Log

Each entry records a decision, the date, and the reason. Newest last.

| # | Date | Decision | Reason |
|---|------|----------|--------|
| 1 | 2026-11-07 | Models in scope: game margin (moneyline and puck line), totals, shots on goal props, points props | Shots on goal is the most predictable target (high counts driven by stable ice time and shot rate); points is next. Moneyline is the sharpest market but its features feed every other model |
| 2 | 2026-11-07 | No goalie saves props | Out of scope by choice; also depends on a pregame starter that no data source provides reliably |
| 3 | 2026-11-07 | No Monte Carlo simulation | Over/under probabilities come from closed-form count distributions instead |
| 4 | 2026-11-07 | Puck line modeled through goal margin, not derived from the moneyline model | Overtime games always end one goal apart and empty-net goals decide many -1.5 covers; a margin model captures both |
| 5 | 2026-11-07 | Probability-level ensembles with calibration, not majority-vote ensembles | Expected value and bet sizing need calibrated probabilities; majority votes produce picks only |
| 6 | 2026-11-07 | Walk-forward validation by season only; no k-fold | Games are time-ordered; k-fold leaks future information into training |
| 7 | 2026-11-07 | Start a new project and treat both earlier projects as read-only sources | Keeps the earlier work as a permanent record and gives one clean, documented pipeline |
| 8 | 2026-11-07 | DuckDB on Parquet as the working database; Google Cloud for storage and an optional warehouse | Data is small; DuckDB lets anyone reproduce results without a cloud account. The data already lives in Google Drive and Colab |
| 9 | 2026-11-07 | Data files are not committed to this repository | Some sources have terms of use (scraped odds, MoneyPuck non-commercial license); the repository documents how to obtain them instead |
| 10 | 2026-11-07 | Files over 10 MB are transferred by splitting them into 3 MB parts in Colab, with a SHA-256 manifest checked on reassembly | The Drive connection used for the audit stops at 10 MB, and 9 MB parts proved too large for it in practice. Splitting needs no extra credentials and keeps a byte-for-byte check on every file |
| 11 | 2026-11-07 | Large files that are duplicates or can be rebuilt are not transferred: older dated versions of the research features, the OddsHarvester browser capture, the raw MoneyPuck CSV, and SportsGameOdds raw responses already parsed into a table | Keeps the transfer to the files the audit needs (about 280 MB) |
| 12 | 2026-11-07 | Spend remaining PuckAPI credits on player prop price history, spread across 22 player-markets in 25-game windows | Neither project had prop prices, so no prop model had been tested against the market. Pricing is 10 credits per 25 games, so the total sample is fixed by budget; spreading it across players reduces dependence on any one player |
| 13 | 2026-11-07 | Keep purchased PuckAPI data in `data_snapshots/` in the private repository | The working machine is temporary and the data cost credits. To be removed from the repository, including history, before it is made public |
| 14 | 2026-11-07 | Grade every prop from official NHL box scores, not the provider's result field | The provider's result disagreed with the NHL box score on 2 of 626 games, one of which changes the grade |
| 15 | 2026-11-07 | Use SportsGameOdds player-prop closing prices from 2026-01-16 as the prop backtest set; ignore earlier events | Earlier events have no opening or closing prices, only in-game last prices. From 2026-01-16, 96% of lines have both closing prices |
| 16 | 2026-11-07 | API keys are stored only as environment network secrets, never in files, code or the repository | Keys in files or chat can leak; the proxy injects them into requests |
| 17 | 2026-11-07 | SportsDataIO not used | The available key is a trial key and returns deliberately scrambled data (fake sportsbooks, wrong scores) |
| 18 | 2026-11-07 | Game-market backtests (moneyline, puck line, totals) use SportsGameOdds consensus closing prices from 2026-01-16 plus the external closing CSV for earlier 2025-26 games; per-book SportsGameOdds prices are never used | Consensus open and close fields are pregame (closing moneyline log loss 0.667). Per-book fields are last-seen prices and include in-game quotes |
| 19 | 2026-11-07 | Shots-on-goal props are judged by whether the model adds information to the closing price (a market + model blend fitted on earlier lines), not by the model alone | The stand-alone model is less sharp than the market but its weight in the blend is clearly above zero; this is the honest test of usefulness |
| 20 | 2026-11-07 | The shots-on-goal confirmation test is fixed in advance: blend fitted on lines through 2026-11-06, 4-point edge, closing price, every 2026-27 game after 2026-11-06 | The first ROI estimate used thresholds chosen after seeing results; a pre-set rule on new games removes that bias |
| 21 | 2026-11-07 | Game models predict regulation goals with a fitted tie adjustment, and derive moneyline, puck line and totals from the same score distribution | One distribution keeps the three markets consistent; overtime and shootout rules make regulation the natural unit, and independent Poisson scores under-predict ties by a quarter |
| 22 | 2026-11-07 | The totals model is not used for betting or as a projection | It scores worse than a constant in every comparison against closing lines |
| 23 | 2026-11-07 | The puck line is tracked live with a fixed rule: calibrated GLM, 2-point edge, SportsGameOdds consensus close, games after 2026-11-06 | It matched or beat the closing line in three of four comparisons but not significantly; a fixed rule on new games is the honest test |
| 24 | 2026-11-07 | Current-season schedule, scores and goalie lines come from the NHL web API once the audited v2 tables end (2026-11-02) | The feed matched v2 exactly on all 470 overlapping team-games and 940 goalie lines |
| 25 | 2026-11-07 | The shots-on-goal blend for live tracking is fitted on all 15,777 lines through 2026-11-06 | Decision 20 fixed the cutoff; using every line before it gives the most stable coefficients |
| 26 | 2026-11-07 | Lineup features use the skaters who dressed for the game (assumption L-1) | Teams announce lineups before puck drop; the features use only those players' pregame form, never game results |
| 27 | 2026-11-07 | Points model v2 (with lineup features) replaces v1; v1 results are kept | v2 is better in every season on all games and on prop lines; neither shows an edge over the market |
| 28 | 2026-11-07 | Elo is a documented power rating and an optional game-model feature, not part of the default game model | Adding it moved moneyline log loss by 0.0002 to 0.0024 either way across six seasons, within noise; the default model is also the one frozen for live tracking |

