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
| 10 | 2026-11-07 | Files over 10 MB are transferred by splitting them into 9 MB parts in Colab, with a SHA-256 manifest checked on reassembly | The Drive connection used for the audit stops at 10 MB. Splitting needs no extra credentials and keeps a byte-for-byte check on every file |
| 11 | 2026-11-07 | Large files that are duplicates or can be rebuilt are not transferred: older dated versions of the research features, the OddsHarvester browser capture, the raw MoneyPuck CSV, and SportsGameOdds raw responses already parsed into a table | Keeps the transfer to the files the audit needs (about 280 MB) |
