# NHL Prediction System

Machine learning models for NHL betting markets: moneyline, puck line, total
goals, and player props for shots on goal and points. Every model is trained
and evaluated with walk-forward validation and backtested against sportsbook
closing lines.

This README is a living document. Each phase of the project adds a section
recording what was done, what was found, and the numbers behind it. Decisions
and their reasons are logged in [docs/decision-log.md](docs/decision-log.md).

**Status:** Phase 1, inventory and audit of the two earlier versions of this
project. No new models have been trained yet.

---

## Project goals

1. Clean, audited data with documented provenance for every source.
2. Models whose methods match what their documentation claims.
3. Backtests that are walk-forward only and compared against closing lines.
4. Full documentation, so every reported number can be traced to code and data.

## Models in scope

| Model | Market | Target |
|-------|--------|--------|
| Game margin | Moneyline and puck line | Home goals minus away goals |
| Totals | Total goals over/under | Total goals in the game |
| Shots on goal | Player prop | Player shots on goal |
| Points | Player prop | Player points |

Out of scope: goalie saves props, and Monte Carlo simulation of any kind.
Over/under probabilities come from fitted count distributions (Poisson or
negative binomial).

## Project history

This project consolidates two earlier versions built in Google Colab:

- `NHL_Prediction_System` (started 2026-10-14): holds the production model
  bundles and manifest.
- `NHL_Prediction_System_ChatGPT` (started 2026-10-18): a staged research
  pipeline (about 60 stages) covering data acquisition, source audits, feature
  construction, moneyline research, calibration, and a forward odds pilot.

Both are kept unchanged as read-only sources. This repository rebuilds the
system from audited inputs, keeping the earlier models as benchmarks.

### Earlier results being audited

As reported in the earlier production manifest (not yet verified):

| Model | Holdout | Reported result |
|-------|---------|-----------------|
| Moneyline and totals (bivariate negative binomial, 200 features) | 727 games | Moneyline log loss 0.6705 vs Pinnacle closing 0.6706; ROI +4.4% (SE 4.5%) |
| Shots on goal (hurdle negative binomial) | 50,182 skater-games, 2025-26 | Predicted zero-shot rate 27.0% vs actual 27.2% |
| Points (joint negative binomial) | 50,182 skater-games, 2025-26 | Predicted zero-point rate 65.1% vs actual 65.3% |
| Blocked shots (negative binomial) | 16,342 skater-games | Predicted zero-block rate 90.6% vs actual 94.6% |

The latest research reports (2026-10-31) found the moneyline model behind
every sportsbook closing line in 2025-26 (log loss about 0.690 vs 0.674-0.682).

---

## Architecture

| Layer | Tool | Purpose |
|-------|------|---------|
| Working database | DuckDB on Parquet | Cleaning, auditing, features, backtests; reproducible on a laptop |
| Storage | Google Drive, later Google Cloud Storage | Audited raw and clean Parquet files |
| Warehouse (optional) | BigQuery | Daily pipeline and dashboards |

### Data layout (Google Drive: `NHL_Prediction_System_Claude/`)

| Folder | Contents |
|--------|----------|
| `00_inventory/` | Every file in both earlier projects, deduplicated by checksum |
| `01_raw/` | Source data carried forward, with provenance notes |
| `02_clean/` | Audited, cleaned tables |
| `03_features/` | As-of feature tables (only information known before each game) |
| `04_models/` | Trained models and their predictions |
| `05_backtests/` | Walk-forward results against closing lines |
| `06_audit/` | Data and model audit reports |

### Repository layout

| Path | Contents |
|------|----------|
| `docs/` | Plan, decision log, data dictionary, model cards, audit reports |
| `src/` | Cleaning, features, models, backtest |
| `tests/` | Data checks and leakage checks |

Data files are not committed. See `.gitignore`.

---

## Phase log

### Phase 0: Setup (2026-11-07)

- Created this repository and the `NHL_Prediction_System_Claude` Drive folder.
- Recorded the plan in [docs/plan.md](docs/plan.md).

### Phase 1: Inventory and audit

**1a. Inventory (done 2026-11-07).** Listed every file in both earlier
projects: 143 files (400 MB) in v1 and 10,819 files (1.4 GB) in v2, with
almost no overlap between them. Details and findings in
[docs/inventory.md](docs/inventory.md). Early findings:

- v1 contains no training code, only a loader, so its models are audited from
  the saved bundles, out-of-fold predictions and reports.
- v1 has seven pairs of raw files that are probably duplicates.
- v1 skater tables for 2021-22 to 2023-24 are a quarter the size of later
  seasons, which may mean truncated downloads.

**1b. Data audit (in progress).** The 10 files over the 10 MB download
limit were transferred in 3 MB parts and all verified by SHA-256 against the
originals. Findings so far in [docs/audit/data-audit.md](docs/audit/data-audit.md):

- **D-1 (High):** v1 skater data for 2021-22 to 2023-24 holds about 5
  skaters per game instead of 36. The gap carries into the player feature
  table used to train the shots-on-goal and points models. The 2025-26
  holdout is complete. **Raw data fixed:** the three seasons were re-fetched
  (about 50,400 rows each, every regular-season game present, values
  identical to the rows that already existed). Player features still need
  rebuilding.
- **D-2 (Low):** the `_full` raw files are byte-identical duplicates.
- **Game results and labels: clean.** Both projects' labels match the NHL
  schedule and an independent copy from PuckAPI on every game (9,242 in v1,
  9,392 in v2), with correct regulation, overtime and shootout handling.
  Modeling note: the official final score credits the shootout winner with a
  goal, so the rebuild models real goals and uses the official score only to
  grade bets.
- **Odds, O-1 (Critical):** v1 graded about 74% of its bets at prices captured
  during or after the game (log loss 0.29 to 0.46, where a real closing line
  scores about 0.67). Results that rely on those prices are invalid. v2 caught
  the problem and excluded those quotes. Clean pregame closing prices exist
  for about 1,560 games (2025-26 and early 2026-27); there are no bookmaker
  odds before February 2024 in either project.

**1c. Backtest review.** [docs/audit/backtest-review.md](docs/audit/backtest-review.md).
On clean closing prices the earlier moneyline models score close to, but
not better than, the market (v1: 0.6735 vs 0.6706 log loss on 727 games,
reproduced exactly; v2 2025-26: 0.687 vs 0.682 on 1,292 games). Betting the
model's positive-EV picks at the best of nine books returns -1.5% (SE 2.9%).
The +4.4% ROI in the v1 manifest cannot be reproduced from the saved ledger.
No prop betting backtest exists, because there is no historical prop-price data.

**1d. Prop price history.** Bought 745 player-games of closing prop prices
from PuckAPI (shots on goal and points, 16 star players, mostly 2025-26):
[docs/audit/prop-history.md](docs/audit/prop-history.md). Results agree with
NHL box scores on 624 of 626 games. Closing shots-on-goal prices are close to
coin flips (log loss 0.690 vs 0.692 for a constant), with a 6.5% margin.

**1e. Prop closing prices at scale (SportsGameOdds).** Downloaded every NHL
game from 2026-01-16 to 2026-11-06 and extracted about 120,000 player-prop
closing lines: [docs/audit/sgo-props.md](docs/audit/sgo-props.md). Graded on
NHL box scores, 2025-26 gives about 53,000 lines across shots on goal, points,
assists and goals. Found that the provider's `points` stat is goals and
`goals+assists` is points. This is the prop backtest set.
