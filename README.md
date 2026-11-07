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

In progress.
