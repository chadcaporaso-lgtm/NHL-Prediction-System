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

See [docs/data-coverage.md](docs/data-coverage.md) for what each model needs and the open gaps.

**1f. Game-market closing prices (SportsGameOdds).** Extracted full-game
moneyline, puck line and total opening and closing prices from the same
downloads: 881 completed games from 2026-01-16 to 2026-11-02, every one with
closing prices in all three markets (data audit finding O-6). Closing log
loss: moneyline 0.667, puck line 0.641, totals 0.688, against 0.692, 0.686
and 0.693 for a constant. Clean game-market prices still cover only 2025-26
and the start of 2026-27; earlier seasons need PuckAPI's archive odds (O-5).

**1g. Data gaps filled.** Downloaded skater game data for 2019-20 and
2020-21 (two more training seasons), re-fetched the incomplete 2026-27
skater data (now 36 skaters per game), and refreshed MoneyPuck through
2026-11-06. Skater game data now covers 2019-20 to the present with every
game complete. MoneyPuck revised a few dozen past values out of 106,040 rows,
so feature builds record which snapshot they use. Details in the data audit.

**1h. Goalie game logs: clean.** 37,568 goalie-games covering every completed
game from 2019-20, exactly one starter per team-game, and goals and shots
against that reconcile with official team totals (differences are empty-net
goals). 81 rows (0.2%) are off by one between saves, goals and shots, a quirk
in the NHL source. With this, every training table is in place.

### Phase 2: Features

**2a. Skater game table and as-of features (2026-11-07).**
`src/features/build_skater_games.py` joins the audited skater summary and
ice-time tables into one row per player-game: 339,062 rows covering all
9,421 completed games from 2019-20 to 2026-11-06, with no duplicates and ice
time on every row. `src/features/build_skater_features.py` adds features
built only from earlier games: weighted recent form, last 10 games, season
to date and career means for shots, ice time, power-play time, goals,
assists and points; shot and point rates per 60 minutes; previous-season
rates; rest days; and team context (own team's shots for, opponent's shots
and goals allowed over their last 20 games).
`tests/test_skater_feature_leakage.py` recomputes the features for 500
random player-games and 300 team-games using only games dated before each
one, and checks that no feature reproduces the same-game target. All tests
pass.

### Phase 3: Models

**3a. Shots on goal (2026-11-07).** Code: `src/models/prop_model.py`. Model card:
[docs/models/sog-model.md](docs/models/sog-model.md). A negative binomial
shot model (gradient boosting with Poisson loss), validated walk-forward by
season from 2021-22, beats a recent-form baseline in every season. Against
15,777 closing prop lines, it alone is slightly worse than the market (log
loss 0.6821 vs 0.6815). Combined with the market price, it adds information:
fitted on 2025-26 lines and tested on 2026-27, the model's weight is 0.35
(SE 0.09) and the blend scores 0.6766 vs the market's 0.6797. Betting the
blend at a 4-point edge returned +8.3% (SE 2.7%) on 907 bets, but this is
one short test period with thresholds chosen afterward, so it is a
hypothesis to confirm on later 2026-27 games, not a result.

**3b. Points (2026-11-07).** Model card:
[docs/models/points-model.md](docs/models/points-model.md). Same walk-forward
method, with goal, assist, power-play, team-scoring and opposing-goalie
features. It beats the baseline every season, but against 18,851 closing
lines it is worse than the market (2026-27 log loss 0.6405 vs 0.6340), and a
market + model blend gains nothing on held-out 2026-27 lines. No betting edge;
used as a projection only. The likely missing piece is lineup information
(linemates, power-play unit) that the closing price already reflects.

**3c. Game models: moneyline, puck line, totals (2026-11-07).** Model card:
[docs/models/game-model.md](docs/models/game-model.md). Poisson model of each
team's regulation goals from MoneyPuck team form, the opposing starter, rest
and league scoring level, with a tie adjustment (independent scores predict
16% regulation ties; 21-25% happen) and walk-forward calibration. On the full
2025-26 season the moneyline log loss is 0.6830 against the closing line's
0.6808, better than both earlier projects (best 0.6867) but still behind the
market, and it adds nothing to the closing price in a blend. On the puck line
it matches or slightly beats the close in three of four comparisons (blend
weight 1.9 standard errors), which is suggestive but not significant. Totals
show no skill. No game market shows a betting edge.

### Phase 4: Live tracking

**4a. Frozen rules and weekly pipeline (2026-11-07).**
[docs/live/README.md](docs/live/README.md). The shots-on-goal and puck-line
rules are fixed in advance and graded only on games after 2026-11-06, with
model fingerprints checked each week so the models cannot drift. A weekly
Colab cell pulls this season's scores, goalie lines, skater games and
MoneyPuck from public sources; on all 470 team-games and 940 goalie lines it
shares with the audited tables, it matched exactly. One command then
rebuilds features, reruns the leakage tests and models, and grades the rules.

### Phase 5: Portfolio write-up

**5a. Case study page (2026-11-07).** [docs/portfolio/index.html](docs/portfolio/index.html):
the audit findings, methods and results against the closing line in one page,
with every figure taken from the results files in `docs/models/`.

### Phase 6: Improvements

**6a. Player-name matching (2026-11-07).** An alias list for 12 players the
odds provider lists under another first name cut unmatched shots and points
lines from 3.4% to 3.1%; the rest are scratched players whose bets are void.
Results moved only slightly (shots blend on 2026-27: 0.6764 vs the market's
0.6796; 907 bets at a 4-point edge, +8.3%, SE 2.7%).

**6b. Lineup features for points (2026-11-07).** Five features for each
player's role in the night's lineup (ice-time and power-play rank among
dressed teammates, a linemate-quality proxy, and missing-scorer gaps), built
from pregame form only and leakage-tested. The points model improves in
every season (prop-line log loss 0.6551 to 0.6542), but still trails the
closing price (0.6515), and a market blend still gains nothing. Details in the
points model card.

**6c. Team Elo ratings (2026-11-07).** Model card:
[docs/models/elo-model.md](docs/models/elo-model.md). Margin-of-victory Elo
built from scratch, with its settings chosen walk-forward from earlier seasons
only and two leakage tests. It beats a constant every season and a flat-K
Elo in five of six, but on its own trails the game model from 2023-24 on and
trails the closing moneyline by about 0.01 log loss. As a game-model feature
it changes log loss by 0.0002 to 0.0024 either way, within noise, so the
default model stays without it. Current ratings serve as a power ranking.

