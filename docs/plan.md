# NHL Prediction Model Plan

Working plan for a set of NHL betting models built with the skills in this
repo and data from PuckAPI plus the maintainer's own statistics. Nothing has
been built yet. This file records the decisions made so far and the questions
still open, so work can resume in a new session.

Status: planning. No code written.

---

## Scope

### In scope

| Model | Market | Target |
|-------|--------|--------|
| Game margin model | Moneyline and puck line | Home goals minus away goals (one model serves both markets) |
| Totals model | Total goals over/under | Total goals in the game (count) |
| Shots on goal | Player prop | Player shots on goal in the game (count) |
| Points | Player prop | Player points in the game (count; most lines are 0.5) |

### Possible later additions

| Model | How it would be built |
|-------|-----------------------|
| Goals | Projected shots on goal x regressed shooting percentage |
| Assists | From the points model, after goals are taken out |

### Out of scope

- Goalie saves props (dropped)
- Monte Carlo simulation of any kind. The `playoff-simulation` skill is not used.
  Over/under probabilities come from closed-form count distributions (Poisson or
  negative binomial), not simulation. Confirm this is acceptable (open question 6).

---

## Ranking of predictability

Most to least predictable, with the reason:

1. **Shots on goal**: high counts (2-4 per game), driven by ice time and shot
   rate, both stable. Shot rate settles after about 20 games. Prop markets are
   softer than game markets.
2. **Points**: higher counts than goals or assists alone. Driven by ice time,
   power-play share, and team scoring.
3. **Total goals**: goals per game are noisy and most lines sit at 5.5 or 6.5.
   The signal is goalie quality, pace, and special teams.
4. **Puck line**: an extension of the margin model. Overtime games always end
   one goal apart, and empty-net goals often decide -1.5 covers.
5. **Moneyline**: the sharpest market. A good model reaches about 58-60%
   accuracy, close to the closing line. Its features feed every other model.
6. Goals and assists (not planned yet): shooting percentage and secondary
   assists are close to random over small samples.

---

## Build order

1. **Data layer**: pull games, goalie starts, team stats, player game logs, and
   odds into a local database. Load the maintainer's own statistics.
2. **Shared features**: lagged rolling stats (`.shift(1)` before every rolling
   window), five Elo variants, rest and back-to-back flags, goalie quality from
   the API's `gsax`, `highDangerSavePct`, `rollingSavePct`.
3. **Game margin model**: moneyline and puck line. Logistic regression baseline
   first; gradient boosting only if walk-forward results improve by at least
   1 percentage point.
4. **Totals model**: count regression on total goals; P(over line) read from
   the fitted distribution.
5. **Ice-time model**: shared foundation for every player prop.
6. **Shots on goal**: projected ice time x shot rate per 60, adjusted for
   opponent.
7. **Points**: projected ice time, power-play share, and team projected goals
   from the totals model.
8. **Validation and calibration** at every step (see below).
9. **Backtest** against historical lines for return and closing-line value.

---

## Departures from the skills

1. **prop-modeling data source is out of date.** The skill says PuckAPI has no
   skater game logs or ice time. The live server now has
   `get_player_game_log`, `get_skater_season_stats`, `get_shifts`,
   `get_situation_time`, `get_player_props`, and `get_prop_history`. Verify what
   these return before relying on them; use the NHL Stats API only as a
   fallback.
2. **No hard-vote ensembles.** `model-building` recommends majority-vote
   ensembles, which output picks, not probabilities. Expected value and bet
   sizing need calibrated probabilities, so average or stack model
   probabilities, then calibrate.
3. **Puck line is modeled as goal margin**, not derived from the moneyline
   model. The skills have no puck-line method.
4. **Count distributions instead of simulation** for totals and props.

---

## Data notes

- Games from 2010-11. Odds from 2019-20 (sparse that season), full coverage from
  2020-21. Training and backtest windows for anything that needs odds start
  there.
- Starting goalies from `get_game_detail` are post-game (who actually started).
  That is correct for training. For upcoming games the starter must come from
  the maintainer or a news source.
- Team possession (`corsiPct`, `fenwickPct`, team xG) is 5v5. Goalie expected
  goals cover every strength. Never difference the two.
- Credit costs (from the PuckAPI endpoint reference and the live server):
  `get_games` 5, `get_player_game_log` 5, `get_game_detail` 10, `get_odds` 10,
  `get_player_props` 10, `get_prop_history` 10, `get_shifts` 10,
  `get_situation_time` 10 per game. One regular season is about 1,312 games, so
  any per-game 10-credit pull is about 13,000 credits per season.

---

## Validation rules

- Walk-forward by season only (expanding window, at least 3 folds). No k-fold.
- Metrics: log loss and Brier score per fold; RMSE for count targets; accuracy
  is secondary.
- Baselines: home team always wins, devigged closing line, prior-season record.
- Calibration on held-out folds only (Platt first, isotonic with 500+ samples).
- Backtest with closing-line value and bootstrap confidence intervals.

---

## Open questions

Answered on 2026-11-07:

- Question 1 (own statistics): the two earlier Drive projects,
  `NHL_Prediction_System` and `NHL_Prediction_System_ChatGPT`. They are
  inventoried and audited in Phase 1.
- Question 4 (code location): this repository.
- Question 5 (stack): Python with DuckDB on Parquet; Google Cloud for storage
  and an optional BigQuery warehouse.

Still open: questions 2, 3 and 6. The original list follows.

1. **Own statistics.** What are they and at what level: per game, per player,
   per season? Which seasons do they cover? Is every row dated so it is clear
   what was known before each game? (Decides whether they can be used without
   leaking results.) How do they link to PuckAPI game and player IDs?
2. **Credit budget.** Which PuckAPI plan? Full game detail for every season
   since 2010-11 is roughly 210,000 credits; odds for the seasons that have them
   are roughly 80,000. The number of seasons may need limiting.
3. **Moneyline definition.** Include overtime and shootouts (standard two-way
   line), or also build a 60-minute regulation model?
4. **Code location.** This repo is a public skills library, and model code here
   would publish the maintainer's statistics. Use a separate private repo?
5. **Stack.** Python with scikit-learn, LightGBM, and a local SQLite or DuckDB
   database?
6. **Count distributions.** Confirm that reading over/under probabilities from a
   Poisson or negative binomial distribution fits the no-Monte-Carlo rule.

---

## First steps in the next session

1. Read the maintainer's statistics from Google Drive and Google Docs. For each
   file report columns, seasons covered, join keys, and whether rows are dated
   pre-game.
2. Work through the open questions above.
3. Check which prop markets have historical lines in `get_player_props` and
   `get_prop_history` (shots on goal and points). Without past lines a market
   cannot be backtested, which may change the build order.
