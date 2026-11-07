# Live Tracking

The shots-on-goal and puck-line results so far come from a short test period
with thresholds chosen after seeing the data. Live tracking tests the same
rules, fixed in advance, on games played after the cutoff.

- Rules: [frozen_rules_v2.json](frozen_rules_v2.json) (written by
  `src/live/freeze_rules_v2.py` on 2026-11-07; decision 29). Version 1,
  [frozen_rules.json](frozen_rules.json) (`src/live/freeze_rules.py`;
  decisions 20, 23, 25), is kept as the record; no game was graded under it.
- Results: [tracking.json](tracking.json), cumulative and by week
- Bet ledgers: `data/live/ledger_*.parquet` (not committed)

## Frozen rules (version 2)

| Market | Model | Bet when | Price | Games |
|--------|-------|----------|-------|-------|
| Shots on goal | Gradient-boosted negative binomial (trained on seasons before 2026-27), blended with the closing price; blend fitted on the 15,777 lines through 2026-11-06 (model weight 0.42, SE 0.07). Unchanged from version 1 | blend and market differ by more than 4 points | SportsGameOdds consensus close | after 2026-11-06 |
| Puck line | Calibrated Poisson GLM, probable-starter variant: the opposing goalie's inputs are weighted by start probabilities from earlier games, so nothing depends on the announced starter | model and market differ by more than 2 points | SportsGameOdds consensus close | after 2026-11-07 |

Before grading, `track.py` checks fingerprints (the 2026-27 dispersion, the
mean prediction over 2026-27 games up to a fixed date, and for the puck line
the tie factor and the mean expected goalie save %) and stops if a retrained
model differs from the frozen one. Data after the cutoff can never
change those numbers, because every feature uses only earlier games.

## Weekly steps

1. **Colab:** run `src/ingest/colab_weekly_refresh.py`. It pulls this
   season's schedule, scores, goalie lines, skater games and MoneyPuck, and
   stages them in 3 MB parts with SHA-256 checksums under
   `01_raw/_transfer_3mb/live_<date>/`.
2. **Transfer:** download the parts into `data/live/<date>/` and verify each
   file against the manifest's SHA-256.
3. **Update:** `python3 src/live/weekly_update.py <date>`. This fetches new
   SportsGameOdds lines, checks the feed against the audited tables, rebuilds
   features, runs the leakage tests, reruns the models and grades the rules.
4. **Record:** commit `docs/live/tracking.json` and add the week to the README.

## Feed check (2026-11-07)

The NHL web API feed agreed with the audited v2 tables on every overlapping
2026-27 game: 470 team-games and 940 goalie lines, zero differences in final
score, real goals, shots, decision type, starter, saves, shots against or
goals against (`docs/audit/live_feed_check_2026-11-07.json`).

## When to read the result

At about 830 bets a test period, the standard error on ROI is about 2.8
points. Treat the live result as meaningful after at least 500 bets, and
report it whatever it shows.
