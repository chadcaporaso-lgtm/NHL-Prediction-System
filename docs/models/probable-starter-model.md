# Model Card: Probable Starting Goalie

Code: `src/features/build_probable_starter.py`. Test:
`tests/test_probable_starter_leakage.py`. Results:
`docs/models/probable_starter_results.json`. Used by the game-model variant
`python3 src/models/game_model.py probable` and by the version 2 live
puck-line rule (decision 29).

## Why

The game models use the goalie who actually started (gap G-E). A bettor who
bets before the starter is confirmed does not know that, so the backtests had
a small look-ahead. This model gives each team's goalies a start probability
from earlier games only. The game model then uses the probability-weighted
save % and experience instead of the actual starter's.

## Method

- **Candidates** for a team-game: every goalie who dressed for the team in
  its previous 10 games, plus an "other" option for a goalie not seen in that
  window (call-up, trade, return from injury). The actual starter is a named
  candidate in 98.4-99.3% of team-games each season.
- **Features** (all from earlier games): share of the team's last 5 and last
  10 games he started, share of the team's starts this season, started the
  previous game, dressed for the previous game, started the previous game
  *and* the team is on the second night of a back-to-back, consecutive-start
  streak, days since his previous start, and his save % over the previous 40
  appearances (shrunk to .905). Days since start and save % use only
  appearances that started at least 12 hours earlier (see the leakage test
  below).
- **Model:** conditional logit (a softmax over the team-game's candidates),
  maximum likelihood with a small ridge penalty, 10 coefficients.
- **Walk-forward by season:** each season is predicted by a model fitted on
  earlier seasons only. 2019-20 has no earlier season in the goalie table, so
  its probabilities are in-sample. They serve only as training rows for the
  game model and are not reported as a test result.
- **Game-model inputs:** expected save % = sum of probability x save %, and
  expected appearances likewise. The "other" option counts as .905 and 0
  appearances.

## Results (who starts)

| Season | Team-games | | Model top pick | Most starts in last 10 | Previous game's starter | Log loss | Night 2 of a back-to-back after starting night 1: team-games | Model | Most starts |
|--------|------:|---|------:|------:|------:|------:|------:|------:|------:|
| 2019-20 | 2,424 | in-sample | 0.710 | 0.595 | 0.485 | 0.600 | 388 | 0.773 | 0.379 |
| 2020-21 | 1,904 | | 0.695 | 0.571 | 0.485 | 0.719 | 306 | 0.774 | 0.333 |
| 2021-22 | 2,802 | | 0.702 | 0.594 | 0.499 | 0.672 | 436 | 0.764 | 0.367 |
| 2022-23 | 2,800 | | 0.711 | 0.589 | 0.472 | 0.673 | 410 | 0.837 | 0.320 |
| 2023-24 | 2,800 | | 0.716 | 0.593 | 0.449 | 0.665 | 359 | 0.850 | 0.304 |
| 2024-25 | 2,796 | | 0.732 | 0.609 | 0.451 | 0.624 | 398 | 0.879 | 0.322 |
| 2025-26 | 2,788 | | 0.708 | 0.580 | 0.435 | 0.659 | 430 | 0.849 | 0.337 |
| 2026-27 | 522 | | 0.665 | 0.563 | 0.418 | 0.688 | 57 | 0.912 | 0.246 |

Out of sample (2020-21 to 2026-27, 16,412 team-games) the top pick is right
**71.0%** of the time, against 58.9% for "most starts in the last 10 games"
and 46.3% for "the previous game's starter". Teams rotate goalies more than
one might expect: the previous game's starter starts again only 42-50% of the
time, and this has fallen every season since 2021-22. The model is strongest
on the second night of a back-to-back after the goalie started the first
night (77-91% correct), where both simple rules are usually wrong.

Largest standardized coefficients (2026-27 fit): "other" -3.16, dressed for
the previous game +0.93, started night 1 of a back-to-back -0.60, days since
his last start -0.35, save % +0.30, start shares +0.17 to +0.25.

## Effect on the game models

Log loss, calibrated Poisson GLM, all games. "Actual" is the default model
(actual starter); "probable" uses the expected goalie values in training
and test rows.

| Market | Seasons | Change in log loss (probable - actual) |
|--------|---------|------|
| Moneyline | 2021-22 to 2026-27 | -0.0007 to +0.0003 |
| Home wins by 2+ | same | -0.0013 to +0.0011 |
| Away wins by 2+ | same | -0.0008 to +0.0005 |

Against closing lines (log loss):

| Market | Source, season | Games | Market | Actual starter | Probable starter |
|--------|----------------|------:|------:|------:|------:|
| Puck line | SGO 2025-26 | 646 | 0.6380 | 0.6364 | 0.6368 |
| | SGO 2026-27 | 261 | 0.6428 | 0.6421 | 0.6429 |
| | external 2025-26 | 1,385 | 0.6272 | 0.6246 | 0.6246 |
| | external 2026-27 | 182 | 0.6563 | 0.6565 | 0.6585 |
| Moneyline | SGO 2025-26 | 646 | 0.6739 | 0.6759 | 0.6763 |
| | SGO 2026-27 | 261 | 0.6470 | 0.6527 | 0.6528 |

The frozen puck-line rule (2-point edge, SGO close) goes from -0.1% ROI on
557 bets to -1.2% on 556 bets, with a standard error of 4.2% for both. The
puck-line blend weight is 0.61 (SE 0.31) against 0.61 (SE 0.32) before.

**Conclusion:** the look-ahead from using the actual starter was real but
small. The expected save % correlates 0.83 with the actual starter's, and
the opposing starter's effect in the GLM (-0.016 log goals per standard
deviation) is about a third of the team-form features'. Every number moves
within its noise. The live puck-line rule now uses the probable-starter
version (decision 29), so the live test has no look-ahead.

## Leakage test

For four random cutoff days, the candidate table is rebuilt from goalie rows
before that day, plus that day's rows with every result scrambled (who
started, who played, saves, shots, ice time). Later games are dropped. The
features of that day's games must equal the full build. The first version
of the test failed on one goalie: traded in the off-season, he was still a
candidate for his old team, and his game for his new team had started 30
minutes earlier the same evening, so its result entered his save %. The
12-hour rule fixed it. The main game features cannot have this problem,
because an actual starter cannot play two games at once.

## Limits

- No injury, recall or announcement data: a goalie who is hurt stays a
  candidate until he stops dressing, and a recalled goalie is "other" until
  he dresses once.
- No team-specific rotation habits beyond what the shares capture. A
  team-level random effect or a tree model could add a few points of
  accuracy.
- Morning-skate reports and confirmed starters, when available, beat any
  model. This one is for prices taken before those reports.
