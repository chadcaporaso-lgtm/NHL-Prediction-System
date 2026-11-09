# NHL Prediction System: instructions for Claude

Read this first in every session. It is the project's memory: earlier chat
sessions are not carried over, so everything a new session needs is here or
linked from here. Update the "Current state" and "Next steps" sections at the
end of each working session.

## The project

NHL betting models (moneyline, puck line, totals; player props for shots on
goal and points), rebuilt from an audit of two earlier Google Colab projects
(`NHL_Prediction_System` = v1, `NHL_Prediction_System_ChatGPT` = v2, both in
the owner's Google Drive, read-only). Goal: a portfolio-quality project for
employment in data science, with clean audited data, walk-forward backtests
against sportsbook closing lines, and documentation of every step.

- Living write-up: `README.md` (phase log). Decisions: `docs/decision-log.md`.
- Model cards: `docs/models/*.md`. Audit: `docs/audit/`. Live test: `docs/live/`.
- Case study page: `docs/portfolio/index.html`, published as a private
  artifact: https://claude.ai/artifact/Qgc46vBMEuuh5rPmAvfSw8 (republish by
  passing that URL).
- Drive working folder: `NHL_Prediction_System_Claude/` (00_inventory to 06_audit, 01_raw).

## Rules the owner set

- No Monte Carlo simulation; over/under probabilities come from fitted count
  distributions. No goalie saves props.
- Walk-forward validation by season only, never k-fold. Every feature uses
  only earlier games, and each feature table has a leakage test in `tests/`.
- Do not cut corners. Report every result, including failures, and keep
  earlier versions' results when a model changes (see the v1/v2 files).
- Keep the README phase log and decision log current with each change.
- API keys only as environment network secrets (request_credential), never in
  files, code, chat, Drive or the repo. SportsGameOdds and PuckAPI keys are
  stored that way.
- The repository is public. `data/` is gitignored; never commit purchased or
  licensed data (PuckAPI prop history, SportsGameOdds responses, MoneyPuck
  tables), keys, or personal details. Purchased PuckAPI data was removed from
  the repository and its history before publication (decision 13).
- The owner runs Colab cells when asked: give the full cell in a code block.
- A second repo, `claude-sports-analytics`, is public and has its own
  CLAUDE.md rules (no AI attribution there).

## Frozen live test: do not change

`docs/live/frozen_rules_v2.json` (decision 29; version 1 in
`frozen_rules.json`, decisions 20, 23, 25, kept as the record) fixes the
betting rules: shots on goal for games after 2026-11-06, puck line for games
after 2026-11-07. `src/live/track.py` checks model fingerprints and stops if
the default shots model (`prop_model.py shots`), the probable-starter model
(`build_probable_starter.py`) or the probable-starter game model
(`game_model.py probable`) changes. The default game model (`game_model.py`)
is no longer graded live but stays the reference for backtests. Add new
features as variants or separate outputs, never by editing those (see the
`elo` and `probable` variants). If the rules must change, write a new frozen
file with a new start date and a decision-log entry.

## Starting a new session (for the owner)

1. Open Claude Code on the web and choose the `NHL-Prediction-System`
   repository (branch `main`, after the newest work branch is merged).
2. Say: "Read CLAUDE.md, restore the data, then continue with the next step."
3. Run the Colab cell Claude gives you (`colab_bundle_core_inputs.py`) and
   paste its output back.
4. Claude restores and checks the data, rebuilds the pipeline, and picks up
   the next step from the list below. Old chats stay in the session list at
   claude.ai/code if you want to reread them.

## Restoring the data in a new session

A cloud session starts on a fresh machine with only this repository.

1. `pip install -r requirements.txt`.
2. Ask the owner to run `src/inventory/colab_bundle_core_inputs.py` in Colab.
   It stages the 19 audited inputs (about 28 MB) and the newest weekly
   refresh as 3 MB parts plus a manifest under
   `01_raw/_transfer_3mb/core_<date>/`.
3. Download the manifest and every part with the Google Drive connector
   (search by title `core__`), save the manifest locally, then run
   `python3 run.py restore <manifest.json>`. It
   verifies every SHA-256. The MoneyPuck table appears twice in the manifest
   with the same SHA-256 (core input and weekly refresh): download its 5 parts
   once; the script reuses identical files. Parts of 3 MB come back as saved
   tool results, smaller ones inline; the script searches both.
   The bundle also stages the owner-supplied odds files (decisions 37-40)
   when they are in Drive under `01_raw/external/` (`nhl_2026_27_odds.csv`,
   `sportsbook-nhl-2021-2022.csv`, `sportsbook-nhl-2022-2023.csv`) or
   `01_raw/inspect/` (`archive (8).zip`); `run.py restore` then places them
   in `data/external/` and checks their SHA-256
   (`src/inventory/place_external.py`). Without them `game_market.py` falls
   back to v2's crosswalk and `game_market_history.py` cannot run.
4. Re-download SportsGameOdds history (uses the stored network secret):
   `python3 src/ingest/fetch_sgo_events.py 2026-01-16 <yesterday>`.
5. Optional: `python3 run.py extras` (run it in the background, about 3
   hours; the environment allows api.nhle.com, api-web.nhle.com and
   moneypuck.com). It fetches NHL shift charts, NHL Edge and MoneyPuck game
   logs, builds and leakage-tests the features, and reruns the variant
   models and the earlier-season market tests (decisions 34-39). Only needed
   to work on those variants; the weekly live test does not use them.
6. Rebuild everything: `python3 run.py update` (runs
   `src/live/weekly_update.py`: fetches new odds, rebuilds features and
   models, runs the leakage tests, grades the live test, redraws the charts).

`python3 run.py check` runs the pipeline and leakage tests on synthetic data
without any real data (what GitHub Actions runs). When a builder or its
inputs change, keep `tests/synthetic/make_data.py` producing the same columns,
and run `check` before pushing.

Purchased PuckAPI prop history is not in the repository (decision 13). It is
used only by the prop-price audit (`src/audit/prop_history_profile.py`), whose
results are already in `docs/`; the weekly pipeline does not need it. Keep any
copy in the owner's private Drive.

## Weekly routine (live test)

Owner runs `src/ingest/colab_weekly_refresh.py`; download the parts from
`01_raw/_transfer_3mb/live_<date>/` into `data/live/<date>/`, verify against
the manifest, then `python3 run.py update <date>`. Commit
`docs/live/tracking.json` and the redrawn `docs/figures/`, and add the week
to the README. Treat the result as meaningful after about 500 bets.

## Current state (2026-11-09, end of fifth session)

- Data audit done (findings D-1, D-2, O-1 to O-7, goalie and live-feed checks).
- Models: shots on goal (no demonstrated edge: +8.3% on 907 held-out 2026-27
  bets, SE 2.7%, but 2,260 older-season PuckAPI lines in two independent
  batches do not repeat it: frozen rule -7.7% on 349 bets, SE 5.0, model
  blend weight 0.19, SE 0.20; decisions 32-33; live test decides), puck line (level
  with the close, not significant), moneyline (close, behind), points v2
  (no edge), totals (no skill), team Elo (power rating and optional feature),
  probable starting goalie (71.0% top-pick accuracy out of sample; the
  actual-starter look-ahead moves game-model log loss by at most 0.002).
- New player data (decisions 34-36, README 6i): NHL shift charts
  (linemates, power-play units), NHL Edge (previous-season profile only:
  the NHL publishes season totals), MoneyPuck game-by-game skater data.
  Tested as variants (`prop_model.py shots_edge|shots_mp|points_lm|
  points_lm_edge|points_mp|points_all`). Better forecasts (shots +
  MoneyPuck: 0.6771 vs 0.6781 on 2026-27 lines, 0.6840 vs 0.6854 on older
  lines; points + all three: 0.6359 vs 0.6382, market 0.6333), but no
  variant beats the close on held-out lines or has a profitable rule.
- Earlier-season market prices (decisions 37-40, README 6j): a newer
  2026-27 external file (282 games to 2026-11-08; copies of it disagree on
  past closes, finding E-1; later games join when the weekly update reruns
  `build_external_crosswalk.py`), average closing three-way prices 2021-22
  to 2026-27 and favorite-only moneylines 2021-23. The game model trails the market in
  every season (three-way 1.0364 vs 1.0313 over 6,992 games); the gap is
  team rating, not ties.
- Opponent-adjusted team ratings (decision 41, README 6k): time-decayed
  ridge ratings (`build_team_rating.py`, leakage-tested) as a game-model
  variant (`game_model.py rating`). Closer to the market's ordering but no
  better: three-way log loss 1.0363 vs 1.0364 (market 1.0313). The gap is
  largest in a team's first 10 games (0.013 per game): offseason roster
  change, which past results cannot supply. Not adopted.
- Fifth session restored without Colab from the existing Drive bundle
  (manifest_core_2026-11-07 plus archive (8).zip from
  manifest_inspect_2026-11-09 and the three CSVs in 01_raw/external/).
  `restore_core_inputs.py` now finds every part of non-core bundles.
- Closing-line value (shots): the market moves toward the model between open
  and close (slope 0.21, SE 0.007), but bets at the open lose to the fair
  close on price (CLV -2.1 points); opening-price profit comes mainly from
  lines that later moved. Not adopted (decision 30).
- Live test: rules version 2 frozen on 2026-11-07 (puck line on the
  probable-starter model, games after 2026-11-07); no games graded yet. The
  newest data (refresh 2026-11-07) ends on 2026-11-05 for team and goalie
  lines and 2026-11-06 for skaters.
- Repository is public. `main` was rewritten to remove purchased data. The
  old branch `claude/peaceful-volta-mygk7c` is deleted (2026-11-09); only
  pull request #1's hidden ref (`refs/pull/1/head`) still holds it, and the
  owner has asked GitHub Support to purge it.
- Case study page republished 2026-11-09 (artifact version 8, adds the team-rating
  test of decision 41; version 6 added the new player
  data and the earlier-season market tests.
- Owner-supplied files checked and recorded in `docs/audit/external-odds.md`;
  `src/inventory/colab_stage_file.py` stages any large file for checking.
- Restore and full weekly rebuild checked on a fresh machine: every committed
  results file reproduced exactly; charts are now byte-stable.
- Reproducibility: `run.py` entry point; GitHub Actions runs the pipeline and
  all 15 leakage tests on synthetic data on every push (decision 31).
- Charts: `src/reporting/make_figures.py` -> `docs/figures/*.svg`.

## Next steps (agreed order)

1. Keep running the weekly live test; nothing in it changed.
2. Candidate for a future frozen rule, only if it keeps improving on new
   games: the shots model with MoneyPuck features (`shots_mp`). Track its
   2026-27 predictions beside the frozen model before deciding (new
   decision and a new frozen file with a new start date if adopted).
3. Optional, costs credits: more older-season shots lines
   (`fetch_puckapi_sog_history.py --batch 3 --block 25`) to compare
   `shots_mp` with the frozen model on more held-out lines.
4. Game model: the gap to the market is team rating (decision 39), mostly
   in a team's first 10 games. Opponent-adjusted ratings from past results
   did not help (decision 41; MoneyPuck situation rates were already in).
   Next: roster strength from the skater models (sum of dressed players'
   prior individual expected-goal and ice-time profiles, carried across
   team changes), then goalie quality; test against the three-way prices
   2021-22 to 2025-26 (`game_market_history.py <variant>`).
5. Totals: in-season scoring level or a bivariate score model.
6. Optional: improve the probable-starter model (team rotation habits, a
   tree model) or add confirmed-starter data if a free source exists.
