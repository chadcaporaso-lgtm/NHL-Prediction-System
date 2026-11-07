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
- `data/` is gitignored. Purchased odds in `data_snapshots/` must be removed
  from the repo and its history before it is made public.
- The owner runs Colab cells when asked: give the full cell in a code block.
- A second repo, `claude-sports-analytics`, is public and has its own
  CLAUDE.md rules (no AI attribution there).

## Frozen live test: do not change

`docs/live/frozen_rules.json` (decisions 20, 23, 25) fixes the shots-on-goal
and puck-line betting rules for games after 2026-11-06. `src/live/track.py`
checks model fingerprints and stops if the default shots model
(`prop_model.py shots`) or the default game model (`game_model.py`) changes.
Add new features as variants or separate outputs, never by editing those
defaults (see the `elo` variant). If the rules must change, write a new
frozen file with a new start date and a decision-log entry.

## Restoring the data in a new session

A cloud session starts on a fresh machine with only this repository.

1. `pip install -r requirements.txt`.
2. Ask the owner to run `src/inventory/colab_bundle_core_inputs.py` in Colab.
   It stages the 19 audited inputs (about 28 MB) and the newest weekly
   refresh as 3 MB parts plus a manifest under
   `01_raw/_transfer_3mb/core_<date>/`.
3. Download the manifest and every part with the Google Drive connector
   (search by title `core__`), save the manifest locally, then run
   `python3 src/inventory/restore_core_inputs.py <manifest.json>`. It
   verifies every SHA-256.
4. Re-download SportsGameOdds history (uses the stored network secret):
   `python3 src/ingest/fetch_sgo_events.py 2026-01-16 <yesterday>`.
5. Rebuild everything: run the steps in `src/live/weekly_update.py` (it also
   fetches new odds, rebuilds features and models, runs the leakage tests,
   grades the live test and redraws the charts).

Purchased PuckAPI prop history is committed in `data_snapshots/`.

## Weekly routine (live test)

Owner runs `src/ingest/colab_weekly_refresh.py`; download the parts from
`01_raw/_transfer_3mb/live_<date>/` into `data/live/<date>/`, verify against
the manifest, then `python3 src/live/weekly_update.py <date>`. Commit
`docs/live/tracking.json` and the redrawn `docs/figures/`, and add the week
to the README. Treat the result as meaningful after about 500 bets.

## Current state (2026-11-07)

- Data audit done (findings D-1, D-2, O-1 to O-7, goalie and live-feed checks).
- Models: shots on goal (adds information to the closing price; blend +8.3%
  ROI on 907 held-out bets, SE 2.7%, not yet confirmed), puck line (level
  with the close, not significant), moneyline (close, behind), points v2
  (no edge), totals (no skill), team Elo (power rating and optional feature).
- Live test frozen on 2026-11-07; no games graded yet.
- Charts: `src/reporting/make_figures.py` -> `docs/figures/*.svg`.

## Next steps (agreed order)

1. Starting goalie: the game models use the actual starter, a small
   look-ahead for live bets. Build a probable-starter model (rest, recent
   starts, back-to-backs), weight goalie features by it, and re-freeze the
   puck-line rule before its first graded game.
2. Closing-line value: test whether the shots model's picks at the opening
   price beat the close (SportsGameOdds has opening prices from 2026-01-16).
3. Reproducibility: tests in GitHub Actions with small fixtures, one command
   to run the pipeline, update the git remote to the repo's new location
   (`chadcaporaso-lgtm/NHL-Prediction-System`).
4. Linemates from NHL shift charts (free; Colab) for the points model.
5. Optional, costs credits: check PuckAPI archive odds (about 30 games a
   season) before buying older seasons.
6. Totals: in-season scoring level or a bivariate score model.
