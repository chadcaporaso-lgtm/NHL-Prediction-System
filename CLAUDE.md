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
4. Re-download SportsGameOdds history (uses the stored network secret):
   `python3 src/ingest/fetch_sgo_events.py 2026-01-16 <yesterday>`.
5. Rebuild everything: `python3 run.py update` (runs
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

## Current state (2026-11-08, third session)

- Data audit done (findings D-1, D-2, O-1 to O-7, goalie and live-feed checks).
- Models: shots on goal (no demonstrated edge: +8.3% on 907 held-out 2026-27
  bets, SE 2.7%, but 2,260 older-season PuckAPI lines in two independent
  batches do not repeat it: frozen rule -7.7% on 349 bets, SE 5.0, model
  blend weight 0.19, SE 0.20; decisions 32-33; live test decides), puck line (level
  with the close, not significant), moneyline (close, behind), points v2
  (no edge), totals (no skill), team Elo (power rating and optional feature),
  probable starting goalie (71.0% top-pick accuracy out of sample; the
  actual-starter look-ahead moves game-model log loss by at most 0.002).
- Closing-line value (shots): the market moves toward the model between open
  and close (slope 0.21, SE 0.007), but bets at the open lose to the fair
  close on price (CLV -2.1 points); opening-price profit comes mainly from
  lines that later moved. Not adopted (decision 30).
- Live test: rules version 2 frozen on 2026-11-07 (puck line on the
  probable-starter model, games after 2026-11-07); no games graded yet. The
  newest data (refresh 2026-11-07) ends on 2026-11-05 for team and goalie
  lines and 2026-11-06 for skaters.
- Repository is public. `main` was rewritten to remove purchased data; the
  old branch `claude/peaceful-volta-mygk7c` and pull request #1 still hold
  it until the owner deletes the branch and GitHub purges the PR refs.
- Restore and full weekly rebuild checked on a fresh machine: every committed
  results file reproduced exactly; charts are now byte-stable.
- Reproducibility: `run.py` entry point; GitHub Actions runs the pipeline and
  all 13 leakage tests on synthetic data on every push (decision 31).
- Charts: `src/reporting/make_figures.py` -> `docs/figures/*.svg`.

## Next steps (agreed order)

1. Linemates from NHL shift charts (free; Colab) for the points model.
2. Optional, costs credits: more older-season shots lines would tighten the
   decision-33 comparison (two batches, 2,260 lines; 10 credits per 25 games;
   `fetch_puckapi_sog_history.py --batch 3 --block 25` draws new players). Also possible: a per-book
   timestamped check of opening shots lines (`get_prop_movement`).
3. Totals: in-season scoring level or a bivariate score model.
4. Optional: improve the probable-starter model (team rotation habits, a
   tree model) or add confirmed-starter data if a free source exists.
