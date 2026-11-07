# Phase 1a: Inventory of the Earlier Projects

Date: 2026-11-07. Script: [`src/inventory/build_inventory.py`](../src/inventory/build_inventory.py).
The full file list ([`inventory/inventory.csv`](inventory/inventory.csv), one row per file with
Drive id, path, size and timestamps), a summary by project and file type, and
the duplicate candidates are in [`docs/inventory/`](inventory/).

## Method

1. Listed every Drive item created on or after 2026-10-13 through the Drive
   search API (15 pages of files, 3 pages of folders).
2. Resolved each item's full path by following parent folders up to one of the
   two project roots.
3. Verified completeness by listing a sample of folders directly. 19 of 20
   items matched; the one miss was an empty `tests` folder. Four top-level
   folders of the older project that the paged search had skipped were added
   by direct lookup.
4. Grouped possible duplicates by (file name, byte size). The Drive API does
   not return content checksums, so exact SHA-256 checksums are computed when
   a file is downloaded for the audit (`data/drive/checksums.csv`, produced by
   [`src/inventory/decode_download.py`](../src/inventory/decode_download.py)).

Items outside both projects (an Android backup folder and one Colab notebook)
were excluded.

## Summary

| Project | Folders | Files | Size |
|---------|--------:|------:|-----:|
| `NHL_Prediction_System` (v1, started 2026-10-14) | 21 | 143 | 400 MB |
| `NHL_Prediction_System_ChatGPT` (v2, started 2026-10-18) | 2,294 | 10,819 | 1,406 MB |

The two projects barely overlap: 142 of the 143 v1 files have no file with
the same name in v2.

### v1: `NHL_Prediction_System` (the production system)

| Area | Contents |
|------|----------|
| `data/raw/nhl_api/` | NHL API schedules (2019-20 to 2026-27), skater summary, realtime and time-on-ice tables, team summary and realtime tables |
| `data/raw/sgo_events/` | SportsGameOdds event and odds responses, Feb 2024 to Oct 2026 (55 files, two-week windows) |
| `data/raw/moneypuck/` | MoneyPuck team data, 2018 onward |
| `data/normalized/` | `games`, `odds`, `player_games`, `team_games_stats`, `edge_observations` |
| `data/features/` | `game_features_T24h`, `game_features_T60m`, `player_features_T60m` (as-of 24 hours and 60 minutes before puck drop) |
| `data/labels/` | `game_labels` |
| `models/production/` | Game market model, player prop model (shots on goal and points), blocked shots model |
| `models/candidates/` | Stage 6 game models, phase 2 player models |
| `predictions/` | Daily predictions and gameday slate locks, 2026-10-14 to 2026-11-04 |
| `reports/` | Out-of-fold predictions (stages 5 and 6), verified pregame ledger (stage 8), graded ledger 2026-11-03 |
| `manifests/` | Production manifest and SportsGameOdds ingestion manifests |
| `src/` | `load_production_system.py` only |

**Finding I-1: no training code in v1.** The only source file is a loader.
The `notebooks/` folder is empty. The code that trained the production models
is not in the project, so the model audit must work from the saved model
bundles, out-of-fold predictions and reports.

**Finding I-2: duplicate raw files.** Seven pairs of `<table>` and
`<table>_full` files in `data/raw/nhl_api/` have identical sizes and are
probably identical. To be confirmed by checksum.

**Finding I-3: possible truncated skater tables.** Skater summary files for
2021-22 to 2023-24 are about 150 KB, while 2024-25 onward are about 625 KB.
Skater time-on-ice files show the same 4x gap. This may mean the earlier
seasons were fetched with a row limit. To be confirmed by row counts.

### v2: `NHL_Prediction_System_ChatGPT` (the research pipeline)

| Area | Files | Size | Contents |
|------|------:|-----:|----------|
| `data/raw/` | 7,851 | 487 MB | NHL API responses (5,363 files), OddsHarvester pilot, MoneyPuck, historical and dataset audits, NHL EDGE (859), SportsGameOdds, a user-supplied closing-odds CSV |
| `data/legacy_imports/` | 102 | 330 MB | Content-addressed copies of earlier data |
| `data/normalized/` | 51 | 210 MB | Qualified odds (quote and paired-market observations, closing benchmarks), reconciled MoneyPuck team games |
| `data/features/` | 104 | 182 MB | As-of research features, five dated versions |
| `models/research/` | 1,285 | 19 MB | 361 trained models (joblib), 340 prediction tables, 584 run records |
| `reports/` | about 600 | 50 MB | About 60 numbered stage reports |
| `src/` | 153 Python files | | Stage scripts and updaters |
| `notebooks/` | 19 | 98 MB | Colab notebooks and packaged outputs |
| `vendor/` | 130 | 1 MB | Third-party libraries (nhl-api-py, OddsHarvester) |

v2 records provenance carefully: runs are stored in timestamped folders with
run status, input fingerprints and artifact hashes.

## Download limit

The Drive connection used for this audit downloads files up to 10 MB. 39
files are larger, totalling 975 MB of the 1.8 GB. Most are duplicates or can
be rebuilt:

- Five dated versions of the same 25 MB research feature table (v2).
- A 125 MB browser capture (`.har`) from the OddsHarvester pilot.
- A 127 MB MoneyPuck CSV, available again from the source.
- 11 SportsGameOdds raw responses (v1), already parsed into
  `sgo_game_odds.parquet` (1.6 MB).

The files that matter and are over the limit are the v1 feature tables
(`game_features_T24h`, `game_features_T60m`, `player_features_T60m`), the v1
MoneyPuck table, and the v2 qualified odds tables. How these are transferred
is recorded in the decision log.
