# Data snapshots

Data that was paid for and cannot be re-downloaded for free. Kept here because
the working machine is temporary. This repository is private; remove this
folder (and purge it from history) before making the repository public, as
provider terms may not allow redistribution.

- `puckapi_prop_history_2026-11-07.csv`: 745 player-games of closing prop prices (PuckAPI, 300 credits). Built from the raw responses in `puckapi_prop_history_raw/`.
- `sgo_player_props_matched_2026-11-07.parquet`: about 120,000 SportsGameOdds player-prop closing lines (2026-01-16 to 2026-11-06, 920 games), matched to NHL game and player ids. Built by `src/ingest/fetch_sgo_events.py`, `extract_sgo_props.py` and `match_sgo_props.py` from a Pro-tier subscription.
