"""Single source for official schedule, team-game results and goalie lines.

Audited v2 tables (run 20261103T160656Z_4414a3fb, through 2026-11-02) are the
base. Games that are not in them come from the newest weekly NHL web API
refresh in data/live/<date>/ (src/ingest/colab_weekly_refresh.py), which was
checked against v2 on every overlapping game (docs/audit/live_feed_check_*.json).
Skater tables and MoneyPuck also prefer the newest refresh when present.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/data/normalized/official_incremental/20261103T160656Z_4414a3fb")
MONEYPUCK_BASE = os.path.join(ROOT, "data/drive/NHL_Prediction_System_Claude/01_raw/moneypuck/moneypuck_all_teams_2018_plus_2026-11-07.parquet")


def latest_live() -> str | None:
    dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "data/live/*")) if os.path.isdir(d))
    return dirs[-1] if dirs else None


def _union(v2_name: str, live_name: str) -> pd.DataFrame:
    base = pd.read_parquet(os.path.join(V2, v2_name))
    live = latest_live()
    if live and os.path.exists(os.path.join(live, live_name)):
        new = pd.read_parquet(os.path.join(live, live_name))
        new = new[~new.game_id.isin(base.game_id)]
        base = pd.concat([base, new], ignore_index=True)
    return base


def load_schedule() -> pd.DataFrame:
    s = _union("games_schedule.parquet", "games_schedule_live.parquet")
    # A game scheduled in v2 but completed later appears in both: take the completed row.
    live = latest_live()
    if live:
        new = pd.read_parquet(os.path.join(live, "games_schedule_live.parquet"))
        done = new[new.home_final_score.notna()]
        stale = s.game_id.isin(done.game_id) & s.home_final_score.isna()
        s = pd.concat([s[~stale], done[done.game_id.isin(s.loc[stale, "game_id"])]], ignore_index=True)
    return s.drop_duplicates("game_id", keep="last")


def load_team_games() -> pd.DataFrame:
    t = _union("official_team_games.parquet", "team_games_live.parquet")
    t["is_home"] = t.is_home.astype(int)
    t["won_final"] = t.won_final.astype(bool)
    return t


def load_goalies() -> pd.DataFrame:
    g = _union("goalie_history_candidates.parquet", "goalie_games_live.parquet")
    g["starter"] = g.starter.astype(bool)
    g["participated"] = g.participated.astype(bool)
    return g


def team_abbrev() -> dict:
    s = load_schedule()
    return dict(zip(s.home_team_id, s.home_abbrev)) | dict(zip(s.away_team_id, s.away_abbrev))


def skater_file(report: str, season: str, default: str) -> str:
    live = latest_live()
    p = os.path.join(live, f"skater_{report}_{season}.parquet") if live else None
    return p if p and os.path.exists(p) else default


def moneypuck_file() -> str:
    live = latest_live()
    p = os.path.join(live, "moneypuck_all_teams_2018_plus.parquet") if live else None
    return p if p and os.path.exists(p) else MONEYPUCK_BASE
