"""Match SportsGameOdds prop lines to official NHL games and player box scores.

Inputs
  data/sgo/player_props_close.parquet                       (extract_sgo_props.py)
  v2 games_schedule.parquet                                 (official NHL schedule and results)
  skater summary tables: 2025-26 from v1 raw, which is complete

Output
  data/sgo/player_props_matched.parquet: prop lines with NHL game_id,
  NHL playerId and the player's official goals, assists, points and shots.

Matching
  Game:   same home and away team (provider codes mapped to NHL codes) and
          scheduled start within 12 hours.
  Player: same NHL game and the same name after removing accents,
          punctuation and case. Unmatched rows are kept with null ids and
          counted.
"""

from __future__ import annotations

import os
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAM = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL"}
SCHED = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/data/normalized/official_incremental/20261103T160656Z_4414a3fb/games_schedule.parquet")
SKATERS = [os.path.join(ROOT, "data/drive/NHL_Prediction_System/data/raw/nhl_api/skater_summary_20252026.parquet")]


def norm(name: str | None) -> str:
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


props = pd.read_parquet(os.path.join(ROOT, "data/sgo/player_props_close.parquet"))
props["home_nhl"] = props.home.map(lambda t: TEAM.get(t, t))
props["away_nhl"] = props.away.map(lambda t: TEAM.get(t, t))
props["ts"] = pd.to_datetime(props.starts_at, utc=True)

sched = pd.read_parquet(SCHED)
sched["ts_nhl"] = pd.to_datetime(sched.scheduled_utc, utc=True)
games = props[["event_id", "home_nhl", "away_nhl", "ts"]].drop_duplicates("event_id")
gm = games.merge(sched[["game_id", "home_abbrev", "away_abbrev", "ts_nhl", "game_type"]], left_on=["home_nhl", "away_nhl"], right_on=["home_abbrev", "away_abbrev"])
gm = gm[(gm.ts - gm.ts_nhl).abs() < pd.Timedelta(hours=12)]
gm = gm.drop_duplicates("event_id")
props = props.merge(gm[["event_id", "game_id", "game_type"]], on="event_id", how="left")

box = pd.concat([pd.read_parquet(p) for p in SKATERS if os.path.exists(p)])
box = box[["gameId", "playerId", "skaterFullName", "teamAbbrev", "goals", "assists", "points", "shots"]].rename(columns={"gameId": "game_id"})
box["name_key"] = box.skaterFullName.map(norm)
props["name_key"] = props.player_name.map(norm)
props = props.merge(box.drop(columns="skaterFullName"), on=["game_id", "name_key"], how="left")
props.to_parquet(os.path.join(ROOT, "data/sgo/player_props_matched.parquet"), index=False)

print("events matched to NHL games:", props.groupby("event_id").game_id.first().notna().mean().round(3), "of", props.event_id.nunique())
in_box = props.game_id.isin(box.game_id)
print("prop rows in games with box scores:", int(in_box.sum()), "player matched:", round(float(props[in_box].playerId.notna().mean()), 3))
