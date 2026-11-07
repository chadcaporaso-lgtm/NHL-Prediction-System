"""Build the clean skater game table, 2019-20 to the present.

One row per (game_id, player_id): goals, assists, points, shots, ice time
(total, even strength, power play, shorthanded), shifts, position, team,
opponent and home/road. Regular season and playoffs.

Sources (all audited, see docs/audit/data-audit.md)
  2019-20, 2020-21, 2021-22 to 2023-24, 2026-27: Claude re-fetch (01_raw/nhl_api_refetch)
  2024-25, 2025-26: v1 raw NHL API tables (complete; D-1 affected only 2021-24)

Checks: no duplicate (game, player); every game in the official schedule;
every completed schedule game present; rows per game reported.
"""

from __future__ import annotations

import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEW = os.path.join(ROOT, "data/drive/NHL_Prediction_System_Claude/01_raw/nhl_api_refetch")
V1 = os.path.join(ROOT, "data/drive/NHL_Prediction_System/data/raw/nhl_api")
SCHED = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/data/normalized/official_incremental/20261103T160656Z_4414a3fb/games_schedule.parquet")
SOURCES = {s: NEW for s in ["20192020", "20202021", "20212022", "20222023", "20232024", "20262027"]}
SOURCES.update({"20242025": V1, "20252026": V1})

SUMMARY_COLS = ["gameId", "playerId", "skaterFullName", "positionCode", "teamAbbrev", "opponentTeamAbbrev",
                "homeRoad", "gameDate", "goals", "assists", "points", "shots", "ppPoints", "ppGoals"]
TOI_COLS = ["gameId", "playerId", "timeOnIce", "evTimeOnIce", "ppTimeOnIce", "shTimeOnIce", "shifts"]

frames, report = [], {}
for season, base in SOURCES.items():
    s = pd.read_parquet(f"{base}/skater_summary_{season}.parquet")[SUMMARY_COLS]
    t = pd.read_parquet(f"{base}/skater_timeonice_{season}.parquet")[TOI_COLS]
    d = s.merge(t, on=["gameId", "playerId"], how="left", validate="one_to_one")
    d["season"] = int(season)
    report[season] = {"rows": len(d), "games": int(d.gameId.nunique()),
                      "rows_without_toi": int(d.timeOnIce.isna().sum())}
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
df = df.rename(columns={"gameId": "game_id", "playerId": "player_id", "skaterFullName": "player_name",
                        "positionCode": "position", "teamAbbrev": "team", "opponentTeamAbbrev": "opponent",
                        "gameDate": "game_date", "timeOnIce": "toi", "evTimeOnIce": "toi_ev",
                        "ppTimeOnIce": "toi_pp", "shTimeOnIce": "toi_sh", "ppPoints": "pp_points", "ppGoals": "pp_goals"})
df["is_home"] = (df.pop("homeRoad") == "H").astype(int)
df["game_type"] = df.game_id // 10000 % 100
for c in ["toi", "toi_ev", "toi_pp", "toi_sh"]:
    df[c] = df[c] / 60.0  # seconds to minutes

sched = pd.read_parquet(SCHED)
done = sched[sched.home_final_score.notna() & sched.game_type.isin([2, 3])]
checks = {
    "rows": len(df), "games": int(df.game_id.nunique()),
    "duplicate_game_player": int(df.duplicated(["game_id", "player_id"]).sum()),
    "games_not_in_schedule": int((~df.game_id.drop_duplicates().isin(sched.game_id)).sum()),
    "completed_schedule_games_missing": int((~done.game_id.isin(df.game_id)).sum()),
    "games_not_36_rows": int((df.groupby("game_id").size() != 36).sum()),
    "by_season": report,
}
assert checks["duplicate_game_player"] == 0
assert checks["games_not_in_schedule"] == 0
os.makedirs(os.path.join(ROOT, "data/clean"), exist_ok=True)
df.sort_values(["game_date", "game_id", "player_id"]).to_parquet(os.path.join(ROOT, "data/clean/skater_games.parquet"), index=False)
json.dump(checks, open(os.path.join(ROOT, "docs/audit/skater_games_summary.json"), "w"), indent=2)
print(json.dumps(checks, indent=2))
