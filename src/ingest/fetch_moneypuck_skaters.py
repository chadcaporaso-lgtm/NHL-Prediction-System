"""Fetch MoneyPuck game-by-game skater data (expected goals, shot danger, zone starts).

Data: MoneyPuck.com (https://moneypuck.com/data.htm), credited in the README.
Licensed data: kept in data/moneypuck/ (gitignored), never committed.

One career file per player and game type:
  https://moneypuck.com/moneypuck/playerData/careers/gameByGame/{regular|playoffs}/skaters/<id>.csv
Rows: one per game and situation (all, 5on5, 5on4, 4on5, other). Kept: seasons
2018-19 on, situations all / 5on5 / 5on4, and the columns below. Players: every
skater in data/clean/skater_games.parquet. Files are cached per player
(data/moneypuck/skaters/raw/<type>/<id>.parquet); --refresh refetches players
who played in the newest season (their career file grows each game).

Output: data/moneypuck/skaters/mp_skater_games.parquet

Usage: python3 src/ingest/fetch_moneypuck_skaters.py [--workers 2] [--refresh]
"""

from __future__ import annotations

import argparse
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data/moneypuck/skaters")
URL = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/{}/skaters/{}.csv"
KEEP = ["playerId", "season", "gameId", "gameDate", "playerTeam", "position", "situation", "icetime", "iceTimeRank",
        "I_F_xGoals", "I_F_xOnGoal", "I_F_shotsOnGoal", "I_F_shotAttempts", "I_F_unblockedShotAttempts", "I_F_goals",
        "I_F_primaryAssists", "I_F_secondaryAssists", "I_F_lowDangerShots", "I_F_mediumDangerShots", "I_F_highDangerShots",
        "I_F_highDangerxGoals", "I_F_oZoneShiftStarts", "I_F_dZoneShiftStarts", "I_F_neutralZoneShiftStarts",
        "I_F_flyShiftStarts", "OnIce_F_xGoals", "OnIce_A_xGoals", "OnIce_F_goals", "OnIce_F_shotAttempts",
        "OnIce_A_shotAttempts", "OnIce_F_highDangerShots"]
session = requests.Session()
session.headers["User-Agent"] = "nhl-prediction-system (research; caches every file)"


def fetch(kind: str, pid: int, refresh: set) -> str:
    path = os.path.join(OUT, "raw", kind, f"{pid}.parquet")
    if os.path.exists(path) and pid not in refresh:
        return "cached"
    for attempt in range(5):
        try:
            r = session.get(URL.format(kind, pid), timeout=60)
            if r.status_code == 200:
                d = pd.read_csv(io.StringIO(r.text))
                d = d[(d.season >= 2018) & d.situation.isin(["all", "5on5", "5on4"])]
                d[[c for c in KEEP if c in d.columns]].to_parquet(path, index=False)
                time.sleep(0.5)
                return "ok"
            if r.status_code == 404:
                pd.DataFrame(columns=KEEP).to_parquet(path, index=False)  # no file (e.g. no playoff games)
                return "404"
            if r.status_code not in (429, 500, 502, 503, 504):
                return f"http {r.status_code}"
        except (requests.RequestException, ValueError, pd.errors.ParserError):
            pass
        time.sleep(2 ** (attempt + 1))
    return "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    g = pd.read_parquet(os.path.join(ROOT, "data/clean/skater_games.parquet"), columns=["player_id", "season", "game_type"])
    reg = sorted(g[g.game_type == 2].player_id.unique())
    po = sorted(g[g.game_type == 3].player_id.unique())
    refresh = set(g[g.season == g.season.max()].player_id) if a.refresh else set()
    jobs = [("regular", p) for p in reg] + [("playoffs", p) for p in po]
    for k in ("regular", "playoffs"):
        os.makedirs(os.path.join(OUT, "raw", k), exist_ok=True)
    print(f"{len(jobs)} files ({len(reg)} regular, {len(po)} playoffs)", flush=True)
    status = []
    with ThreadPoolExecutor(a.workers) as ex:
        for k, st in enumerate(ex.map(lambda j: fetch(*j, refresh), jobs), 1):
            status.append(st)
            if k % 250 == 0:
                print(f"  {k}/{len(jobs)}", pd.Series(status).value_counts().to_dict(), flush=True)
    print(pd.Series(status).value_counts().to_dict())
    parts = [pd.read_parquet(os.path.join(OUT, "raw", k, f"{p}.parquet")) for k, p in jobs
             if os.path.exists(os.path.join(OUT, "raw", k, f"{p}.parquet"))]
    d = pd.concat([x for x in parts if len(x)], ignore_index=True).drop_duplicates(["playerId", "gameId", "situation"])
    d.to_parquet(os.path.join(OUT, "mp_skater_games.parquet"), index=False)
    print(len(d), "rows;", d.gameId.nunique(), "games;", d.playerId.nunique(), "players")


if __name__ == "__main__":
    main()
