"""Fetch NHL Edge (puck and player tracking) season profiles for skaters.

Source: api-web.nhle.com/v1/edge/skater-detail/<player>/<season>/2 (free, no
key; the same endpoint as nhl-api-py's edge.skater_detail). Edge starts in
2021-22. The API returns season totals only (no game-by-game history), so a
season's values include games after any given date in that season. The
models therefore use only the PREVIOUS season's profile
(src/features/build_edge_features.py); the current season's numbers are
never joined to its own games.

Players: everyone in data/clean/skater_games.parquet with a regular-season
game in the season. One call per player-season; raw JSON cached in
data/edge/raw/<season>/<player>.json (gitignored), so reruns fetch only
what is missing.

Output: data/edge/edge_skater_season.parquet, one row per player-season:
  gp, top_shot_speed (mph), max_skating_speed (mph), bursts_over20,
  distance_total (miles), distance_max_game, shots, shots_high, shots_mid,
  shots_long, goals_high, oz_pct, oz_ev_pct, dz_pct

Usage: python3 src/ingest/fetch_edge.py [--workers 3]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data/edge")
URL = "https://api-web.nhle.com/v1/edge/skater-detail/{}/{}/2"
SEASONS = [20212022, 20222023, 20232024, 20242025, 20252026, 20262027]
session = requests.Session()


def fetch(pid: int, season: int) -> str:
    path = os.path.join(OUT, "raw", str(season), f"{pid}.json")
    if os.path.exists(path):
        return "cached"
    for attempt in range(5):
        try:
            r = session.get(URL.format(pid, season), timeout=30)
            if r.status_code == 200:
                json.dump(r.json(), open(path, "w"))
                return "ok"
            if r.status_code == 404:
                json.dump({}, open(path, "w"))  # no Edge profile for this player-season
                return "404"
            if r.status_code not in (429, 500, 502, 503, 504):
                return f"http {r.status_code}"
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 ** (attempt + 1))
    return "failed"


def parse(d: dict) -> dict:
    if not d:
        return {}
    g = lambda *ks: _get(d, ks)
    sog = {r["locationCode"]: r for r in d.get("sogSummary", [])}
    z = d.get("zoneTimeDetails") or {}
    return {"gp": g("player", "gamesPlayed"), "top_shot_speed": g("topShotSpeed", "imperial"),
            "max_skating_speed": g("skatingSpeed", "speedMax", "imperial"),
            "bursts_over20": g("skatingSpeed", "burstsOver20", "value"),
            "distance_total": g("totalDistanceSkated", "imperial"), "distance_max_game": g("distanceMaxGame", "imperial"),
            "shots": sog.get("all", {}).get("shots"), "shots_high": sog.get("high", {}).get("shots"),
            "shots_mid": sog.get("mid", {}).get("shots"), "shots_long": sog.get("long", {}).get("shots"),
            "goals_high": sog.get("high", {}).get("goals"),
            "oz_pct": z.get("offensiveZonePctg"), "oz_ev_pct": z.get("offensiveZoneEvPctg"), "dz_pct": z.get("defensiveZonePctg")}


def _get(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    g = pd.read_parquet(os.path.join(ROOT, "data/clean/skater_games.parquet"), columns=["player_id", "season", "game_type"])
    todo = g[(g.game_type == 2) & g.season.isin(SEASONS)].drop_duplicates(["player_id", "season"])
    for s in SEASONS:
        os.makedirs(os.path.join(OUT, "raw", str(s)), exist_ok=True)
    print(f"{len(todo)} player-seasons", flush=True)
    status = []
    with ThreadPoolExecutor(a.workers) as ex:
        for k, st in enumerate(ex.map(lambda r: fetch(*r), todo[["player_id", "season"]].itertuples(index=False)), 1):
            status.append(st)
            if k % 500 == 0:
                print(f"  {k}/{len(todo)}", pd.Series(status).value_counts().to_dict(), flush=True)
    print(pd.Series(status).value_counts().to_dict())
    rows = []
    for pid, s in todo[["player_id", "season"]].itertuples(index=False):
        path = os.path.join(OUT, "raw", str(s), f"{pid}.json")
        if os.path.exists(path):
            rows.append({"player_id": pid, "season": s, **parse(json.load(open(path)))})
    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(OUT, "edge_skater_season.parquet"), index=False)
    print(out.groupby("season")[["gp", "top_shot_speed", "oz_pct"]].count())


if __name__ == "__main__":
    main()
