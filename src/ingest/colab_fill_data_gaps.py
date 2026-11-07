"""Colab cell: fill data gaps G-A, G-B and G-C (docs/data-coverage.md).

  G-B  Re-fetch 2026-27 skater game data (about 24 rows per game, should be 36).
  G-C  Fetch 2019-20 and 2020-21 skater game data (two more training seasons).
  G-A  Refresh the MoneyPuck team game-by-game file (2026-27 had 89 of 235 games).

Skater data uses the same method as colab_refetch_skater_games.py: weekly
windows, fixed sort (gameId, playerId), paging to a short page, retries, and
a check for 36 rows per game with no duplicates.

MoneyPuck: downloads all_teams.csv, keeps seasons 2018 onward (the same
scope as the earlier file), and compares it with the earlier file on the
games both contain, so any change in MoneyPuck's history shows up.
MoneyPuck data is free for non-commercial use with credit to MoneyPuck.com.

Output:
  NHL_Prediction_System_Claude/01_raw/nhl_api_refetch/
      skater_{summary,timeonice}_{20192020,20202021,20262027}.parquet
  NHL_Prediction_System_Claude/01_raw/moneypuck/
      moneypuck_all_teams_2018_plus_<date>.parquet
  NHL_Prediction_System_Claude/01_raw/fill_gaps_report_<date>.json

Run in Google Colab: paste into one cell and run. About 15-25 minutes.
Re-running skips completed seasons; 2026-27 is always fetched again because
the season is in progress.
"""

import hashlib
import io
import json
import os
import time
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd
import requests
from google.colab import drive

drive.mount("/content/drive")

ROOT = "/content/drive/MyDrive/NHL_Prediction_System_Claude/01_raw"
SKATER_DEST = f"{ROOT}/nhl_api_refetch"
MP_DEST = f"{ROOT}/moneypuck"
OLD_MP = "/content/drive/MyDrive/NHL_Prediction_System/data/raw/moneypuck/moneypuck_all_teams_2018_plus.parquet"
TODAY = date.today().isoformat()
YESTERDAY = (date.today() - timedelta(days=1)).isoformat()
os.makedirs(SKATER_DEST, exist_ok=True)
os.makedirs(MP_DEST, exist_ok=True)

SEASONS = {
    # season id: (first date, last date, always re-fetch)
    "20192020": ("2019-10-01", "2020-09-30", False),  # includes the 2020 bubble playoffs
    "20202021": ("2021-01-12", "2021-07-10", False),
    "20262027": ("2026-09-25", YESTERDAY, True),  # season opened 2026-09-29
}
REPORTS = ["summary", "timeonice"]
GAME_TYPES = (2, 3)
PAGE = 100
BASE = "https://api.nhle.com/stats/rest/en/skater/{report}"
SORT = quote(json.dumps([{"property": "gameId", "direction": "ASC"}, {"property": "playerId", "direction": "ASC"}]))
session = requests.Session()
session.headers["User-Agent"] = "nhl-prediction-research/1.0"
report = {"run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "skater_files": [], "moneypuck": {}}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def get(url, tries=5):
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=120)
            if r.status_code == 200:
                return r
            print("  HTTP", r.status_code, "retrying")
        except requests.RequestException as exc:
            print("  error", exc, "retrying")
        time.sleep(2 ** attempt)
    raise RuntimeError("failed after retries: " + url)


def fetch_window(rep, season, start, end):
    types = " or ".join(f"gameTypeId={t}" for t in GAME_TYPES)
    cayenne = quote(f'seasonId={season} and ({types}) and gameDate>="{start}" and gameDate<="{end}"')
    rows, offset = [], 0
    while True:
        url = (f"{BASE.format(report=rep)}?isAggregate=false&isGame=true"
               f"&sort={SORT}&start={offset}&limit={PAGE}&cayenneExp={cayenne}")
        payload = get(url).json()
        batch = payload.get("data", [])
        rows.extend(batch)
        total = payload.get("total")
        offset += len(batch)
        if len(batch) < PAGE or (total is not None and offset >= total):
            if total is not None and len(rows) != total:
                print(f"  WARNING {rep} {start}..{end}: got {len(rows)} rows, API total {total}")
            return rows
        time.sleep(0.15)


def fetch_season(rep, season, first, last):
    d0, d1 = date.fromisoformat(first), date.fromisoformat(last)
    rows = []
    while d0 <= d1:
        end = min(d0 + timedelta(days=6), d1)
        rows.extend(fetch_window(rep, season, d0.isoformat(), end.isoformat()))
        d0 = end + timedelta(days=1)
    return pd.DataFrame(rows)


# ---- G-B and G-C: skater game data ----
for season, (first, last, refresh) in SEASONS.items():
    for rep in REPORTS:
        path = f"{SKATER_DEST}/skater_{rep}_{season}.parquet"
        if os.path.exists(path) and not refresh:
            print("already saved, skipping:", path)
            df = pd.read_parquet(path)
        else:
            print(f"fetching {rep} {season} ({first} to {last}) ...")
            df = fetch_season(rep, season, first, last)
            df.to_parquet(path, index=False)
        per_game = df.groupby("gameId").size()
        unusual = per_game[per_game != 36]
        entry = {
            "file": os.path.basename(path), "rows": len(df), "games": int(per_game.size),
            "first_game_date": str(df.gameDate.min()), "last_game_date": str(df.gameDate.max()),
            "rows_per_game_mean": round(float(per_game.mean()), 2),
            "duplicate_game_player_rows": int(df.duplicated(["gameId", "playerId"]).sum()),
            "games_not_36_rows": int(unusual.size),
            "games_not_36_rows_sample": {str(k): int(v) for k, v in unusual.head(20).items()},
            "sha256": sha256(path),
        }
        report["skater_files"].append(entry)
        print(f"  {entry['file']}: {entry['rows']:,} rows, {entry['games']} games "
              f"({entry['first_game_date']} to {entry['last_game_date']}), {entry['rows_per_game_mean']} per game, "
              f"{entry['duplicate_game_player_rows']} duplicates, {entry['games_not_36_rows']} games without 36 rows")

# ---- G-A: MoneyPuck team game-by-game ----
print("downloading MoneyPuck all_teams.csv (large, a few minutes) ...")
mp_url = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv"
mp = pd.read_csv(io.BytesIO(get(mp_url).content), low_memory=False)
mp = mp[mp.season >= 2018].reset_index(drop=True)
mp_path = f"{MP_DEST}/moneypuck_all_teams_2018_plus_{TODAY}.parquet"
mp.to_parquet(mp_path, index=False)
games_by_season = mp.groupby("season").gameId.nunique()
mp_info = {
    "source_url": mp_url, "file": os.path.basename(mp_path), "rows": len(mp), "sha256": sha256(mp_path),
    "last_game_date": str(mp.gameDate.max()),
    "games_by_season": {str(k): int(v) for k, v in games_by_season.items()},
    "duplicate_team_game_situation_rows": int(mp.duplicated(["team", "gameId", "situation"]).sum()),
}

# Compare with the earlier file on the rows both contain.
if os.path.exists(OLD_MP):
    old = pd.read_parquet(OLD_MP)
    key = ["team", "gameId", "situation"]
    cols = [c for c in ["xGoalsFor", "xGoalsAgainst", "shotsOnGoalFor", "goalsFor", "corsiPercentage", "iceTime"]
            if c in old.columns and c in mp.columns]
    both = old[key + cols].merge(mp[key + cols], on=key, suffixes=("_old", "_new"))
    changed = {c: int((both[f"{c}_old"].round(4) != both[f"{c}_new"].round(4)).sum()) for c in cols}
    mp_info["compare_with_earlier_file"] = {
        "earlier_rows": len(old), "rows_in_both": len(both),
        "earlier_rows_missing_now": int(len(old) - len(both)), "values_changed_by_column": changed,
    }
report["moneypuck"] = mp_info
print(json.dumps(mp_info, indent=2))

report_path = f"{ROOT}/fill_gaps_report_{TODAY}.json"
with open(report_path, "w") as fh:
    json.dump(report, fh, indent=2)
print("Done. Report written to", report_path)
