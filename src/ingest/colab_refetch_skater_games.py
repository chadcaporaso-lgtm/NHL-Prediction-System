"""Colab cell: re-fetch per-game skater data for 2021-22 to 2023-24.

Audit finding D-1: the earlier project's skater tables for these seasons hold
about 5 skaters per game instead of 36. This cell downloads them again from
the NHL stats API and checks completeness before saving.

How it avoids the earlier gap:
  * Queries one week at a time instead of a whole season.
  * Pages through results with a fixed sort (gameId, playerId). Paging
    without a fixed sort lets the API reorder rows between pages, which
    skips some rows and repeats others.
  * Keeps paging until a short page comes back, and retries failed requests.
  * Checks every game for 36 skater rows and no duplicate (gameId, playerId).

Output: NHL_Prediction_System_Claude/01_raw/nhl_api_refetch/
  skater_summary_<season>.parquet, skater_timeonice_<season>.parquet,
  fetch_report.json (row counts, games with unusual counts, SHA-256 of files)

Run in Google Colab: paste into one cell and run. Takes roughly 10-20 minutes.
It can be re-run: seasons already saved are skipped.
"""

import hashlib
import json
import os
import time
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd
import requests
from google.colab import drive

drive.mount("/content/drive")

DEST = "/content/drive/MyDrive/NHL_Prediction_System_Claude/01_raw/nhl_api_refetch"
os.makedirs(DEST, exist_ok=True)

BASE = "https://api.nhle.com/stats/rest/en/skater/{report}"
REPORTS = ["summary", "timeonice"]
SEASONS = {
    # season id: (first date, last date) covering regular season and playoffs
    "20212022": ("2021-10-12", "2022-06-30"),
    "20222023": ("2022-10-07", "2023-06-30"),
    "20232024": ("2023-10-10", "2024-06-30"),
}
GAME_TYPES = (2, 3)  # 2 = regular season, 3 = playoffs
PAGE = 100
SORT = quote(json.dumps([{"property": "gameId", "direction": "ASC"}, {"property": "playerId", "direction": "ASC"}]))
session = requests.Session()


def get_json(url, tries=5):
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            print("  HTTP", r.status_code, "retrying")
        except requests.RequestException as exc:
            print("  error", exc, "retrying")
        time.sleep(2 ** attempt)
    raise RuntimeError("failed after retries: " + url)


def fetch_window(report, season, start, end):
    """All rows for one report and date window, paged with a fixed sort."""
    types = " or ".join(f"gameTypeId={t}" for t in GAME_TYPES)
    cayenne = quote(f'seasonId={season} and ({types}) and gameDate>="{start}" and gameDate<="{end}"')
    rows, offset = [], 0
    while True:
        url = (
            f"{BASE.format(report=report)}?isAggregate=false&isGame=true"
            f"&sort={SORT}&start={offset}&limit={PAGE}&cayenneExp={cayenne}"
        )
        payload = get_json(url)
        batch = payload.get("data", [])
        rows.extend(batch)
        total = payload.get("total")
        offset += len(batch)
        if len(batch) < PAGE or (total is not None and offset >= total):
            if total is not None and len(rows) != total:
                print(f"  WARNING {report} {start}..{end}: got {len(rows)} rows, API total {total}")
            return rows
        time.sleep(0.15)


def fetch_season(report, season, first, last):
    d0, d1 = date.fromisoformat(first), date.fromisoformat(last)
    rows = []
    while d0 <= d1:
        end = min(d0 + timedelta(days=6), d1)
        rows.extend(fetch_window(report, season, d0.isoformat(), end.isoformat()))
        d0 = end + timedelta(days=1)
    return pd.DataFrame(rows)


report_log = {"fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": []}
for season, (first, last) in SEASONS.items():
    for report in REPORTS:
        path = f"{DEST}/skater_{report}_{season}.parquet"
        if os.path.exists(path):
            print("already saved, skipping:", path)
            df = pd.read_parquet(path)
        else:
            print(f"fetching {report} {season} ...")
            df = fetch_season(report, season, first, last)
            df.to_parquet(path, index=False)

        dupes = int(df.duplicated(["gameId", "playerId"]).sum())
        per_game = df.groupby("gameId").size()
        unusual = per_game[per_game != 36]
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        entry = {
            "file": os.path.basename(path),
            "rows": len(df),
            "games": int(per_game.size),
            "rows_per_game_mean": round(float(per_game.mean()), 2),
            "duplicate_game_player_rows": dupes,
            "games_not_36_rows": int(unusual.size),
            "games_not_36_rows_sample": {str(k): int(v) for k, v in unusual.head(20).items()},
            "sha256": sha,
        }
        report_log["files"].append(entry)
        print(
            f"  {entry['file']}: {entry['rows']:,} rows, {entry['games']} games, "
            f"{entry['rows_per_game_mean']} per game, {dupes} duplicates, "
            f"{entry['games_not_36_rows']} games without 36 rows"
        )

with open(f"{DEST}/fetch_report.json", "w") as fh:
    json.dump(report_log, fh, indent=2)
print("Done. Report written to", f"{DEST}/fetch_report.json")
