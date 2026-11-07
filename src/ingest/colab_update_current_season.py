"""Colab cell: refresh 2026-27 skater game data and stage it for transfer.

Re-fetches the in-progress season from its opener (2026-09-29) to yesterday,
checks 36 rows per game and no duplicates, saves to 01_raw/nhl_api_refetch/,
and copies each file into 01_raw/_transfer_3mb/current/ with a SHA-256
manifest. Rerun whenever the season data needs updating.
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

ROOT = "/content/drive/MyDrive/NHL_Prediction_System_Claude/01_raw"
DEST = f"{ROOT}/nhl_api_refetch"
STAGE = f"{ROOT}/_transfer_3mb/current"
os.makedirs(STAGE, exist_ok=True)
SEASON, FIRST, LAST = "20262027", "2026-09-25", (date.today() - timedelta(days=1)).isoformat()
BASE = "https://api.nhle.com/stats/rest/en/skater/{report}"
SORT = quote(json.dumps([{"property": "gameId", "direction": "ASC"}, {"property": "playerId", "direction": "ASC"}]))
PAGE = 100
session = requests.Session()
session.headers["User-Agent"] = "nhl-prediction-research/1.0"


def get(url, tries=5):
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            print("  HTTP", r.status_code, "retrying")
        except requests.RequestException as exc:
            print("  error", exc, "retrying")
        time.sleep(2 ** attempt)
    raise RuntimeError("failed after retries: " + url)


def fetch(rep):
    rows, d0, d1 = [], date.fromisoformat(FIRST), date.fromisoformat(LAST)
    while d0 <= d1:
        end = min(d0 + timedelta(days=6), d1)
        cay = quote(f'seasonId={SEASON} and (gameTypeId=2 or gameTypeId=3) and gameDate>="{d0}" and gameDate<="{end}"')
        offset = 0
        while True:
            p = get(f"{BASE.format(report=rep)}?isAggregate=false&isGame=true&sort={SORT}"
                    f"&start={offset}&limit={PAGE}&cayenneExp={cay}")
            batch = p.get("data", [])
            rows.extend(batch)
            offset += len(batch)
            if len(batch) < PAGE:
                break
            time.sleep(0.15)
        d0 = end + timedelta(days=1)
    return pd.DataFrame(rows)


manifest = []
for rep in ("summary", "timeonice"):
    df = fetch(rep)
    path = f"{DEST}/skater_{rep}_{SEASON}.parquet"
    df.to_parquet(path, index=False)
    data = open(path, "rb").read()
    per_game = df.groupby("gameId").size()
    stage_name = f"skater_{rep}_{SEASON}.parquet.part000"
    open(f"{STAGE}/{stage_name}", "wb").write(data)
    entry = {"file": os.path.basename(path), "rows": len(df), "games": int(per_game.size),
             "first_game_date": str(df.gameDate.min()), "last_game_date": str(df.gameDate.max()),
             "games_not_36_rows": int((per_game != 36).sum()),
             "duplicates": int(df.duplicated(["gameId", "playerId"]).sum()),
             "sha256": hashlib.sha256(data).hexdigest(), "parts": [stage_name]}
    manifest.append(entry)
    print(json.dumps({k: v for k, v in entry.items() if k != "parts"}))

json.dump(manifest, open(f"{STAGE}/transfer_manifest_current.json", "w"), indent=2)
print("Done.")
