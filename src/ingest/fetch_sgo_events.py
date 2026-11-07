"""Download finished NHL events from SportsGameOdds, one day at a time.

The API key is injected by the environment's network proxy for
api.sportsgameodds.com; this script never reads or stores it.

Only events from 2026-01-16 on carry closing prices (openBookOdds /
closeBookOdds); earlier events hold only the last price seen, which was
captured during or after the game (data audit O-1), so they are not fetched.

Output: data/sgo/raw/<YYYY-MM-DD>.json.gz, one file per day (UTC noon to noon,
so an evening game stays on its own date). Existing files are skipped.

Usage: python src/ingest/fetch_sgo_events.py [first_date] [last_date]
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data/sgo/raw")
BASE = "https://api.sportsgameodds.com/v2/events"


def get(params: dict) -> dict:
    url = BASE + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(10 * (attempt + 1))
                continue
            raise


def fetch_day(day: date) -> list:
    nxt = day + timedelta(days=1)
    params = {
        "leagueID": "NHL",
        "startsAfter": f"{day.isoformat()}T12:00:00Z",
        "startsBefore": f"{nxt.isoformat()}T12:00:00Z",
        "finalized": "true",
        "limit": 50,
    }
    events, cursor = [], None
    while True:
        if cursor:
            params["cursor"] = cursor
        payload = get(params)
        events.extend(payload.get("data", []))
        cursor = payload.get("nextCursor")
        if not cursor:
            return events


def main(first: str, last: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    day, end = date.fromisoformat(first), date.fromisoformat(last)
    total = 0
    while day <= end:
        path = os.path.join(OUT, f"{day.isoformat()}.json.gz")
        if not os.path.exists(path):
            events = fetch_day(day)
            with gzip.open(path, "wt") as fh:
                json.dump({"date": day.isoformat(), "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "events": events}, fh)
            total += len(events)
            if events:
                print(day, len(events), "events")
            time.sleep(0.25)
        day += timedelta(days=1)
    print("done; events fetched this run:", total)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-01-16", sys.argv[2] if len(sys.argv) > 2 else (date.today() - timedelta(days=1)).isoformat())
