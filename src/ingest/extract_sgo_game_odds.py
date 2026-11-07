"""Extract pregame opening and closing game lines from SportsGameOdds event files.

Input:  data/sgo/raw/*.json.gz (fetch_sgo_events.py)
Output: data/sgo/game_odds_close.parquet, one row per event with
        moneyline (full game, incl. OT/SO), puck line (full game) and total
        goals (full game): opening and closing consensus prices and lines.

Only the provider's consensus open*/close* fields are used. Per-book `odds`
and the top-level `bookOdds` are last-seen prices that are often in-game
(for example -50000 after the result was known), the same contamination
found in the v1 odds (finding O-1), so they are never read.

Odd IDs: points-home-game-ml-home, points-away-game-ml-away,
         points-home-game-sp-home, points-away-game-sp-away,
         points-all-game-ou-over,  points-all-game-ou-under.
"""

from __future__ import annotations

import glob
import gzip
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARKETS = {
    "ml_home": "points-home-game-ml-home", "ml_away": "points-away-game-ml-away",
    "pl_home": "points-home-game-sp-home", "pl_away": "points-away-game-sp-away",
    "tot_over": "points-all-game-ou-over", "tot_under": "points-all-game-ou-under",
}


def num(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return None


rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "data/sgo/raw/*.json.gz"))):
    for e in json.load(gzip.open(path, "rt"))["events"]:
        r = {"event_id": e["eventID"], "starts_at": e["status"].get("startsAt"),
             "home": e["teams"]["home"]["names"].get("short"), "away": e["teams"]["away"]["names"].get("short")}
        for k, oid in MARKETS.items():
            o = e.get("odds", {}).get(oid, {})
            r[f"{k}_open"] = num(o.get("openBookOdds"))
            r[f"{k}_close"] = num(o.get("closeBookOdds"))
            r[f"{k}_books"] = len(o.get("byBookmaker", {}))
            if k.startswith("pl"):
                r[f"{k}_open_line"] = num(o.get("openBookSpread"))
                r[f"{k}_close_line"] = num(o.get("closeBookSpread"))
            if k.startswith("tot"):
                r[f"{k}_open_line"] = num(o.get("openBookOverUnder"))
                r[f"{k}_close_line"] = num(o.get("closeBookOverUnder"))
        rows.append(r)

df = pd.DataFrame(rows).drop_duplicates("event_id")
out = os.path.join(ROOT, "data/sgo/game_odds_close.parquet")
df.to_parquet(out, index=False)
print("events", len(df))
for m in ["ml", "pl", "tot"]:
    a, b = (("home", "away") if m != "tot" else ("over", "under"))
    both = df[[f"{m}_{a}_close", f"{m}_{b}_close"]].notna().all(axis=1)
    print(m, "both closing prices:", int(both.sum()), round(float(both.mean()), 3))
