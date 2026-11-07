"""Extract closing player-prop prices from SportsGameOdds event files.

Input:  data/sgo/raw/*.json.gz (fetch_sgo_events.py)
Output: data/sgo/player_props_close.parquet, one row per event, player, stat
        and line, with over and under closing prices (consensus across books),
        the opening line and prices, and the provider's recorded stat value.

Odd IDs have the form  <statID>-<PLAYER_ID>-game-ou-<over|under>.
"""

from __future__ import annotations

import glob
import gzip
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAM_SIDES = {"home", "away", "all"}

rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "data/sgo/raw/*.json.gz"))):
    payload = json.load(gzip.open(path, "rt"))
    for e in payload["events"]:
        players = e.get("players", {})
        start = e["status"].get("startsAt")
        home = e["teams"]["home"]["names"].get("short")
        away = e["teams"]["away"]["names"].get("short")
        for odd_id, o in e.get("odds", {}).items():
            parts = odd_id.split("-")
            if len(parts) != 5 or parts[2] != "game" or parts[3] != "ou" or parts[1].lower() in TEAM_SIDES:
                continue
            stat, player, _, _, side = parts
            p = players.get(player, {})
            rows.append({
                "event_id": e["eventID"], "starts_at": start, "home": home, "away": away,
                "player_key": player, "player_name": p.get("name"), "player_team": p.get("teamID"),
                "stat": stat, "side": side,
                "close_line": o.get("closeBookOverUnder"), "close_odds": o.get("closeBookOdds"), "close_fair_odds": o.get("closeFairOdds"),
                "open_line": o.get("openBookOverUnder"), "open_odds": o.get("openBookOdds"),
                "score": o.get("score"), "books": len(o.get("byBookmaker", {})),
            })

df = pd.DataFrame(rows)
for c in ["close_line", "open_line", "score"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in ["close_odds", "close_fair_odds", "open_odds"]:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace("+", "", regex=False), errors="coerce")
key = ["event_id", "starts_at", "home", "away", "player_key", "player_name", "player_team", "stat"]
over = df[df.side == "over"].drop(columns="side")
under = df[df.side == "under"].drop(columns="side")
wide = over.merge(under[key + ["close_line", "close_odds", "close_fair_odds", "open_odds"]], on=key, suffixes=("_over", "_under"), how="outer")
out = os.path.join(ROOT, "data/sgo/player_props_close.parquet")
wide.to_parquet(out, index=False)
print("rows", len(wide), "events", wide.event_id.nunique(), "players", wide.player_key.nunique())
print(wide.stat.value_counts().head(15).to_string())
print("share with both closing prices:", round(float(wide[["close_odds_over", "close_odds_under"]].notna().all(axis=1).mean()), 3))
