"""Buy PuckAPI shots-on-goal prop history for older seasons (2023-24 to 2026-01-15).

SportsGameOdds closing prices start on 2026-01-16, so every earlier line is a
period the shots model has never been tested against the market. PuckAPI's
archive holds about 51,700 shots-on-goal lines in that window (median closing
line across books, with prices).

Player sample: regulars who reliably get a shots prop. Pool = players with at
least 150 games in the window (data/models/sog_predictions.parquet) and at
least 30 graded SportsGameOdds lines from 2026. Players are drawn at random
(seed 2026) within forwards and defensemen; nothing about their results
against the market is used to choose them.

Calls go to the PuckAPI MCP endpoint over HTTPS. The API key is injected by
the environment's network proxy; this script never reads or stores it.
Each call costs PuckAPI credits: the script stops at --max-calls and keeps a
ledger (data/puckapi/sog_history/ledger.csv) so a rerun skips what is done.

Output (gitignored, purchased data, never committed):
  data/puckapi/sog_history/<player_id>_<from_date>.json   raw responses
Usage: python3 src/ingest/fetch_puckapi_sog_history.py --players N --max-calls M [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data/puckapi/sog_history")
URL = "https://mcp.puckapi.com/mcp"
FROM, TO = "2023-10-01", "2026-01-15"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
           "Authorization": "Bearer proxy-injected"}  # the proxy replaces this placeholder


def player_sample(n: int, seed: int = 2026) -> pd.DataFrame:
    p = pd.read_parquet(os.path.join(ROOT, "data/models/sog_predictions.parquet"))
    p["game_date"] = pd.to_datetime(p.game_date)
    old = p[(p.game_date >= FROM) & (p.game_date <= TO)]
    games = old.groupby("player_id").size().rename("old_games")
    lines = pd.read_parquet(os.path.join(ROOT, "data/models/sog_market_graded.parquet")).groupby("player_id").size().rename("sgo_lines")
    info = p.sort_values("game_date").groupby("player_id")[["player_name", "position"]].last()
    pool = info.join(games, how="inner").join(lines, how="inner")
    pool = pool[(pool.old_games >= 150) & (pool.sgo_lines >= 30)].reset_index()
    pool["group"] = np.where(pool.position == "D", "D", "F")
    rng = np.random.default_rng(seed)
    pool["draw"] = rng.random(len(pool))
    # Forwards and defensemen in proportion to the pool, in random order.
    pool = pool.sort_values("draw").reset_index(drop=True)
    share_d = (pool.group == "D").mean()
    n_d = int(round(n * share_d))
    pick = pd.concat([pool[pool.group == "D"].head(n_d), pool[pool.group == "F"].head(n - n_d)])
    return pick.sort_values("draw").reset_index(drop=True), len(pool)


class Client:
    def __init__(self):
        self.sid = None
        h, _ = self._post({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                                      "clientInfo": {"name": "nhl-prediction-system", "version": "1"}}})
        self.sid = h.get("Mcp-Session-Id")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _post(self, body):
        h = dict(HEADERS)
        if self.sid:
            h["Mcp-Session-Id"] = self.sid
        req = urllib.request.Request(URL, json.dumps(body).encode(), h)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.headers, r.read().decode()
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))

    def call(self, name, args):
        _, body = self._post({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": name, "arguments": args}})
        if body.startswith("event:") or body.startswith("data:"):
            body = next(l[5:] for l in body.splitlines() if l.startswith("data:"))
        res = json.loads(body)
        if "error" in res or res.get("result", {}).get("isError"):
            raise RuntimeError(str(res)[:500])
        return json.loads(res["result"]["content"][0]["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, required=True)
    ap.add_argument("--max-calls", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    pick, pool_size = player_sample(a.players)
    print(f"pool {pool_size} players; sample {len(pick)} ({(pick.group == 'D').sum()} D)")
    os.makedirs(OUT, exist_ok=True)
    pick.to_csv(os.path.join(OUT, "sample.csv"), index=False)
    if a.dry_run:
        print(pick[["player_id", "player_name", "position", "old_games", "sgo_lines"]].to_string())
        return
    ledger = os.path.join(OUT, "ledger.csv")
    done = set()
    if os.path.exists(ledger):
        done = {(int(r["player_id"]), r["from_date"]) for r in csv.DictReader(open(ledger))}
    client, calls = Client(), 0
    for r in pick.itertuples():
        start = FROM
        while start and calls < a.max_calls:
            key = (int(r.player_id), start)
            path = os.path.join(OUT, f"{r.player_id}_{start}.json")
            if key in done and os.path.exists(path):
                start = json.load(open(path)).get("next_from_date") if json.load(open(path)).get("truncated") else None
                continue
            res = client.call("get_prop_history", {"player_id": str(r.player_id), "market": "player_shots_on_goal",
                                                   "from_date": start, "to_date": TO, "limit": 200})
            calls += 1
            json.dump(res, open(path, "w"))
            with open(ledger, "a", newline="") as fh:
                w = csv.writer(fh)
                if fh.tell() == 0:
                    w.writerow(["player_id", "from_date", "games", "next_from_date"])
                w.writerow([r.player_id, start, res.get("count"), res.get("next_from_date")])
            print(f"call {calls}: {r.player_name} from {start}: {res.get('count')} games", flush=True)
            start = res.get("next_from_date") if res.get("truncated") else None
        if calls >= a.max_calls:
            break
    print(f"calls made: {calls}")


if __name__ == "__main__":
    main()
