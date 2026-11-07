"""Fetch player prop history from PuckAPI for the prop backtest.

Calls the PuckAPI MCP server's get_prop_history tool directly over HTTPS and
writes every response to data/puckapi/prop_history/<player_id>_<market>.json.
Each response holds up to 200 games of one player's prop market: the closing
line (median across books), over/under prices, what the player recorded and
the result.

Stops at the first error (for example, when credits run out) so no credits
are spent on calls that cannot succeed. Already-saved files are skipped, so
the script can be re-run.

Pricing observed 2026-11-07: 10 credits per 25 games covered (a 195-game
call cost 80). The total number of player-games is therefore fixed by the
credit budget, so the sample is spread across many players in 25-game
windows rather than a few full histories.

Usage: python src/ingest/fetch_prop_history.py [max_calls]
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data/puckapi/prop_history")
URL = "https://mcp.puckapi.com/mcp"
# Each request: (player_id, expected last name, market, from_date, limit).
# The first request is a full history (2024-25 to date); the rest are
# 25-game windows starting 2025-12-01, the middle of the 2025-26 season.
WINDOW_START = "2025-12-01"
REQUESTS = [(8478402, "McDavid", "player_shots_on_goal", "2024-10-01", 200)]
SOG_PLAYERS = [(8479318, "Matthews"), (8477492, "MacKinnon"), (8477934, "Draisaitl"), (8476453, "Kucherov"),
               (8477956, "Pastrnak"), (8478864, "Kaprizov"), (8480069, "Makar"), (8480800, "Hughes"),
               (8471214, "Ovechkin"), (8478460, "Werenski"), (8484144, "Bedard"), (8481540, "Caufield")]
POINTS_PLAYERS = [(8479318, "Matthews"), (8477492, "MacKinnon"), (8477934, "Draisaitl"), (8476453, "Kucherov"),
                  (8477956, "Pastrnak"), (8478864, "Kaprizov"), (8478550, "Panarin"), (8480069, "Makar"),
                  (8478420, "Rantanen"), (8478403, "Eichel")]
REQUESTS += [(pid, n, "player_shots_on_goal", WINDOW_START, 25) for pid, n in SOG_PLAYERS]
REQUESTS += [(pid, n, "player_points", WINDOW_START, 25) for pid, n in POINTS_PLAYERS]



def call(tool: str, arguments: dict) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            wait = int(exc.headers.get("Retry-After") or 15 * (attempt + 1))
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)


def main(max_calls: int) -> None:
    os.makedirs(OUT, exist_ok=True)
    made = 0
    for player_id, last_name, market, from_date, limit in REQUESTS:
        path = os.path.join(OUT, f"{player_id}_{market}_{from_date}.json")
        if os.path.exists(path):
            continue
        if made >= max_calls:
            print("reached max_calls", max_calls)
            return
        raw = call("get_prop_history", {"player_id": str(player_id), "market": market, "from_date": from_date, "limit": limit})
        made += 1
        result = raw.get("result", {})
        if result.get("isError") or "error" in raw:
            print("STOPPED on error:", json.dumps(raw)[:500])
            return
        data = result.get("structuredContent") or json.loads(result["content"][0]["text"])
        name = data.get("player_name", "")
        if last_name.lower() not in name.lower():
            print(f"WARNING player id {player_id} returned {name!r}, expected {last_name}")
        meta = {k: v for k, v in result.items() if k not in ("content", "structuredContent")}
        json.dump({"request": {"player_id": player_id, "market": market, "from_date": from_date, "limit": limit}, "response": data, "raw_meta": meta}, open(path, "w"))
        credits = meta.get("_creditsMeta", {})
        print(f"{made:3} {name:22} {market:22} games {data.get('count'):4}  credits used {credits.get('used')}  remaining {credits.get('remaining')}")
        if credits.get("remaining") is not None and credits["remaining"] < 10:
            print("stopping: fewer than 10 credits left")
            return
        time.sleep(3)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
