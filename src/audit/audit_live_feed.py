"""Check the weekly NHL web API feed against the audited v2 tables.

For every 2026-27 game in both sources, compares final scores, real goals,
shootout credits, shots on goal, decision type, and each goalie's starter
flag, saves, shots against and goals against.

Input:  data/live/<date>/ (colab_weekly_refresh.py) and the v2 run
Output: docs/audit/live_feed_check_<date>.json
Usage:  python3 src/audit/audit_live_feed.py 2026-11-07
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/data/normalized/official_incremental/20261103T160656Z_4414a3fb")
DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-11-07"
LIVE = os.path.join(ROOT, f"data/live/{DATE}")

def norm(s: pd.Series) -> pd.Series:
    """Compare by value: booleans and 0/1 integers, ints and floats, are equal when their values are."""
    return s.map(lambda v: float(v) if isinstance(v, (bool, int, float, np.bool_, np.integer, np.floating)) else str(v))


tl = pd.read_parquet(f"{LIVE}/team_games_live.parquet")
tv = pd.read_parquet(f"{RUN}/official_team_games.parquet")
tv = tv[tv.season == 20262027]
key = ["game_id", "team_id"]
cols = ["final_score_for", "final_score_against", "real_goals_for", "real_goals_against", "sog_for", "sog_against",
        "decision_type", "is_home", "won_final"]
m = tv[key + cols].merge(tl[key + cols], on=key, suffixes=("_v2", "_live"))
out = {"date": DATE, "team_games_v2": len(tv), "team_games_live": len(tl), "team_games_in_both": len(m),
       "v2_games_missing_from_live": int((~tv.game_id.isin(tl.game_id)).sum()),
       "team_game_mismatches": {c: int((norm(m[f"{c}_v2"]) != norm(m[f"{c}_live"])).sum()) for c in cols}}

gl = pd.read_parquet(f"{LIVE}/goalie_games_live.parquet")
gv = pd.read_parquet(f"{RUN}/goalie_history_candidates.parquet")
gv = gv[gv.season == 20262027]
gk = ["game_id", "player_id"]
gc = ["starter", "saves", "shots_against", "goals_against", "participated", "team_id"]
g = gv[gk + gc].merge(gl[gk + gc], on=gk, how="outer", suffixes=("_v2", "_live"), indicator=True)
both = g[g._merge == "both"]
out["goalie_rows_v2"], out["goalie_rows_live"] = len(gv), len(gl)
out["goalie_rows_only_v2"] = int((g._merge == "left_only").sum())
out["goalie_rows_only_live"] = int(((g._merge == "right_only") & g.game_id.isin(gv.game_id)).sum())
out["goalie_mismatches"] = {c: int((norm(both[f"{c}_v2"]) != norm(both[f"{c}_live"])).sum()) for c in gc}
out["goalie_mismatch_sample"] = both[(both.saves_v2 != both.saves_live) | (both.starter_v2 != both.starter_live)].head(5).astype(str).values.tolist()
out["only_v2_sample"] = g[g._merge == "left_only"].head(5).astype(str).values.tolist()
json.dump(out, open(os.path.join(ROOT, f"docs/audit/live_feed_check_{DATE}.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
