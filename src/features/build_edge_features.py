"""NHL Edge features: each skater's PREVIOUS-season tracking profile.

Edge (src/ingest/fetch_edge.py) gives season totals only, so a season's
values include games after any date inside it. Every player-game in season
S therefore gets the player's Edge profile from season S-1 (whole season,
finished before S starts). Edge begins in 2021-22, so the features exist from
2022-23 on; earlier rows and players without a previous NHL season are NaN.

  edge_hd_share      high-danger shots / all shots on goal
  edge_hd_shots_pg   high-danger shots per game
  edge_hd_sh_pct     high-danger shooting percentage (goals / shots, high danger)
  edge_oz_pct        share of his ice time in the offensive zone (all strengths)
  edge_oz_ev_pct     the same at even strength
  edge_shot_speed    hardest shot (mph)
  edge_max_speed     top skating speed (mph)
  edge_bursts_pg     skating bursts over 20 mph per game
  edge_dist_pg       miles skated per game
  edge_gp            games behind the profile

Leakage test: tests/test_edge_leakage.py.
Output: data/features/edge_features.parquet (game_id, player_id, features)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES = ["edge_hd_share", "edge_hd_shots_pg", "edge_hd_sh_pct", "edge_oz_pct", "edge_oz_ev_pct", "edge_shot_speed",
            "edge_max_speed", "edge_bursts_pg", "edge_dist_pg", "edge_gp"]


def prev_season(s):
    s = np.asarray(s, dtype=np.int64)
    return (s // 10000 - 1) * 10000 + (s % 10000 - 1)


def profile(e: pd.DataFrame) -> pd.DataFrame:
    gp = e.gp.where(e.gp > 0)
    return pd.DataFrame({
        "player_id": e.player_id, "season": e.season,
        "edge_hd_share": e.shots_high / e.shots.where(e.shots > 0),
        "edge_hd_shots_pg": e.shots_high / gp,
        "edge_hd_sh_pct": e.goals_high / e.shots_high.where(e.shots_high > 0),
        "edge_oz_pct": e.oz_pct, "edge_oz_ev_pct": e.oz_ev_pct,
        "edge_shot_speed": e.top_shot_speed, "edge_max_speed": e.max_skating_speed,
        "edge_bursts_pg": e.bursts_over20 / gp, "edge_dist_pg": e.distance_total / gp, "edge_gp": e.gp})


def build(edge: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    p = profile(edge.astype({c: float for c in edge.columns if c not in ("player_id", "season")}))
    k = f[["game_id", "player_id", "season"]].copy()
    k["prev"] = prev_season(k.season)
    out = k.merge(p.rename(columns={"season": "prev"}), on=["player_id", "prev"], how="left")
    return out[["game_id", "player_id", *FEATURES]]


if __name__ == "__main__":
    edge = pd.read_parquet(os.path.join(ROOT, "data/edge/edge_skater_season.parquet"))
    f = pd.read_parquet(os.path.join(ROOT, "data/features/skater_features.parquet"))
    res = build(edge, f)
    res.to_parquet(os.path.join(ROOT, "data/features/edge_features.parquet"), index=False)
    print(len(res), "player-games")
    print(res.merge(f[["game_id", "player_id", "season"]]).groupby("season")[FEATURES[:3]].count())
    print(res[FEATURES].describe().T.round(3).to_string())
