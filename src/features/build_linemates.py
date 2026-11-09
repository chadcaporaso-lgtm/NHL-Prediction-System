"""Linemate features from NHL shift charts (points model, variant `lm`).

Inputs
  data/shiftcharts/shift_pairs.parquet  teammate pair seconds per game (5v5, power play)
  data/shiftcharts/shift_toi.parquet    skater seconds by strength per game
  data/features/skater_features.parquet pregame form (ewm_points) and tonight's dressed skaters

For each player-game, from the player's previous WINDOW games only (shift
data from tonight's game is never used):

  lm_ev_pts      mean ewm_points of tonight's dressed teammates, weighted by
                 seconds shared with the player at 5-on-5 over those games
                 (his usual linemates and defence partner dominate)
  lm_pp_pts      the same, weighted by seconds shared on the power play
  lm_ev_present  share of his recent 5-on-5 time with teammates who are
                 dressed tonight (1 = his usual line is intact)
  pp_unit_share  his power-play seconds over the team's power-play skater
                 seconds / 5 in those games (about 1 on the first unit,
                 near 0 if he does not play the power play)
  shift_games    how many of the previous WINDOW games have shift data

Teammates' ewm_points is their pregame value for tonight (already uses
earlier games only). Tonight's dressed lineup is known before the game
(assumption L-1, the same as the v2 lineup features). Rows with no earlier
shift data get NaN (the model fills them).

Leakage test: tests/test_linemate_leakage.py.
Output: data/features/linemate_features.parquet (game_id, player_id, features)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WINDOW = 5
FEATURES = ["lm_ev_pts", "lm_pp_pts", "lm_ev_present", "pp_unit_share", "shift_games"]


def build(pairs: pd.DataFrame, toi: pd.DataFrame, f: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    f = f[["game_id", "player_id", "team", "game_date", "ewm_points"]].copy()
    f["game_date"] = pd.to_datetime(f.game_date)
    f = f.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    f["k"] = f.groupby("player_id").cumcount()  # player's game number
    key = f[["game_id", "player_id", "k"]]

    # Pairs both ways, tagged with the player's game number k.
    p = pd.concat([pairs[["game_id", "player_id", "mate_id", "s_5v5", "s_pp"]],
                   pairs.rename(columns={"player_id": "mate_id", "mate_id": "player_id"})[
                       ["game_id", "player_id", "mate_id", "s_5v5", "s_pp"]]], ignore_index=True)
    p = p.merge(key, on=["game_id", "player_id"], how="inner")

    # Team power-play skater seconds per game, and the player's own seconds.
    t = toi.merge(key, on=["game_id", "player_id"], how="inner")
    team_pp = t.groupby(["game_id", "team_id"]).s_pp.sum().rename("team_pp").reset_index()
    t = t.merge(team_pp, on=["game_id", "team_id"])

    out = []
    for lag in range(1, window + 1):
        # Source game k contributes to the player's game k + lag.
        x = p.assign(k=p.k + lag)
        out.append(x)
    hist = pd.concat(out, ignore_index=True).groupby(["player_id", "k", "mate_id"])[["s_5v5", "s_pp"]].sum().reset_index()
    hist = hist.merge(f[["player_id", "k", "game_id"]], on=["player_id", "k"], how="inner")  # tonight's game
    mates = f[["game_id", "player_id", "ewm_points"]].rename(columns={"player_id": "mate_id", "ewm_points": "mate_pts"})
    hist = hist.merge(mates, on=["game_id", "mate_id"], how="left")
    hist["dressed"] = hist.mate_pts.notna()
    hist["mate_pts"] = hist.mate_pts.fillna(0.0)
    hist["w5_d"] = hist.s_5v5 * hist.dressed
    hist["wpp_d"] = hist.s_pp * hist.dressed
    hist["w5_pts"] = hist.w5_d * hist.mate_pts
    hist["wpp_pts"] = hist.wpp_d * hist.mate_pts
    g = hist.groupby(["game_id", "player_id"])[["s_5v5", "w5_d", "wpp_d", "w5_pts", "wpp_pts"]].sum()
    res = pd.DataFrame(index=g.index)
    res["lm_ev_pts"] = g.w5_pts / g.w5_d.where(g.w5_d > 0)
    res["lm_pp_pts"] = g.wpp_pts / g.wpp_d.where(g.wpp_d > 0)
    res["lm_ev_present"] = g.w5_d / g.s_5v5.where(g.s_5v5 > 0)
    res = res.reset_index()

    pp = []
    for lag in range(1, window + 1):
        pp.append(t[["player_id", "k", "s_pp", "team_pp"]].assign(k=t.k + lag))
    pp = pd.concat(pp, ignore_index=True).groupby(["player_id", "k"]).agg(s_pp=("s_pp", "sum"), team_pp=("team_pp", "sum"),
                                                                         shift_games=("s_pp", "size")).reset_index()
    pp["pp_unit_share"] = (pp.s_pp / (pp.team_pp / 5).where(pp.team_pp > 0)).clip(upper=1.0)
    pp = pp.merge(f[["player_id", "k", "game_id"]], on=["player_id", "k"], how="inner")

    res = f[["game_id", "player_id"]].merge(res, on=["game_id", "player_id"], how="left") \
        .merge(pp[["game_id", "player_id", "pp_unit_share", "shift_games"]], on=["game_id", "player_id"], how="left")
    res["shift_games"] = res.shift_games.fillna(0).astype(int)
    return res[["game_id", "player_id", *FEATURES]]


if __name__ == "__main__":
    S = os.path.join(ROOT, "data/shiftcharts")
    pairs = pd.read_parquet(os.path.join(S, "shift_pairs.parquet"))
    toi = pd.read_parquet(os.path.join(S, "shift_toi.parquet"))
    f = pd.read_parquet(os.path.join(ROOT, "data/features/skater_features.parquet"))
    res = build(pairs, toi, f)
    res.to_parquet(os.path.join(ROOT, "data/features/linemate_features.parquet"), index=False)
    print(len(res), "player-games")
    print(res[FEATURES].describe().T.round(3).to_string())
    print("missing:", res[FEATURES].isna().mean().round(3).to_dict())
