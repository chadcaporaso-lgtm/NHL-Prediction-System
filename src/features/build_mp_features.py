"""MoneyPuck game-by-game shot-quality and deployment features (variants *_mp).

Input: data/moneypuck/skaters/mp_skater_games.parquet (src/ingest/fetch_moneypuck_skaters.py),
joined to the player's games in data/clean/skater_games.parquet. Every feature
is an exponentially weighted mean (half-life 10 games, as the other skater
features) of the player's EARLIER games only (shift(1) within the player's
game sequence). Games MoneyPuck lacks are skipped, not counted as zero.

  mp_ixg            individual expected goals per game (all situations)
  mp_attempts       shot attempts per game (all situations)
  mp_xog            expected shots on goal per game (MoneyPuck's on-goal model)
  mp_hd_shots       high-danger shots per game
  mp_pp_ixg         expected goals per game on the 5-on-4 power play
  mp_onice_xgf      on-ice expected goals for per game (all situations; scoring chances around him)
  mp_onice_xgf_pct  on-ice expected-goals share at 5-on-5
  mp_ozs_share      offensive-zone share of his offensive + defensive zone shift starts
  mp_games          earlier games with MoneyPuck data (count)

Leakage test: tests/test_mp_leakage.py.
Output: data/features/mp_features.parquet (game_id, player_id, features)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HALF_LIFE = 10
FEATURES = ["mp_ixg", "mp_attempts", "mp_xog", "mp_hd_shots", "mp_pp_ixg", "mp_onice_xgf", "mp_onice_xgf_pct",
            "mp_ozs_share", "mp_games"]


def per_game(mp: pd.DataFrame) -> pd.DataFrame:
    """MoneyPuck rows (game x situation) -> one row of raw per-game values per player-game."""
    mp = mp.rename(columns={"playerId": "player_id", "gameId": "game_id"})
    a = mp[mp.situation == "all"].set_index(["game_id", "player_id"])
    f5 = mp[mp.situation == "5on5"].set_index(["game_id", "player_id"])
    pp = mp[mp.situation == "5on4"].set_index(["game_id", "player_id"])
    out = pd.DataFrame(index=a.index)
    out["ixg"] = a.I_F_xGoals
    out["attempts"] = a.I_F_shotAttempts
    out["xog"] = a.I_F_xOnGoal
    out["hd_shots"] = a.I_F_highDangerShots
    out["pp_ixg"] = pp.I_F_xGoals.reindex(out.index).fillna(0.0)
    out["onice_xgf"] = a.OnIce_F_xGoals
    tot = (f5.OnIce_F_xGoals + f5.OnIce_A_xGoals).reindex(out.index)
    out["onice_xgf_pct"] = f5.OnIce_F_xGoals.reindex(out.index) / tot.where(tot > 0)
    zs = (a.I_F_oZoneShiftStarts + a.I_F_dZoneShiftStarts)
    out["ozs_share"] = a.I_F_oZoneShiftStarts / zs.where(zs > 0)
    out["has_mp"] = 1.0
    return out.reset_index()


def build(mp: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    g = g[["game_id", "player_id", "game_date"]].copy()
    g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    g = g.merge(per_game(mp), on=["game_id", "player_id"], how="left")
    by = g.groupby("player_id", sort=False)
    out = g[["game_id", "player_id"]].copy()
    for x in ["ixg", "attempts", "xog", "hd_shots", "pp_ixg", "onice_xgf", "onice_xgf_pct", "ozs_share"]:
        prev = by[x].shift(1)
        out[f"mp_{x}"] = prev.groupby(g.player_id).transform(lambda s: s.ewm(halflife=HALF_LIFE, ignore_na=True).mean())
    out["mp_games"] = by.has_mp.shift(1).fillna(0).groupby(g.player_id).cumsum()
    return out[["game_id", "player_id", *FEATURES]]


if __name__ == "__main__":
    mp = pd.read_parquet(os.path.join(ROOT, "data/moneypuck/skaters/mp_skater_games.parquet"))
    g = pd.read_parquet(os.path.join(ROOT, "data/clean/skater_games.parquet"))
    res = build(mp, g)
    res.to_parquet(os.path.join(ROOT, "data/features/mp_features.parquet"), index=False)
    pg = per_game(mp)
    cov = g.merge(pg[["game_id", "player_id"]].assign(mp=1), on=["game_id", "player_id"], how="left")
    print("skater-games with MoneyPuck data by season:")
    print(cov.groupby("season").mp.mean().round(4).to_string())
    # Agreement check: MoneyPuck shots on goal vs the official box score.
    chk = g.merge(mp[mp.situation == "all"].rename(columns={"playerId": "player_id", "gameId": "game_id"})[
        ["game_id", "player_id", "I_F_shotsOnGoal", "I_F_goals"]], on=["game_id", "player_id"])
    print("shots on goal equal to box score:", round(float((chk.I_F_shotsOnGoal == chk.shots).mean()), 4),
          "goals equal:", round(float((chk.I_F_goals == chk.goals).mean()), 4), "n", len(chk))
    print(res[FEATURES].describe().T.round(3).to_string())
