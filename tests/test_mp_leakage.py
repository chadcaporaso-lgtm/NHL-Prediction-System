"""Leakage test for data/features/mp_features.parquet.

1. Future scramble: replace every MoneyPuck value on or after a cut-off date
   with random numbers and rebuild; features for games on the cut-off date
   and earlier must not change (a game's own MoneyPuck row must not matter).
2. Recompute: for a sample of player-games, rebuild mp_ixg and mp_games by
   hand from the player's earlier games and compare.

Run: python3 tests/test_mp_leakage.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src/features"))
from build_mp_features import FEATURES, HALF_LIFE, build  # noqa: E402


def _load():
    mp = pd.read_parquet(os.path.join(ROOT, "data/moneypuck/skaters/mp_skater_games.parquet"))
    g = pd.read_parquet(os.path.join(ROOT, "data/clean/skater_games.parquet"))
    stored = pd.read_parquet(os.path.join(ROOT, "data/features/mp_features.parquet"))
    return mp, g, stored


def test_future_scramble(cut="2024-01-15", seed=4):
    mp, g, stored = _load()
    dates = g.drop_duplicates("game_id").set_index("game_id").game_date
    late = mp.gameId.map(dates) >= cut
    num = [c for c in mp.columns if c.startswith(("I_F_", "OnIce_"))]
    rng = np.random.default_rng(seed)
    mp = mp.copy()
    mp.loc[late, num] = rng.random((late.sum(), len(num))) * 10
    new = build(mp, g).set_index(["game_id", "player_id"])
    a = stored.set_index(["game_id", "player_id"])
    keep = a.index.get_level_values(0).map(dates) <= cut
    x, y = a[keep][FEATURES].to_numpy(float), new.loc[a[keep].index, FEATURES].to_numpy(float)
    assert np.allclose(x, y, equal_nan=True), "features changed when same-day or later MoneyPuck data changed"
    later = ~keep
    assert not np.allclose(a[later][FEATURES].to_numpy(float), new.loc[a[later].index, FEATURES].to_numpy(float),
                           equal_nan=True), "scramble had no effect (test is not testing anything)"


def test_recompute(n=300, seed=8):
    mp, g, stored = _load()
    g["game_date"] = pd.to_datetime(g.game_date)
    a = mp[mp.situation == "all"].set_index(["playerId", "gameId"]).I_F_xGoals
    st = stored.set_index(["game_id", "player_id"])
    for gid, pid in stored[stored.mp_games >= 1].sample(n, random_state=seed)[["game_id", "player_id"]].itertuples(index=False):
        hist = g[g.player_id == pid].sort_values(["game_date", "game_id"])
        today = hist[hist.game_id == gid].game_date.iloc[0]
        prior = hist[(hist.game_date < today) | ((hist.game_date == today) & (hist.game_id < gid))]
        vals = pd.Series([a.get((pid, x), np.nan) for x in prior.game_id], dtype=float)
        exp = vals.ewm(halflife=HALF_LIFE, ignore_na=True).mean().iloc[-1]
        assert np.isclose(exp, st.loc[(gid, pid), "mp_ixg"], equal_nan=True), (gid, pid, exp, st.loc[(gid, pid), "mp_ixg"])
        assert vals.notna().sum() == st.loc[(gid, pid), "mp_games"], (gid, pid)


if __name__ == "__main__":
    test_future_scramble()
    test_recompute()
    print("MoneyPuck leakage tests passed")
