"""Leakage test for data/features/edge_features.parquet.

Edge data are season totals, so a game in season S may use only season S-1.
1. Scramble every season's own Edge rows for the season the game is in:
   replace all Edge values of season S with random numbers and rebuild;
   features for games in season S must not change.
2. Recompute: for a sample of player-games, the stored features equal the
   profile built from the player's season S-1 row.

Run: python3 tests/test_edge_leakage.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src/features"))
from build_edge_features import FEATURES, build, prev_season, profile  # noqa: E402


def _load():
    edge = pd.read_parquet(os.path.join(ROOT, "data/edge/edge_skater_season.parquet"))
    f = pd.read_parquet(os.path.join(ROOT, "data/features/skater_features.parquet"))
    stored = pd.read_parquet(os.path.join(ROOT, "data/features/edge_features.parquet"))
    return edge, f, stored


def test_same_season_scramble(seed=5):
    edge, f, stored = _load()
    rng = np.random.default_rng(seed)
    a = stored.set_index(["game_id", "player_id"]).sort_index()
    seasons = f.set_index(["game_id", "player_id"]).season.reindex(a.index)
    for s in sorted(edge.season.unique()):
        e = edge.copy()
        cols = [c for c in e.columns if c not in ("player_id", "season")]
        m = e.season == s
        e.loc[m, cols] = rng.random((m.sum(), len(cols))) * 100
        b = build(e, f).set_index(["game_id", "player_id"]).loc[a.index]
        same = (seasons == s).to_numpy()
        assert np.allclose(a[same][FEATURES].to_numpy(float), b[same][FEATURES].to_numpy(float), equal_nan=True), \
            f"season {s} features changed when season {s} Edge data changed"
        nxt = (prev_season(seasons.to_numpy()) == s)
        if nxt.any() and a[nxt].edge_gp.notna().any():
            assert not np.allclose(a[nxt][FEATURES].to_numpy(float), b[nxt][FEATURES].to_numpy(float), equal_nan=True), \
                "scramble did not reach the next season (test is not testing anything)"


def test_recompute(n=500, seed=9):
    edge, f, stored = _load()
    p = profile(edge.astype({c: float for c in edge.columns if c not in ("player_id", "season")})).set_index(["player_id", "season"])
    s = stored.merge(f[["game_id", "player_id", "season"]]).sample(n, random_state=seed)
    for r in s.itertuples(index=False):
        key = (r.player_id, int(prev_season([r.season])[0]))
        exp = p.loc[key, FEATURES].to_numpy(float) if key in p.index else np.full(len(FEATURES), np.nan)
        got = np.array([getattr(r, c) for c in FEATURES], float)
        assert np.allclose(exp, got, equal_nan=True), (r.game_id, r.player_id)


if __name__ == "__main__":
    test_same_season_scramble()
    test_recompute()
    print("edge leakage tests passed")
