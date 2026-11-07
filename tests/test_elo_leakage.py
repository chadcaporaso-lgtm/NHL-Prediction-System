"""Leakage test for the Elo ratings.

Pre-game ratings must depend only on earlier games: rewriting every result
after a cutoff date (flipping winners and changing margins) must leave every
pre-game rating and probability on or before that date unchanged, and must
change at least some later ones. Also checks that the parameters selected
for each season were chosen without that season's results.
Run: python3 tests/test_elo_leakage.py   (or pytest)
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src/features"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import build_elo as be  # noqa: E402


def test_future_results_do_not_move_past_ratings():
    g = be.game_results()
    params = {"K": 8, "HFA": 20, "C": 0.8, "S_OT": 1.0}
    rh, ra, ph, _ = be.run(g, **params)
    for cutoff in ["2021-03-01", "2024-01-15", "2026-10-20"]:
        h = g.copy()
        later = h.start.dt.strftime("%Y-%m-%d") > cutoff
        h.loc[later, "home_win"] = 1 - h.loc[later, "home_win"]
        h.loc[later, ["hg", "ag"]] = h.loc[later, ["ag", "hg"]].values + np.array([3, 0])
        rh2, ra2, ph2, _ = be.run(h, **params)
        early = (~later).values
        assert np.allclose(ph[early], ph2[early]) and np.allclose(rh[early], rh2[early]), cutoff
        assert not np.allclose(ph[later.values], ph2[later.values]), cutoff


def test_selection_ignores_the_test_season():
    """Flipping every result in 2023-24 must not change the parameters chosen for 2023-24."""
    g = be.game_results()
    grid = {"K": [4, 8, 16], "HFA": [0, 35], "C": [0.65, 1.0], "S_OT": [1.0]}
    _, chosen = be.tune(g, grid, mov=True)
    h = g.copy()
    s = h.season == 20232024
    h.loc[s, "home_win"] = 1 - h.loc[s, "home_win"]
    _, chosen2 = be.tune(h, grid, mov=True)
    assert chosen[20232024] == chosen2[20232024]
    assert chosen[20222023] == chosen2[20222023]


if __name__ == "__main__":
    for t in [test_future_results_do_not_move_past_ratings, test_selection_ignores_the_test_season]:
        t()
        print("PASS", t.__name__)
