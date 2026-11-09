"""Leakage test for the opponent-adjusted team ratings (build_team_rating.py).

Pre-game ratings must depend only on games on earlier dates: rewriting every
result on or after a cutoff date must leave every rating for dates up to the
cutoff unchanged, and must change later ones. Also checks that the
parameters selected for a season were chosen without that season's results.
Run: python3 tests/test_team_rating_leakage.py   (or pytest)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src/features"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import build_team_rating as br  # noqa: E402


def test_future_results_do_not_move_past_ratings():
    tg = br.team_games()
    dates = np.array(sorted(tg.date.unique()))
    params = {"H": 120, "C": 0.6, "LAMBDA": 8.0}
    f = br.run(tg, dates, **params)
    checked = 0
    for cutoff in pd.to_datetime(dates[[len(dates) // 4, len(dates) // 2, len(dates) - 5]]):
        h = tg.copy()
        later = h.date >= cutoff
        for s in br.STATS:
            h.loc[later, s] = h.loc[later, s] * 3 + 1
        f2 = br.run(h, dates, **params)
        upto = pd.to_datetime(dates) <= cutoff
        for s in br.STATS:
            for k in range(4):
                a, b = np.asarray(f[s][k])[upto], np.asarray(f2[s][k])[upto]
                assert np.allclose(a, b, equal_nan=True), (cutoff, s, k)
            assert not np.allclose(np.asarray(f[s][2])[~upto], np.asarray(f2[s][2])[~upto], equal_nan=True), (cutoff, s)
        checked += 1
    assert checked == 3


def test_selection_ignores_the_test_season():
    """Rewriting 2023-24 must not change the parameters chosen for 2023-24 or earlier."""
    tg = br.team_games()
    grid = {"H": [60, 240], "C": [0.3, 1.0], "LAMBDA": [8.0]}
    _, combos, _, _, chosen = br.tune(tg, grid)
    h = tg.copy()
    s = h.season == 20232024
    rng = np.random.default_rng(0)
    for st in br.STATS:
        h.loc[s, st] = rng.permutation(h.loc[s, st].values) * 2
    _, combos2, _, _, chosen2 = br.tune(h, grid)
    for st in br.STATS:
        for S in [20212022, 20222023, 20232024]:
            assert combos[chosen[st][S]] == combos2[chosen2[st][S]], (st, S)


if __name__ == "__main__":
    for t in [test_future_results_do_not_move_past_ratings, test_selection_ignores_the_test_season]:
        t()
        print("PASS", t.__name__)
