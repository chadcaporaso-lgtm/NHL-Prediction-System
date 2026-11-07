"""Leakage test for the probable-starter candidate features.

For several cutoff dates, rebuilds the candidate table from goalie rows that
start before the cutoff day plus that day's rows with every result field
scrambled (who started, who played, saves, shots, time on ice), and drops
every later game. The features of the cutoff day's games must be identical
to the full build: they may use only earlier games.
Run: python3 tests/test_probable_starter_leakage.py   (or pytest)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src/features"))
import build_probable_starter as bps  # noqa: E402

CHECK = [f for f in bps.FEATURES] + ["prior_apps"]


def _scramble(day: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    day = day.copy()
    for (_, _), idx in day.groupby(["game_id", "team_id"]).groups.items():
        flip = rng.random() < 0.5
        if flip:
            day.loc[idx, "starter"] = ~day.loc[idx, "starter"].to_numpy()
    day["participated"] = True
    day["saves"] = rng.integers(0, 50, len(day))
    day["shots_against"] = day.saves + rng.integers(0, 8, len(day))
    day["toi_total_sec"] = rng.integers(0, 3900, len(day))
    return day


def test_candidate_features_use_only_earlier_games(seed=11):
    gl = bps.goalie_history()
    full = bps.build_candidates(gl)
    rng = np.random.default_rng(seed)
    days = sorted(gl.start.dt.normalize().unique())
    picks = [days[i] for i in rng.choice(np.arange(200, len(days)), 4, replace=False)]
    for d in picks:
        before = gl[gl.start < d]
        today = gl[(gl.start >= d) & (gl.start < d + pd.Timedelta(days=1))]
        cut = bps.build_candidates(pd.concat([before, _scramble(today, seed)]))
        ids = set(today.game_id)
        a = full[full.game_id.isin(ids)].set_index(["game_id", "team_id", "player_id"]).sort_index()[CHECK]
        b = cut[cut.game_id.isin(ids)].set_index(["game_id", "team_id", "player_id"]).sort_index()[CHECK]
        assert a.index.equals(b.index), f"candidate sets differ on {d}"
        assert np.allclose(a.to_numpy(float), b.to_numpy(float)), f"features differ on {d}"


def test_probabilities_sum_to_one():
    c = pd.read_parquet(os.path.join(ROOT, "data/features/probable_starter_candidates.parquet"))
    s = c.groupby(["game_id", "team_id"]).p.sum()
    assert np.allclose(s, 1.0)
    t = pd.read_parquet(os.path.join(ROOT, "data/features/probable_starter_team.parquet"))
    assert t.exp_sv_l40.between(0.85, 0.95).all()


if __name__ == "__main__":
    for t in [test_candidate_features_use_only_earlier_games, test_probabilities_sum_to_one]:
        t()
        print("ok", t.__name__)
