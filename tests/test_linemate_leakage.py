"""Leakage test for data/features/linemate_features.parquet.

1. Recompute from scratch: for a random sample of player-games, rebuild the
   features by hand from the player's previous 5 games' shift pairs and
   tonight's dressed teammates, and compare with the stored values.
2. Scramble the future: replace every shift pair and ice time on or after a
   cut-off date with random numbers, rebuild, and check that every feature
   for games on the cut-off date and earlier is unchanged (tonight's own
   shifts must not matter).

Run: python3 tests/test_linemate_leakage.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src/features"))
from build_linemates import FEATURES, WINDOW, build  # noqa: E402

S = os.path.join(ROOT, "data/shiftcharts")


def _load():
    pairs = pd.read_parquet(os.path.join(S, "shift_pairs.parquet"))
    toi = pd.read_parquet(os.path.join(S, "shift_toi.parquet"))
    f = pd.read_parquet(os.path.join(ROOT, "data/features/skater_features.parquet"))
    f["game_date"] = pd.to_datetime(f.game_date)
    stored = pd.read_parquet(os.path.join(ROOT, "data/features/linemate_features.parquet"))
    return pairs, toi, f, stored


def test_recompute(n=300, seed=11):
    pairs, toi, f, stored = _load()
    st = stored.set_index(["game_id", "player_id"])
    both = pd.concat([pairs, pairs.rename(columns={"player_id": "mate_id", "mate_id": "player_id"})], ignore_index=True)
    sample = stored[stored.shift_games >= 1].sample(n, random_state=seed)
    fs = f.sort_values(["game_date", "game_id"])
    bad = 0
    for gid, pid in sample[["game_id", "player_id"]].itertuples(index=False):
        hist = fs[fs.player_id == pid]
        today = hist[hist.game_id == gid].iloc[0]
        prev = hist[(hist.game_date < today.game_date)].tail(WINDOW).game_id
        pr = both[(both.player_id == pid) & both.game_id.isin(prev)].groupby("mate_id")[["s_5v5", "s_pp"]].sum()
        dressed = f[f.game_id == gid].set_index("player_id").ewm_points.fillna(0.0)
        d = pr[pr.index.isin(dressed.index)]
        exp = {"lm_ev_pts": (d.s_5v5 * dressed[d.index]).sum() / d.s_5v5.sum() if d.s_5v5.sum() > 0 else np.nan,
               "lm_pp_pts": (d.s_pp * dressed[d.index]).sum() / d.s_pp.sum() if d.s_pp.sum() > 0 else np.nan,
               "lm_ev_present": d.s_5v5.sum() / pr.s_5v5.sum() if pr.s_5v5.sum() > 0 else np.nan}
        t = toi[toi.game_id.isin(prev)]
        team = t[t.player_id == pid][["game_id", "team_id"]]
        mine = t[t.player_id == pid].s_pp.sum()
        team_pp = t.merge(team, on=["game_id", "team_id"]).s_pp.sum()
        exp["pp_unit_share"] = min(mine / (team_pp / 5), 1.0) if team_pp > 0 else np.nan
        exp["shift_games"] = len(team)
        for k, v in exp.items():
            got = st.loc[(gid, pid), k]
            if not ((np.isnan(v) and np.isnan(got)) or np.isclose(v, got, rtol=1e-9, atol=1e-9)):
                bad += 1
                print("mismatch", gid, pid, k, v, got)
    assert bad == 0, f"{bad} mismatches"


def test_future_scramble(cut="2024-01-15", seed=3):
    pairs, toi, f, stored = _load()
    dates = f.drop_duplicates("game_id").set_index("game_id").game_date
    rng = np.random.default_rng(seed)
    late_p = pairs.game_id.map(dates) >= cut
    late_t = toi.game_id.map(dates) >= cut
    pairs, toi = pairs.copy(), toi.copy()
    pairs.loc[late_p, "s_5v5"] = rng.integers(0, 1500, late_p.sum()).astype("int16")
    pairs.loc[late_p, "s_pp"] = rng.integers(0, 300, late_p.sum()).astype("int16")
    toi.loc[late_t, "s_pp"] = rng.integers(0, 300, late_t.sum()).astype("int16")
    new = build(pairs, toi, f)
    keep = stored.game_id.map(dates) <= cut
    a = stored[keep].set_index(["game_id", "player_id"]).sort_index()[FEATURES]
    b = new.set_index(["game_id", "player_id"]).loc[a.index, FEATURES]
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True), "features changed when later shifts changed"
    later = stored.game_id.map(dates) > cut
    c = stored[later].set_index(["game_id", "player_id"]).sort_index()[FEATURES]
    assert not np.allclose(c.to_numpy(float), new.set_index(["game_id", "player_id"]).loc[c.index, FEATURES].to_numpy(float),
                           equal_nan=True), "scramble had no effect on later games (test is not testing anything)"


if __name__ == "__main__":
    test_recompute()
    test_future_scramble()
    print("linemate leakage tests passed")
