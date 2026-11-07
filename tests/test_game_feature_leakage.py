"""Leakage test for data/features/game_team_features.parquet.

Recomputes team form and goalie features for random team-games from raw
MoneyPuck and goalie rows dated strictly before the game, and checks the
regulation-goal targets against the official scores.
Run: python3 tests/test_game_feature_leakage.py   (or pytest)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src/features"))
import build_game_features as bgf  # noqa: E402


def test_team_form_uses_only_prior_games(n=300, seed=5):
    f = pd.read_parquet(os.path.join(ROOT, "data/features/game_team_features.parquet"))
    h = bgf.team_history()
    sample = pd.concat([f.dropna(subset=["ewm20_xgf"]).sample(n, random_state=seed),
                        f[f.team == "UTA"].head(5)])  # first Utah games use Arizona history
    bad = []
    for _, r in sample.iterrows():
        th = h[h.hist_team == h.loc[h.team == r.team, "hist_team"].iloc[0]]
        d0 = th.loc[th.game_id == r.game_id, "game_date"].iloc[0]
        prior = th[th.game_date < d0]
        for col, hl in [("xgf", 20), ("ga", 20), ("xga60_5v5", 20), ("xgf", 60)]:
            expect = prior[col].ewm(halflife=hl, ignore_na=True).mean().iloc[-1]
            if not np.isclose(r[f"ewm{hl}_{col}"], expect):
                bad.append((r.game_id, r.team, col, hl, r[f"ewm{hl}_{col}"], expect))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:5]}"


def test_starter_save_pct_uses_only_prior_appearances(n=300, seed=9):
    f = pd.read_parquet(os.path.join(ROOT, "data/features/game_team_features.parquet"))
    gl = bgf.official.load_goalies()
    gl = gl[gl.participated].sort_values(["scheduled_utc", "game_id"])
    bad = []
    for _, r in f.dropna(subset=["starter_sv_l40"]).sample(n, random_state=seed).iterrows():
        prior = gl[(gl.player_id == r.starter_id) & (gl.scheduled_utc < r.scheduled_utc)].tail(40)
        expect = (prior.saves.sum() + 0.905 * 300) / (prior.shots_against.sum() + 300)
        if not np.isclose(r.starter_sv_l40, expect):
            bad.append((r.game_id, r.team, r.starter_sv_l40, expect))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:5]}"


def test_league_scoring_uses_only_earlier_dates(n=50, seed=2):
    f = pd.read_parquet(os.path.join(ROOT, "data/features/game_team_features.parquet"))
    h = f[f.is_home == 1].sort_values(["game_date", "game_id"])
    h = h.assign(goals=h.reg_goals_for + h.reg_goals_against)
    for _, r in h.dropna(subset=["league_reg_gpg_l300"]).sample(n, random_state=seed).iterrows():
        prior = h[h.game_date < r.game_date].tail(300)
        assert np.isclose(r.league_reg_gpg_l300, prior.goals.mean() / 2), (r.game_id, r.league_reg_gpg_l300)


def test_regulation_targets():
    f = pd.read_parquet(os.path.join(ROOT, "data/features/game_team_features.parquet"))
    reg = f.reg_goals_for - f.reg_goals_against
    assert (reg[f.decision_type == "REG"] != 0).all()
    assert (reg[f.decision_type != "REG"] == 0).all(), "OT and SO games must be tied after regulation"
    assert ((f.final_score_for - f.final_score_against).abs()[f.decision_type != "REG"] == 1).all()


if __name__ == "__main__":
    for t in [test_team_form_uses_only_prior_games, test_starter_save_pct_uses_only_prior_appearances,
              test_league_scoring_uses_only_earlier_dates, test_regulation_targets]:
        t()
        print("PASS", t.__name__)
