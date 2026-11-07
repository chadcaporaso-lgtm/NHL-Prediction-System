"""Leakage test for data/features/skater_features.parquet.

For a random sample of player-games, recompute features from scratch using
only that player's (or team's) rows dated strictly before the game, and
check they equal the stored values. Also checks that no feature is
perfectly predictive of the same-game target.

Run: python3 tests/test_skater_feature_leakage.py   (or pytest)
"""

from __future__ import annotations

import os

import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import official  # noqa: E402
RUN = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/data/normalized/official_incremental/20261103T160656Z_4414a3fb")


def _load():
    f = pd.read_parquet(os.path.join(ROOT, "data/features/skater_features.parquet"))
    g = pd.read_parquet(os.path.join(ROOT, "data/clean/skater_games.parquet"))
    g["game_date"] = pd.to_datetime(g.game_date)
    return f, g


def test_player_features_use_only_prior_games(n=500, seed=7):
    f, g = _load()
    sample = f[f.n_prior >= 1].sample(n, random_state=seed)
    gp = {pid: d.sort_values(["game_date", "game_id"]) for pid, d in g[g.player_id.isin(sample.player_id)].groupby("player_id")}
    bad = []
    for _, r in sample.iterrows():
        hist = gp[r.player_id]
        prior = hist[hist.game_date < r.game_date]
        checks = {
            "n_prior": len(prior),
            "car_shots": prior.shots.mean(),
            "l10_shots": prior.shots.tail(10).mean(),
            "ewm_shots": prior.shots.ewm(halflife=10).mean().iloc[-1],
            "l10_toi": prior.toi.tail(10).mean(),
            "car_points": prior.points.mean(),
            "l10_pp_points": prior.pp_points.tail(10).mean(),
        }
        ps = prior[prior.season == r.season]
        if len(ps):
            checks["std_shots"] = ps.shots.mean()
        for k, v in checks.items():
            if not np.isclose(r[k], v, rtol=1e-9, atol=1e-9):
                bad.append((r.game_id, r.player_id, k, r[k], v))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:5]}"


def test_team_context_uses_only_prior_games(n=300, seed=11):
    f, _ = _load()
    tg = official.load_team_games()
    tg["team"] = tg.team_id.map(official.team_abbrev())
    tg = tg[tg.game_type.isin([2, 3]) & tg.final_score_for.notna()].sort_values(["scheduled_utc", "game_id"])
    sample = f.dropna(subset=["opp_sog_against_l20"]).drop_duplicates(["game_id", "opponent"]).sample(n, random_state=seed)
    bad = []
    for _, r in sample.iterrows():
        opp = tg[tg.team == r.opponent]
        t0 = opp.loc[opp.game_id == r.game_id, "scheduled_utc"].iloc[0]
        expect = opp[opp.scheduled_utc < t0].sog_against.tail(20).mean()
        if not np.isclose(r.opp_sog_against_l20, expect):
            bad.append((r.game_id, r.opponent, r.opp_sog_against_l20, expect))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:5]}"


def test_opposing_goalie_uses_only_prior_appearances(n=300, seed=3):
    f, _ = _load()
    gl = official.load_goalies()
    gl = gl[gl.participated].sort_values(["scheduled_utc", "game_id"])
    sample = f.dropna(subset=["opp_goalie_sv_l40"]).drop_duplicates(["game_id", "opponent"]).sample(n, random_state=seed)
    bad = []
    for _, r in sample.iterrows():
        st = gl[(gl.game_id == r.game_id) & (gl.team_abbrev == r.opponent) & gl.starter].iloc[0]
        prior = gl[(gl.player_id == st.player_id) & (gl.scheduled_utc < st.scheduled_utc)].tail(40)
        expect = (prior.saves.sum() + 0.905 * 300) / (prior.shots_against.sum() + 300)
        if not np.isclose(r.opp_goalie_sv_l40, expect):
            bad.append((r.game_id, r.opponent, r.opp_goalie_sv_l40, expect))
    assert not bad, f"{len(bad)} mismatches, first: {bad[:5]}"


def test_lineup_features_use_only_pregame_form(n=200, seed=13):
    """Recompute lineup features for random team-games from the stored pregame columns."""
    f, _ = _load()
    keys = f[["game_id", "team"]].drop_duplicates().sample(n, random_state=seed)
    for gid, team in keys.itertuples(index=False, name=None):
        d = f[(f.game_id == gid) & (f.team == team)]
        assert np.isclose(d.team_dressed_pts.iloc[0], d.ewm_points.fillna(0).sum()), (gid, team)
        for grp, dd in d.groupby("is_defense"):
            expect = dd.ewm_toi.rank(ascending=False, method="first")
            # rank order must follow pregame ice time; ties may break either way
            assert (dd.sort_values("toi_rank_pos").ewm_toi.diff().dropna() <= 1e-12).all(), (gid, team, grp)
        assert d.pp_rank.notna().sum() == d.ewm_toi_pp.notna().sum()


def test_no_feature_reproduces_target():
    f, _ = _load()
    feats = [c for c in f.columns if c.startswith(("ewm_", "l10_", "std_", "car_", "prev_", "team_", "opp_"))]
    feats += ["toi_rank_pos", "pp_rank", "linemates_pts"]
    for target in ["shots", "points"]:
        corr = f[feats].corrwith(f[target]).abs().max()
        assert corr < 0.9, f"a feature correlates {corr:.3f} with same-game {target}"


if __name__ == "__main__":
    for t in [test_player_features_use_only_prior_games, test_team_context_uses_only_prior_games,
              test_opposing_goalie_uses_only_prior_appearances, test_lineup_features_use_only_pregame_form,
              test_no_feature_reproduces_target]:
        t()
        print("PASS", t.__name__)
