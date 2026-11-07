"""Leakage test for data/features/skater_features.parquet.

For a random sample of player-games, recompute features from scratch using
only that player's (or team's) rows dated strictly before the game, and
check they equal the stored values. Also checks that no feature is
perfectly predictive of the same-game target.

Run: python3 tests/test_skater_feature_leakage.py   (or pytest)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    tg = pd.read_parquet(f"{RUN}/official_team_games.parquet")
    teams = pd.read_parquet(f"{RUN}/games_schedule.parquet")[["home_team_id", "home_abbrev"]].drop_duplicates()
    tg["team"] = tg.team_id.map(dict(zip(teams.home_team_id, teams.home_abbrev)))
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
    gl = pd.read_parquet(f"{RUN}/goalie_history_candidates.parquet")
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


def test_no_feature_reproduces_target():
    f, _ = _load()
    feats = [c for c in f.columns if c.startswith(("ewm_", "l10_", "std_", "car_", "prev_", "team_", "opp_"))]
    for target in ["shots", "points"]:
        corr = f[feats].corrwith(f[target]).abs().max()
        assert corr < 0.9, f"a feature correlates {corr:.3f} with same-game {target}"


if __name__ == "__main__":
    for t in [test_player_features_use_only_prior_games, test_team_context_uses_only_prior_games,
              test_opposing_goalie_uses_only_prior_appearances, test_no_feature_reproduces_target]:
        t()
        print("PASS", t.__name__)
