"""As-of skater features for the shots-on-goal and points models.

Every feature for a player-game uses only games played before that game's
date (pandas shift(1) inside each player's or team's history). Tested in
tests/test_skater_feature_leakage.py.

Input:  data/clean/skater_games.parquet, v2 official_team_games.parquet
Output: data/features/skater_features.parquet (one row per player-game,
        targets plus features)

Player features (prior games only)
  ewm_<x>      exponentially weighted mean, half-life 10 games
  l10_<x>      mean of the last 10 games
  std_<x>      season-to-date mean (NaN for a player's first game of a season)
  car_<x>      mean over all prior games in the data (career to date, from 2019-20)
  for x in shots, toi, toi_pp, goals, assists, points, pp_points
  shots_per60_ewm, points_per60_ewm (from the EWMA sums)
  prev_season_shots_pg, prev_season_points_pg, prev_season_toi
  n_prior, n_prior_season, days_rest, back_to_back
Team context (team-game level, prior games only, last 20)
  team_sog_for_l20, opp_sog_against_l20, opp_goals_against_l20, team_goals_for_l20
Opposing goalie (the goalie who started for the opponent; see gap G-E)
  opp_goalie_sv_l40  save percentage over that goalie's previous 40
                     appearances, shrunk toward .905 with 300 pseudo-shots
Static: is_home, is_defense, is_playoff, season
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/data/normalized/official_incremental/20261103T160656Z_4414a3fb")
STATS = ["shots", "toi", "toi_pp", "goals", "assists", "points", "pp_points"]
HALF_LIFE = 10
PRIOR_SV, PRIOR_SHOTS = 0.905, 300


def build() -> pd.DataFrame:
    g = pd.read_parquet(os.path.join(ROOT, "data/clean/skater_games.parquet"))
    g["game_date"] = pd.to_datetime(g.game_date)
    g = g.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    by = g.groupby("player_id", sort=False)

    f = g[["game_id", "player_id", "player_name", "team", "opponent", "game_date", "season", "game_type",
           "is_home", "position"] + STATS].copy()
    f["is_defense"] = (g.position == "D").astype(int)
    f["is_playoff"] = (g.game_type == 3).astype(int)
    f["n_prior"] = by.cumcount()
    f["n_prior_season"] = g.groupby(["player_id", "season"]).cumcount()
    f["days_rest"] = by.game_date.diff().dt.days
    f["back_to_back"] = (f.days_rest == 1).astype(int)

    for x in STATS:
        prev = by[x].shift(1)
        f[f"ewm_{x}"] = prev.groupby(g.player_id).transform(lambda s: s.ewm(halflife=HALF_LIFE, ignore_na=True).mean())
        f[f"l10_{x}"] = prev.groupby(g.player_id).transform(lambda s: s.rolling(10, min_periods=1).mean())
        f[f"car_{x}"] = prev.groupby(g.player_id).transform(lambda s: s.expanding().mean())
        sprev = g.groupby(["player_id", "season"])[x].shift(1)
        f[f"std_{x}"] = sprev.groupby([g.player_id, g.season]).transform(lambda s: s.expanding().mean())
    f["shots_per60_ewm"] = 60 * f.ewm_shots / f.ewm_toi.replace(0, np.nan)
    f["points_per60_ewm"] = 60 * f.ewm_points / f.ewm_toi.replace(0, np.nan)

    # Previous full season per game (regular season and playoffs).
    ps = g.groupby(["player_id", "season"]).agg(prev_season_shots_pg=("shots", "mean"),
                                                 prev_season_points_pg=("points", "mean"),
                                                 prev_season_toi=("toi", "mean")).reset_index()
    ps["season"] = ps.season + 10001  # 20242025 -> 20252026
    f = f.merge(ps, on=["player_id", "season"], how="left")

    # Team context from official team-game totals, prior 20 games.
    tg = pd.read_parquet(f"{RUN}/official_team_games.parquet")
    teams = pd.read_parquet(f"{RUN}/games_schedule.parquet")[["home_team_id", "home_abbrev"]].drop_duplicates()
    abbrev = dict(zip(teams.home_team_id, teams.home_abbrev))
    tg = tg[tg.game_type.isin([2, 3]) & tg.final_score_for.notna()].copy()
    tg["team"] = tg.team_id.map(abbrev)
    tg = tg.sort_values(["team", "scheduled_utc", "game_id"])
    tb = tg.groupby("team", sort=False)
    for src, name in [("sog_for", "sog_for_l20"), ("sog_against", "sog_against_l20"),
                      ("real_goals_against", "goals_against_l20"), ("real_goals_for", "goals_for_l20")]:
        tg[name] = tb[src].shift(1).groupby(tg.team).transform(lambda s: s.rolling(20, min_periods=5).mean())
    team = tg[["game_id", "team", "sog_for_l20", "sog_against_l20", "goals_against_l20", "goals_for_l20"]]
    f = f.merge(team[["game_id", "team", "sog_for_l20", "goals_for_l20"]].rename(
                    columns={"sog_for_l20": "team_sog_for_l20", "goals_for_l20": "team_goals_for_l20"}),
                on=["game_id", "team"], how="left")
    f = f.merge(team.rename(columns={"team": "opponent", "sog_against_l20": "opp_sog_against_l20",
                                     "goals_against_l20": "opp_goals_against_l20"})[
                    ["game_id", "opponent", "opp_sog_against_l20", "opp_goals_against_l20"]],
                on=["game_id", "opponent"], how="left")
    # Opposing starting goalie: save percentage over his previous 40 appearances.
    gl = pd.read_parquet(f"{RUN}/goalie_history_candidates.parquet")
    gl = gl[gl.participated].sort_values(["player_id", "scheduled_utc", "game_id"]).copy()
    gb = gl.groupby("player_id", sort=False)
    for c in ["saves", "shots_against"]:
        gl[f"{c}_l40"] = gb[c].shift(1).groupby(gl.player_id).transform(lambda s: s.rolling(40, min_periods=1).sum()).fillna(0)
    gl["goalie_sv_l40"] = (gl.saves_l40 + PRIOR_SV * PRIOR_SHOTS) / (gl.shots_against_l40 + PRIOR_SHOTS)
    starters = gl[gl.starter][["game_id", "team_abbrev", "goalie_sv_l40"]].rename(
        columns={"team_abbrev": "opponent", "goalie_sv_l40": "opp_goalie_sv_l40"})
    f = f.merge(starters, on=["game_id", "opponent"], how="left")
    return f


if __name__ == "__main__":
    f = build()
    os.makedirs(os.path.join(ROOT, "data/features"), exist_ok=True)
    f.to_parquet(os.path.join(ROOT, "data/features/skater_features.parquet"), index=False)
    print(f.shape)
    print("team context missing:", f[["team_sog_for_l20", "opp_sog_against_l20"]].isna().mean().round(4).to_dict())
    print(f[f.n_prior >= 20].describe().T[["mean", "min", "max"]].round(3).to_string())
