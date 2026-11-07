"""As-of team features for the game models (moneyline, puck line, totals).

One row per team-game (two per game) for every completed regular-season and
playoff game from 2019-20. Every feature uses only the team's (or goalie's)
earlier games: history is ordered by scheduled start and shifted by one game
before any rolling or weighted mean. MoneyPuck history from 2018-19 warms the
averages up so 2019-20 starts with information. Tested in
tests/test_game_feature_leakage.py.

Targets
  reg_goals_for / reg_goals_against  goals in regulation (an overtime winner
                                     is removed; shootouts add nothing)
  final_score_for / against, decision_type (REG, OT, SO), won_final

Team features (exponentially weighted means of previous games; half-life
20 games, plus a 60-game version of the core rates)
  xgf, xga             expected goals for and against per game, all situations
  gf, ga               real goals for and against per game
  sf, sa               shots on goal for and against per game
  cf_pct               shot-attempt share (Corsi), all situations
  xgf60_5v5, xga60_5v5 even-strength expected goal rates per 60
  pp_xgf60, pk_xga60   power-play and penalty-kill expected goal rates per 60
  pp_toi, pk_toi       power-play and shorthanded minutes per game
Goalie: starter_sv_l40 (the goalie who started; gap G-E), shrunk to .905
Schedule: days_rest, back_to_back, games_played_season
League: league_reg_gpg_l300  regulation goals per team-game over the last 300
        games played on earlier dates (tracks the scoring level, which shifts
        from season to season)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import official  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/data/normalized/official_incremental/20261103T160656Z_4414a3fb")
MP = os.path.join(ROOT, "data/drive/NHL_Prediction_System_Claude/01_raw/moneypuck/moneypuck_all_teams_2018_plus_2026-11-07.parquet")
MP_TEAM = {"N.J": "NJD", "T.B": "TBL", "S.J": "SJS", "L.A": "LAK"}
# Utah (2024-25 on) took over Arizona's roster, so Arizona's games continue
# Utah's history for the weighted means. Official game rows keep their own ids.
HISTORY_CONTINUES = {"ARI": "UTA"}
PRIOR_SV, PRIOR_SHOTS = 0.905, 300


def team_history() -> pd.DataFrame:
    """Per team-game stats from MoneyPuck, 2018-19 onward."""
    mp = pd.read_parquet(official.moneypuck_file())
    mp["team"] = mp.team.replace(MP_TEAM)
    key = ["gameId", "team"]
    a = mp[mp.situation == "all"].set_index(key)
    e = mp[mp.situation == "5on5"].set_index(key)
    pp = mp[mp.situation == "5on4"].set_index(key)
    pk = mp[mp.situation == "4on5"].set_index(key)
    h = pd.DataFrame({
        "game_date": pd.to_datetime(a.gameDate.astype(str)),
        "xgf": a.xGoalsFor, "xga": a.xGoalsAgainst, "gf": a.goalsFor, "ga": a.goalsAgainst,
        "sf": a.shotsOnGoalFor, "sa": a.shotsOnGoalAgainst,
        "cf_pct": a.shotAttemptsFor / (a.shotAttemptsFor + a.shotAttemptsAgainst),
        # Two MoneyPuck games record zero 5v5 ice time; their rates are missing, not infinite.
        "xgf60_5v5": 3600 * e.xGoalsFor / e.iceTime.replace(0, np.nan),
        "xga60_5v5": 3600 * e.xGoalsAgainst / e.iceTime.replace(0, np.nan),
        "pp_xgf60": 3600 * pp.xGoalsFor / pp.iceTime.replace(0, np.nan),
        "pk_xga60": 3600 * pk.xGoalsAgainst / pk.iceTime.replace(0, np.nan),
        "pp_toi": pp.iceTime / 60, "pk_toi": pk.iceTime / 60,
    }).reset_index().rename(columns={"gameId": "game_id"})
    h["hist_team"] = h.team.replace(HISTORY_CONTINUES)
    return h.sort_values(["hist_team", "game_date", "game_id"]).reset_index(drop=True)


RATE_COLS = ["xgf", "xga", "gf", "ga", "sf", "sa", "cf_pct", "xgf60_5v5", "xga60_5v5", "pp_xgf60", "pk_xga60", "pp_toi", "pk_toi"]


def add_team_form(h: pd.DataFrame) -> pd.DataFrame:
    """Weighted means of earlier games; missing rates (no power-play time) are skipped."""
    g = h.groupby("hist_team", sort=False)
    out = h[["game_id", "team"]].copy()
    for c in RATE_COLS:
        prev = g[c].shift(1)
        out[f"ewm20_{c}"] = prev.groupby(h.hist_team).transform(lambda s: s.ewm(halflife=20, ignore_na=True).mean())
    for c in ["xgf", "xga", "gf", "ga"]:
        prev = g[c].shift(1)
        out[f"ewm60_{c}"] = prev.groupby(h.hist_team).transform(lambda s: s.ewm(halflife=60, ignore_na=True).mean())
    return out


def goalie_starters() -> pd.DataFrame:
    gl = official.load_goalies()
    gl = gl[gl.participated].sort_values(["player_id", "scheduled_utc", "game_id"]).copy()
    gb = gl.groupby("player_id", sort=False)
    for c in ["saves", "shots_against"]:
        gl[f"{c}_l40"] = gb[c].shift(1).groupby(gl.player_id).transform(lambda s: s.rolling(40, min_periods=1).sum()).fillna(0)
    gl["starter_sv_l40"] = (gl.saves_l40 + PRIOR_SV * PRIOR_SHOTS) / (gl.shots_against_l40 + PRIOR_SHOTS)
    gl["starter_prior_apps"] = gb.cumcount()
    return gl[gl.starter][["game_id", "team_id", "player_id", "starter_sv_l40", "starter_prior_apps"]].rename(
        columns={"player_id": "starter_id"})


def league_scoring(tg: pd.DataFrame, n_games: int = 300) -> pd.DataFrame:
    """Per date: mean regulation goals per team-game over the last n games before that date."""
    games = tg[tg.is_home == 1][["game_id", "game_date", "reg_goals_for", "reg_goals_against"]].copy()
    games["goals"] = games.reg_goals_for + games.reg_goals_against
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    csum = np.concatenate([[0.0], games.goals.cumsum().values])
    out = []
    for d, idx in games.groupby("game_date").groups.items():
        end = idx.min()  # games before this date: positions [0, end)
        start = max(0, end - n_games)
        out.append((d, (csum[end] - csum[start]) / (2 * (end - start)) if end > start else np.nan))
    return pd.DataFrame(out, columns=["game_date", "league_reg_gpg_l300"])


def build() -> pd.DataFrame:
    tg = official.load_team_games()
    abbrev = official.team_abbrev()
    tg = tg[tg.game_type.isin([2, 3]) & tg.final_score_for.notna()].copy()
    tg["team"] = tg.team_id.map(abbrev)
    tg["opponent"] = tg.opponent_id.map(abbrev)
    ot_win = (tg.decision_type == "OT") & tg.won_final.astype(bool)
    ot_loss = (tg.decision_type == "OT") & ~tg.won_final.astype(bool)
    tg["reg_goals_for"] = tg.real_goals_for - ot_win.astype(int)
    tg["reg_goals_against"] = tg.real_goals_against - ot_loss.astype(int)
    tg["start"] = pd.to_datetime(tg.scheduled_utc, utc=True)
    tg = tg.sort_values(["team", "start", "game_id"])
    tb = tg.groupby("team", sort=False)
    tg["days_rest"] = tb.start.diff().dt.total_seconds() / 86400
    tg["back_to_back"] = (tg.days_rest < 1.5).astype(int)
    tg["games_played_season"] = tg.groupby(["team", "season"]).cumcount()

    tg = tg.merge(league_scoring(tg), on="game_date", how="left")
    form = add_team_form(team_history())
    tg = tg.merge(form, on=["game_id", "team"], how="left", validate="one_to_one")
    tg = tg.merge(goalie_starters(), on=["game_id", "team_id"], how="left", validate="one_to_one")

    keep = ["game_id", "season", "game_type", "game_date", "scheduled_utc", "team", "opponent", "team_id", "opponent_id",
            "is_home", "reg_goals_for", "reg_goals_against", "real_goals_for", "real_goals_against",
            "final_score_for", "final_score_against", "decision_type", "won_final",
            "days_rest", "back_to_back", "games_played_season", "league_reg_gpg_l300", "starter_id", "starter_sv_l40", "starter_prior_apps"]
    keep += [c for c in tg.columns if c.startswith(("ewm20_", "ewm60_"))]
    return tg[keep].sort_values(["scheduled_utc", "game_id", "is_home"]).reset_index(drop=True)


if __name__ == "__main__":
    f = build()
    f.to_parquet(os.path.join(ROOT, "data/features/game_team_features.parquet"), index=False)
    print(f.shape, "games", f.game_id.nunique())
    print("missing share:", f.isna().mean()[lambda s: s > 0].round(4).to_dict())
    print(f[f.team.isin(["ARI", "UTA"])].groupby("team").ewm20_xgf.apply(lambda x: x.isna().sum()).to_dict())
    print(f.groupby("season").agg(games=("game_id", "nunique"), reg_gf=("reg_goals_for", "mean"),
                                  ewm_xgf=("ewm20_xgf", "mean")).round(3).to_string())
