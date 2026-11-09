"""Opponent-adjusted team ratings, refitted before every game date (decision 41).

Why: the game model rates teams with exponentially weighted means of their
own raw results (build_game_features.py). Those means ignore who the games
were against and carry last season's form unchanged into October. Against the
average closing three-way prices of 2021-22 to 2025-26 the model trails the
market most in a team's first 10 games of a season (0.013 log loss per game,
SE 0.004) and its home/away split is narrower than the market's (SD of the
log odds 0.53 vs 0.57). Decision 39 located the gap in team rating.

Model (one per statistic; fitted separately for every game date D)
  y[team, game] = mu + home * is_home + off[team] + def[opponent] + noise
  fitted by weighted ridge regression on every team-game played before D
  (strictly earlier dates), weights
      w = 0.5 ** (age_days / H) * C ** (seasons between the game and D)
  and a ridge penalty LAMBDA on off and def (shrinks a team to league
  average when it has little recent data, e.g. Seattle in October 2021).
  off[team] is how many more of the statistic the team produces than an
  average team against an average opponent; def[team] is how many more it
  allows. Both are pre-game: a rating for a game on date D uses only games
  on earlier dates.
Statistics (MoneyPuck, all situations, per game)
  xg   flurry- and score-venue-adjusted expected goals
  g    goals (MoneyPuck's count: no shootout goals)
MoneyPuck history from 2018-19 warms the ratings up. Arizona's games continue
Utah's history (as in build_game_features.py).

Tuning: every grid combination (H, C, LAMBDA) is run over all dates. For each
game the one-step-ahead prediction mu + home + off + def is compared with the
actual value (squared error). For test season S the combination with the
lowest mean squared error over seasons 2020-21 to S-1 is selected (2019-20 is
burn-in), separately for each statistic. Selection never looks at season S.
Tested in tests/test_team_rating_leakage.py.

Outputs
  data/features/team_rating.parquet  one row per (official game, team):
      rating_off_xg, rating_def_xg, rating_off_g, rating_def_g
  docs/models/team_rating_results.json  selected parameters and errors
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import official  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MP_TEAM = {"N.J": "NJD", "T.B": "TBL", "S.J": "SJS", "L.A": "LAK"}
CONTINUES = {"ARI": "UTA"}
STATS = {"xg": "flurryScoreVenueAdjustedxGoalsFor", "g": "goalsFor"}
# First grid: H 30-240, C 0.3-1.0, LAMBDA 2-32. Goals selections landed on the H
# and LAMBDA edges and H = 30 was never chosen, so the grid was moved (2026-11-09).
GRID = {"H": [60, 120, 240, 480], "C": [0.3, 0.6, 1.0], "LAMBDA": [4.0, 8.0, 32.0, 96.0]}
TEST_SEASONS = [20212022, 20222023, 20232024, 20242025, 20252026, 20262027]
WINDOW_DAYS = 1100  # about three seasons back


def season_of(d: pd.Series) -> pd.Series:
    """NHL season id (e.g. 20212022) for a date: seasons start in July."""
    y = d.dt.year - (d.dt.month < 7).astype(int)
    return y * 10000 + y + 1


def team_games() -> pd.DataFrame:
    """One row per MoneyPuck team-game, all situations."""
    mp = pd.read_parquet(official.moneypuck_file())
    a = mp[mp.situation == "all"].copy()
    for c in ["team", "opposingTeam"]:
        a[c] = a[c].replace(MP_TEAM).replace(CONTINUES)
    out = pd.DataFrame({"game_id": a.gameId.astype(int), "date": pd.to_datetime(a.gameDate.astype(str)),
                        "team": a.team, "opp": a.opposingTeam, "is_home": (a.home_or_away == "HOME").astype(int)})
    for k, col in STATS.items():
        out[k] = a[col].astype(float).values
    out["season"] = season_of(out.date)
    return out.sort_values(["date", "game_id", "is_home"]).reset_index(drop=True)


def run(tg: pd.DataFrame, dates: np.ndarray, H: float, C: float, LAMBDA: float) -> dict:
    """Pre-game ratings for every date in `dates`, from games strictly before it.

    Returns {stat: (mu, home, off[date, team], def[date, team])} with teams in
    sorted order of tg.team.
    """
    teams = np.array(sorted(set(tg.team) | set(tg.opp)))
    ti = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    k = 2 + 2 * n
    gd = tg.date.values.astype("datetime64[D]")
    gs = tg.season.values
    t_idx, o_idx = tg.team.map(ti).values, tg.opp.map(ti).values
    home = tg.is_home.values.astype(float)
    ys = {s: tg[s].values for s in STATS}
    pen = np.zeros(k)
    pen[2:] = LAMBDA
    out = {s: (np.full(len(dates), np.nan), np.full(len(dates), np.nan), np.full((len(dates), n), np.nan),
               np.full((len(dates), n), np.nan)) for s in STATS}
    d_season = season_of(pd.Series(pd.to_datetime(dates))).values
    for j, d in enumerate(np.asarray(dates).astype("datetime64[D]")):
        lo, hi = np.searchsorted(gd, d - np.timedelta64(WINDOW_DAYS, "D")), np.searchsorted(gd, d)
        if hi - lo < 50:
            continue
        sl = slice(lo, hi)
        age = (d - gd[sl]).astype(float)
        nseason = (d_season[j] // 10000) - (gs[sl] // 10000)
        w = 0.5 ** (age / H) * C ** nseason
        m = hi - lo
        X = np.zeros((m, k))
        X[:, 0] = 1.0
        X[:, 1] = home[sl]
        X[np.arange(m), 2 + t_idx[sl]] = 1.0
        X[np.arange(m), 2 + n + o_idx[sl]] = 1.0
        XtW = X.T * w
        A = XtW @ X + np.diag(pen)
        for s in STATS:
            b = np.linalg.solve(A, XtW @ ys[s][sl])
            out[s][0][j], out[s][1][j] = b[0], b[1]
            out[s][2][j], out[s][3][j] = b[2:2 + n], b[2 + n:]
    out["_teams"] = teams
    return out


def one_step_errors(tg: pd.DataFrame, dates: np.ndarray, fit: dict) -> dict:
    """Squared error of mu + home + off[team] + def[opp] for each team-game (NaN before the first fit)."""
    ti = {t: i for i, t in enumerate(fit["_teams"])}
    j = np.searchsorted(dates.astype("datetime64[D]"), tg.date.values.astype("datetime64[D]"))
    t, o = tg.team.map(ti).values, tg.opp.map(ti).values
    errs = {}
    for s in STATS:
        mu, hm, off, de = fit[s]
        pred = mu[j] + hm[j] * tg.is_home.values + off[j, t] + de[j, o]
        errs[s] = (pred - tg[s].values) ** 2
    return errs


def tune(tg: pd.DataFrame, grid: dict = GRID):
    """Run every combination; per test season pick the lowest error over 2020-21 .. S-1."""
    dates = np.array(sorted(tg.date.unique()))
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    fits, mse = [], {s: [] for s in STATS}
    for c in combos:
        f = run(tg, dates, **c)
        e = one_step_errors(tg, dates, f)
        fits.append(f)
        for s in STATS:
            mse[s].append(pd.Series(e[s]).groupby(tg.season.values).mean())
    chosen = {s: {} for s in STATS}
    for s in STATS:
        tab = pd.DataFrame(mse[s])
        for S in TEST_SEASONS:
            cols = [c for c in tab.columns if 20202021 <= c < S]
            chosen[s][S] = int(tab[cols].mean(axis=1).idxmin())
    return dates, combos, fits, mse, chosen


def build():
    tg = team_games()
    dates, combos, fits, mse, chosen = tune(tg)
    # Official team-games (including any game newer than the MoneyPuck table).
    f = pd.read_parquet(os.path.join(ROOT, "data/features/game_team_features.parquet"),
                        columns=["game_id", "season", "game_date", "team", "opponent", "is_home"])
    f["d"] = pd.to_datetime(f.game_date.astype(str).str[:10])
    all_dates = np.array(sorted(set(dates.astype("datetime64[ns]")) | set(f.d.values)))
    out = f[["game_id", "team"]].copy()
    for s in STATS:
        for col in [f"rating_off_{s}", f"rating_def_{s}"]:
            out[col] = np.nan
        for S in sorted(set(f.season)):
            idx = chosen[s].get(S, chosen[s][TEST_SEASONS[0]]) if S >= TEST_SEASONS[0] else chosen[s][TEST_SEASONS[0]]
            rows = f.season == S
            fit = run(tg, np.array(sorted(f.loc[rows, "d"].unique())), **combos[idx])
            ti = {t: i for i, t in enumerate(fit["_teams"])}
            dd = np.array(sorted(f.loc[rows, "d"].unique()))
            j = np.searchsorted(dd, f.loc[rows, "d"].values)
            team = f.loc[rows, "team"].replace(CONTINUES).map(ti)
            opp_team = f.loc[rows, "opponent"].replace(CONTINUES).map(ti)
            ok = team.notna() & opp_team.notna()
            jj, tt = j[ok.values], team[ok].astype(int).values
            out.loc[rows[rows].index[ok.values], f"rating_off_{s}"] = fit[s][2][jj, tt]
            out.loc[rows[rows].index[ok.values], f"rating_def_{s}"] = fit[s][3][jj, tt]
    res = {"stats": STATS, "grid": GRID,
           "selected": {s: {str(S): combos[i] for S, i in chosen[s].items()} for s in STATS},
           "mse_by_season_selected": {s: {str(S): round(float(mse[s][i].get(S, np.nan)), 4) for S, i in chosen[s].items()}
                                      for s in STATS},
           "mse_constant_by_season": {s: {str(S): round(float(tg[tg.season == S][s].var()), 4) for S in TEST_SEASONS}
                                      for s in STATS}}
    return out, res


if __name__ == "__main__":
    out, res = build()
    out.to_parquet(os.path.join(ROOT, "data/features/team_rating.parquet"), index=False)
    json.dump(res, open(os.path.join(ROOT, "docs/models/team_rating_results.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))
    print(out.describe().round(3).to_string())
    print("missing share:", out.isna().mean().round(4).to_dict())
