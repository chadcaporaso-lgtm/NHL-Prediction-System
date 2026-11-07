"""Team Elo ratings, built from scratch and tuned walk-forward.

Results
  2018-19: MoneyPuck team game results (warm-up only; shootout winners are
           not recorded there, so shootout games count as draws)
  2019-20 onward: official results (common.official), real goals for the margin
Arizona's rating continues into Utah. Seattle starts at the league mean.

Rating update (per game, after it is played)
  p_home   = 1 / (1 + 10 ** (-(R_home + HFA - R_away) / 400))
  S_home   = 1 for a regulation win, S_OT for an overtime or shootout win
             (and 1 - S_OT for the loser)
  standard K_eff = K * ln(|real goal margin| + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
           (a game tied on real goals, decided by shootout, uses the multiplier
           for a one-goal margin)
  flat     K_eff = K
  R_home  += K_eff * (S_home - p_home);  R_away -= the same amount
Between seasons every rating moves toward 1500: R = 1500 + C * (R - 1500).

A team's rating changes only when it plays, and every team plays at most once
a day, so a pre-game rating depends only on that team's and its opponents'
earlier games.

Tuning: every grid combination is run once over all games (online, so each
pre-game probability uses only earlier games). For each test season S, the
combination with the lowest moneyline log loss over seasons 2020-21 to S-1
is selected (2019-20 is burn-in). The selected run supplies season S's
pre-game ratings. Selection never looks at season S.

Outputs
  data/features/elo_games.parquet  one row per official game: pre-game ratings
                                   and home win probability, standard and flat
  docs/models/elo_results.json     selected parameters per season, log loss
  docs/models/elo_current.json     current ratings (power ranking)
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
# First grid: K 4-16, HFA 20-50, C 0.5-0.8. Selections landed on the HFA and C
# edges, so the grid was widened to the values below (2026-11-07).
GRID = {"K": [4, 6, 8, 10, 12, 16], "HFA": [0, 10, 20, 35], "C": [0.65, 0.8, 0.9, 1.0], "S_OT": [1.0, 0.75]}
FLAT_GRID = GRID
TEST_SEASONS = [20212022, 20222023, 20232024, 20242025, 20252026, 20262027]


def game_results() -> pd.DataFrame:
    """One row per game: season, start, home, away, real goals, decision, home win."""
    mp = pd.read_parquet(official.moneypuck_file())
    mp = mp[(mp.situation == "all") & (mp.season == 2018) & (mp.home_or_away == "HOME")].copy()
    mp["team"] = mp.team.replace(MP_TEAM)
    mp["opposingTeam"] = mp.opposingTeam.replace(MP_TEAM)
    warm = pd.DataFrame({
        "game_id": mp.gameId, "season": 20182019, "start": pd.to_datetime(mp.gameDate.astype(str)),
        "home": mp.team, "away": mp.opposingTeam, "hg": mp.goalsFor.astype(int), "ag": mp.goalsAgainst.astype(int)})
    warm["decision"] = np.where(warm.hg == warm.ag, "SO", "REG")  # one-goal OT games look like REG here
    warm["home_win"] = np.where(warm.hg > warm.ag, 1.0, np.where(warm.hg < warm.ag, 0.0, 0.5))

    tg = official.load_team_games()
    tg = tg[tg.game_type.isin([2, 3]) & tg.final_score_for.notna()]
    ab = official.team_abbrev()
    h = tg[tg.is_home == 1]
    off = pd.DataFrame({
        "game_id": h.game_id, "season": h.season, "start": pd.to_datetime(h.scheduled_utc, utc=True).dt.tz_localize(None),
        "home": h.team_id.map(ab), "away": h.opponent_id.map(ab),
        "hg": h.real_goals_for.astype(int), "ag": h.real_goals_against.astype(int),
        "decision": h.decision_type, "home_win": h.won_final.astype(bool).astype(float)})
    g = pd.concat([warm, off], ignore_index=True)
    for c in ["home", "away"]:
        g[c] = g[c].replace(CONTINUES)
    return g.sort_values(["start", "game_id"]).reset_index(drop=True)


def run(g: pd.DataFrame, K, HFA, C, S_OT, mov=True):
    """Online Elo over all games. Returns pre-game home and away ratings and p_home."""
    R: dict[str, float] = {}
    season = None
    n = len(g)
    rh, ra, ph = np.empty(n), np.empty(n), np.empty(n)
    home, away, hg, ag = g.home.values, g.away.values, g.hg.values, g.ag.values
    dec, hw, seas = g.decision.values, g.home_win.values, g.season.values
    for i in range(n):
        if seas[i] != season:
            if season is not None:
                for t in R:
                    R[t] = 1500 + C * (R[t] - 1500)
            season = seas[i]
        a, b = R.get(home[i], 1500.0), R.get(away[i], 1500.0)
        p = 1 / (1 + 10 ** (-(a + HFA - b) / 400))
        rh[i], ra[i], ph[i] = a, b, p
        if hw[i] == 0.5:
            s = 0.5
        elif dec[i] in ("OT", "SO"):
            s = S_OT if hw[i] == 1 else 1 - S_OT
        else:
            s = hw[i]
        k = K
        if mov:
            margin = max(abs(hg[i] - ag[i]), 1)
            win_diff = (a + HFA - b) if hw[i] >= 0.5 else (b - a - HFA)
            k = K * np.log(margin + 1) * 2.2 / (win_diff * 0.001 + 2.2)
        d = k * (s - p)
        R[home[i]], R[away[i]] = a + d, b - d
    return rh, ra, ph, R


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune(g: pd.DataFrame, grid: dict, mov: bool):
    runs = {}
    y = g.home_win.values
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        rh, ra, ph, R = run(g, mov=mov, **params)
        ll = {s: logloss(ph[(g.season == s).values & (y != 0.5)], y[(g.season == s).values & (y != 0.5)])
              for s in g.season.unique() if s >= 20192020}
        runs[combo] = {"params": params, "rh": rh, "ra": ra, "ph": ph, "R": R, "ll": ll}
    chosen = {}
    for s in TEST_SEASONS:
        prior = [x for x in range(20202021, s, 10001)]
        best = min(runs, key=lambda c: np.mean([runs[c]["ll"][p] for p in prior]))
        chosen[s] = best
    # seasons before the first test season use the 2021-22 choice (burn-in only)
    return runs, chosen


if __name__ == "__main__":
    g = game_results()
    out = g[g.season >= 20192020][["game_id", "season", "start", "home", "away", "home_win", "decision"]].copy()
    res = {"games": int(len(out)), "warmup_games_2018_19": int((g.season == 20182019).sum())}
    for name, grid, mov in [("std", GRID, True), ("flat", FLAT_GRID, False)]:
        runs, chosen = tune(g, grid, mov)
        idx = g.season >= 20192020
        for col in ["rh", "ra", "ph"]:
            out[f"{name}_{col}"] = np.nan
        for s in sorted(out.season.unique()):
            c = chosen.get(s, chosen[TEST_SEASONS[0]])
            m = (g.season == s).values
            mo = (out.season == s).values
            for col in ["rh", "ra", "ph"]:
                out.loc[mo, f"{name}_{col}"] = runs[c][col][m]
        res[name] = {
            "selected": {str(s): runs[c]["params"] for s, c in chosen.items()},
            "test_log_loss": {str(s): round(runs[c]["ll"][s], 4) for s, c in chosen.items()},
        }
        if name == "std":
            last = chosen[TEST_SEASONS[-1]]
            cur = pd.Series(runs[last]["R"]).sort_values(ascending=False)
            active = set(out[out.season == 20262027].home) | set(out[out.season == 20262027].away)
            cur = cur[cur.index.isin(active)]
            json.dump({"as_of": str(out.start.max().date()), "params": runs[last]["params"],
                       "ratings": {t: round(float(v), 1) for t, v in cur.items()}},
                      open(os.path.join(ROOT, "docs/models/elo_current.json"), "w"), indent=2)
    y = out.home_win.values
    res["constant_log_loss"] = {str(s): round(logloss(np.full(int(((out.season == s) & (out.home_win != 0.5)).sum()),
                                    y[(out.season.values < s) & (y != 0.5)].mean() if s > 20192020 else 0.5),
                                    y[(out.season.values == s) & (y != 0.5)]), 4) for s in TEST_SEASONS}
    out.to_parquet(os.path.join(ROOT, "data/features/elo_games.parquet"), index=False)
    json.dump(res, open(os.path.join(ROOT, "docs/models/elo_results.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))
