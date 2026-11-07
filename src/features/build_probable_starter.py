"""Probable-starter model: who will start in goal, from information before the game.

The game models use the goalie who actually started (gap G-E). Before an
announcement a bettor does not know that, so this model gives each team's
candidate goalies a start probability from earlier games only, and the game
model variant `probable` uses the probability-weighted goalie features.

Candidates for a team-game: every goalie who dressed for the team in its
previous 10 games, plus an "other" option (a goalie not seen in that window:
a call-up, a trade, a return from injury). Candidate features, all from
earlier games:
  start_share_l5, start_share_l10  share of the team's last 5 / 10 games he started
  season_start_share   share of the team's games this season he started
  started_last         started the team's previous game
  dressed_last         dressed for the team's previous game
  b2b_started_last     started_last x the team is on the second night of a back-to-back
  streak               consecutive team starts up to the previous game (capped at 10)
  days_since_start     days since his previous start for any team (capped at 30)
  sv_l40               save % over his previous 40 appearances, shrunk to .905
                       (these two use only appearances that started 12 hours or
                       more before this game, so a game for another team the
                       same evening is never counted)
  is_other             1 for the "other" option only (its other features are 0)

Model: conditional logit (softmax over a team-game's candidates), fitted by
maximum likelihood with a small ridge penalty. Walk-forward by season: each
season is predicted by a model fitted on earlier seasons only. 2019-20 has
no earlier season in the goalie table, so its probabilities come from a
model fitted on 2019-20 itself; they are used only as training rows for the
game model, never scored as a test season.

Outputs
  data/features/probable_starter_candidates.parquet  one row per candidate
  data/features/probable_starter_team.parquet        one row per team-game:
      p_top (probability of the most likely candidate), top_id, actual starter,
      exp_sv_l40 and exp_prior_apps (probability-weighted; "other" counts as
      .905 and 0 appearances)
  docs/models/probable_starter_results.json          accuracy and log loss by season
Tested in tests/test_probable_starter_leakage.py.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import official  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WINDOW = 10
PRIOR_SV, PRIOR_SHOTS = 0.905, 300
FEATURES = ["start_share_l5", "start_share_l10", "season_start_share", "started_last", "dressed_last",
            "b2b_started_last", "streak", "days_since_start", "sv_l40", "is_other"]
SEASONS = [20192020, 20202021, 20212022, 20222023, 20232024, 20242025, 20252026, 20262027]
RIDGE = 1e-3


def goalie_history() -> pd.DataFrame:
    """Every dressed goalie row (regular season and playoffs)."""
    gl = official.load_goalies()
    gl = gl[gl.game_type.isin([2, 3])].copy()
    gl["start"] = pd.to_datetime(gl.scheduled_utc, utc=True)
    return gl.sort_values(["player_id", "start", "game_id"])


def build_candidates(gl: pd.DataFrame | None = None) -> pd.DataFrame:
    gl = goalie_history() if gl is None else gl
    tg = (gl[["game_id", "team_id", "season", "game_type", "game_date", "start", "scheduled_utc"]]
          .drop_duplicates(["game_id", "team_id"]).sort_values(["team_id", "start", "game_id"]).reset_index(drop=True))
    tg["team_game_no"] = tg.groupby("team_id").cumcount()
    tg["prev_start_time"] = tg.groupby("team_id").start.shift(1)
    tg["b2b"] = ((tg.start - tg.prev_start_time).dt.total_seconds() / 86400 < 1.5).astype(int)
    tg["season_game_no"] = tg.groupby(["team_id", "season"]).cumcount()

    dressed = gl.merge(tg[["game_id", "team_id", "team_game_no"]], on=["game_id", "team_id"])
    dressed = dressed[["team_id", "team_game_no", "player_id", "starter", "season"]]
    starters = dressed[dressed.starter]

    # Every goalie's start times (any team), for days since previous start and streaks.
    starts_all = gl[gl.starter][["player_id", "start"]].sort_values("start")

    rows = []
    by_team = {t: d for t, d in tg.groupby("team_id")}
    dressed_by_team = {t: d for t, d in dressed.groupby("team_id")}
    start_by_team = {t: d for t, d in starters.groupby("team_id")}
    for team, games in by_team.items():
        dr = dressed_by_team[team]
        st = start_by_team[team]
        dr_idx = dr.groupby("team_game_no").player_id.apply(set).to_dict()
        st_idx = st.set_index("team_game_no").player_id.to_dict()
        for g in games.itertuples(index=False):
            n = g.team_game_no
            window = range(max(0, n - WINDOW), n)
            cand = set().union(*[dr_idx.get(i, set()) for i in window]) if n > 0 else set()
            last5 = [st_idx.get(i) for i in range(max(0, n - 5), n)]
            last10 = [st_idx.get(i) for i in window]
            season_prev = range(n - g.season_game_no, n)
            season_starts = [st_idx.get(i) for i in season_prev]
            prev_starter = st_idx.get(n - 1) if n > 0 else None
            prev_dressed = dr_idx.get(n - 1, set()) if n > 0 else set()
            for pid in cand:
                streak = 0
                for i in range(n - 1, max(-1, n - 11), -1):
                    if st_idx.get(i) == pid:
                        streak += 1
                    else:
                        break
                rows.append((g.game_id, team, g.season, g.game_date, g.start, pid,
                             last5.count(pid) / max(len(last5), 1), last10.count(pid) / max(len(last10), 1),
                             season_starts.count(pid) / len(season_starts) if season_starts else 0.0,
                             int(prev_starter == pid), int(pid in prev_dressed), int(prev_starter == pid) * g.b2b,
                             min(streak, 10), 0))
            rows.append((g.game_id, team, g.season, g.game_date, g.start, -1, 0.0, 0.0, 0.0, 0, 0, 0, 0, 1))
    c = pd.DataFrame(rows, columns=["game_id", "team_id", "season", "game_date", "start", "player_id", "start_share_l5",
                                    "start_share_l10", "season_start_share", "started_last", "dressed_last",
                                    "b2b_started_last", "streak", "is_other"])

    # Days since the candidate's previous start (any team) and as-of save %:
    # merge_asof on start time with strictly earlier rows.
    # The lookup key is 12 hours before the start, so a goalie's game for another
    # team earlier the same evening (still in progress) is never counted.
    real = c[c.player_id >= 0].assign(asof=lambda d: d.start - pd.Timedelta(hours=12)).sort_values("asof")
    sa = starts_all.rename(columns={"start": "prev_start"}).sort_values("prev_start")
    real = pd.merge_asof(real, sa, left_on="asof", right_on="prev_start", by="player_id",
                         allow_exact_matches=True, direction="backward")
    real["days_since_start"] = ((real.start - real.prev_start).dt.total_seconds() / 86400).fillna(30).clip(upper=30)
    # As-of appearance stats: the latest appearance strictly before this start,
    # with that appearance's own result added to its before-game sums.
    plx = gl[gl.participated][["player_id", "game_id", "start", "saves", "shots_against"]].sort_values(["player_id", "start"])
    gb = plx.groupby("player_id", sort=False)
    plx["saves_incl"] = gb.saves.transform(lambda s: s.rolling(40, min_periods=1).sum())
    plx["shots_incl"] = gb.shots_against.transform(lambda s: s.rolling(40, min_periods=1).sum())
    plx["apps_incl"] = gb.cumcount() + 1
    look = plx[["player_id", "start", "saves_incl", "shots_incl", "apps_incl"]].rename(columns={"start": "app_start"})
    real = pd.merge_asof(real.sort_values("asof"), look.sort_values("app_start"), left_on="asof", right_on="app_start",
                         by="player_id", allow_exact_matches=True, direction="backward")
    real["sv_l40"] = (real.saves_incl.fillna(0) + PRIOR_SV * PRIOR_SHOTS) / (real.shots_incl.fillna(0) + PRIOR_SHOTS)
    real["prior_apps"] = real.apps_incl.fillna(0)
    other = c[c.player_id < 0].assign(days_since_start=0.0, sv_l40=0.0, prior_apps=0.0)
    out = pd.concat([real.drop(columns=["asof", "prev_start", "app_start", "saves_incl", "shots_incl", "apps_incl"]), other])

    actual = gl[gl.starter][["game_id", "team_id", "player_id"]].rename(columns={"player_id": "actual_id"})
    out = out.merge(actual, on=["game_id", "team_id"], how="left", validate="many_to_one")
    in_set = out[out.player_id >= 0].groupby(["game_id", "team_id"]).apply(
        lambda d: (d.player_id == d.actual_id).any(), include_groups=False).rename("actual_in_set")
    out = out.merge(in_set.reset_index(), on=["game_id", "team_id"], how="left")
    out["actual_in_set"] = out.actual_in_set.fillna(False).astype(bool)
    out["chosen"] = np.where(out.player_id >= 0, out.player_id == out.actual_id, ~out.actual_in_set).astype(int)
    return out.sort_values(["start", "game_id", "team_id", "player_id"]).reset_index(drop=True)


def _design(c: pd.DataFrame, mean=None, sd=None):
    X = c[FEATURES].to_numpy(float)
    if mean is None:
        real = c.is_other.to_numpy() == 0
        mean, sd = X[real].mean(0), X[real].std(0)
        mean[FEATURES.index("is_other")], sd[FEATURES.index("is_other")] = 0.0, 1.0
        sd[sd == 0] = 1.0
    Xs = (X - mean) / sd
    # The "other" option keeps zeros in every real feature (only its own dummy).
    Xs[c.is_other.to_numpy() == 1] = 0.0
    Xs[c.is_other.to_numpy() == 1, FEATURES.index("is_other")] = 1.0
    return Xs, mean, sd


def _groups(c: pd.DataFrame) -> np.ndarray:
    key = c.game_id.astype(np.int64) * 100 + c.team_id.astype(np.int64)
    _, g = np.unique(key.to_numpy(), return_inverse=True)
    return g


def fit_clogit(c: pd.DataFrame):
    X, mean, sd = _design(c)
    g, y = _groups(c), c.chosen.to_numpy(float)
    ng = g.max() + 1

    def nll(b):
        u = X @ b
        umax = np.full(ng, -np.inf)
        np.maximum.at(umax, g, u)
        e = np.exp(u - umax[g])
        den = np.bincount(g, e, ng)
        ll = (y * (u - umax[g])).sum() - np.log(den).sum()
        p = e / den[g]
        grad = X.T @ (y - p)
        return -ll + RIDGE * b @ b, -grad + 2 * RIDGE * b

    res = minimize(nll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    return {"coef": res.x, "mean": mean, "sd": sd, "converged": bool(res.success)}


def predict(model, c: pd.DataFrame) -> np.ndarray:
    X, _, _ = _design(c, model["mean"], model["sd"])
    g = _groups(c)
    u = X @ model["coef"]
    umax = np.full(g.max() + 1, -np.inf)
    np.maximum.at(umax, g, u)
    e = np.exp(u - umax[g])
    return e / np.bincount(g, e)[g]


def walk_forward(c: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parts, coefs = [], {}
    for s in SEASONS:
        tr = c[c.season < s] if s != SEASONS[0] else c[c.season == s]
        m = fit_clogit(tr)
        te = c[c.season == s].copy()
        te["p"] = predict(m, te)
        te["in_sample"] = s == SEASONS[0]
        parts.append(te)
        coefs[str(s)] = {f: round(float(b), 4) for f, b in zip(FEATURES, m["coef"])} | {"converged": m["converged"]}
    return pd.concat(parts), coefs


def team_level(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["sv_val"] = np.where(c.is_other == 1, PRIOR_SV, c.sv_l40)
    c["apps_val"] = np.where(c.is_other == 1, 0.0, c.prior_apps)
    c["w_sv"], c["w_apps"] = c.p * c.sv_val, c.p * c.apps_val
    key = ["game_id", "team_id"]
    agg = c.groupby(key).agg(season=("season", "first"), game_date=("game_date", "first"), start=("start", "first"), in_sample=("in_sample", "first"),
                             exp_sv_l40=("w_sv", "sum"), exp_prior_apps=("w_apps", "sum"),
                             n_candidates=("is_other", lambda s: int((s == 0).sum())),
                             p_actual=("p", lambda s: float(s[c.loc[s.index, "chosen"] == 1].sum())),
                             actual_id=("actual_id", "first"), actual_in_set=("actual_in_set", "first"))
    top = c.sort_values("p", ascending=False).drop_duplicates(key).set_index(key)[["player_id", "p"]]
    agg = agg.join(top.rename(columns={"player_id": "top_id", "p": "p_top"}))
    agg["top_correct"] = (agg.top_id == agg.actual_id) & agg.actual_in_set
    last = c[c.started_last == 1].drop_duplicates(key).set_index(key).player_id.rename("last_starter_id")
    agg = agg.join(last)
    agg["last_starter_correct"] = agg.last_starter_id == agg.actual_id
    # Baseline 2: the goalie with the most starts in the team's last 10 games
    # (ties go to the one who started more recently, then the previous game's starter).
    mode = (c[c.is_other == 0].sort_values(["start_share_l10", "start_share_l5", "started_last"], ascending=False)
            .drop_duplicates(key).set_index(key).player_id.rename("most_starts_id"))
    agg = agg.join(mode)
    agg["most_starts_correct"] = agg.most_starts_id == agg.actual_id
    b2b = c[c.b2b_started_last == 1].drop_duplicates(key).set_index(key).b2b_started_last.rename("b2b_flag")
    agg = agg.join(b2b)
    agg["b2b_flag"] = agg.b2b_flag.fillna(0).astype(int)
    return agg.reset_index()


def results(t: pd.DataFrame) -> dict:
    out = {}
    for s, d in t.groupby("season"):
        row = {"team_games": len(d), "in_sample": bool(d.in_sample.iloc[0]),
               "actual_in_candidate_set": round(float(d.actual_in_set.mean()), 4),
               "top_pick_accuracy": round(float(d.top_correct.mean()), 4),
               "last_starter_baseline_accuracy": round(float(d.last_starter_correct.mean()), 4),
               "most_starts_l10_baseline_accuracy": round(float(d.most_starts_correct.mean()), 4),
               "log_loss": round(float(-np.log(np.clip(d.p_actual, 1e-6, 1)).mean()), 4),
               "mean_p_top": round(float(d.p_top.mean()), 4)}
        b = d[d.b2b_flag == 1]
        row["after_starting_night_one_of_b2b"] = {"team_games": len(b),
                                                  "top_pick_accuracy": round(float(b.top_correct.mean()), 4) if len(b) else None,
                                                  "last_starter_baseline_accuracy": round(float(b.last_starter_correct.mean()), 4) if len(b) else None,
                                                  "most_starts_l10_baseline_accuracy": round(float(b.most_starts_correct.mean()), 4) if len(b) else None}
        out[str(s)] = row
    return out


if __name__ == "__main__":
    c = build_candidates()
    c, coefs = walk_forward(c)
    c.to_parquet(os.path.join(ROOT, "data/features/probable_starter_candidates.parquet"), index=False)
    t = team_level(c)
    t.to_parquet(os.path.join(ROOT, "data/features/probable_starter_team.parquet"), index=False)
    res = {"features": FEATURES, "window_team_games": WINDOW, "coefficients_standardized": coefs, "by_season": results(t)}
    oos = t[~t.in_sample]
    res["out_of_sample_total"] = {"team_games": len(oos), "top_pick_accuracy": round(float(oos.top_correct.mean()), 4),
                                  "last_starter_baseline_accuracy": round(float(oos.last_starter_correct.mean()), 4),
                                  "most_starts_l10_baseline_accuracy": round(float(oos.most_starts_correct.mean()), 4),
                                  "log_loss": round(float(-np.log(np.clip(oos.p_actual, 1e-6, 1)).mean()), 4)}
    json.dump(res, open(os.path.join(ROOT, "docs/models/probable_starter_results.json"), "w"), indent=2)
    print(json.dumps(res["by_season"], indent=1))
    print(res["out_of_sample_total"])
    print(json.dumps(coefs["20262027"]))
