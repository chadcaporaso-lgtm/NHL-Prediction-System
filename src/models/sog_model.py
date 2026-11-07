"""Shots-on-goal model: walk-forward by season, graded on all games and on
SportsGameOdds closing prop prices.

Models (all predict the mean shot count mu for a player-game)
  baseline  mu = player's exponentially weighted shots per game (no fitting)
  glm       Poisson GLM, log link, standardized features
  gbm       Histogram gradient boosting with Poisson loss
Distribution: negative binomial, Var = mu + alpha * mu^2, alpha fitted by
maximum likelihood on the training seasons only. Over probability for a
line L: P(X > L); whole-number lines are graded with pushes removed.

Walk-forward: for each test season S from 2021-22, train on every season
before S. No information from S or later is used to fit anything.

Outputs
  data/models/sog_predictions.parquet   every test player-game with mu per model
  docs/models/sog_results.json          metrics per season and model, market comparison
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_SEASONS = [20212022, 20222023, 20232024, 20242025, 20252026, 20262027]
MIN_PRIOR = 5  # player must have at least 5 earlier games in the data
FEATURES = [
    "ewm_shots", "l10_shots", "std_shots", "car_shots", "prev_season_shots_pg",
    "ewm_toi", "l10_toi", "std_toi", "ewm_toi_pp", "l10_toi_pp", "prev_season_toi",
    "shots_per60_ewm", "ewm_points", "n_prior_season",
    "is_defense", "is_home", "back_to_back", "is_playoff",
    "team_sog_for_l20", "opp_sog_against_l20",
]


def prepare(f: pd.DataFrame) -> pd.DataFrame:
    f = f[f.n_prior >= MIN_PRIOR].copy()
    for x in ["shots", "toi", "toi_pp"]:
        f[f"std_{x}"] = f[f"std_{x}"].fillna(f[f"ewm_{x}"])
    f["prev_season_shots_pg"] = f.prev_season_shots_pg.fillna(f.car_shots)
    f["prev_season_toi"] = f.prev_season_toi.fillna(f.car_toi)
    for c in ["team_sog_for_l20", "opp_sog_against_l20"]:
        f[c] = f[c].fillna(30.0)
    f["shots_per60_ewm"] = f.shots_per60_ewm.fillna(f.shots_per60_ewm.median())
    return f


def nb_logpmf(y, mu, alpha):
    r = 1.0 / alpha
    return stats.nbinom.logpmf(y, r, r / (r + mu))


def fit_alpha(y, mu):
    res = optimize.minimize_scalar(lambda a: -nb_logpmf(y, mu, a).sum(), bounds=(1e-4, 2.0), method="bounded")
    return float(res.x)


def p_over(line, mu, alpha):
    """P(X > line) and P(X == line) under the negative binomial."""
    r = 1.0 / alpha
    p = r / (r + mu)
    k = np.floor(line)
    over = stats.nbinom.sf(k, r, p)
    push = np.where(line == k, stats.nbinom.pmf(k, r, p), 0.0)
    return over, push


def american_to_prob(a):
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def american_payout(a):
    a = np.asarray(a, dtype=float)
    return np.where(a < 0, 100 / -a, a / 100)


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def walk_forward(f):
    preds, alphas = [], {}
    for s in TEST_SEASONS:
        tr, te = f[f.season < s], f[f.season == s].copy()
        glm = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=1000)).fit(tr[FEATURES], tr.shots)
        gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                            min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(tr[FEATURES], tr.shots)
        te["mu_baseline"] = te.ewm_shots.clip(lower=0.05)
        te["mu_glm"] = glm.predict(te[FEATURES])
        te["mu_gbm"] = gbm.predict(te[FEATURES])
        alphas[s] = {}
        for m, est in [("baseline", None), ("glm", glm), ("gbm", gbm)]:
            mu_tr = tr.ewm_shots.clip(lower=0.05) if est is None else est.predict(tr[FEATURES])
            alphas[s][m] = fit_alpha(tr.shots.values, np.asarray(mu_tr))
        preds.append(te)
        print(s, "train", len(tr), "test", len(te), "alpha", {k: round(v, 4) for k, v in alphas[s].items()})
    return pd.concat(preds, ignore_index=True), alphas


def score_all_games(p, alphas):
    out = {}
    for s, d in p.groupby("season"):
        row = {"player_games": len(d), "mean_shots": round(float(d.shots.mean()), 4)}
        for m in ["baseline", "glm", "gbm"]:
            mu = d[f"mu_{m}"].values
            row[m] = {
                "nb_log_lik_per_game": round(float(nb_logpmf(d.shots.values, mu, alphas[s][m]).mean()), 5),
                "mae": round(float(np.abs(d.shots - mu).mean()), 4),
                "pred_zero_rate": round(float(np.exp(nb_logpmf(np.zeros(len(mu)), mu, alphas[s][m])).mean()), 4),
            }
        row["actual_zero_rate"] = round(float((d.shots == 0).mean()), 4)
        out[str(s)] = row
    return out


def score_market(p, alphas, edge=0.03):
    props = pd.read_parquet(os.path.join(ROOT, "data/sgo/player_props_matched.parquet"))
    props = props[(props.stat == "shots_onGoal") & props.playerId.notna()
                  & props.close_odds_over.notna() & props.close_odds_under.notna()
                  & (props.close_line_over == props.close_line_under)].copy()
    props["player_id"] = props.playerId.astype(int)
    props["game_id"] = props.game_id.astype(int)
    d = props.merge(p, on=["game_id", "player_id"], how="inner", suffixes=("_prop", ""))
    d["line"] = d.close_line_over.astype(float)
    po, pu = american_to_prob(d.close_odds_over), american_to_prob(d.close_odds_under)
    d["p_market"] = po / (po + pu)
    d["margin"] = po + pu - 1
    d = d[d.shots != d.line].copy()  # drop pushes on whole-number lines
    d["y"] = (d.shots > d.line).astype(int)
    res = {"lines": int(len(d)), "games": int(d.game_id.nunique()), "median_margin": round(float(d.margin.median()), 4),
           "over_rate": round(float(d.y.mean()), 4), "line_counts": {str(k): int(v) for k, v in d.line.value_counts().head(6).items()},
           "market_log_loss": round(logloss(d.p_market, d.y), 4),
           "constant_log_loss": round(logloss(np.full(len(d), d.y.mean()), d.y), 4), "models": {}, "by_season": {}}
    for m in ["baseline", "glm", "gbm"]:
        a = d.season.map({s: alphas[s][m] for s in alphas})
        over, push = p_over(d.line.values, d[f"mu_{m}"].values, a.values)
        d[f"p_{m}"] = over / (1 - push)
        # Bet the side with model edge over the de-vigged close, at the closing price.
        ev_over = d[f"p_{m}"] - d.p_market
        side_over = ev_over > edge
        side_under = ev_over < -edge
        win = np.where(side_over, d.y == 1, d.y == 0)
        pay = np.where(side_over, american_payout(d.close_odds_over), american_payout(d.close_odds_under))
        profit = np.where(win, pay, -1.0)
        bets = side_over | side_under
        pr = pd.Series(profit[bets]).groupby(d.game_id.values[bets]).sum()
        n = int(bets.sum())
        res["models"][m] = {
            "log_loss": round(logloss(d[f"p_{m}"], d.y), 4),
            "mean_p_over": round(float(d[f"p_{m}"].mean()), 4),
            f"bets_edge_{edge}": n,
            "roi": round(float(profit[bets].mean()), 4) if n else None,
            "roi_se_game_clustered": round(float(pr.std(ddof=1) * np.sqrt(len(pr)) / n), 4) if n > 1 else None,
            "over_bets": int(side_over.sum()), "under_bets": int(side_under.sum()),
        }
    for s, ds in d.groupby("season"):
        res["by_season"][str(s)] = {"lines": int(len(ds)), "market_log_loss": round(logloss(ds.p_market, ds.y), 4),
                                    **{f"{m}_log_loss": round(logloss(ds[f"p_{m}"], ds.y), 4) for m in ["baseline", "glm", "gbm"]}}
    d.to_parquet(os.path.join(ROOT, "data/models/sog_market_graded.parquet"), index=False)
    return res


if __name__ == "__main__":
    f = prepare(pd.read_parquet(os.path.join(ROOT, "data/features/skater_features.parquet")))
    p, alphas = walk_forward(f)
    p.to_parquet(os.path.join(ROOT, "data/models/sog_predictions.parquet"), index=False)
    results = {"features": FEATURES, "min_prior_games": MIN_PRIOR,
               "alpha": {str(k): v for k, v in alphas.items()},
               "all_games": score_all_games(p, alphas), "market": score_market(p, alphas)}
    json.dump(results, open(os.path.join(ROOT, "docs/models/sog_results.json"), "w"), indent=2)
    print(json.dumps({k: results[k] for k in ["all_games", "market"]}, indent=2))
