"""Freeze the live-tracking rules (run once, 2026-11-07).

Shots on goal (decision 20)
  model      gbm from prop_model.py, trained on seasons before 2026-27
  blend      y ~ logit(p_market) + logit(p_gbm), fitted on every graded
             shots-on-goal line dated on or before 2026-11-06
  bet        |p_blend - p_market| > 0.04, the side the blend favors, at the
             consensus closing price; whole-number pushes are void
Puck line (decision 23)
  model      calibrated GLM puck-line probability from game_model.py
  bet        |p_model - p_market| > 0.02 at the SportsGameOdds consensus close
Window: games after 2026-11-06 only.

Fingerprints let later runs prove the retrained models are identical: the
2026-27 negative binomial alpha and the mean prediction over 2026-27 rows up
to 2026-11-06, which later data cannot change.

Output: docs/live/frozen_rules.json (committed)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUTOFF = "2026-11-06"


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


if __name__ == "__main__":
    lines = pd.read_parquet(os.path.join(ROOT, "data/models/sog_market_graded.parquet"))
    lines["game_date"] = pd.to_datetime(lines.game_date).dt.strftime("%Y-%m-%d")
    fit_rows = lines[lines.game_date <= CUTOFF]
    X = sm.add_constant(np.column_stack([logit(fit_rows.p_market), logit(fit_rows.p_gbm)]))
    fit = sm.GLM(fit_rows.y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": fit_rows.game_id})
    sog = pd.read_parquet(os.path.join(ROOT, "data/models/sog_predictions.parquet"))
    sog["game_date"] = pd.to_datetime(sog.game_date).dt.strftime("%Y-%m-%d")
    sog_fp = sog[(sog.season == 20262027) & (sog.game_date <= CUTOFF)]
    sog_res = json.load(open(os.path.join(ROOT, "docs/models/sog_results.json")))
    games = pd.read_parquet(os.path.join(ROOT, "data/models/game_predictions.parquet"))
    games_fp = games[(games.season == 20262027) & (games.game_date <= CUTOFF)]
    rules = {
        "frozen_on": "2026-11-07", "window_starts_after": CUTOFF,
        "sog": {"model": "gbm", "threshold": 0.04, "price": "SportsGameOdds consensus close",
                "blend_fit_lines": int(len(fit_rows)), "blend_fit_dates": [fit_rows.game_date.min(), fit_rows.game_date.max()],
                "blend_coef_const_market_model": [float(v) for v in fit.params],
                "blend_se_game_clustered": [float(v) for v in fit.bse],
                "fingerprint": {"alpha_20262027_gbm": sog_res["alpha"]["20262027"]["gbm"],
                                "rows_to_cutoff": int(len(sog_fp)), "mean_mu_gbm_to_cutoff": round(float(sog_fp.mu_gbm.mean()), 8)}},
        "puck_line": {"model": "glm calibrated", "threshold": 0.02, "price": "SportsGameOdds consensus close",
                      "fingerprint": {"games_to_cutoff": int(len(games_fp)),
                                      "mean_lh_glm_to_cutoff": round(float(games_fp.lh_glm.mean()), 8),
                                      "theta_glm": round(float(games_fp.theta_glm.iloc[0]), 8),
                                      "q_ot_home": round(float(games_fp.q_ot_home.iloc[0]), 8)}},
    }
    json.dump(rules, open(os.path.join(ROOT, "docs/live/frozen_rules.json"), "w"), indent=2)
    print(json.dumps(rules, indent=2))
