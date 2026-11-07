"""Draw the project's charts as SVG files in docs/figures/.

Every chart reads the same results files the model cards cite, so the
figures can be regenerated after any rerun:  python3 src/reporting/make_figures.py

  elo_power_ranking.svg     current Elo rating for all 32 teams
  elo_history.svg           Elo ratings since 2019-20, three teams highlighted
  market_gap.svg            model log loss minus closing-line log loss, by market
  sog_calibration.svg       shots model vs closing line, predicted vs actual over rate
  sog_walk_forward.svg      shots model vs baseline log-likelihood by season
  tie_rate.svg              regulation ties: actual vs independent Poisson vs tie-adjusted
  elo_vs_models.svg         moneyline log loss by season: Elo, game model, constant
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src/models"))
OUT = os.path.join(ROOT, "docs/figures")
os.makedirs(OUT, exist_ok=True)

# Validated categorical slots (dataviz reference palette, light mode) and text tokens.
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, INK3, GRID, BG, MUTED = "#12202b", "#4a5964", "#6f7d87", "#e6ebee", "#ffffff", "#c9d1d6"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10.5,
    "svg.fonttype": "none", "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 26,
    "xtick.color": INK3, "ytick.color": INK3, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "legend.frameon": False, "legend.fontsize": 10, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "svg.hashsalt": "nhl-prediction-system",  # fixed element ids, so unchanged charts give unchanged files
})


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=10.5, va="bottom")


def save(fig, name, source):
    fig.text(0.01, -0.04, f"Source: {source}", color=INK3, fontsize=8.5)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", pad_inches=0.25, metadata={"Date": None})
    plt.close(fig)
    print("wrote", name)


def season_label(s):
    s = str(s)
    return f"{s[:4]}-{s[6:]}"


def elo_power_ranking():
    cur = json.load(open(os.path.join(ROOT, "docs/models/elo_current.json")))
    r = pd.Series(cur["ratings"]).sort_values()
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = [BLUE if v >= 1500 else RED for v in r.values]
    ax.barh(r.index, r.values - 1500, left=1500, color=colors, height=0.62)
    ax.axvline(1500, color=INK3, lw=1)
    for i, (t, v) in enumerate(r.items()):
        ax.text(v + (3 if v >= 1500 else -3), i, f"{v:.0f}", va="center", ha="left" if v >= 1500 else "right",
                color=INK2, fontsize=9)
    ax.set_xlim(r.min() - 40, r.max() + 40)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_xlabel("Elo rating (league average = 1500)")
    ax.set_title("NHL Elo power ranking")
    subtitle(ax, f"Margin-of-victory Elo, through {cur['as_of']}. Blue: above average; red: below.")
    save(fig, "elo_power_ranking.svg", "docs/models/elo_current.json (src/features/build_elo.py)")


def elo_history():
    e = pd.read_parquet(os.path.join(ROOT, "data/features/elo_games.parquet"))
    rows = pd.concat([
        pd.DataFrame({"date": e.start, "season": e.season, "team": e.home, "elo": e.std_rh}),
        pd.DataFrame({"date": e.start, "season": e.season, "team": e.away, "elo": e.std_ra})]).sort_values("date")
    cur = json.load(open(os.path.join(ROOT, "docs/models/elo_current.json")))["ratings"]
    teams = list(cur)
    highlight = [(teams[0], BLUE), (teams[1], AQUA), (teams[-1], ORANGE)]
    fig, ax = plt.subplots(figsize=(11, 5.2))

    def series(t):
        d = rows[rows.team == t].copy()
        # break the line across any gap of more than 30 days (off-seasons, the 2020 pause)
        block = (d.date.diff() > pd.Timedelta(days=30)).cumsum()
        out = []
        for _, x in d.groupby(block):
            out.append(x)
            out.append(pd.DataFrame({"date": [x.date.max() + pd.Timedelta(days=1)], "elo": [np.nan]}))
        return pd.concat(out)
    for t in rows.team.unique():
        if t not in dict(highlight):
            d = series(t)
            ax.plot(d.date, d.elo, color=MUTED, lw=0.8, alpha=0.7)
    for t, c in highlight:
        d = series(t)
        ax.plot(d.date, d.elo, color=c, lw=2, label=t)
        last = d.dropna().iloc[-1]
        ax.text(last.date + pd.Timedelta(days=8), last.elo, t, color=INK, fontsize=9.5, va="center")
    ax.axhline(1500, color=INK3, lw=1)
    ax.set_ylabel("Pre-game Elo rating")
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("Elo ratings since 2019-20")
    subtitle(ax, "Every team in grey; the current top two and bottom team highlighted. Lines break across off-seasons and the 2020 pause.")
    save(fig, "elo_history.svg", "data/features/elo_games.parquet (src/features/build_elo.py)")


def market_gap():
    sb = json.load(open(os.path.join(ROOT, "docs/models/sog_blend_results.json")))["models"]["gbm"]
    s = json.load(open(os.path.join(ROOT, "docs/models/sog_results.json")))["market"]
    p = json.load(open(os.path.join(ROOT, "docs/models/points_results.json")))["market"]
    g = json.load(open(os.path.join(ROOT, "docs/models/game_market_results.json")))["log_loss"]
    items = [
        ("Shots on goal\n(model + market)", sb["test_log_loss_blend"] - sb["test_log_loss_market"]),
        ("Puck line", g["external|pl|20252026"]["model_cal"] - g["external|pl|20252026"]["market"]),
        ("Shots on goal\n(model alone)", s["models"]["gbm"]["log_loss"] - s["market_log_loss"]),
        ("Moneyline", g["external|ml|20252026"]["model_raw"] - g["external|ml|20252026"]["market"]),
        ("Points", p["models"]["glm"]["log_loss"] - p["market_log_loss"]),
        ("Total goals", g["external|tot|20252026"]["model_cal"] - g["external|tot|20252026"]["market"]),
    ][::-1]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names, vals = zip(*items)
    ax.barh(names, vals, color=[BLUE if v < 0 else RED for v in vals], height=0.55)
    ax.axvline(0, color=INK3, lw=1)
    for i, v in enumerate(vals):
        ax.text(v + (0.00012 if v >= 0 else -0.00012), i, f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right",
                color=INK2, fontsize=9.5)
    ax.set_xlim(min(vals) - 0.0015, max(vals) + 0.0015)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_xlabel("Model log loss minus closing-line log loss (left of zero beats the market)")
    ax.set_title("Distance from the closing line")
    subtitle(ax, "Blue beats the close, red trails it. Shots blend: 2026-27 held-out lines; games: full 2025-26 season.")
    save(fig, "market_gap.svg", "docs/models/*_results.json")


def sog_calibration():
    d = pd.read_parquet(os.path.join(ROOT, "data/models/sog_market_graded.parquet"))
    bins = [0, .3, .35, .4, .45, .5, .55, .6, .65, 1]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.plot([0.2, 0.7], [0.2, 0.7], color=INK3, lw=1)
    ax.text(0.47, 0.435, "perfect calibration", color=INK3, fontsize=9, ha="left", rotation=42)
    for col, c, label in [("p_market", ORANGE, "Closing line"), ("p_gbm", BLUE, "Model alone")]:
        b = d.groupby(pd.cut(d[col], bins), observed=True).agg(p=(col, "mean"), y=("y", "mean"), n=("y", "size"))
        b = b[b.n >= 100]
        ax.plot(b.p, b.y, color=c, label=label, marker="o", ms=6, mec=BG, mew=1.5)
    ax.set_xlim(0.2, 0.7); ax.set_ylim(0.2, 0.7)
    ax.set_xticks(np.arange(0.2, 0.71, 0.1)); ax.set_yticks(np.arange(0.2, 0.71, 0.1))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}"); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Predicted chance of the over"); ax.set_ylabel("How often the over hit")
    ax.legend(loc="upper left")
    ax.set_title("Shots on goal: calibration")
    subtitle(ax, f"{len(d):,} closing lines. The model alone is too extreme; blended with the price it adds information.")
    save(fig, "sog_calibration.svg", "data/models/sog_market_graded.parquet (src/models/prop_model.py)")


def sog_walk_forward():
    r = json.load(open(os.path.join(ROOT, "docs/models/sog_results.json")))["all_games"]
    seasons = list(r)
    x = np.arange(len(seasons))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    gain = [(r[s]["gbm"]["nb_log_lik_per_game"] - r[s]["baseline"]["nb_log_lik_per_game"]) * 1000 for s in seasons]
    ax.bar(x, gain, color=BLUE, width=0.5)
    for i, v in enumerate(gain):
        ax.text(i, v + 0.2, f"+{v:.1f}", ha="center", color=INK2, fontsize=9.5)
    ax.set_xticks(x, [season_label(s) for s in seasons])
    ax.set_ylabel("Log-likelihood gain per 1,000 player-games")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(gain) * 1.25)
    ax.set_title("Shots on goal: walk-forward gain over a recent-form baseline")
    subtitle(ax, "Each season predicted by a model trained only on earlier seasons. Positive every season.")
    save(fig, "sog_walk_forward.svg", "docs/models/sog_results.json (src/models/prop_model.py)")


def tie_rate():
    from game_model import joint_scores
    g = pd.read_parquet(os.path.join(ROOT, "data/models/game_predictions.parquet"))
    rows = []
    for s, d in g.groupby("season"):
        j0 = joint_scores(d.lh_glm.values, d.la_glm.values, 0.0)
        rows.append((s, (d.decision_type != "REG").mean(), j0.diagonal(axis1=1, axis2=2).sum(1).mean(), d.p_tie_glm_raw.mean()))
    t = pd.DataFrame(rows, columns=["season", "actual", "indep", "adjusted"])
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.26
    for k, (col, c, label) in enumerate([("actual", INK2, "Actual"), ("indep", RED, "Independent Poisson"),
                                         ("adjusted", BLUE, "Tie-adjusted model")]):
        ax.bar(x + (k - 1) * (w + 0.02), t[col], width=w, color=c, label=label)
    ax.set_xticks(x, [season_label(s) for s in t.season])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("Games tied after regulation")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0, 0.32)
    ax.set_title("Why the game model adjusts for ties")
    subtitle(ax, "Independent Poisson scores under-predict regulation ties by about a quarter; a factor fitted on training seasons fixes it.")
    save(fig, "tie_rate.svg", "data/models/game_predictions.parquet (src/models/game_model.py)")


def elo_vs_models():
    e = json.load(open(os.path.join(ROOT, "docs/models/elo_eval.json")))["all_games_moneyline"]
    seasons = list(e)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for col, c, label in [("constant", MUTED, "Constant (home win rate)"), ("elo_std", ORANGE, "Elo alone"),
                          ("game_model", BLUE, "Game model")]:
        ax.plot([season_label(s) for s in seasons], [e[s][col] for s in seasons], color=c, label=label, marker="o",
                ms=6, mec=BG, mew=1.5)
    ax.set_ylabel("Moneyline log loss (lower is better)")
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0.64, 0.70)
    ax.set_title("Elo against the game model")
    subtitle(ax, "Elo matches the game model early on, then falls behind; expected goals and goalie form carry more.")
    save(fig, "elo_vs_models.svg", "docs/models/elo_eval.json (src/models/elo_eval.py)")


if __name__ == "__main__":
    for f in [elo_power_ranking, elo_history, market_gap, sog_calibration, sog_walk_forward, tie_rate, elo_vs_models]:
        f()
