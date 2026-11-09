"""Unit test for the shift-chart reduction in src/ingest/colab_fetch_shiftcharts.py.

A hand-built game with known answers: two teams of five skaters and a goalie
for one 1,200-second period. Team 1 (skaters 11-15) takes a penalty from 600
to 720 (skater 15 off), so team 2 is on a power play for 120 seconds; team 2
swaps skater 25 for 26 at 300 (a line change at the same second). Checks
seconds by strength per skater and pair seconds at 5v5 and on the power play,
and that raw rows are parsed (goal events and broken rows dropped).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src/ingest"))
from colab_fetch_shiftcharts import game_aggregates, parse_shifts  # noqa: E402


def mmss(s):
    return f"{s // 60:02d}:{s % 60:02d}"


def row(pid, team, a, b, period=1, code=517):
    return {"playerId": pid, "teamId": team, "period": period, "startTime": mmss(a), "endTime": mmss(b), "typeCode": code}


rows = [row(1, 1, 0, 1200), row(2, 2, 0, 1200)]                      # goalies
rows += [row(p, 1, 0, 1200) for p in (11, 12, 13, 14)]
rows += [row(15, 1, 0, 600), row(15, 1, 720, 1200)]                   # penalty 600-720
rows += [row(p, 2, 0, 1200) for p in (21, 22, 23, 24)]
rows += [row(25, 2, 0, 300), row(26, 2, 300, 1200)]                   # change at 300
rows += [row(21, 2, 650, 650, code=505), {"playerId": 22, "teamId": 2, "period": 1,
                                          "startTime": "05:00", "endTime": None, "typeCode": 517}]
skaters = {11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 26}

sh = parse_shifts(rows, 2025020001)
assert len(sh) == 14, len(sh)  # goal event and the row without an end time are dropped
toi, pairs = game_aggregates(sh, skaters)
t = toi.set_index("player_id")
assert t.loc[11, "s_all"] == 1200 and t.loc[15, "s_all"] == 1080
assert t.loc[11, "s_5v5"] == 1080 and t.loc[11, "s_sh"] == 120 and t.loc[11, "s_pp"] == 0
assert t.loc[21, "s_pp"] == 120 and t.loc[21, "s_5v5"] == 1080 and t.loc[21, "s_ev"] == 1080
assert t.loc[25, "s_all"] == 300 and t.loc[26, "s_all"] == 900 and t.loc[26, "s_pp"] == 120
assert t.loc[15, "s_sh"] == 0 and t.loc[15, "n_shifts"] == 2
assert 1 not in t.index and 2 not in t.index  # goalies are not in the skater table

pr = {(min(a, b), max(a, b)): (s5, spp) for a, b, s5, spp in pairs[["player_id", "mate_id", "s_5v5", "s_pp"]].itertuples(index=False)}
assert pr[(11, 15)] == (1080, 0)
assert pr[(21, 26)] == (900 - 120, 120)    # 5v5 after the change, minus the power play
assert pr[(21, 25)] == (300, 0)
assert (25, 26) not in pr                   # never on the ice together
assert not any((a < 20) != (b < 20) for a, b in pr)  # pairs are teammates only
print("shift pair test passed")
