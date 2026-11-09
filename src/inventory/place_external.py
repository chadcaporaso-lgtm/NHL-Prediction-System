"""Put restored owner-supplied odds files where the code expects them (decisions 37-39).

`python3 run.py restore <manifest>` writes Drive files to data/drive/<Drive path>.
The owner-supplied odds live in Drive under NHL_Prediction_System_Claude/01_raw/
external/ (or inspect/ for the zip staged on 2026-11-09). This script copies
each known file into data/external/, checks its SHA-256 against the version
the results were computed from, and extracts the NHL rows from the
October 2026 scrape. Missing files are reported, not fatal: the code that
uses them is optional and falls back where it can (game_market.py uses v2's
crosswalk without the newer 2026-27 file).

Run automatically by `python3 run.py restore`, or on its own.
"""

from __future__ import annotations

import glob
import hashlib
import io
import os
import shutil
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRIVE = os.path.join(ROOT, "data/drive/NHL_Prediction_System_Claude/01_raw")
# Drive file name -> (data/external path, SHA-256 of the version used)
FILES = {
    "nhl_2026_27_odds.csv": ("closing_csv/nhl_2026_27_odds.csv",  # 282 games to 2026-11-08 (decision 40)
                             "f324cc328e664cbee4b8f5cf396663ae518514d56a61b6813cbe811849eeaee6"),
    "sportsbook-nhl-2021-2022.csv": ("sbr/sportsbook-nhl-2021-2022.csv",
                                     "891229ab16a8aac0d1137e24e1d71ed6f544be28fc3b557b36d6a7ff59c7bfa2"),
    "sportsbook-nhl-2022-2023.csv": ("sbr/sportsbook-nhl-2022-2023.csv",
                                     "b164f907608bc89ae20eb0d78079eeb2937965ee985061cdd9f0553e8fe05297"),
    "archive (8).zip": ("odds_scrape_2026-10/archive (8).zip",
                        "761b3c5605fdb2d8e508936870c0f4739731c163fb54ef52a35285f1a5743789"),
}


def main() -> int:
    found = 0
    for name, (dest, sha) in FILES.items():
        hits = [p for p in glob.glob(os.path.join(DRIVE, "**", "*"), recursive=True)
                if os.path.isfile(p) and os.path.basename(p) == name]
        ok = [p for p in hits if hashlib.sha256(open(p, "rb").read()).hexdigest() == sha]
        if not ok:
            print(f"missing   {name}" + (" (a copy exists but its SHA-256 differs)" if hits else ""))
            continue
        out = os.path.join(ROOT, "data/external", dest)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(ok[0], out)
        found += 1
        print(f"placed    {name} -> data/external/{dest}")
        if name.endswith(".zip"):
            z = zipfile.ZipFile(out)
            g = pd.read_parquet(io.BytesIO(z.read("games.parquet")))
            g[g.competition == "NHL"].to_parquet(os.path.join(os.path.dirname(out), "games_nhl.parquet"), index=False)
            print("          extracted games_nhl.parquet")
    print(f"{found} of {len(FILES)} owner-supplied odds files in place")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
