"""Colab cell: pick a large data file (from your computer or Google Drive),
profile it, and stage it in 3 MB parts for a Claude session.

Chat uploads stop at about 30 MB. This cell:
  1. lets you choose a file: upload from your computer, or pick one from a
     numbered list of data files in your Google Drive;
  2. keeps a copy in NHL_Prediction_System_Claude/01_raw/inspect/;
  3. prints a profile (rows, columns, types, first rows, date range; for a
     zip, the files inside) so you can paste it into the chat;
  4. splits the file into 3 MB parts with a SHA-256 manifest under
     01_raw/_transfer_3mb/inspect_<date>/ (part names start with inspect__).
The Claude session downloads the parts with the Drive connector and runs
`python3 run.py restore <manifest.json>`, which checks every SHA-256 and
writes the file to data/drive/NHL_Prediction_System_Claude/01_raw/inspect/.
Nothing is committed to the repository.

Run in Google Colab: paste into one cell and run; answer the prompts.
Run it again for each file (each run adds to the same day's manifest).
"""

import glob
import hashlib
import io
import json
import os
import shutil
import zipfile
from datetime import date

import pandas as pd
from google.colab import drive, files

drive.mount("/content/drive")
MY = "/content/drive/MyDrive"
KEEP = f"{MY}/NHL_Prediction_System_Claude/01_raw/inspect"
TODAY = date.today().isoformat()
STAGE = f"{MY}/NHL_Prediction_System_Claude/01_raw/_transfer_3mb/inspect_{TODAY}"
PART = 3 * 1024 * 1024
EXT = (".csv", ".parquet", ".xlsx", ".xls", ".json", ".zip", ".gz", ".txt", ".tsv")
os.makedirs(KEEP, exist_ok=True)
os.makedirs(STAGE, exist_ok=True)

# 1. Choose the file.
choice = input("Where is the file? Type 1 = my computer, 2 = Google Drive: ").strip()
if choice == "1":
    up = files.upload()  # opens a file picker; large files take a while
    name = next(iter(up))
    path = f"{KEEP}/{name}"
    open(path, "wb").write(up[name])
else:
    found = []
    for p in glob.glob(f"{MY}/**/*", recursive=True):
        if p.lower().endswith(EXT) and "/_transfer_3mb/" not in p and os.path.isfile(p):
            found.append(p)
    flt = input(f"{len(found)} data files in Drive. Type part of the name to filter (or press Enter for all): ").strip().lower()
    found = sorted(p for p in found if flt in os.path.basename(p).lower())
    for i, p in enumerate(found[:200]):
        print(f"{i:3d}  {os.path.getsize(p) / 1e6:8.1f} MB  {os.path.relpath(p, MY)}")
    if len(found) > 200:
        print(f"... {len(found) - 200} more; run again with a narrower filter")
    src = found[int(input("Number of the file to stage: ").strip())]
    name = os.path.basename(src)
    path = f"{KEEP}/{name}"
    if os.path.abspath(src) != os.path.abspath(path):
        shutil.copy2(src, path)
data = open(path, "rb").read()
print(f"\nFILE {name}  {len(data) / 1e6:.1f} MB")


# 2. Profile it (paste this output into the chat).
def profile(df: pd.DataFrame, label: str):
    print(f"\n== {label}: {len(df):,} rows x {df.shape[1]} columns")
    print(df.dtypes.to_string())
    with pd.option_context("display.max_columns", 60, "display.width", 250):
        print(df.head(5).to_string())
    for c in df.columns:
        if any(k in c.lower() for k in ("date", "time", "season", "start")):
            try:
                print(f"  {c}: {df[c].min()} .. {df[c].max()}  ({df[c].nunique():,} distinct)")
            except TypeError:
                pass
    nulls = df.isna().mean()
    if (nulls > 0).any():
        print("  share missing:", nulls[nulls > 0].round(3).to_dict())


low = name.lower()
try:
    if low.endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(data))
        for info in z.infolist()[:50]:
            print(f"  {info.file_size / 1e6:8.1f} MB  {info.filename}")
        for info in z.infolist():
            if info.filename.lower().endswith(".csv"):
                profile(pd.read_csv(z.open(info.filename), nrows=200_000, low_memory=False), info.filename + " (first 200,000 rows)")
                break
    elif low.endswith(".parquet"):
        profile(pd.read_parquet(path), name)
    elif low.endswith((".xlsx", ".xls")):
        for sheet, df in pd.read_excel(path, sheet_name=None).items():
            profile(df, f"{name} / sheet {sheet}")
    elif low.endswith(".json"):
        try:
            profile(pd.read_json(path), name)
        except ValueError:
            profile(pd.read_json(path, lines=True), name)
    else:
        sep = "\t" if low.endswith((".tsv", ".txt")) else ","
        profile(pd.read_csv(path, sep=sep, low_memory=False), name)
except Exception as e:  # still stage the file; Claude can inspect it after restore
    print("could not profile:", type(e).__name__, str(e)[:300])

# 3. Stage in 3 MB parts and add to today's manifest.
stem = f"NHL_Prediction_System_Claude/01_raw/inspect/{name}"
parts = []
for i in range(0, len(data), PART):
    part = f"inspect__{name}.part{i // PART:03d}"
    open(f"{STAGE}/{part}", "wb").write(data[i:i + PART])
    parts.append(part)
mpath = f"{STAGE}/manifest_inspect_{TODAY}.json"
manifest = json.load(open(mpath)) if os.path.exists(mpath) else []
manifest = [m for m in manifest if m["source"] != stem]
manifest.append({"source": stem, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "parts": parts})
json.dump(manifest, open(mpath, "w"), indent=2)
print(f"\nSTAGED {name}: {len(parts)} parts -> {STAGE}")
print(f"manifest_inspect_{TODAY}.json now lists {len(manifest)} file(s)")
