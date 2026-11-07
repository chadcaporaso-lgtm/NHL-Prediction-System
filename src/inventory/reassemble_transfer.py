"""Reassemble files split by colab_split_large_files.py and verify them.

Reads Drive download results (JSON with base64 ``content`` and ``title``)
from one or more directories, rebuilds each original file from its parts in
order, and checks the size and SHA-256 against the transfer manifest. A file
is only written to data/drive/ if its checksum matches; otherwise it is
reported and left out.

Usage:
  python src/inventory/reassemble_transfer.py <manifest.json> <download_dir> [<download_dir> ...]
"""

from __future__ import annotations

import base64
import csv
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "data", "drive")
CHECKSUMS = os.path.join(OUT_DIR, "checksums.csv")


def collect_parts(download_dirs: list[str]) -> dict[str, bytes]:
    parts: dict[str, bytes] = {}
    for directory in download_dirs:
        for path in glob.glob(os.path.join(directory, "*.txt")) + glob.glob(os.path.join(directory, "*.json")):
            try:
                payload = json.load(open(path))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict) and ".part" in payload.get("title", "") and "content" in payload:
                parts[payload["title"]] = base64.b64decode(payload["content"])
    return parts


def main(manifest_path: str, download_dirs: list[str]) -> int:
    manifest = json.load(open(manifest_path))
    parts = collect_parts(download_dirs)
    os.makedirs(OUT_DIR, exist_ok=True)
    new_file = not os.path.exists(CHECKSUMS)
    failures = 0
    with open(CHECKSUMS, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["drive_id", "project", "path", "local_path", "size_bytes", "size_matches", "md5", "sha256"])
        for entry in manifest:
            if entry.get("status") != "split":
                print("SKIPPED (not split):", entry["source"])
                continue
            missing = [p for p in entry["parts"] if p not in parts]
            if missing:
                print(f"MISSING {len(missing)} of {len(entry['parts'])} parts:", entry["source"])
                failures += 1
                continue
            data = b"".join(parts[p] for p in entry["parts"])
            sha = hashlib.sha256(data).hexdigest()
            ok = sha == entry["sha256"] and len(data) == entry["size_bytes"]
            if not ok:
                print("CHECKSUM MISMATCH:", entry["source"])
                failures += 1
                continue
            local = os.path.join(OUT_DIR, entry["source"])
            os.makedirs(os.path.dirname(local), exist_ok=True)
            with open(local, "wb") as out:
                out.write(data)
            writer.writerow(
                [
                    "split-transfer",
                    entry["source"].split("/")[0],
                    entry["source"],
                    os.path.relpath(local, ROOT),
                    len(data),
                    True,
                    hashlib.md5(data).hexdigest(),
                    sha,
                ]
            )
            print(f"VERIFIED  {len(data) / 1e6:6.1f} MB  {entry['source']}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1], sys.argv[2:]) else 0)
