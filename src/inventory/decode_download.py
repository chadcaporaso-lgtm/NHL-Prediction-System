"""Decode a Google Drive download (base64 JSON) into a local file.

Checks the decoded size against the inventory and records the MD5 and
SHA-256 of every file in data/drive/checksums.csv, so later audit steps
can show exactly which bytes were examined.

Usage:
  python src/inventory/decode_download.py <download.json> [<download.json> ...]
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INVENTORY = os.path.join(ROOT, "data", "inventory", "out", "inventory.csv")
OUT_DIR = os.path.join(ROOT, "data", "drive")
CHECKSUMS = os.path.join(OUT_DIR, "checksums.csv")


def load_inventory() -> dict[str, dict]:
    with open(INVENTORY) as fh:
        return {row["id"]: row for row in csv.DictReader(fh)}


def main(paths: list[str]) -> None:
    inventory = load_inventory()
    os.makedirs(OUT_DIR, exist_ok=True)
    new_file = not os.path.exists(CHECKSUMS)
    with open(CHECKSUMS, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(["drive_id", "project", "path", "local_path", "size_bytes", "size_matches", "md5", "sha256"])
        for path in paths:
            payload = json.load(open(path))
            data = base64.b64decode(payload["content"])
            entry = inventory.get(payload["id"], {})
            drive_path = entry.get("path", payload["title"])
            local = os.path.join(OUT_DIR, drive_path)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            with open(local, "wb") as out:
                out.write(data)
            expected = int(entry.get("size_bytes") or -1)
            matches = expected == len(data)
            writer.writerow(
                [
                    payload["id"],
                    entry.get("project", ""),
                    drive_path,
                    os.path.relpath(local, ROOT),
                    len(data),
                    matches,
                    hashlib.md5(data).hexdigest(),
                    hashlib.sha256(data).hexdigest(),
                ]
            )
            print(("OK  " if matches else "SIZE MISMATCH  ") + drive_path, len(data))


if __name__ == "__main__":
    main(sys.argv[1:])
