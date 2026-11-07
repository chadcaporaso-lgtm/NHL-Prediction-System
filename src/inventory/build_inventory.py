"""Build a file inventory of the earlier Drive projects.

Input: JSON pages exported from the Google Drive search API (one object per
page with a ``files`` list). Pages may overlap; items are deduplicated by
Drive file id.

Output (written to the output directory):
  inventory.csv          one row per file or folder, with full path and project
  unresolved.csv         items whose ancestry could not be traced to a project
  summary.json           counts and sizes by project, area and file type
  duplicate_groups.csv   files sharing the same title and byte size

Drive metadata from this API has no content checksum, so duplicates are
grouped by (title, size). Exact checksums are computed later for files that
are downloaded for the audit.

Usage:
  python src/inventory/build_inventory.py data/inventory data/inventory/out
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

PROJECT_ROOTS = {
    "1TBigHWN1kFJ-ZlbDv8zszir5JIiV1bpP": "NHL_Prediction_System",
    "1ZERUpSQTo12Kmfjm_v0cX8-DZLYj7yoY": "NHL_Prediction_System_ChatGPT",
    "10t3iZ9sbDqPNyLhntD8rnss_EgdMeBIZ": "NHL_Prediction_System_Claude",
}
FOLDER_MIME = "application/vnd.google-apps.folder"


def load_items(input_dir: str) -> dict[str, dict]:
    items: dict[str, dict] = {}
    for page in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        try:
            data = json.load(open(page))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict) or "files" not in data:
            continue
        for item in data["files"]:
            items[item["id"]] = item
    return items


def resolve_paths(items: dict[str, dict]) -> dict[str, str]:
    """Map every item id to a slash path rooted at its project name.

    Items whose chain of parents leaves the known items without reaching a
    project root get a path starting with ``?<missing parent id>``.
    """
    cache: dict[str, str] = {}

    def path_of(item_id: str, depth: int = 0) -> str:
        if item_id in PROJECT_ROOTS:
            return PROJECT_ROOTS[item_id]
        if item_id in cache:
            return cache[item_id]
        item = items.get(item_id)
        if item is None or depth > 100:
            return "?" + item_id
        result = path_of(item.get("parentId", ""), depth + 1) + "/" + item["title"]
        cache[item_id] = result
        return result

    return {item_id: path_of(item_id) for item_id in items}


def main(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    items = load_items(input_dir)
    paths = resolve_paths(items)

    rows = []
    for item_id, item in items.items():
        path = paths[item_id]
        project = path.split("/")[0] if not path.startswith("?") else ""
        parts = path.split("/")
        rows.append(
            {
                "id": item_id,
                "project": project,
                "area": "/".join(parts[1:3]) if project and len(parts) > 2 else "",
                "path": path,
                "title": item["title"],
                "is_folder": item["mimeType"] == FOLDER_MIME,
                "mime_type": item["mimeType"],
                "extension": item.get("fileExtension", ""),
                "size_bytes": int(item.get("fileSize") or 0),
                "created": item["createdTime"],
                "modified": item["modifiedTime"],
            }
        )
    rows.sort(key=lambda r: (r["project"] or "~", r["path"]))

    fields = list(rows[0].keys())
    in_projects = [r for r in rows if r["project"]]
    with open(os.path.join(output_dir, "inventory.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(in_projects)
    with open(os.path.join(output_dir, "unresolved.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(r for r in rows if not r["project"])

    # Duplicate candidates: same title and size within the projects.
    groups: dict[tuple, list] = defaultdict(list)
    for r in in_projects:
        if not r["is_folder"] and r["size_bytes"] > 0:
            groups[(r["title"], r["size_bytes"])].append(r)
    with open(os.path.join(output_dir, "duplicate_groups.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["title", "size_bytes", "copies", "paths"])
        for (title, size), members in sorted(groups.items(), key=lambda kv: -kv[0][1] * len(kv[1])):
            if len(members) > 1:
                writer.writerow([title, size, len(members), " | ".join(m["path"] for m in members)])

    files_only = [r for r in in_projects if not r["is_folder"]]
    summary = {
        "items_seen": len(rows),
        "unresolved_items": sum(1 for r in rows if not r["project"]),
        "by_project": {},
    }
    for project in sorted({r["project"] for r in in_projects}):
        proj_files = [r for r in files_only if r["project"] == project]
        dup_bytes = sum(
            m[0]["size_bytes"] * (len(m) - 1)
            for (t, s), m in groups.items()
            if len(m) > 1 and all(x["project"] == project for x in m)
        )
        summary["by_project"][project] = {
            "folders": sum(1 for r in in_projects if r["project"] == project and r["is_folder"]),
            "files": len(proj_files),
            "bytes": sum(r["size_bytes"] for r in proj_files),
            "duplicate_bytes_within_project": dup_bytes,
            "by_extension": dict(Counter(r["extension"] or r["mime_type"] for r in proj_files).most_common()),
            "by_area": dict(
                sorted(
                    Counter(r["area"].split("/")[0] for r in proj_files).items(),
                    key=lambda kv: -kv[1],
                )
            ),
        }
    json.dump(summary, open(os.path.join(output_dir, "summary.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_project"}, indent=2))
    for project, s in summary["by_project"].items():
        print(project, s["folders"], "folders", s["files"], "files", round(s["bytes"] / 1e6, 1), "MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
