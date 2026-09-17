"""Every run writes one CSV with its own raw fields; aggregation is a separate, re-runnable pass."""

import csv
import glob
import os


def write(path, rows, fields=None):
    rows = [rows] if isinstance(rows, dict) else list(rows)
    fields = fields or list(rows[0])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[saved] {path}", flush=True)


def read(pattern):
    """All rows matching a glob, tagged with the file they came from."""
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_file"] = os.path.basename(path)
                rows.append(r)
    return rows


def joined(values):
    return "|".join(str(v) for v in values)


def split(field, cast=float):
    return [cast(v) for v in field.split("|")] if field else []
