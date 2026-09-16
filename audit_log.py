"""
Append-only JSONL audit log for reviewed decisions.

One JSON object per line: easy to append to concurrently-ish, easy to grep,
easy to load fully into memory for diagnostics without a CSV dialect to
fight with. This is a NEW log stream, deliberately separate from the
existing `review_log.csv` (the earlier shadow-mode layer's log): that file
already has ~270 rows in an older, narrower schema (no separate override
fields, no Q-values per action, no discretized state). Rewriting it in
place would silently corrupt historical rows. `review_log.csv` is left
exactly as-is, as a historical record of the earlier layer; all decisions
made by the new HumanReviewGate go to `audit_log.jsonl` going forward -
one log, one schema, no dual-writing.
"""

import json
import os


def append_record(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_records(path: str) -> list:
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
