"""Validate the two released domains and their frozen train/test splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPECTED_ROWS = {"financial": 2390, "telecom": 526}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    report: dict[str, object] = {"domains": {}}
    for domain, expected_rows in EXPECTED_ROWS.items():
        data_path = DATA / f"{domain}.csv"
        split_path = DATA / f"{domain}_train_test_split.csv"
        frame = pd.read_csv(data_path, low_memory=False)
        split = pd.read_csv(split_path, low_memory=False)
        if len(frame) != expected_rows or frame["Id"].duplicated().any():
            raise RuntimeError(f"Unexpected {domain} data shape")
        if set(split["Id"].astype(int)) != set(frame["Id"].astype(int)):
            raise RuntimeError(f"{domain} split does not cover exactly the released rows")
        if set(split["split"]) != {"train", "test"}:
            raise RuntimeError(f"{domain} split labels are invalid")
        report["domains"][domain] = {
            "rows": int(len(frame)),
            "train_rows": int((split["split"] == "train").sum()),
            "test_rows": int((split["split"] == "test").sum()),
            "data_sha256": sha256(data_path),
            "split_sha256": sha256(split_path),
        }
    output = ROOT / "results" / "data_validation_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
