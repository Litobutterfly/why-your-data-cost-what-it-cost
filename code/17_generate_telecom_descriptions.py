from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
GENERATOR = STAGE / "04_generate_product_descriptions.py"
INPUTS = ROOT / "results" / "work" / "telecom" / "model_inputs"
OUTPUT = ROOT / "results" / "work" / "telecom" / "descriptions"


def load_module():
    spec = importlib.util.spec_from_file_location("symbiotrade_telecom_descriptions", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("AI_EDGE_API_KEY") or ""
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY or AI_EDGE_API_KEY")
    generator = load_module()
    reports = {}
    frames = []
    for split, filename in [
        ("train", "train_negotiation_inputs.csv"),
        ("test", "test_model_inputs.csv"),
    ]:
        split_output = OUTPUT / split
        reports[split] = generator.run(
            input_path=INPUTS / filename,
            output_dir=split_output,
            api_key=api_key,
            workers=4,
            timeout=120,
            retries=3,
        )
        frame = pd.read_csv(split_output / "train_product_descriptions.csv")
        frame.insert(1, "split", split)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values("Id")
    combined.to_csv(OUTPUT / "telecom_product_descriptions.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "completed" if all(item["status"] == "completed" for item in reports.values()) else "completed_with_errors",
        "train_rows": len(frames[0]),
        "test_rows": len(frames[1]),
        "total_rows": len(combined),
        "observed_outcome_visible": False,
        "reports": reports,
    }
    (OUTPUT / "telecom_description_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
