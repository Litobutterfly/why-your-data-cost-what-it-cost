from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parent
SOURCE = STAGE / "support" / "multi_aspect_discovery.py"
INPUTS = ROOT / "results" / "work" / "telecom" / "model_inputs"
AGENTS = ROOT / "results" / "work" / "telecom" / "agents"
OUTPUT = ROOT / "results" / "work" / "telecom" / "discovery"


def load_module():
    spec = importlib.util.spec_from_file_location("symbiotrade_telecom_discovery", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("AI_EDGE_API_KEY") or ""
    if not key:
        raise RuntimeError("Set DEEPSEEK_API_KEY or AI_EDGE_API_KEY")
    runner = load_module()
    configurations = [
        (
            "telecom_train_discovery_v1",
            AGENTS / "train" / "train_fixed_agent_pairs.csv",
            INPUTS / "train_negotiation_inputs.csv",
        ),
        (
            "telecom_test_discovery_v1",
            AGENTS / "test" / "train_fixed_agent_pairs.csv",
            INPUTS / "test_model_inputs.csv",
        ),
    ]
    for stem, selection, facts in configurations:
        runner.main(
            api_key=key,
            output_stem=stem,
            selection_path=selection,
            facts_path=facts,
            output_dir=OUTPUT,
            limit=None,
            batch_size=3,
            workers=4,
            timeout=240,
            retries=2,
        )


if __name__ == "__main__":
    main()
