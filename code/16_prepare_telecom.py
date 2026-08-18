"""Train Telecom M0 and prepare its target-separated negotiation inputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
WORK = ROOT / "results" / "work" / "telecom"


def main() -> None:
    subprocess.run(
        [sys.executable, str(CODE / "02_train_m0.py"), "--domain", "telecom"],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [
            sys.executable,
            str(CODE / "03_prepare_negotiation_inputs.py"),
            "--data", str(ROOT / "data" / "telecom.csv"),
            "--split", str(ROOT / "data" / "telecom_train_test_split.csv"),
            "--oof", str(ROOT / "results" / "work" / "telecom_m0" / "m0_train_oof_predictions.csv"),
            "--test", str(ROOT / "results" / "work" / "telecom_m0" / "m0_test_predictions.csv"),
            "--model", str(ROOT / "models" / "telecom" / "telecom_m0_reproduced.joblib"),
            "--output", str(WORK / "model_inputs"),
        ],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
