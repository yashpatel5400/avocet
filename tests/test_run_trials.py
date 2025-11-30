import subprocess
import sys

import pytest


try:
    import pandas  # noqa: F401
except Exception:
    pandas_available = False
else:
    pandas_available = True


def test_run_trials_pvalue():
    if not pandas_available:
        pytest.skip("pandas not installed; skipping run_trials integration test.")
    cmd = [
        sys.executable,
        "scripts/run_trials.py",
        "--example",
        "robust_bike_newsvendor.py",
        "--trials",
        "5",
        "--alpha",
        "0.1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    output = proc.stdout
    # Expect paired t-test line and p-value < 0.05
    marker = "Paired t-test (robust < nominal):"
    assert marker in output, "t-test result not printed"
    # Extract p-value
    for line in output.splitlines():
        if marker in line:
            parts = line.split(",")
            for part in parts:
                if "p=" in part:
                    p_val = float(part.split("=")[1])
                    assert p_val < 0.05, f"Expected p < 0.05, got {p_val}"
                    return
    raise AssertionError("p-value not found in output")
