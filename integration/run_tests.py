"""Run every module's tests with that module's own command, from the repository root.

    python -m integration.run_tests

Each suite runs in its own process (M2's four original tests are standalone scripts; M4's
tests are a package), so no module needs changing to be collected by one runner.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SUITES = [
    ("M1", [PY, "-m", "unittest", "discover", "-s", "m1/tests"]),
    ("M2", [PY, "-m", "unittest", "discover", "-s", "m2/tests", "-p", "test_repository.py"]),
    *[(f"M2 {s}", [PY, f"m2/tests/{s}.py"]) for s in ("test_factors", "test_volatility", "test_pipeline", "test_audit")],
    ("M3", [PY, "-m", "unittest", "discover", "-s", "m3/tests"]),
    ("M3 eras study", [PY, "-m", "unittest", "discover", "-s", "m3/eras/tests", "-t", "."]),
    ("M4", [PY, "-m", "unittest", "m4.tests.test_m4"]),
    ("M5", [PY, "-m", "unittest", "discover", "-s", "m5/tests"]),
    ("M6", [PY, "-m", "unittest", "discover", "-s", "m6/tests"]),
    ("backtest", [PY, "-m", "unittest", "discover", "-s", "backtest/tests"]),
    ("historical inference", [PY, "-m", "unittest", "backtest.test_historical_sweep"]),
    ("integration", [PY, "-m", "unittest", "discover", "-s", "integration/tests", "-t", "."]),
]


def summary(out: str) -> str:
    ran = [line for line in out.splitlines() if line.startswith("Ran ")]
    state = [line for line in out.splitlines() if line.startswith(("OK", "FAILED"))]
    return (ran[-1] + (' ' + state[-1] if state else '')) if ran else (out.strip().splitlines() or ["(no output)"])[-1][:80]


def main() -> int:
    failed = []
    for name, cmd in SUITES:
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        ok = r.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'} {name:20s} {summary(r.stderr + r.stdout):45s} {time.perf_counter() - t0:5.1f}s",
              flush=True)
        if not ok:
            failed.append(name)
            print((r.stderr or r.stdout)[-3000:])
    print("all suites passed" if not failed else f"failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
