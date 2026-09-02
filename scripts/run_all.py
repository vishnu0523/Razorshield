"""Cross-platform reproduction runner for RazorShield.

This mirrors the Makefile's `reproduce` target using Python subprocesses so a
reviewer on Windows PowerShell can regenerate the project without installing
make or bash.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    [sys.executable, "-m", "ml.generate", "--seed", "42"],
    [sys.executable, "-m", "ml.inspect_overlap"],
    [sys.executable, "-m", "ml.features"],
    [sys.executable, "-m", "ml.train", "--seed", "42"],
    [sys.executable, "-m", "ml.evaluate"],
    [sys.executable, "-m", "ml.inspect_graph"],
    [sys.executable, "-m", "ml.rings"],
    [sys.executable, "-m", "ml.spike"],
    [sys.executable, "-m", "ml.inspect_spike"],
    [sys.executable, "-m", "ml.fusion"],
    [sys.executable, "-m", "ml.simulate"],
    [sys.executable, "-m", "backend.app.core.seed"],
    [sys.executable, "-m", "pytest"],
]


def main() -> int:
    for i, command in enumerate(STEPS, start=1):
        printable = " ".join(command)
        print(f"\n[{i}/{len(STEPS)}] {printable}", flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        if completed.returncode:
            print(f"\nFailed at step {i}: {printable}", file=sys.stderr)
            return completed.returncode

    print("\nRegenerated from seed 42. Nothing above was hardcoded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
