"""Cross-platform reproduction runner for RazorShield.

This mirrors the Makefile's `reproduce` target using Python subprocesses so a
reviewer on Windows PowerShell can regenerate the project without installing
make or bash.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
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


LOG_PATH = ROOT / "artifacts" / "pipeline_log.txt"


def main() -> int:
    """Run every step, streaming output live and teeing it to an artifact.

    The captured log is what the dashboard shows on its Live demo tab. It is
    the real stdout of a real run -- the fusion verdict, the ring counts, the
    test summary -- rather than a screenshot or a retyped summary, so a
    reader can check the numbers on screen against the run that produced
    them. Streaming and capturing at once, instead of subprocess capture,
    because a three-minute pipeline with no visible output looks hung.
    """
    lines: list[str] = []

    def emit(text: str) -> None:
        print(text, flush=True)
        lines.append(text)

    started = datetime.now(timezone.utc)
    emit(f"RazorShield pipeline — started {started.isoformat(timespec='seconds')}")

    for i, command in enumerate(STEPS, start=1):
        printable = " ".join(command)
        emit(f"\n[{i}/{len(STEPS)}] {printable}")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            emit(line.rstrip("\n"))
        returncode = process.wait()
        if returncode:
            emit(f"\nFailed at step {i}: {printable}")
            LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
            return returncode

    emit("\nRegenerated from seed 42. Nothing above was hardcoded.")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\npipeline log written: {LOG_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
