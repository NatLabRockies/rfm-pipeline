"""Validate the built wheel from outside the source checkout."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _built_wheel() -> Path:
    wheels = sorted((ROOT / "dist").glob("rfm_pipeline-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one built wheel in dist/, found {wheels}")
    return wheels[0].resolve()


def _run_from_wheel(wheel: Path, code: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(wheel)
    with tempfile.TemporaryDirectory(prefix="rfm-wheel-smoke-") as directory:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode != 0:
        raise SystemExit(f"wheel smoke failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def main() -> None:
    """Check imports and entry-point modules using only the built wheel."""
    wheel = _built_wheel()
    commands = "; ".join(
        [
            "import rfm_pipeline",
            f"assert {wheel.name!r} in rfm_pipeline.__file__, rfm_pipeline.__file__",
            "from rfm_pipeline import run_canonical_workflow",
            "assert callable(run_canonical_workflow)",
        ]
    )
    _run_from_wheel(wheel, commands)
    print(f"validated {wheel.name} from outside the source checkout")


if __name__ == "__main__":
    main()
