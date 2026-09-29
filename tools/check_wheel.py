"""Validate the built wheel from outside the source checkout."""

from __future__ import annotations

import configparser
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENTRY_POINTS = {
    "rfm-hpc-reduce": "rfm_pipeline.hpc_reduce:main",
    "rfm-hpc-submit": "rfm_pipeline.hpc_submit:main",
    "rfm-hpc-worker": "rfm_pipeline.hpc_shard_worker:main",
}


def _built_wheel() -> Path:
    wheels = sorted((ROOT / "dist").glob("rfm_pipeline-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one built wheel in dist/, found {wheels}")
    return wheels[0].resolve()


def _entry_points(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        matches = [
            name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")
        ]
        if len(matches) != 1:
            raise SystemExit(f"expected one entry_points.txt in {wheel.name}, found {matches}")
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(matches[0]).decode("utf-8"))
    return dict(parser["console_scripts"])


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
    actual_entry_points = _entry_points(wheel)
    if actual_entry_points != EXPECTED_ENTRY_POINTS:
        raise SystemExit(
            f"console entry points differ: expected {EXPECTED_ENTRY_POINTS}, "
            f"found {actual_entry_points}"
        )

    commands = "; ".join(
        [
            "import rfm_pipeline",
            f"assert {wheel.name!r} in rfm_pipeline.__file__, rfm_pipeline.__file__",
            "from rfm_pipeline import run_canonical_workflow",
            "assert callable(run_canonical_workflow)",
            "from rfm_pipeline.hpc_reduce import main as reduce_main",
            "from rfm_pipeline.hpc_submit import main as submit_main",
            "from rfm_pipeline.hpc_shard_worker import main as worker_main",
            "assert all(callable(item) for item in (reduce_main, submit_main, worker_main))",
        ]
    )
    _run_from_wheel(wheel, commands)
    print(f"validated {wheel.name} from outside the source checkout")


if __name__ == "__main__":
    main()
