"""Source-tree import smoke checks for the repository gate."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Import the package from the source tree and print a small summary."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import rfm_pipeline  # noqa: PLC0415

    exported = len(getattr(rfm_pipeline, "__all__", []))
    print(f"Imported rfm_pipeline from {src}; exported names: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
