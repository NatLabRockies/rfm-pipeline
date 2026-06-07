"""Locate a headless Chrome/Chromium binary for SVG→PDF conversion.

Resolution order:
1. ``RFM_CHROME_BINARY`` env var (explicit override).
2. Platform-specific default install locations.
3. ``shutil.which`` over a list of common executable names on PATH.

Raises a :class:`FileNotFoundError` with a clear remediation message if no
binary can be located.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_MAC_DEFAULTS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)

_LINUX_DEFAULTS = (
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)

_PATH_NAMES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
)


def find_chrome() -> str:
    """Return a path to a headless-capable Chrome/Chromium binary."""
    override = os.environ.get("RFM_CHROME_BINARY")
    if override:
        if not Path(override).exists():
            raise FileNotFoundError(f"RFM_CHROME_BINARY={override} does not exist")
        return override

    candidates: tuple[str, ...]
    if sys.platform == "darwin":
        candidates = _MAC_DEFAULTS
    elif sys.platform.startswith("linux"):
        candidates = _LINUX_DEFAULTS
    else:
        candidates = ()

    for path in candidates:
        if Path(path).exists():
            return path

    for name in _PATH_NAMES:
        found = shutil.which(name)
        if found:
            return found

    raise FileNotFoundError(
        "No Chrome/Chromium binary found for SVG→PDF conversion. "
        "Install Google Chrome or Chromium, or set RFM_CHROME_BINARY to "
        "an explicit binary path."
    )
