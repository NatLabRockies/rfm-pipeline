"""Fail closed on malformed Generation-11 slice controls."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "AUTONOMOUS_ANALYSIS_PLAN_G11.md"
PROMPT = ROOT / "scripts" / "autonomous_slice_prompt_g11.md"
SLICE = re.compile(r"^G11-(CTRL|G0|A|P|HPC)-S[12]$")


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not SLICE.fullmatch(argv[1]):
        print("usage: validate_autonomous_slice_g11.py G11-<CTRL|G0|A|P|HPC>-S<1|2>")
        return 2
    missing = [str(path.relative_to(ROOT)) for path in (PLAN, PROMPT) if not path.is_file()]
    if missing:
        print(f"missing Generation-11 controls: {', '.join(missing)}")
        return 1
    text = PLAN.read_text(encoding="utf-8")
    if f"### Slice {argv[1]}:" not in text:
        print(f"slice {argv[1]} is not defined by the Generation-11 plan")
        return 1
    required = (
        "HPC_SUBMISSION_READY",
        "never submits a scheduler job",
        "historical P9/G10",
    )
    absent = [value for value in required if value not in text]
    if absent:
        print(f"Generation-11 plan lacks fail-closed controls: {', '.join(absent)}")
        return 1
    print(f"{argv[1]} Generation-11 control validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
