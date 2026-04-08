from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_markdown_formatter_targets_untracked_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    tracked = tmp_path / "README.md"
    tracked.write_text("# Title\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    untracked = tmp_path / "docs" / "new_doc.md"
    untracked.parent.mkdir()
    untracked.write_text("# Bad   spacing\n", encoding="utf-8")

    script = Path("tools/format_markdown.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "docs/new_doc.md" in result.stderr or "docs/new_doc.md" in result.stdout
