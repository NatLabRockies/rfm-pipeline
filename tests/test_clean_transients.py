from pathlib import Path

from tools.clean_transients import remove_transients


def test_clean_transients_keeps_pixi_site_packages_build(tmp_path: Path) -> None:
    repo_root = tmp_path

    pixi_build = (
        repo_root / ".pixi" / "envs" / "default" / "lib" / "python3.12" / "site-packages" / "build"
    )
    pixi_build.mkdir(parents=True)
    (pixi_build / "__init__.py").write_text("x = 1\n", encoding="utf-8")

    repo_build = repo_root / "build"
    repo_build.mkdir()
    (repo_build / "artifact.txt").write_text("temp\n", encoding="utf-8")

    removed = remove_transients(repo_root)

    assert pixi_build.exists()
    assert not repo_build.exists()
    assert Path("build") in removed
