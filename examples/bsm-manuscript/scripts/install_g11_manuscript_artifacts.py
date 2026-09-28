"""Install one independently audited G11 result macro and figure set."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # Support direct CLI execution from the repository.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_g11_publication_artifacts import audit_publication_artifacts
from scripts.reproduce_artifacts import MANUSCRIPT_FIGURES


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install(source: Path, destination: Path, *, replace: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _sha256(source) == _sha256(destination):
        return
    if destination.exists() and not replace:
        raise ValueError(f"refusing to replace manuscript artifact: {destination}")
    pending = destination.with_name(destination.name + ".pending")
    if pending.exists():
        raise ValueError(f"stale pending manuscript artifact exists: {pending}")
    shutil.copy2(source, pending)
    pending.replace(destination)


def install_manuscript_artifacts(
    *,
    publication_root: str | Path,
    figures_root: str | Path,
    audit_path: str | Path,
    manuscript_root: str | Path,
    replace: bool = False,
) -> dict[str, Any]:
    publication = Path(publication_root).resolve()
    figures = Path(figures_root).resolve()
    manuscript = Path(manuscript_root).resolve()
    if not (manuscript / "manuscript.tex").is_file():
        raise ValueError("manuscript root does not contain manuscript.tex")
    recorded = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    observed = audit_publication_artifacts(
        publication_root=publication,
        figures_root=figures,
    )
    if (
        recorded != observed
        or recorded.get("status") != "PUBLICATION_ARTIFACT_AUDIT_PASS"
    ):
        raise ValueError("publication audit is stale or differs from current artifacts")
    identity = {key: value for key, value in recorded.items() if key != "audit_sha256"}
    if recorded.get("audit_sha256") != _stable_hash(identity):
        raise ValueError("publication audit self-hash differs")

    installed: dict[str, str] = {}
    generated_sources = sorted((publication / "manuscript").glob("*.tex"))
    if {path.name for path in generated_sources} != {
        "manuscript_results.tex",
        "recovery_fwer_rows.tex",
        "recovery_scenario_rows.tex",
    }:
        raise ValueError("publication bundle has an unexpected generated-LaTeX surface")
    for source in generated_sources:
        destination = manuscript / "generated" / source.name
        _install(source, destination, replace=replace)
        installed[destination.relative_to(manuscript).as_posix()] = _sha256(destination)
    for name in MANUSCRIPT_FIGURES:
        destination = manuscript / "figures" / name
        _install(figures / name, destination, replace=replace)
        installed[destination.relative_to(manuscript).as_posix()] = _sha256(destination)

    install_identity = {
        "schema_version": 1,
        "status": "MANUSCRIPT_ARTIFACTS_INSTALLED",
        "run_id": recorded["run_id"],
        "audit_sha256": recorded["audit_sha256"],
        "manuscript_root": str(manuscript),
        "installed_sha256": installed,
    }
    result = {**install_identity, "install_sha256": _stable_hash(install_identity)}
    manifest_path = manuscript / "generated" / "final_artifact_install_manifest.json"
    if manifest_path.exists() and not replace:
        raise ValueError(
            f"refusing to replace manuscript install manifest: {manifest_path}"
        )
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", required=True)
    parser.add_argument("--figures-root", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--manuscript-root", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    result = install_manuscript_artifacts(
        publication_root=args.publication_root,
        figures_root=args.figures_root,
        audit_path=args.audit,
        manuscript_root=args.manuscript_root,
        replace=args.replace,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
