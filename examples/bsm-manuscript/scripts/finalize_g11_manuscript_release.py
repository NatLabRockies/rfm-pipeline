"""Turn a completed G11 publication bundle into an audited JDS release.

This is intentionally a local-laptop workflow. It performs no fitting,
resampling, or other scientific computation and requires no HPC allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable

if __package__ in {
    None,
    "",
}:  # Support the documented ``python scripts/...`` entry point.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_g11_publication_artifacts import audit_publication_artifacts
from scripts.install_g11_manuscript_artifacts import install_manuscript_artifacts
from scripts.reproduce_artifacts import (
    MANUSCRIPT_FIGURES,
    validate_manuscript_figure_pdfs,
)


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_pdf(path: Path) -> None:
    if (
        not path.is_file()
        or path.stat().st_size <= 1024
        or not path.read_bytes().startswith(b"%PDF-")
        or b"%%EOF" not in path.read_bytes()[-1024:]
    ):
        raise ValueError(f"compiled manuscript PDF is missing or invalid: {path}")


def _validate_latex_log(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"LaTeX build log is missing: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if (
        "There were undefined references" in text
        or re.search(r"(?:Citation|Reference) .+ undefined", text, re.IGNORECASE)
        or re.search(r"undefined citations?", text, re.IGNORECASE)
    ):
        raise ValueError(f"LaTeX build contains unresolved references: {path}")


def _pdf_page_count(path: Path, *, run_command: Runner) -> int:
    completed = run_command(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(?m)^Pages:\s*([0-9]+)\s*$", str(completed.stdout))
    if match is None:
        raise ValueError(f"pdfinfo did not report a page count: {path}")
    return int(match.group(1))


def _write_deterministic_zip(destination: Path, *, files: dict[str, Path]) -> str:
    if destination.exists():
        raise ValueError(f"refusing to overwrite submission archive: {destination}")
    if not files or any(not path.is_file() for path in files.values()):
        raise ValueError(
            f"submission archive inputs are incomplete: {destination.name}"
        )
    with zipfile.ZipFile(
        destination,
        mode="x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, source in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with (
                source.open("rb") as input_handle,
                archive.open(info, mode="w", force_zip64=True) as output_handle,
            ):
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(files):
            raise ValueError(f"submission archive validation failed: {destination}")
    return _sha256(destination)


def finalize_manuscript_release(
    *,
    publication_root: str | Path,
    bsm_repo_root: str | Path,
    manuscript_root: str | Path,
    release_root: str | Path,
    replace: bool = False,
    run_command: Runner = subprocess.run,
) -> dict[str, Any]:
    """Regenerate figures, audit/install results, and compile both JDS PDFs."""
    publication = Path(publication_root).resolve()
    bsm = Path(bsm_repo_root).resolve()
    manuscript = Path(manuscript_root).resolve()
    release = Path(release_root).resolve()
    publication_manifest = publication / "publication_artifact_manifest.json"
    if not publication_manifest.is_file():
        raise ValueError("publication bundle has no artifact manifest")
    reproduction_files = (
        "scripts/reproduce_artifacts.py",
        "scripts/audit_g11_publication_artifacts.py",
        "pixi.toml",
        "pixi.lock",
    )
    if any(not (bsm / name).is_file() for name in reproduction_files):
        raise ValueError("BSM repository lacks a pinned artifact-reproduction input")
    if not (bsm / "LICENSE").is_file():
        raise ValueError("BSM repository lacks the supplement license")
    source_files = (
        "manuscript.tex",
        "coverpage.tex",
        "ref.bib",
        "acmart.cls",
        "ACM-Reference-Format.bst",
        "acm-ims-jds-logo.pdf",
        "acm-jdslogo.png",
    )
    for source in source_files:
        if not (manuscript / source).is_file():
            raise ValueError(f"manuscript repository lacks {source}")
    if release.exists() and any(release.iterdir()):
        raise ValueError("release output root must be new or empty")
    release.mkdir(parents=True, exist_ok=True)
    figures = release / "figures"
    command = [
        sys.executable,
        str(bsm / "scripts" / "reproduce_artifacts.py"),
        "--artifact-root",
        str(publication),
        "--output-dir",
        str(figures),
    ]
    run_command(command, cwd=bsm, check=True, capture_output=True, text=True)
    validate_manuscript_figure_pdfs(figures)
    svgs = sorted(figures.glob("*.svg"))
    if len(svgs) != 11 or any(path.stat().st_size <= 1024 for path in svgs):
        raise ValueError("figure regeneration did not produce all 11 nontrivial SVGs")

    audit_path = release / "publication_artifact_audit.json"
    audit = audit_publication_artifacts(
        publication_root=publication,
        figures_root=figures,
        output_path=audit_path,
    )
    install = install_manuscript_artifacts(
        publication_root=publication,
        figures_root=figures,
        audit_path=audit_path,
        manuscript_root=manuscript,
        replace=replace,
    )

    build = release / "latex_build"
    for source in ("manuscript.tex", "coverpage.tex"):
        run_command(
            [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-outdir={build}",
                source,
            ],
            cwd=manuscript,
            check=True,
            capture_output=True,
            text=True,
        )
        _validate_latex_log(build / Path(source).with_suffix(".log").name)
    pdf_dir = release / "pdf"
    pdf_dir.mkdir()
    pdf_hashes = {}
    page_counts = {}
    for name in ("manuscript.pdf", "coverpage.pdf"):
        source = build / name
        _validate_pdf(source)
        page_counts[name] = _pdf_page_count(source, run_command=run_command)
        destination = pdf_dir / name
        shutil.copy2(source, destination)
        pdf_hashes[name] = _sha256(destination)
    if page_counts["manuscript.pdf"] > 25:
        raise ValueError(
            "JDS manuscript exceeds the 25-page limit: "
            f"{page_counts['manuscript.pdf']} pages"
        )
    if page_counts["coverpage.pdf"] != 1:
        raise ValueError(
            "JDS cover page must be exactly one page: "
            f"{page_counts['coverpage.pdf']} pages"
        )

    source_archive_files = {name: manuscript / name for name in source_files}
    source_archive_files.update(
        {
            f"generated/{name}": manuscript / "generated" / name
            for name in (
                "manuscript_results.tex",
                "recovery_fwer_rows.tex",
                "recovery_scenario_rows.tex",
            )
        }
    )
    source_archive_files.update(
        {
            f"figures/{name}": manuscript / "figures" / name
            for name in MANUSCRIPT_FIGURES
        }
    )
    source_archive_files.update(
        {name: pdf_dir / name for name in ("manuscript.pdf", "coverpage.pdf")}
    )
    source_archive = release / "jds-submission-source.zip"
    supplement_archive = release / "bsm-public-rf-supplement.zip"
    supplement_readme = release / "supplement_README.md"
    supplement_readme.write_text(
        "\n".join(
            (
                "# BSM reduced-form manuscript supplement",
                "",
                f"Confirmatory run: `{audit['run_id']}`",
                "",
                "This archive contains the complete checksummed publication bundle,",
                "an independent `PUBLICATION_ARTIFACT_AUDIT_PASS` record, and the",
                "pinned code and environment needed to regenerate all 11 figures.",
                "Output eligibility is recorded for all 23,495 outputs; 9,954 pass",
                "the prespecified signal-to-noise and nonzero-variance rule.",
                "",
                "## Reproduce the figures",
                "",
                "```bash",
                "cd reproduction",
                "pixi install --locked",
                "pixi run python scripts/reproduce_artifacts.py --artifact-root .. --output-dir ../figures",
                "```",
                "",
                "The publication manifest checksums every released artifact. The",
                "independent audit additionally verifies the model coefficient and",
                "intercept algebra rather than relying on file hashes alone.",
                "",
            )
        ),
        encoding="utf-8",
    )
    archive_hashes = {
        source_archive.name: _write_deterministic_zip(
            source_archive, files=source_archive_files
        ),
        supplement_archive.name: _write_deterministic_zip(
            supplement_archive,
            files={
                **{
                    path.relative_to(publication).as_posix(): path
                    for path in publication.rglob("*")
                    if path.is_file()
                },
                "publication_artifact_audit.json": audit_path,
                "README.md": supplement_readme,
                "LICENSE": bsm / "LICENSE",
                **{f"reproduction/{name}": bsm / name for name in reproduction_files},
            },
        ),
    }

    identity = {
        "schema_version": 1,
        "status": "JDS_RELEASE_BUILD_PASS",
        "run_id": audit["run_id"],
        "publication_manifest_sha256": _sha256(publication_manifest),
        "publication_audit_sha256": audit["audit_sha256"],
        "manuscript_install_sha256": install["install_sha256"],
        "svg_count": len(svgs),
        "figure_svg_sha256": {path.name: _sha256(path) for path in svgs},
        "pdf_sha256": pdf_hashes,
        "manuscript_page_count": page_counts["manuscript.pdf"],
        "coverpage_page_count": page_counts["coverpage.pdf"],
        "submission_archive_sha256": archive_hashes,
    }
    result = {**identity, "release_sha256": _stable_hash(identity)}
    (release / "JDS_RELEASE_BUILD_PASS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", required=True)
    parser.add_argument("--bsm-repo-root", required=True)
    parser.add_argument("--manuscript-root", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    result = finalize_manuscript_release(
        publication_root=args.publication_root,
        bsm_repo_root=args.bsm_repo_root,
        manuscript_root=args.manuscript_root,
        release_root=args.release_root,
        replace=args.replace,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
