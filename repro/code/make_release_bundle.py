"""
Build a redacted journal/preprint release bundle.

The bundle is intentionally whitelist-based.  It copies manuscript source,
aggregate public artifacts, reproducibility code, tests, and provenance files
from this project only.  It refuses smoke-generated manuscript artifacts and
paths that would leak unrelated nonpublic workspace material.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "paper"
RELEASE_DIR = PROJECT_ROOT / "release"
MF_MAIN_BLINDED = "MF_sharpe_uvif_main_blinded.pdf"
MF_APPENDIX_BLINDED = "MF_sharpe_uvif_appendix_blinded.pdf"
MF_TITLE_PAGE = "MF_sharpe_uvif_title_page.md"
MF_COVER_LETTER = "MF_sharpe_uvif_cover_letter.md"
MF_REPLICATION_ZIP = "MF_sharpe_uvif_replication_blinded.zip"
MF_SHA256 = "MF_sharpe_uvif_sha256.txt"
PAPER_TITLE = "Sampling-Boundary Distortion in Sharpe Ratio: Evidence from Financial Panels"
FORBIDDEN_PARTS = {
    "venv",
    ".venv",
    "__pycache__",
    "artifacts",
    "quantum_alpha",
    "output_smoke",
    "output_perm_smoke",
    "output_power_parallel_smoke",
}
FORBIDDEN_NAMES = {
    "trades.csv",
    "raw_predictions.csv",
    "live_status.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_clean_generated_sources() -> None:
    for name in ["generated_empirical_artifacts.tex", "generated_simulation_artifacts.tex", "generated_artifacts.tex"]:
        path = PAPER_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if "smoke" in text:
            raise ValueError(f"refusing to package smoke-generated manuscript artifact: {path}")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def guarded_copy(src: Path, dst: Path) -> None:
    rel = src.relative_to(PROJECT_ROOT) if src.is_relative_to(PROJECT_ROOT) else src
    parts = {p.lower() for p in rel.parts}
    if parts & FORBIDDEN_PARTS or src.name.lower() in FORBIDDEN_NAMES:
        raise ValueError(f"forbidden release path: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree_whitelist(src_root: Path, dst_root: Path, suffixes: set[str]) -> None:
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        if any(part.lower() in FORBIDDEN_PARTS for part in path.parts):
            continue
        guarded_copy(path, dst_root / path.relative_to(src_root))


def copy_manuscript(dst: Path) -> None:
    for name in [
        "sddm_paper_v2.tex",
        "sddm_paper_v2_blinded.tex",
        "sddm_technical_appendix.tex",
        "sddm_technical_appendix_blinded.tex",
        "generated_empirical_main_artifacts.tex",
        "generated_simulation_main_artifacts.tex",
        "generated_empirical_artifacts.tex",
        "generated_simulation_artifacts.tex",
        "generated_positive_control_artifacts.tex",
        "generated_provenance.md",
    ]:
        guarded_copy(PAPER_DIR / name, dst / name)
    if (PAPER_DIR / "sddm_paper_v2.pdf").exists():
        guarded_copy(PAPER_DIR / "sddm_paper_v2.pdf", dst / "sddm_paper_v2.pdf")
    if (PAPER_DIR / "sddm_paper_v2_blinded.pdf").exists():
        guarded_copy(PAPER_DIR / "sddm_paper_v2_blinded.pdf", dst / "sddm_paper_v2_blinded.pdf")
    if (PAPER_DIR / "sddm_technical_appendix.pdf").exists():
        guarded_copy(PAPER_DIR / "sddm_technical_appendix.pdf", dst / "sddm_technical_appendix.pdf")
    if (PAPER_DIR / "sddm_technical_appendix_blinded.pdf").exists():
        guarded_copy(PAPER_DIR / "sddm_technical_appendix_blinded.pdf", dst / "sddm_technical_appendix_blinded.pdf")
    fig_src = PAPER_DIR / "generated_figures"
    if fig_src.exists():
        copy_tree_whitelist(fig_src, dst / "generated_figures", {".png", ".pdf"})


def copy_campaign_aggregates(campaign_root: Path, dst: Path) -> None:
    aggregate_names = {
        "campaign_attempts.csv",
        "campaign_metadata.json",
        "source_registry_snapshot.json",
        "candidate_gate_sensitivity.csv",
        "candidate_gate_counts.csv",
        "horizon_effect.csv",
        "artifact_provenance.csv",
    }
    for name in aggregate_names:
        path = campaign_root / name
        if path.exists():
            guarded_copy(path, dst / name)
    sim = campaign_root / "simulation"
    if sim.exists():
        for name in ["coverage_all_merged.csv", "coverage_pivot.csv", "design_sweep.csv", "dgp_configs.csv", "size_test_audit.csv", "power_audit.csv", "power_audit_pivot.csv"]:
            path = sim / name
            if path.exists():
                guarded_copy(path, dst / "simulation" / name)
    empirical = campaign_root / "empirical"
    if empirical.exists():
        for cdir in sorted(p for p in empirical.iterdir() if p.is_dir()):
            for name in [
                "metadata.json",
                "audit_gate.json",
                "failure.json",
                "methods.csv",
                "hac_delta.csv",
                "hac_prewhite.csv",
                "hac_bandwidth.csv",
                "fixed_b_hac.csv",
                "threshold_menu.csv",
                "romano_wolf.csv",
                "permutation.csv",
                "factor_alpha.csv",
                "data_snooping.csv",
                "stationarity.csv",
                "costs.csv",
                "holdout_subperiods.csv",
                "selected_counts.csv",
                "source_spec.json",
            ]:
                path = cdir / name
                if path.exists():
                    guarded_copy(path, dst / "empirical" / cdir.name / name)
    synthetic = campaign_root / "synthetic_positive_control"
    if synthetic.exists():
        for path in sorted(synthetic.iterdir()):
            if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
                guarded_copy(path, dst / "synthetic_positive_control" / path.name)


def write_manifest(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(f"{sha256(path)}  {path.relative_to(root)}")
    (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_archive(folder: Path) -> Path:
    archive = folder.with_suffix(".tar.gz")
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(folder, arcname=folder.name)
    return archive


def write_anonymized_review_zip(repro_root: Path, journal_root: Path) -> Path:
    anon = journal_root / "MF_sharpe_uvif_replication_blinded"
    reset_dir(anon)
    (anon / "README.md").write_text(
        "# Anonymous Replication Package\n\n"
        "This double-blind review package contains public-data loaders, simulation code, generated aggregate artifacts, and tests needed to reproduce the manuscript tables and figures. "
        "Author-identifying title-page material, cover letters, personal repository links, and public-archive metadata are excluded.\n\n"
        "Run `python -m pytest -q tests/test_audit_methods.py` for the method checks. "
        "The manuscript artifacts were rendered from the public aggregates using the included rendering utility.\n",
        encoding="utf-8",
    )
    for name in ["requirements-paper.txt"]:
        guarded_copy(PROJECT_ROOT / name, anon / name)

    anon_code = anon / "code"
    for name in [
        "sddm_bootstrap.py",
        "threshold_analysis.py",
        "public_data.py",
        "campaign_sources.py",
        "run_full_campaign.py",
        "run_power_size.py",
        "run_prod_parallel.py",
        "simulation_study.py",
        "synthetic_positive_control.py",
        "render_manuscript_artifacts.py",
        "walk_forward.py",
        "campaign_registry.json",
    ]:
        guarded_copy(PROJECT_ROOT / "code" / name, anon_code / name)
    copy_tree_whitelist(PROJECT_ROOT / "tests", anon / "tests", {".py"})
    public_aggregates = repro_root / "artifacts"
    for path in sorted(public_aggregates.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md"}:
            continue
        dst = anon / "public_aggregates" / path.relative_to(public_aggregates)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)

    forbidden = [
        "Dhananjay",
        "Chauhan",
        "dschauhan08",
        "0009-0003-1049-2213",
        "DsChauhan08",
        "Open" + "AI",
        "Co" + "dex",
    ]
    for path in sorted(anon.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".py", ".json", ".csv", ".md", ".txt", ".cff"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [term for term in forbidden if term in text]
        if hits:
            raise ValueError(f"author-identifying text in anonymous review package: {path}: {hits}")

    zip_path = journal_root / MF_REPLICATION_ZIP
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(anon.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(journal_root))
    shutil.rmtree(anon)
    return zip_path


def build(campaign_root: Path, version: str, compile_pdf: bool = False) -> None:
    campaign_root = campaign_root.resolve()
    if "smoke" in str(campaign_root).lower():
        raise ValueError(f"refusing smoke campaign root: {campaign_root}")
    ensure_clean_generated_sources()

    root = RELEASE_DIR / version
    reset_dir(root)
    arxiv = root / "arxiv"
    ssrn = root / "ssrn"
    journal = root / "journal"
    repro = root / "repro"
    for folder in [arxiv, ssrn, journal, repro]:
        folder.mkdir(parents=True, exist_ok=True)

    copy_manuscript(arxiv)
    if (PAPER_DIR / "sddm_paper_v2.pdf").exists():
        guarded_copy(PAPER_DIR / "sddm_paper_v2.pdf", ssrn / "sddm_paper_v2.pdf")
    if (PAPER_DIR / "sddm_paper_v2_blinded.pdf").exists():
        guarded_copy(PAPER_DIR / "sddm_paper_v2_blinded.pdf", journal / MF_MAIN_BLINDED)
    if (PAPER_DIR / "sddm_technical_appendix_blinded.pdf").exists():
        guarded_copy(PAPER_DIR / "sddm_technical_appendix_blinded.pdf", journal / MF_APPENDIX_BLINDED)

    for name in ["README_REPRO.md", "requirements-paper.txt", "LICENSE", "CITATION.cff"]:
        guarded_copy(PROJECT_ROOT / name, repro / name)
    (repro / "README.md").write_text(
        (PROJECT_ROOT / "README_REPRO.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for name in [
        "sddm_bootstrap.py",
        "threshold_analysis.py",
        "public_data.py",
        "campaign_sources.py",
        "run_full_campaign.py",
        "run_power_size.py",
        "run_prod_parallel.py",
        "simulation_study.py",
        "synthetic_positive_control.py",
        "render_manuscript_artifacts.py",
        "make_release_bundle.py",
        "walk_forward.py",
        "validate_full_campaign.py",
        "campaign_registry.json",
    ]:
        guarded_copy(PROJECT_ROOT / "code" / name, repro / "code" / name)
    copy_tree_whitelist(PROJECT_ROOT / "tests", repro / "tests", {".py"})
    copy_campaign_aggregates(campaign_root, repro / "artifacts")
    write_anonymized_review_zip(repro, journal)

    (ssrn / "README_SSRN.txt").write_text(
        "SSRN package: upload the PDF only after the platform disclosure requirements are resolved.\n",
        encoding="utf-8",
    )
    (journal / MF_COVER_LETTER).write_text(
        "# Cover Letter\n\n"
        "Dear Editor,\n\n"
        f"Please consider the manuscript, \"{PAPER_TITLE},\" for publication as a Research Article in Modern Finance.\n\n"
        "The paper is a financial econometrics methodology study using public Kenneth French panels, AQR factor series, and Monte Carlo designs. It examines how Sharpe-ratio evidence can be misstated when selected rows in an entity-time panel are treated as independent observations even though the economically relevant payoff is a date-level portfolio return.\n\n"
        "The simulations show that row-naive testing rejects about 35 percent of the time at a nominal 5 percent level under dependent zero-edge nulls, while date-level HAC and dependent-bootstrap procedures remain much closer to nominal size. The public benchmarks show that the correction preserves canonical momentum evidence but removes weaker stress-panel claims. The manuscript is positioned as a methods paper for evaluating panel-based Sharpe evidence rather than as a new anomaly or trading-rule paper.\n\n"
        "The submission is original, is not under consideration elsewhere, and has not been published in a peer-reviewed outlet. Separate blinded manuscript files, a separate blinded appendix, an anonymized replication ZIP, and a separate title page have been prepared in accordance with the journal's review requirements.\n\n"
        "Potential reviewers:\n\n"
        "The submission system requests three potential reviewers. Reviewer names, affiliations, and emails should be entered in the portal before final upload; this draft cover letter does not fabricate reviewer identities.\n\n"
        "Sincerely,\n\n"
        "Dhananjay S. Chauhan\n"
        "Independent Researcher, India\n"
        "ORCID: https://orcid.org/0009-0003-1049-2213\n"
        "Email: dschauhan08.me@gmail.com\n",
        encoding="utf-8",
    )
    (journal / MF_TITLE_PAGE).write_text(
        "# Title Page\n\n"
        f"Title: {PAPER_TITLE}\n\n"
        "Author: Dhananjay S. Chauhan\n\n"
        "Affiliation: Independent Researcher, India\n\n"
        "Corresponding author: Dhananjay S. Chauhan, Independent Researcher, India\n\n"
        "Email: dschauhan08.me@gmail.com\n\n"
        "ORCID: https://orcid.org/0009-0003-1049-2213\n\n"
        "Public repository: https://github.com/DsChauhan08/dependence-aware-evaluation-of-panel-trading-strategy-evidence\n\n"
        "Keywords: Sharpe ratio; panel data; cross-sectional dependence; serial dependence; HAC/bootstrap inference; portfolio evaluation\n\n"
        "JEL Codes: C12; C15; G11; G12; G17\n",
        encoding="utf-8",
    )
    (journal / "reproducibility_statement.md").write_text(
        "# Reproducibility Statement\n\n"
        "The public manuscript tables, figures, and conclusions are generated from public Kenneth French and AQR data sources and from simulation code included in the reproducibility package. "
        "The source registry records source URLs, including the Kenneth French Data Library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and the AQR datasets page (https://www.aqr.com/Insights/Datasets), plus candidate-specific download URLs and checksums. "
        "No nonpublic or workspace-specific production artifacts enter the reported evidence.\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        f"# {PAPER_TITLE}\n\n"
        "This repository contains the public manuscript, technical appendix, aggregate public artifacts, and reproducibility code for the paper.\n\n"
        "Public repository: https://github.com/DsChauhan08/dependence-aware-evaluation-of-panel-trading-strategy-evidence\n\n"
        "## Main Files\n\n"
        f"- `journal/{MF_MAIN_BLINDED}`: blinded peer-review manuscript PDF for journals that require separate title pages.\n"
        f"- `journal/{MF_TITLE_PAGE}`: author title page.\n"
        f"- `journal/{MF_APPENDIX_BLINDED}`: blinded technical appendix for supplementary upload.\n"
        f"- `journal/{MF_REPLICATION_ZIP}`: anonymized replication materials for double-blind review.\n"
        "- `arxiv/`: editable LaTeX source and generated manuscript artifacts.\n"
        "- `repro/`: public-data loaders, simulations, tests, aggregate artifacts, and reproduction instructions.\n\n"
        "## Release Archives\n\n"
        "`arxiv.tar.gz`, `journal.tar.gz`, `ssrn.tar.gz`, and `repro.tar.gz` mirror the corresponding release folders. "
        "Each release subfolder includes a `MANIFEST.sha256` file for integrity checks.\n\n"
        "## Redaction Boundary\n\n"
        "The manuscript tables, figures, and conclusions use public Kenneth French and AQR data plus simulations only. "
        "No nonpublic or workspace-specific production artifacts enter the reported evidence.\n",
        encoding="utf-8",
    )
    scope_note = PAPER_DIR / "modern_finance_scope_control.md"
    if scope_note.exists():
        guarded_copy(scope_note, root / "MODERN_FINANCE_SCOPE_CONTROL.md")

    if compile_pdf:
        for _ in range(3):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "sddm_paper_v2.tex"], cwd=arxiv, check=True)

    for folder in [arxiv, ssrn, journal, repro]:
        write_manifest(folder)
        make_archive(folder)
    (journal / MF_SHA256).write_text((journal / "MANIFEST.sha256").read_text(encoding="utf-8"), encoding="utf-8")
    make_archive(journal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--compile-pdf", action="store_true")
    args = parser.parse_args()
    build(Path(args.campaign_root), args.version, compile_pdf=args.compile_pdf)


if __name__ == "__main__":
    main()
