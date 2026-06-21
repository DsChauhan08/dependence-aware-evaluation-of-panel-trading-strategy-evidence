"""
Build a redacted journal/preprint release bundle.

The bundle is intentionally whitelist-based.  It copies manuscript source,
aggregate public artifacts, reproducibility code, tests, and provenance files
from this project only.  It refuses smoke-generated manuscript artifacts and
paths that would leak unrelated workspace or proprietary strategy material.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "paper"
RELEASE_DIR = PROJECT_ROOT / "release"
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
        "sddm_technical_appendix.tex",
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
    if (PAPER_DIR / "sddm_technical_appendix.pdf").exists():
        guarded_copy(PAPER_DIR / "sddm_technical_appendix.pdf", dst / "sddm_technical_appendix.pdf")
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
        "artifact_provenance.csv",
    }
    for name in aggregate_names:
        path = campaign_root / name
        if path.exists():
            guarded_copy(path, dst / name)
    sim = campaign_root / "simulation"
    if sim.exists():
        for name in ["coverage_all_merged.csv", "coverage_pivot.csv", "dgp_configs.csv", "size_test_audit.csv", "power_audit.csv", "power_audit_pivot.csv"]:
            path = sim / name
            if path.exists():
                guarded_copy(path, dst / "simulation" / name)
    empirical = campaign_root / "empirical"
    if empirical.exists():
        for cdir in sorted(p for p in empirical.iterdir() if p.is_dir()):
            for name in [
                "metadata.json",
                "viability.json",
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
    copy_manuscript(journal / "source")
    if (PAPER_DIR / "sddm_paper_v2.pdf").exists():
        guarded_copy(PAPER_DIR / "sddm_paper_v2.pdf", ssrn / "sddm_paper_v2.pdf")
        guarded_copy(PAPER_DIR / "sddm_paper_v2.pdf", journal / "sddm_paper_v2.pdf")
    guarded_copy(PAPER_DIR / "reviewer_experiment_memo.md", journal / "reviewer_experiment_memo.md")
    guarded_copy(PAPER_DIR / "submission_checklist_private.md", journal / "submission_checklist_private.md")

    for name in ["README.md", "README_REPRO.md", "requirements-paper.txt", "LICENSE", "CITATION.cff"]:
        guarded_copy(PROJECT_ROOT / name, repro / name)
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

    (ssrn / "README_SSRN.txt").write_text(
        "SSRN package: upload the PDF only after the platform disclosure requirements are resolved.\n",
        encoding="utf-8",
    )
    (journal / "cover_letter_template.md").write_text(
        "# Cover Letter Template\n\n"
        "Dear Editor,\n\n"
        "I am submitting my manuscript, \"Sharpe-ratio variance inflation under cross-sectional and serial dependence in trading panels,\" for consideration as a Research article in Quantitative Finance and Economics.\n\n"
        "The paper develops a dependence-aware inference framework for trading-panel evidence. Its main contribution is the Sharpe Unified Variance Inflation Factor, which compares the long-run delta-method variance of a date-level Sharpe estimator with the row-pooled IID variance that would be reported when selected symbol-date rows are treated as independent. The empirical section uses public Kenneth French and AQR data for benchmark validation and stress testing, while the Monte Carlo section shows that row-naive inference can substantially overreject under dependent nulls.\n\n"
        "The manuscript should be of interest to readers working in quantitative finance, financial econometrics, portfolio-performance evaluation, trading-strategy evidence, and empirical asset-pricing methodology. The manuscript is not under consideration by another journal and has not been formally published in a peer-reviewed venue. I understand that the manuscript will be considered for publication by AIMS Press in open access format.\n\n"
        "I am the sole author. I received no external funding for this research and declare no conflict of interest. The replication code, public aggregate artifacts, configuration files, simulation scripts, and reproducibility materials are available at https://github.com/DsChauhan08/dependence-aware-evaluation-of-panel-trading-strategy-evidence.\n\n"
        "Sincerely,\n\n"
        "Dhananjay S. Chauhan\n"
        "Independent Researcher, India\n"
        "ORCID: https://orcid.org/0009-0003-1049-2213\n"
        "Email: dschauhan08.me@gmail.com\n",
        encoding="utf-8",
    )
    (journal / "reproducibility_statement.md").write_text(
        "# Reproducibility Statement\n\n"
        "The public manuscript tables are generated from public Kenneth French and AQR data sources and from simulation code included in the reproducibility package. "
        "The release excludes proprietary predictions, raw trade records, private tickers, and production strategy features.\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Sharpe-ratio variance inflation under cross-sectional and serial dependence in trading panels\n\n"
        "This repository contains the public manuscript, technical appendix, aggregate public artifacts, and reproducibility code for the paper.\n\n"
        "Public repository: https://github.com/DsChauhan08/dependence-aware-evaluation-of-panel-trading-strategy-evidence\n\n"
        "## Main Files\n\n"
        "- `journal/sddm_paper_v2.pdf`: peer-review manuscript PDF.\n"
        "- `journal/source/sddm_technical_appendix.pdf`: technical appendix for supplementary upload.\n"
        "- `journal/source/`: editable LaTeX source and generated manuscript artifacts.\n"
        "- `repro/`: public-data loaders, simulations, tests, aggregate artifacts, and reproduction instructions.\n\n"
        "## Release Archives\n\n"
        "`arxiv.tar.gz`, `journal.tar.gz`, `ssrn.tar.gz`, and `repro.tar.gz` mirror the corresponding release folders. "
        "Each release subfolder includes a `MANIFEST.sha256` file for integrity checks.\n\n"
        "## Redaction Boundary\n\n"
        "The public release excludes proprietary predictions, raw trade records, private tickers, production feature definitions, and unrelated workspace artifacts.\n",
        encoding="utf-8",
    )

    if compile_pdf:
        for _ in range(3):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "sddm_paper_v2.tex"], cwd=arxiv, check=True)

    for folder in [arxiv, ssrn, journal, repro]:
        write_manifest(folder)
        make_archive(folder)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--compile-pdf", action="store_true")
    args = parser.parse_args()
    build(Path(args.campaign_root), args.version, compile_pdf=args.compile_pdf)


if __name__ == "__main__":
    main()
