# Sharpe-ratio variance inflation under cross-sectional and serial dependence in trading panels

This directory contains the revised manuscript and reproducibility code for
`Sharpe-ratio variance inflation under cross-sectional and serial dependence in trading panels`.

The code file names retain some older `sddm_*` names for compatibility, but
the manuscript no longer presents SDDM as a branded method.  The public-facing
object is a financial-econometrics evaluation of Sharpe-ratio variance
inflation in trading panels with cross-sectional and serial dependence.

Public repository:
https://github.com/DsChauhan08/dependence-aware-evaluation-of-panel-trading-strategy-evidence

## Quick Checks

```bash
cd /home/regulus/Trade
python -m py_compile sddm-paper-v2/code/sddm_bootstrap.py \
  sddm-paper-v2/code/threshold_analysis.py \
  sddm-paper-v2/code/public_data.py \
  sddm-paper-v2/code/run_full_campaign.py \
  sddm-paper-v2/code/run_power_size.py \
  sddm-paper-v2/code/synthetic_positive_control.py \
  sddm-paper-v2/code/render_manuscript_artifacts.py \
  sddm-paper-v2/code/make_release_bundle.py
python -m pytest -q sddm-paper-v2/tests/test_audit_methods.py
```

## Smoke Runs

```bash
cd /home/regulus/Trade/sddm-paper-v2/code
SDDM_N_BOOT=99 SDDM_N_PERMS=99 python public_data.py
SDDM_N_SIM=5 SDDM_N_BOOT=99 SDDM_N_JOBS=2 python run_power_size.py
```

Production runs should use the full campaign runner and should regenerate all
manuscript tables from CSV artifacts rather than hand-typed numbers.  Do not
render submission tables from an output path containing `smoke`.

```bash
cd /home/regulus/Trade/sddm-paper-v2/code
python run_full_campaign.py \
  --output-root output_release_YYYYMMDD \
  --n-sim 1000 --n-boot 5000 --n-perms 10000 \
  --n-jobs 12 --coverage-parallel 4 --no-resume --write-memo
cd /home/regulus/Trade/sddm-paper-v2
python code/render_manuscript_artifacts.py --campaign-root code/output_release_YYYYMMDD --paper-dir paper
python code/synthetic_positive_control.py --output-dir code/output_synthetic_positive_control --paper-dir paper
python code/make_release_bundle.py --campaign-root code/output_release_YYYYMMDD --version vYYYYMMDD
```

## Main Components

```text
paper/sddm_paper_v2.tex       Peer-review manuscript draft
paper/CLAIM_LEDGER.md         Claim-to-evidence ledger
paper/submission_checklist_private.md
code/sddm_bootstrap.py        Date aggregation, bootstrap, HAC-delta inference
code/threshold_analysis.py    Holm/BH/BY/Storey/Romano-Wolf/WRC/DSR tools
code/public_data.py           Kenneth French momentum and placebo panels
code/run_full_campaign.py     Full public candidate and simulation campaign
code/run_power_size.py        Size and power curves across audit methods
code/synthetic_positive_control.py  Reproducible strong-edge positive control
code/render_manuscript_artifacts.py  Manuscript table/figure renderer
code/make_release_bundle.py   arXiv/SSRN/journal/repro release packager
tests/test_audit_methods.py   Unit tests for mathematical/statistical primitives
```

## Release Boundary

The main paper is designed to be reproducible from public Kenneth French data.
Proprietary results may appear only as appendix diagnostics when the displayed
aggregate tables can be reproduced from released redacted artifacts.  The
release must not contain model IP, raw predictions, raw trade dates, tickers,
feature definitions, or anything that reveals a production strategy.
