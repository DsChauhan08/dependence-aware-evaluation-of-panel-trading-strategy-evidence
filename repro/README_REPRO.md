# Reproducibility Instructions

This package reproduces the public-data and simulation evidence for
`Sharpe-Ratio Inference at the Date Boundary in Financial Panels`.

Public repository:
https://github.com/DsChauhan08/dependence-aware-evaluation-of-panel-trading-strategy-evidence

## Environment

Use Python 3.11 or newer.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-paper.txt
```

## Checks

```bash
python -m py_compile code/sddm_bootstrap.py code/threshold_analysis.py \
  code/public_data.py code/run_full_campaign.py code/render_manuscript_artifacts.py
python -m pytest -q tests
```

## Full Release Campaign

The release campaign is intentionally expensive.

```bash
python code/run_full_campaign.py \
  --output-root output_release_YYYYMMDD \
  --n-sim 1000 --n-boot 5000 --n-perms 10000 \
  --n-jobs 12 --coverage-parallel 4 --no-resume --write-memo
```

If this `repro/` directory is being used inside the full release tree,
regenerate manuscript tables into the journal source folder with:

```bash
python code/render_manuscript_artifacts.py \
  --campaign-root output_release_YYYYMMDD --paper-dir ../journal/source
```

Compile the peer-review manuscript and technical appendix from
`../journal/source/` with `pdflatex` twice when the source folder is present.

## Release Boundary

The public release includes code, public-data fetchers, aggregate artifacts,
simulation outputs, and provenance hashes.  It excludes proprietary
predictions, raw trades, private tickers, production feature definitions, and
unrelated workspace artifacts.

Code is released under Apache-2.0.  Manuscript text and generated public
aggregate artifacts are intended for release under CC-BY 4.0 unless a target
venue requires different terms.
