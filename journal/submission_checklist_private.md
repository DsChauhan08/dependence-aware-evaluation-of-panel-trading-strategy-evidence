# Private Submission Checklist

This file is not part of the manuscript.

## Journal Policy

- Target journal: AIMS Press / Quantitative Finance and Economics.
- Article type: Original Research Article / Research article.
- Special issue: regular issue / no special issue unless an exact match appears.
- Number of authors: 1.
- Enter the final QFE PDF page count from `pdfinfo`.
- Upload `sddm_technical_appendix.pdf` as supplementary material because the
  manuscript cites proofs, robustness tables, capacity details, and the full
  simulation grid from the appendix.
- Keep the AI/tool-assistance disclosure out of the manuscript; provide it in a
  separate submission letter or portal disclosure field if the journal requires
  it.
- Confirm the manuscript and cover letter contain the real GitHub repository
  URL, not a placeholder.
- Confirm author name, affiliation, and email before submission.

## Mathematical Review

- Independent algebra review of Proposition 1.
- Independent probability/asymptotics review of Proposition 2.
- Confirm every theorem assumption is either tested, caveated, or described as
  a sufficient condition only.

## Artifact Review

- Run unit tests.
- Run public-data production pipeline.
- Run simulation production pipeline.
- Confirm `paper/generated_*artifacts.tex` was rendered from
  `code/output_release_YYYYMMDD`, not a smoke directory.
- Build `release/<version>/arxiv`, `release/<version>/ssrn`,
  `release/<version>/journal`, and `release/<version>/repro` with
  `code/make_release_bundle.py`.
- Confirm the arXiv source tarball compiles from a clean folder.
- Verify every manuscript number against `CLAIM_LEDGER.md`.
- Confirm appendix proprietary tables are reproducible from released redacted
  artifacts before including them.

## Redaction Boundary

- No model IP.
- No raw predictions.
- No raw trade dates.
- No tickers from proprietary runs.
- No proprietary feature definitions.
- No hidden appendix claim that the main paper relies on.
