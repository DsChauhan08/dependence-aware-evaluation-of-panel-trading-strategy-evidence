# Private Submission Checklist

This file is not part of the manuscript.

## Journal Policy

- Check the target journal's author disclosure policy before submission.
- Restore any required tool-assistance, code-generation, or writing-assistance
  disclosure in the cover letter or manuscript metadata if the journal requires
  it.
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
