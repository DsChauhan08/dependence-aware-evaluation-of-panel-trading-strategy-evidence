# Cover Letter

Dear Editors,

Please consider my manuscript, "False Precision in Panel-Based Sharpe Backtests: A Practical Date-Aggregation Diagnostic," for publication in the Journal of Asset Management.

The paper develops a practical diagnostic for portfolio researchers and asset managers who evaluate panel-based trading signals. Many backtests evaluate thousands of selected asset-date rows but realize performance through one portfolio return per date. The manuscript shows how treating those rows as independent can overstate Sharpe-ratio precision and provides a reproducible workflow for reviewing the portfolio-return boundary.

The contribution is deliberately applied. The article does not propose a new trading rule or asset-pricing anomaly. Instead, it gives a portfolio backtest audit workflow: aggregate selected rows to date-level portfolio returns, conduct Sharpe inference on the date series using robust procedures, compare row-pooled and date-level uncertainty, apply fixed-menu and placebo checks where applicable, and report turnover-cost sensitivity.

Public Kenneth French and AQR examples illustrate the diagnostic, and Monte Carlo stress tests show that row-naive Sharpe tests can reject about 35 percent of the time at nominal 5 percent size under dependent null designs with no true edge. The article is intended to be useful for asset managers, portfolio researchers, and reviewers of systematic-investment evidence.

The manuscript is original, is not under review elsewhere, and uses public data and simulation code. A reproducibility package is available.

Sincerely,

Dhananjay S. Chauhan

Independent Researcher, India

ORCID: https://orcid.org/0009-0003-1049-2213

Email: dschauhan08.me@gmail.com
