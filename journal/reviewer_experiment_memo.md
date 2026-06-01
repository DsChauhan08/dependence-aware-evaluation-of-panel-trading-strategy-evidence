# Reviewer Experiment Memo

Generated UTC: 2026-05-28T17:38:02.507573+00:00
Output root: `code/output_release_dryrun`

## Recommendation

No pre-registered public candidate passed all viability gates. Treat this as a null audit result, not as a discovery result: do not claim a viable public strategy, but the no-pass outcome can be reported as evidence that the framework rejects candidates with missing panel structure, weak dependence-adjusted edge, or cost/factor fragility.

## Strict-Reviewer Interpretation

The AQR factor-series candidates are economically meaningful factor returns, but they are pre-aggregated single-series sources. Their `permutation_p` entries are missing because the same-date signal permutation gate is structurally not applicable without row-level constituent signals. They can be discussed as supporting evidence for factor significance, not as candidates that pass a panel-level audit.

The French momentum decile candidate is the cleanest weak-edge example. It can pass the same-date permutation placebo while still failing the dependence-aware Sharpe gate and the researcher-menu gate. The correct interpretation is a detectable cross-sectional signal that is too weak to support a viable date-level trading claim under the audit.

The coverage pivot supports calibration rather than over-conservatism. HAC-delta, moving-block, and stationary methods remain near the nominal target across the main dependence designs, while `row_naive` coverage is materially below nominal in every displayed DGP. The strict gates therefore reject the public candidates because the candidates fail the stated audit conditions, not because the dependence-aware methods are designed to reject everything.

## Methodological Extensions

1. General covariance model. Proposition 1 should remain the transparent equicorrelated derivation, but the manuscript should explicitly state that the audit does not require equicorrelation. A finite factor covariance model with heterogeneous loadings is the natural generalization; the date-return boundary remains the operational solution.
2. Synthetic positive control. Add a reproducible synthetic panel with a known strong edge, serial dependence, cross-sectional dependence, and persistent signals. This directly answers the critique that the audit gates are impossible to pass.
3. Composite audit framing. Present the gates as a staged decision: same-date permutation for panel signal detection, HAC-delta Sharpe inference for dependence-aware significance, Romano-Wolf for the fixed researcher menu, and turnover-scaled net Sharpe for economic viability.

## Peer-Review Revision Notes

The peer-review revision items are clarifications rather than changes to the empirical record: state the Bartlett HAC construction for the joint `(r_t, r_t^2)` process; separate exploratory iteration from a frozen confirmatory researcher menu; state the limits of linear turnover costs for high-frequency, illiquid, and capacity-constrained strategies; read row-naive and dependence-aware outputs side by side; and explain how the date-boundary principle can transfer to other entity-time finance panels.

## Strong-Reject Stress Notes

A hostile review should be handled by reducing overclaiming rather than changing the empirical record: present the audit as a reproducible protocol, not as a new estimator; identify Proposition 1 with design-effect/Moulton logic; describe the composite gate as a conservative pre-specified policy rule; treat Kenneth French momentum as a worked example rather than validation by a famous anomaly; caveat same-date permutation by exchangeability/blocking; report HAC bandwidth and compare important conclusions to block-bootstrap results; and keep `N_eff` as a diagnostic only.

Implemented follow-through: `public_grouped_permutation.csv` repeats the same-date placebo while preserving Size or B/M blocks in the French 25 Size-B/M panel; `public_momentum_hac_bandwidth.csv` and `public_placebo_hac_bandwidth.csv` report HAC-delta sensitivity across automatic and fixed Bartlett bandwidths; and the current pre-registered public-candidate campaign has zero viable candidates at the conventional alpha=0.05 and also zero at alpha=0.01.

## Candidate Attempts

| candidate_id                           | status   |   viable |   gross_sharpe |   hac_p_positive |       rw_p |   permutation_p |   elapsed_sec | source_url                                                                                                      | error                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | generated_at_utc                 |
|:---------------------------------------|:---------|---------:|---------------:|-----------------:|-----------:|----------------:|--------------:|:----------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------|
| french_momentum_deciles_daily_rank     | ok       |        0 |      0.0818873 |      0.242947    |   0.142857 |        0.047619 |      53.0973  | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Portfolios_Prior_12_2_Daily_CSV.zip              | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | nan                              |
| french_size_bm_daily_dynamic_momentum  | ok       |        0 |      0.0356078 |      0.445435    |   0.380952 |        0.904762 |      40.5949  | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/25_Portfolios_5x5_Daily_CSV.zip                     | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | nan                              |
| stooq_fixed_etf_daily_dynamic_momentum | failed   |      nan |    nan         |    nan           | nan        |      nan        |       7.61446 | https://stooq.com/db/h/                                                                                         | fewer than three Stooq series loaded: spy.us: Stooq CSV endpoint now requires STOOQ_APIKEY/captcha-gated API key; qqq.us: Stooq CSV endpoint now requires STOOQ_APIKEY/captcha-gated API key; iwm.us: Stooq CSV endpoint now requires STOOQ_APIKEY/captcha-gated API key; efa.us: Stooq CSV endpoint now requires STOOQ_APIKEY/captcha-gated API key; eem.us: Stooq CSV endpoint now requires STOOQ_APIKEY/captcha-gated API key; tlt.us: Stooq CSV endpoint now requires STOOQ_APIKEY/captcha-gated API key; ief.us: Stooq CSV endpoint now requires STOOQ_APIKEY/captcha-gated API key; gld.us: Stooq CSV endpoint now requires STOOQ_APIKEY/captcha-gated API key; dbc.us: Stooq CSV endpoint now requires STOOQ_APIKEY/captcha-gated API key; vnq.us: Stooq CSV endpoint now requires STOOQ_APIKEY/captcha-gated API key; hyg.us: Stooq CSV endpoint now requires STOOQ_APIKEY/captcha-gated API key | 2026-05-28T17:29:06.329082+00:00 |
| aqr_bab_equity_factors_daily           | ok       |        0 |      0.741431  |      3.07865e-13 |   0.047619 |      nan        |     154.789   | https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Daily.xlsx     | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | nan                              |
| aqr_qmj_factors_daily                  | ok       |        0 |      0.536257  |      2.25707e-05 |   0.047619 |      nan        |     231.996   | https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Quality-Minus-Junk-Factors-Daily.xlsx              | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | nan                              |
| aqr_hml_devil_factors_daily            | ok       |        0 |      0.33607   |      0.00379406  |   0.047619 |      nan        |     137.299   | https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/The-Devil-in-HMLs-Details-Factors-Daily.xlsx       | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | nan                              |
| aqr_value_momentum_everywhere_monthly  | ok       |        0 |      2.62299   |      0.000161609 |   0.047619 |      nan        |       4.67775 | https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Value-and-Momentum-Everywhere-Factors-Monthly.xlsx | nan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | nan                              |

## Coverage Pivot

| DGP              |   blocked |   hac_delta |   iid |   row_naive |   stationary |
|:-----------------|----------:|------------:|------:|------------:|-------------:|
| 01_iid           |       1   |         1   |   1   |         0   |          1   |
| 02_ser_mild      |       0.5 |         1   |   1   |         0   |          1   |
| 03_ser_strong    |       1   |         1   |   0.5 |         0   |          0.5 |
| 04_xs_mild       |       1   |         1   |   0.5 |         0.5 |          0.5 |
| 05_xs_strong     |       0.5 |         1   |   0.5 |         0   |          0.5 |
| 06_both_mod      |       1   |         1   |   1   |         0.5 |          1   |
| 07_both_strong   |       1   |         1   |   1   |         0   |          1   |
| 08_realistic     |       0.5 |         0.5 |   0.5 |         0   |          0.5 |
| 09_garch_mild    |       1   |         1   |   0.5 |         0   |          0.5 |
| 10_garch_strong  |       0.5 |         1   |   1   |         0.5 |          1   |
| 11_multifactor   |       1   |         1   |   1   |         0   |          1   |
| 12_regime        |       0.5 |         1   |   1   |         0   |          1   |
| 13_high_ar1      |       1   |         1   |   0.5 |         0   |          0.5 |
| 14_garch_highvol |       1   |         1   |   1   |         1   |          1   |

## Power Pivot

|   true_sharpe |   date_iid |   hac_delta |   moving_block |   romano_wolf |   row_naive |   stationary |
|--------------:|-----------:|------------:|---------------:|--------------:|------------:|-------------:|
|     0         |        0   |         0   |            0   |           0   |         0   |          0   |
|     0.0944309 |        0   |         0   |            0   |           0   |         0   |          0   |
|     0.188862  |        0   |         0   |            0   |           0   |         0   |          0   |
|     0.377724  |        0   |         0   |            0   |           0   |         0   |          0   |
|     0.755447  |        0   |         0   |            0   |           0   |         0.5 |          0   |
|     1.51089   |        0.5 |         0.5 |            0.5 |           0.5 |         1   |          0.5 |
|     2.83293   |        1   |         1   |            1   |           1   |         1   |          1   |

## Provenance

| path                                                                    |   bytes | sha256                                                           |
|:------------------------------------------------------------------------|--------:|:-----------------------------------------------------------------|
| campaign_attempts.csv                                                   |    2514 | efe32ecc410a32086f12730144f4ab9de1f39c867b302d9a0a1f6b3aad52c5cd |
| campaign_metadata.json                                                  |     690 | e7f04c863cd6a9ecb6a46d25b42d8e9f3c32682d4beb2e5df63779bf6cf56b2b |
| candidate_gate_counts.csv                                               |      29 | 5e3eba60f941d810825041f45d374c4d5ce9f0f53935edaeda4835e7d3ba4416 |
| candidate_gate_sensitivity.csv                                          |    2175 | 35b0fe55a90ab52ba69bb90ee274d6fd3ec39c473f9ee4bbbda83db3a90d6592 |
| empirical/aqr_bab_equity_factors_daily/costs.csv                        |     527 | 0787fec6f0af1afb2933b82319f91b6c9c190f1d52000ccbd938901015ace339 |
| empirical/aqr_bab_equity_factors_daily/data_snooping.csv                |     170 | 35897c3543a8801f42b68081279a8ae4d26b417d0d83bc19131f2ab795fe0c2b |
| empirical/aqr_bab_equity_factors_daily/factor_alpha.csv                 |      42 | 404ec6d95ea7b16c58076245ac20dc08b9895c1de904401821cc8d03da0b7031 |
| empirical/aqr_bab_equity_factors_daily/fixed_b_hac.csv                  |     866 | 8558d26b4c148315b4d9fee37875f36bea8deb1d53f9a5efeb9d9d55cce2f324 |
| empirical/aqr_bab_equity_factors_daily/hac_bandwidth.csv                |     956 | 17f0fc1f2a411992901f19163b29abf681007eed32de0d07a17d5a29581b9df9 |
| empirical/aqr_bab_equity_factors_daily/hac_delta.csv                    |     433 | fd8b16d99526b91816aa304c1244b7c47592601fbe5e76c4346571b8f4c5cbad |
| empirical/aqr_bab_equity_factors_daily/hac_prewhite.csv                 |     292 | 44b749220823fe132db308465d22f807dad944ebc38b3b34ad55324b12777005 |
| empirical/aqr_bab_equity_factors_daily/holdout_subperiods.csv           |     305 | 13d55dde188653264db3a2352927b993661e7309955abb81451cca61ff790cb7 |
| empirical/aqr_bab_equity_factors_daily/metadata.json                    |     649 | 10f92b3038780a6948ca110361c045f51c20506ef827b021d0ac732cddafd87f |
| empirical/aqr_bab_equity_factors_daily/methods.csv                      |     595 | 55f62d48e9605485d26f50dcf209e6d0a92e6991a1298c3eaf7765f9461198d2 |
| empirical/aqr_bab_equity_factors_daily/permutation.csv                  |     169 | 9600c039f9354cddce170bb71d4658e41711a6390a5e30d037f894ff6ff48528 |
| empirical/aqr_bab_equity_factors_daily/romano_wolf.csv                  |     133 | b84255b97a0471d663ba5dac8f93fd0364f4e6be47e63bf252c355ad5ac56855 |
| empirical/aqr_bab_equity_factors_daily/selected_counts.csv              |     154 | 32144e2bb613ba2f57b259c98d92955d6a62eb4523f745de189ef87a9829df8e |
| empirical/aqr_bab_equity_factors_daily/source_returns_head.csv          |     577 | 3cffe4523f05c55f646efd23d63e4ebe18bc8329de990c699a840368cec94d71 |
| empirical/aqr_bab_equity_factors_daily/source_spec.json                 |     572 | 216b38f7396311dbc729c1fb0d4f16e0406356e84d3c47d1d47eadc6b5a0c5f0 |
| empirical/aqr_bab_equity_factors_daily/stationarity.csv                 |     114 | f247bf1389852abd4036c8355b1a4b38c22b1936515e9cf35df3240d945902fd |
| empirical/aqr_bab_equity_factors_daily/threshold_menu.csv               |     417 | 6abc54dde057d26e8105d7fff6b17bf83fd10e6c75fd252498bbf226f9df1273 |
| empirical/aqr_bab_equity_factors_daily/viability.json                   |    1155 | d6c413f84d2c0138a00ce8e411e80d3228cffcbabbebcea5aa3920f33269fe34 |
| empirical/aqr_hml_devil_factors_daily/costs.csv                         |     537 | d2e6a776fa142cb82b7d00ea3b8713465f2514681f900a008b281257a46663af |
| empirical/aqr_hml_devil_factors_daily/data_snooping.csv                 |     168 | ddd09d61e18cbdbe80169cc6c99bb9ba714f0ec7b380337398ade34a69cf8996 |
| empirical/aqr_hml_devil_factors_daily/factor_alpha.csv                  |      42 | 404ec6d95ea7b16c58076245ac20dc08b9895c1de904401821cc8d03da0b7031 |
| empirical/aqr_hml_devil_factors_daily/fixed_b_hac.csv                   |     865 | fc88d93b2513c5acbcacdb66e2bb1eef8903b413ba3595aa10c87994273df6d9 |
| empirical/aqr_hml_devil_factors_daily/hac_bandwidth.csv                 |     948 | 42a6145e82765479c2b5374e1a46b5821abeabaa5c50f8a62d2d01f2079733d4 |
| empirical/aqr_hml_devil_factors_daily/hac_delta.csv                     |     431 | 77c8dc024bc145e066eee1945da9b5388fdb4811b6719c3e7d5c77ebb4445244 |
| empirical/aqr_hml_devil_factors_daily/hac_prewhite.csv                  |     290 | 3a4a4dbe212ea5bf911e1686cc407824ad78f8adebcd549d7e5ca90e96a6511f |
| empirical/aqr_hml_devil_factors_daily/holdout_subperiods.csv            |     308 | e6ef8aa482c4be087c834e5d675dd6401bc0d77a9dc9a10176ec93e9f01790e1 |
| empirical/aqr_hml_devil_factors_daily/metadata.json                     |     646 | 6d5c37ca0a0e328fa6f214370bfca08ea5ef40401876d0142fa7d3345763f17f |
| empirical/aqr_hml_devil_factors_daily/methods.csv                       |     607 | 7c89db3d9e4454f60ba34448f04de0619f53989b673f457b9b8ad8d72bf29a13 |
| empirical/aqr_hml_devil_factors_daily/permutation.csv                   |     170 | 80703655e12a1798a689dc8ae2cbd767b3a8f950ac7c3ac69e26f121c68d7a6e |
| empirical/aqr_hml_devil_factors_daily/romano_wolf.csv                   |     132 | 098757fd1101a10171be9fee0c02e53f699e39198a92371863f9241e73d5691b |
| empirical/aqr_hml_devil_factors_daily/selected_counts.csv               |     154 | e4b33aa6a565eedcea98df19c13d3c281993bbceeb16fe988825c3280224b37b |
| empirical/aqr_hml_devil_factors_daily/source_returns_head.csv           |     586 | 6ae3b74e9097b92a997a8093ca9afb9307334ce1b8d56b18640b7edc13673d35 |
| empirical/aqr_hml_devil_factors_daily/source_spec.json                  |     566 | 47b25905e39fd25a996746f5d338dd21c03381d1bb35df723c7e10252398059e |
| empirical/aqr_hml_devil_factors_daily/stationarity.csv                  |     115 | 36899d2621ef32f0d57d39de5954218389f0ba7b97856484075a8841bec311d9 |
| empirical/aqr_hml_devil_factors_daily/threshold_menu.csv                |     418 | 6b240d6e3412d2e75789ddcff3b234422fdceb4172887833840d46d6a643462a |
| empirical/aqr_hml_devil_factors_daily/viability.json                    |    1153 | 4b9bb95358c64fa095c8f9b30bdcf7f12ee2afb60c945ab33fb0765cb6919df1 |
| empirical/aqr_qmj_factors_daily/costs.csv                               |     527 | 9e0ea6cda648f6e9ed3f943de39d6e9ed142f249159856932b27cf5a6eb310c8 |
| empirical/aqr_qmj_factors_daily/data_snooping.csv                       |     170 | c5c490fdb5bb62d5f80864b77763b89b8a2e623f1692946f43b11c686de7bd41 |
| empirical/aqr_qmj_factors_daily/factor_alpha.csv                        |      42 | 404ec6d95ea7b16c58076245ac20dc08b9895c1de904401821cc8d03da0b7031 |
| empirical/aqr_qmj_factors_daily/fixed_b_hac.csv                         |     872 | 0d05d92f43219a969134250e78d25099199e4794bb1a120c88eb98604034d647 |
| empirical/aqr_qmj_factors_daily/hac_bandwidth.csv                       |     952 | 9e360ba2fd3fceca4bf14cf7fc22bd7f43f16f93c5dc28934ab4655f08c3e162 |
| empirical/aqr_qmj_factors_daily/hac_delta.csv                           |     435 | 41b3c613ae80ef20566671f50e17e1022dbb07b13d08d9ce11b36bac2a2a6660 |
| empirical/aqr_qmj_factors_daily/hac_prewhite.csv                        |     295 | 290e26abbdc6f016a62df6c18bf028c3ad9bef6464b85ba78814b6eb6a95e1a7 |
| empirical/aqr_qmj_factors_daily/holdout_subperiods.csv                  |     305 | 3c46daad2bcf5f8fdac0814f5b683f55951946b576e32795e4cc6a1a24001f21 |
| empirical/aqr_qmj_factors_daily/metadata.json                           |     633 | 66702e9ed9b050091535229af6183dc2883509f983aae67b5af94c89754675b7 |
| empirical/aqr_qmj_factors_daily/methods.csv                             |     599 | 06222a2aba3cbfc652e8ec0d709b1dcef73b6b11a42bb245afb3d5d02ce6ee40 |
| empirical/aqr_qmj_factors_daily/permutation.csv                         |     169 | 658023c5b79702a46ec9c674540da9e8e33ef93091d9e99709b47cb3b2f84037 |
| empirical/aqr_qmj_factors_daily/romano_wolf.csv                         |     132 | d0a9318752ac808481eb736f543aa07356d76b8656810ca30902eac9a7f59fe3 |
| empirical/aqr_qmj_factors_daily/selected_counts.csv                     |     154 | 52f75c4e7a0dfee59420d371cfb461379a9d5e99208d6954ef8f72770ba6a269 |
| empirical/aqr_qmj_factors_daily/source_returns_head.csv                 |     586 | 9e4a99b740ef4569baae3f1998b2daedeaaa8f3e44404b844431d787e252c069 |
| empirical/aqr_qmj_factors_daily/source_spec.json                        |     547 | 2b7a71d9b1a438c9976281c410899600cf261591e3225e9eb1f229432781ef85 |
| empirical/aqr_qmj_factors_daily/stationarity.csv                        |     117 | 45e42f2b3afb45863a549ecd14707868b1018844885cce1e20ad9ea4294e6e5e |
| empirical/aqr_qmj_factors_daily/threshold_menu.csv                      |     419 | c41f69971736554b46a4156953d337cdd89b5e348e86882f409504659b3905a1 |
| empirical/aqr_qmj_factors_daily/viability.json                          |    1140 | cdb765dcd4ba33acba1bfdaf618ae51537eb96357bb400fb03975ec7ba5ea43e |
| empirical/aqr_value_momentum_everywhere_monthly/costs.csv               |     527 | b2cbf0491338a4314c45fd691139ebe3659d813cb68af1235c504be3b7c21e28 |
| empirical/aqr_value_momentum_everywhere_monthly/data_snooping.csv       |     169 | 2559180dad3b45c6d342b7b6f7ac44158c7938dc8a27252abf812e048cde8808 |
| empirical/aqr_value_momentum_everywhere_monthly/factor_alpha.csv        |      42 | 404ec6d95ea7b16c58076245ac20dc08b9895c1de904401821cc8d03da0b7031 |
| empirical/aqr_value_momentum_everywhere_monthly/fixed_b_hac.csv         |     860 | d6cb9fb7345741cbad39773974a0676612e1f42a71d57ac28e324fad20cd20c7 |
| empirical/aqr_value_momentum_everywhere_monthly/hac_bandwidth.csv       |     939 | 72c79cc090f22ac10ecd8d21d7f689e1fd222e960cafa14bc7e487c7348c2164 |
| empirical/aqr_value_momentum_everywhere_monthly/hac_delta.csv           |     428 | b782cd8c95b31bf9db78517b4aae5a2a727063c7823ced0b243b763abeafa5f2 |
| empirical/aqr_value_momentum_everywhere_monthly/hac_prewhite.csv        |     288 | 5809b2e2024897ca0c0c0675d5f688826ffacadb3fea394875ce5a9575d70384 |
| empirical/aqr_value_momentum_everywhere_monthly/holdout_subperiods.csv  |     278 | 30283739c33a76bd93f60e363494df2be23dc67bf7626618c94b146e54d2055b |
| empirical/aqr_value_momentum_everywhere_monthly/metadata.json           |     660 | bcca13ba1fab60b07b30365105ecd12d2e45f8e4cb0aa99d0624bd962d83cbc7 |
| empirical/aqr_value_momentum_everywhere_monthly/methods.csv             |     577 | 14ea1587b5fee1d4a6c89b152d38d6d72e4b244026d4abb85b54615dc45c41cd |
| empirical/aqr_value_momentum_everywhere_monthly/permutation.csv         |     169 | 95c664c73483086a5adde6d935a1f923b49ec4458a62cc77605f736c45f944a8 |
| empirical/aqr_value_momentum_everywhere_monthly/romano_wolf.csv         |     132 | 5224e81bd0914e154d04ef6c632c250d1a5b84e37db2de29e7b89d86d534f479 |
| empirical/aqr_value_momentum_everywhere_monthly/selected_counts.csv     |     148 | a5effdab0781fd58b5055323a98552ee52b7f10fa0304481a75ed0caa8570dbb |
| empirical/aqr_value_momentum_everywhere_monthly/source_returns_head.csv |     644 | 51449ea277d0fda9e8a3d36ebb4d92135d325befb21f87f4c88287b9a024bfb7 |
| empirical/aqr_value_momentum_everywhere_monthly/source_spec.json        |     543 | b537e955aa3ce1fa7c3b6b977f3c507523b7c1cc66b244d7c4e21af68ccb9246 |
| empirical/aqr_value_momentum_everywhere_monthly/stationarity.csv        |     150 | 8f9bf18c8628191b5eda4902efec738a2d1dc9b13e4fcb0a792a478c9540636f |
| empirical/aqr_value_momentum_everywhere_monthly/threshold_menu.csv      |     409 | 249948c99ca2e545cff113a74f7f08eb3370a38cc00c18d1f2a1207b74f3ecf6 |
| empirical/aqr_value_momentum_everywhere_monthly/viability.json          |    1167 | f74acf6bb7656335460079a841d5810a7be425d96c3725339651ac7858293ef2 |
| empirical/french_momentum_deciles_daily_rank/costs.csv                  |     876 | c61e75901f44374af7001348999156e6c381dc424b2e006bbeb724f6387e5918 |
| empirical/french_momentum_deciles_daily_rank/data_snooping.csv          |     186 | ba29784e4450ec895a64f16597be38a056dffe72b886f66e84c3a7f243550276 |
| empirical/french_momentum_deciles_daily_rank/factor_alpha.csv           |     191 | fd0877f8aa5950e8225d7c89d7cfa3ee4165586452e25605e21b31bd7bc7b59c |
| empirical/french_momentum_deciles_daily_rank/fixed_b_hac.csv            |     813 | 724ebe7504ddfd7aa52c69f1cab3988eaa9f5c99bd9e6fefa8dc5c92660c29de |
| empirical/french_momentum_deciles_daily_rank/hac_bandwidth.csv          |     952 | 0342095bc26146291654a0f88f6747c0fc38a6dbaaa4115ac6f94c8fd4c0457f |
| empirical/french_momentum_deciles_daily_rank/hac_delta.csv              |     434 | f7dc7ba0e0007f7059c4ed4cd5131b5c1604b9e8ed0c1dd7e43c8c3f77c3e247 |
| empirical/french_momentum_deciles_daily_rank/hac_prewhite.csv           |     293 | 2ae4b22bbbca2afb7301a238bfca8909e697f979d48fcc8c9333fc8d182fdda3 |
| empirical/french_momentum_deciles_daily_rank/holdout_subperiods.csv     |     316 | 011c2bb86736534bccabcd1c53fa7f17ee550060ae1d443a4103c186d46d3263 |
| empirical/french_momentum_deciles_daily_rank/metadata.json              |     677 | b3646d4121ce787a6cca1351c7000da689a9c4ff1cf0222f78257606ff3d8966 |
| empirical/french_momentum_deciles_daily_rank/methods.csv                |     605 | 5c93cc06bad42982c80499a2cb143330b57b505b2546d0de386f48d25c0b638a |
| empirical/french_momentum_deciles_daily_rank/permutation.csv            |     869 | be8ba146ff3839f120c5a62f9475e5ca8b09a6c42d94369063cfb660e89d2582 |
| empirical/french_momentum_deciles_daily_rank/permutation_null.csv       |    3695 | d60112bc81e3b65ca37c5e3c2f1428e6104671cf23f50e53d2452a7860beec49 |
| empirical/french_momentum_deciles_daily_rank/romano_wolf.csv            |     415 | f0a4ab2029b0c04c4691d03f6261cba1b7014300983b6d3137b516e2616705f7 |
| empirical/french_momentum_deciles_daily_rank/selected_counts.csv        |     331 | 428b3dec3dbfac1eef91fb42ee5194552bc819108539c20e7d3d7d211fa76048 |
| empirical/french_momentum_deciles_daily_rank/source_returns_head.csv    |    2530 | 847623c03eae551e5fca462a4dcc583e45678fccac92d07b7440d7c55db57adc |
| empirical/french_momentum_deciles_daily_rank/source_spec.json           |    1119 | 3c0da78a553f3eb194392730b7e1a7b9090b2e89bb1334483b13be269c7657ef |
| empirical/french_momentum_deciles_daily_rank/stationarity.csv           |     115 | a897993c17f41dfb01c50c914ed92da4430f69fac8afa6c0667eaf1e22c76cfd |
| empirical/french_momentum_deciles_daily_rank/threshold_menu.csv         |    1505 | 54e21db882ae4172f7f2c1a5a32e2b7b55590a560f1edf6cdfcb8b2e273e758f |
| empirical/french_momentum_deciles_daily_rank/viability.json             |    1199 | 12bc9f2bdfe84dff775d7f6a06d2778546d4526d22614adcf5bbdfc8324b8ad4 |
| empirical/french_size_bm_daily_dynamic_momentum/costs.csv               |     856 | 5f34264a8ee65947753649fbaddc37aac816f282c5144116e96d6df4b3f5ca46 |
| empirical/french_size_bm_daily_dynamic_momentum/data_snooping.csv       |     183 | a6709dadfcaaa61f3eb82f0229c3a6bbeabede89d574fc69e87e49959bcb2c02 |
| empirical/french_size_bm_daily_dynamic_momentum/factor_alpha.csv        |     190 | 953d04382a879c22964d839cd0d3837662084e8e2942e03fcee509d011fa7ab4 |
| empirical/french_size_bm_daily_dynamic_momentum/fixed_b_hac.csv         |     857 | 711af25cca99307708cbb245036e944f6c0d5667aca9290dac22c3109eb4048c |
| empirical/french_size_bm_daily_dynamic_momentum/hac_bandwidth.csv       |     949 | 84fbca3704d331890c05737a733f795541a8bc98ddf023c36ce53f2a978fa368 |
| empirical/french_size_bm_daily_dynamic_momentum/hac_delta.csv           |     420 | 122dac7e985cf3f97cfa8b26af643294e4babe4f479296d4e95c2ea3f76e0c48 |
| empirical/french_size_bm_daily_dynamic_momentum/hac_prewhite.csv        |     291 | 2336f79dd575710598c086d9377a6b8e02d82b38ab71ef21d6e94faaefdb2470 |
| empirical/french_size_bm_daily_dynamic_momentum/holdout_subperiods.csv  |     299 | cd8a4893b65e6d03a3c1efd250b20bce822405535d71c065957f91be7fbc9f83 |
| empirical/french_size_bm_daily_dynamic_momentum/metadata.json           |     641 | 5be489da8a6bee4dcc64b9c6644ae40d100bd52e6520f926a4aa8961dfe0507f |
| empirical/french_size_bm_daily_dynamic_momentum/methods.csv             |     553 | ff6683c44ae6902e35a30c807a8c6068c9667f3b7c3fdb03bd54a26fe458c65e |
| empirical/french_size_bm_daily_dynamic_momentum/permutation.csv         |     849 | 881677711eecd5165c68b8dc69ef2c9900b4a87ad32062d575e70d77a0deaf06 |
| empirical/french_size_bm_daily_dynamic_momentum/permutation_null.csv    |    3719 | 1c8bbe9e6096665f93bec46f3e742dededa5b41b15e31863536579adb1acd2c5 |
| empirical/french_size_bm_daily_dynamic_momentum/romano_wolf.csv         |     414 | e2c63e728ff215ce2b89f7aaf21b966b1b712b889937b399ccbc8cd54c3583b0 |
| empirical/french_size_bm_daily_dynamic_momentum/selected_counts.csv     |     330 | b1ee839f453cec9bba843a88afdb14ba87b766983fe348b7055c1310be3d8e96 |
| empirical/french_size_bm_daily_dynamic_momentum/source_returns_head.csv |    6055 | 642ab4ea2b71e7b809e80aa43eec38e87a60c9267f92c00b8d1dfd59498d95f4 |
| empirical/french_size_bm_daily_dynamic_momentum/source_spec.json        |     921 | 3647ea1af9041eb5445e6b2fa04db6c2e94af7336583d8e4a88afa75c21fbec6 |
| empirical/french_size_bm_daily_dynamic_momentum/stationarity.csv        |     134 | 0eed591931c7d232880a324d5d180744a18cc13b27724ac5b1ce9a0a09b25102 |
| empirical/french_size_bm_daily_dynamic_momentum/threshold_menu.csv      |    1421 | 273387cc63b87a032447dcd5781853768af687e6c878f016c4177a5807514665 |
| empirical/french_size_bm_daily_dynamic_momentum/viability.json          |    1163 | ebcfd9487b99c8679a84299beb66a6237b583fecf77c577539dbdd8a284bd8d9 |
| empirical/stooq_fixed_etf_daily_dynamic_momentum/failure.json           |    1073 | 78a90d7fc289bde010f458fe6f910c20b774bbcbacce9871cb7df641519c02dd |
| simulation/coverage_01_iid.csv                                          |     636 | 375dedfc34acc1be78d5c93701d1b40cf671a4e25aeaad3d5967634cf615a5d2 |
| simulation/coverage_02_ser_mild.csv                                     |     662 | da21db192bb1847f9d0e05da348930a903c74eed75985e4674b10265bccb553e |
| simulation/coverage_03_ser_strong.csv                                   |     697 | 182bd1b1fff22ea79268550f801c749133f149a679c5534874256ee4d48e619f |
| simulation/coverage_04_xs_mild.csv                                      |     684 | 2d7a63eaa16140dd214c44cc61d32c552dc27d1875bb29c84fd254d001519918 |
| simulation/coverage_05_xs_strong.csv                                    |     707 | c88f79de26077e2e1e008f48c5827983768716d509d7e1a7aa8d317bb11767c0 |
| simulation/coverage_06_both_mod.csv                                     |     669 | bef953f5c84f82df6e232ded2fc17506a23af63a5d6e9a61016ed2a8417212c0 |
| simulation/coverage_07_both_strong.csv                                  |     665 | a5b82ca0b8d96c8aa09dcf5fe58bcd8bec851250a7dbd733aa809fc91693e286 |
| simulation/coverage_08_realistic.csv                                    |     704 | b5f1a443fecfe6af7295e4fc1a7ed6795c6ada9c1f2949d85aa8974e063f3fbd |
| simulation/coverage_09_garch_mild.csv                                   |     688 | 44fdb6fc1600e1012f459a5893c320e35f4dbe6a781b7d5ba44dab522f9ad626 |
| simulation/coverage_10_garch_strong.csv                                 |     708 | 25787801b664187014b965a6041e4e96df6b1f72d4c126b98a3d41ad4e108da8 |
| simulation/coverage_11_multifactor.csv                                  |     695 | 12f70736315b095d1fcd5615aa2dc72a103d6734ffac187b1a1ed80ad6fbe865 |
| simulation/coverage_12_regime.csv                                       |     684 | 57b9a3f39e5dc4c7b4dcdd878bf59b7b5751da8d4188401c897bc415c002f585 |
| simulation/coverage_13_high_ar1.csv                                     |     731 | ac02dfd1379e6f30d0a1675fe00267f9d94752e7a976a5e824f96f58792ec150 |
| simulation/coverage_14_garch_highvol.csv                                |     723 | 3f4074895259caf0d0892cb88096deaf20ee866bc234bdf25bcb0d9d5763ef61 |
| simulation/coverage_all_merged.csv                                      |    8315 | 7c474fcfe935f0c56e3bbece55488707b2c31ee6e50e3e8615c4c1df2bf51e6d |
| simulation/coverage_pivot.csv                                           |     508 | 2f92be89df1df459d45e2c36ea1fdff463ec151ffedbb16811155ae5474348e3 |
| simulation/dgp_configs.csv                                              |    1325 | 913f94d5081bb64c685122f57c14b61212a65751fbc44576c7d5aceaa6f18c8a |
| simulation/power_audit.csv                                              |    1714 | 9aaa383f4cda2be587670f21690959cc791d010c2e2c7291c0a09061f35b3ff7 |
| simulation/power_audit_pivot.csv                                        |     362 | 103361d894740d2f2d0966b3dbbc57ef024516f1746ef22d37009e15fc06411f |
| simulation/size_test_audit.csv                                          |     972 | f7dc5b81d35c129a4f065a913918a68d46c97e5cd6d177c57c6297c876cc097e |
| source_registry_snapshot.json                                           |    5743 | efca32ecd8a7bc727665a1d4c1885f46908d3b742e051a7f5ea64273a8a813d5 |
