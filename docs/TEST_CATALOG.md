# Validation test catalog

244 tests (236 deterministic, 8 LLM-judge), generated from the registry in `ask/validation`. Regenerate with `python -m ask.validation.catalog_doc`.

Required inputs are in **bold**, the rest are optional. Run a test with `run_validation_test` (or as part of `run_validation_suite`); each run is saved with a citable run_id. `describe_test` gives the full description, H0 and references.

## `aml` (15)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `aml.alert_aging` Alert ageing and backlog | Alert analysis | aml | **date**, closed_date, as_of, segment, buckets |
| `aml.alert_funnel` Alert → case → SAR conversion funnel by scenario | Alert analysis | aml | **target**, case, scenario, period, confidence |
| `aml.alert_rate` Alert rate overall and by scenario, segment and period | Alert analysis | aml | **alert**, scenario, segment, period, confidence |
| `aml.atl_threshold_sweep` Above-the-line threshold sweep (alerts, SARs, marginal yield) | Threshold tuning | aml | **value**, **target**, thresholds, n_thresholds, current_threshold, direction |
| `aml.benford` Benford's law first-digit / first-two-digits test | Data quality | aml, general | **amount**, digits, min_amount |
| `aml.btl_bands` Below-the-line population bands around a scenario threshold | Threshold tuning | aml | **value**, **threshold**, direction, bands, id |
| `aml.btl_review` Productive rate in reviewed below-the-line samples (exact CIs) | Threshold tuning | aml | **target**, band, value, threshold, direction, bands, population_counts, confidence |
| `aml.crr_distribution` Customer risk rating: distribution across risk classes over time | Customer risk rating | aml | **grade**, period, order |
| `aml.crr_migration` Customer risk rating migration matrix | Customer risk rating | aml | **id**, **period**, **grade**, order, reference_value, current_value |
| `aml.crr_sar_concordance` Customer risk rating concordance with SAR outcomes | Customer risk rating | aml | **grade**, **target**, order, confidence |
| `aml.data_completeness` Completeness of TM-critical fields | Data quality | aml | **columns**, amount, period, segment, placeholders |
| `aml.sample_size` Sample size for BTL/ATL testing (discovery and estimation sampling) | Threshold tuning | aml, general | confidence, tolerable_rate, expected_errors, population, margin, expected_rate |
| `aml.scenario_overlap` Scenario overlap (Jaccard) and unique SAR contribution of each rule | Alert analysis | aml | **id**, **scenario**, target |
| `aml.screening_effectiveness` Name-screening effectiveness from test cases | Screening | aml | **expected_hit**, **actual_hit**, variation, score, confidence |
| `aml.segmentation_quality` Segmentation / peer-group quality and drift | Segmentation | aml | **segment**, **features**, period, max_silhouette_rows |

## `ccr` (7)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `ccr.cva` Unilateral CVA (and DVA/BCVA) from an exposure profile | CVA | ccr | time_columns, times, mtm, time, path, ee, nee, lgd, hazard_rate, cds_spread, spread_tenors, spreads, discount_rate, own_hazard_rate, own_cds_spread, own_lgd, integration |
| `ccr.exposure_backtest` Exposure back-testing on PIT values (KS, AD, CvM, Berkowitz) | Back-testing | ccr | pit, actual, id, other, forecast_column, n_sims, bins |
| `ccr.exposure_profile` Exposure profile: EE, PFE, ENE, EPE, Effective EE/EPE, EAD | Exposure | ccr | time_columns, times, mtm, time, path, quantile, horizon, alpha |
| `ccr.mpor_effect` Collateral and margin-period-of-risk effect on exposure | Exposure | ccr | **mtm**, **time**, **path**, **collateral**, mpor, horizon |
| `ccr.netting_check` Netting-set aggregation check and netting benefit | Exposure | ccr | **mtm**, **netting_set**, reported, path, time, tolerance |
| `ccr.sa_ccr_ir` SA-CCR exposure for an unmargined interest-rate netting set | Exposure | ccr | **notional**, **start**, **end**, **delta**, **mtm**, maturity, currency, collateral, alpha |
| `ccr.wrong_way_risk` Wrong-way risk: dependence between exposure and counterparty credit quality | Wrong-way risk | ccr | **mtm**, **credit**, time, use_exposure, tail |

## `data` (19)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `data.duplicates` Duplicate rows and duplicate keys | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | keys, date, ignore, round_digits, id, max_examples |
| `data.grubbs` Grubbs test / generalized ESD for outliers in one column | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | **column**, alternative, alpha, max_outliers, id, max_examples |
| `data.leakage_candidates` Target-leakage candidates in the data | Data quality | general, pd, ifrs9, ews, aml, ml_classification | **target**, features, date, observation_date, id, max_examples, dayfirst |
| `data.littles_mcar` Little's MCAR test | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | **features** |
| `data.missing_patterns` Missingness pattern combinations | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | columns, max_patterns, id, max_examples |
| `data.missing_vs_target` Missingness vs target (event rate when missing vs present) | Data quality | general, pd, ifrs9, ews, aml, ml_classification | **target**, columns, id, max_examples |
| `data.missingness` Missing values by column, period and segment | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | columns, period, segment, id, max_examples |
| `data.outliers` Outliers per column (IQR, z-score, robust MAD z-score, percentiles) | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | columns, iqr_k, z, mad_z, lower_pct, upper_pct, id, max_examples |
| `data.profile` Column profile (types, missing, distinct, distribution) | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | columns, top_n, near_constant, high_cardinality |
| `data.range_checks` Automatic range checks (probabilities, amounts, dates) | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | as_of, min_date, prob_columns, amount_columns, date_columns, id, max_examples, dayfirst |
| `data.reconciliation` Reconciliation against a source / reference table | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | **other**, **keys**, compare, tolerance, rel_tolerance, sum_columns, group, max_examples |
| `data.referential_integrity` Referential integrity (foreign key vs reference table) | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | **column**, **other**, other_column, id, max_examples |
| `data.representativeness` Sample vs population representativeness | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | **other**, features, bins, id |
| `data.target_association` Univariate association with the target (AUC/Gini, Cramér's V, IV) | Data quality | general, pd, ifrs9, ews, aml, ml_classification | **target**, features, bins |
| `data.target_sanity` Target / default-definition sanity | Data quality | general, pd, ifrs9, ews, aml, ml_classification | **target**, period, segment, allowed, event_value, as_of, horizon_months, freq, id, max_examples, dayfirst |
| `data.time_consistency` Time consistency per entity (date order, overlapping intervals) | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | **id**, date, start_date, end_date, inclusive_end, max_examples, dayfirst |
| `data.time_coverage` Time coverage: records per period and gaps | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | **period**, freq, segment, dayfirst |
| `data.type_consistency` Data-type and formatting consistency | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | columns, id, max_examples |
| `data.validity_rules` Validity rules (ranges, allowed values, patterns, not-null) | Data quality | general, pd, lgd, ead, ifrs9, ews, aml, satellite, pricing, ccr, var, ml_classification, ml_regression, ml_unsupervised | **rules**, id, max_examples, dayfirst |

## `ead` (5)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `ead.ccf_backtest` CCF back-test t-test (realised vs estimated, per pool and portfolio) | Calibration | ead | actual, ead, limit, drawn, min_undrawn, **predicted**, grade, alternative |
| `ead.ccf_distribution` Realised / estimated CCF distribution and mass at bounds | Distribution | ead | actual, ead, limit, drawn, min_undrawn, predicted, tol, bins |
| `ead.ccf_ranking` CCF ranking power: generalised AUC and rank correlations | Discrimination | ead | actual, ead, limit, drawn, min_undrawn, **predicted**, segment, confidence |
| `ead.coverage_ratio` EAD coverage ratio (predicted vs realised EAD) | Calibration | ead | **ead**, **predicted**, segment |
| `ead.realised_ccf` Realised CCF computation and profile | Replication | ead | **ead**, **limit**, **drawn**, segment, min_undrawn, utilisation_bands |

## `econ` (24)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `econ.autocorrelation` Residual autocorrelation (Durbin–Watson, Breusch–Godfrey, Ljung–Box) | Residual diagnostics | satellite, pd, lgd, ead, ifrs9, ml_regression | target, features, model_kind, date, residuals, lags |
| `econ.bootstrap_stability` Coefficient and selection stability under bootstrap resampling | Model specification | satellite, pd, lgd, ead, ifrs9, ml_regression | **target**, **features**, model_kind, date, n_boot, alpha, block_length, expected_signs |
| `econ.chow_test` Chow test for a structural break at a known date | Structural stability | satellite, pd, lgd, ead, ifrs9, ml_regression | **target**, **features**, model_kind, date, break_at, break_index |
| `econ.coefficients` Coefficient estimates, significance and replication of documented values | Model specification | satellite, pd, lgd, ead, ifrs9, ml_regression | **target**, **features**, model_kind, date, cov_type, hac_lags, alpha, expected_coefficients, expected_signs |
| `econ.cusum` CUSUM stability tests (recursive and OLS-residual CUSUM, CUSUM of squares) | Structural stability | satellite, ifrs9, lgd, ead, pd | **target**, **features**, model_kind, date |
| `econ.diebold_mariano` Diebold–Mariano test of equal forecast accuracy (HLN-corrected) | Predictive accuracy | satellite, ifrs9, lgd, ead, ml_regression | **actual**, **predicted**, date, sample, current_value, **benchmark**, horizon, loss |
| `econ.engle_granger` Engle–Granger cointegration test | Stationarity | satellite, ifrs9 | **target**, **features**, date, trend |
| `econ.forecast_accuracy` Out-of-time forecast accuracy (RMSE, MAE, MAPE, Theil's U) | Predictive accuracy | satellite, ifrs9, lgd, ead, ml_regression | **actual**, **predicted**, date, sample, current_value, benchmark |
| `econ.goodness_of_fit` Goodness of fit (R², AIC/BIC, log-likelihood, pseudo-R², LR test) | Goodness of fit | satellite, pd, lgd, ead, ifrs9, ml_regression | **target**, **features**, model_kind, date |
| `econ.granger_causality` Granger causality of each driver for the target | Stationarity | satellite, ifrs9 | **target**, **features**, date, max_lag, both_directions |
| `econ.heteroskedasticity` Heteroskedasticity (Breusch–Pagan, White, Goldfeld–Quandt, ARCH-LM) | Residual diagnostics | satellite, pd, lgd, ead, ifrs9, ml_regression | **target**, **features**, model_kind, date, sort_by, drop_fraction, arch_lags |
| `econ.influence` Influential observations (Cook's distance, leverage, DFFITS, DFBETAS) | Model specification | satellite, pd, lgd, ead, ifrs9, ml_regression | **target**, **features**, model_kind, date, top_n |
| `econ.johansen` Johansen cointegration rank test (trace and maximum eigenvalue) | Stationarity | satellite, ifrs9 | **columns**, date, det_order, k_ar_diff, confidence |
| `econ.macro_sensitivity` Sensitivity of the prediction to each driver (±k standard deviations) | Sensitivity analysis | satellite, pd, lgd, ead, ifrs9, ml_regression | **target**, **features**, model_kind, date, shock_sd |
| `econ.recursive_coefficients` Recursive (expanding-window) coefficient estimates | Structural stability | satellite, pd, lgd, ead, ifrs9, ml_regression | **target**, **features**, model_kind, date, min_obs, step |
| `econ.residual_normality` Residual normality (Jarque–Bera, Shapiro–Wilk, Anderson–Darling, D'Agostino) | Residual diagnostics | satellite, pd, lgd, ead, ifrs9, ml_regression | target, features, model_kind, date, residuals |
| `econ.robust_se` Classical vs robust standard errors (HC0–HC3, Newey–West HAC) | Model specification | satellite, pd, lgd, ead, ifrs9, ml_regression | **target**, **features**, model_kind, date, hac_lags |
| `econ.scorecard_points` Scorecard points recomputation (PDO scaling) | Model implementation | pd | pdo, base_score, base_odds, pd, features, coefficients, score, round_points, tolerance |
| `econ.specification` Functional-form tests (Ramsey RESET, Harvey–Collier, Rainbow, link test) | Model specification | satellite, pd, lgd, ead, ifrs9, ml_regression | **target**, **features**, model_kind, date, reset_power |
| `econ.sup_f_break` Unknown-date structural break (Quandt–Andrews sup-F, bootstrap p-value) | Structural stability | satellite, ifrs9 | **target**, **features**, model_kind, date, trim, n_boot |
| `econ.unit_root` Stationarity of each series (ADF, Phillips–Perron, KPSS) | Stationarity | satellite, ifrs9 | **columns**, date, regression, differences |
| `econ.vif` Multicollinearity (VIF, condition number, correlations) | Model specification | satellite, pd, lgd, ead, ifrs9, ml_regression | **features**, target |
| `econ.woe_iv` Weight of Evidence, Information Value and WoE monotonicity | Discrimination | pd, ifrs9 | **target**, **features**, bins, bin_edges |
| `econ.zivot_andrews` Zivot–Andrews unit-root test with an endogenous break | Stationarity | satellite, ifrs9 | **columns**, date, regression, trim |

## `ews` (8)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `ews.alert_workload` Alert workload per period | Operational | ews, aml | **id**, **event_date**, signal_date, signals, sig_id, sig_date, sig_type, sig_score, threshold, observation_end, frequency, lookback_days, min_lead_days |
| `ews.dpd_comparison` EWS timing vs days-past-due (does EWS fire before 30 dpd?) | Timeliness | ews | **id**, **dpd_date**, event_date, signal_date, signals, sig_id, sig_date, sig_type, sig_score, threshold, observation_end, lookback_days |
| `ews.hit_rate` EWS hit rate, false-alarm rate and precision | Discrimination | ews, aml | **id**, **event_date**, signal_date, signals, sig_id, sig_date, sig_type, sig_score, threshold, observation_end, lookback_days, min_lead_days, confidence |
| `ews.lead_time` EWS lead time distribution and cumulative capture | Timeliness | ews | **id**, **event_date**, signal_date, signals, sig_id, sig_date, sig_type, sig_score, threshold, observation_end, lookback_days, min_lead_days, lead_grid |
| `ews.persistence` Signal persistence and flip-flop rate | Stability | ews | **id**, **period**, signal, score, threshold |
| `ews.recall_by_horizon` Recall and precision by warning horizon (e.g. 3 / 6 / 12 months) | Timeliness | ews, aml | **id**, **event_date**, signal_date, signals, sig_id, sig_date, sig_type, sig_score, threshold, observation_end, horizons, min_lead_days, confidence |
| `ews.score_lift` Precision@k and lift by score band | Discrimination | ews, aml | **target**, **score**, n_bins, top_k |
| `ews.trigger_performance` Performance per trigger type (hit rate, precision, lift) | Discrimination | ews, aml | **id**, **event_date**, signal_date, signals, sig_id, sig_date, sig_type, sig_score, threshold, observation_end, lookback_days, min_lead_days |

## `fairness` (14)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `fairness.auc_by_group` Discrimination (AUC) by group, with BPSN / BNSP AUC | Fairness | ml_classification, pd, aml, ews | **target**, **protected**, **score**, reference_group |
| `fairness.calibration_by_group` Calibration within groups (observed vs predicted by group and bin) | Fairness | ml_classification, pd, aml, ews | **target**, **protected**, **score**, bins, reference_group |
| `fairness.conditional_parity` Conditional statistical parity within strata (CMH test) | Fairness | ml_classification, pd, aml, ews | **target**, **protected**, score, predicted, threshold, reference_group, favourable_label, **segment** |
| `fairness.counterfactual_flip` Counterfactual flip test of the protected attribute (real model) | Fairness | ml_classification, pd, aml, ews | **model**, **protected**, features, threshold |
| `fairness.demographic_parity` Statistical parity difference and disparate impact ratio | Fairness | ml_classification, pd, aml, ews | **target**, **protected**, score, predicted, threshold, reference_group, favourable_label |
| `fairness.equal_opportunity` Equal opportunity (true-positive-rate gap) | Fairness | ml_classification, pd, aml, ews | **target**, **protected**, score, predicted, threshold, reference_group, favourable_label |
| `fairness.equalized_odds` Equalized odds (TPR and FPR gaps) | Fairness | ml_classification, pd, aml, ews | **target**, **protected**, score, predicted, threshold, reference_group, favourable_label |
| `fairness.error_rate_balance` Error-rate balance (FNR, FPR, FDR, FOR by group) | Fairness | ml_classification, pd, aml, ews | **target**, **protected**, score, predicted, threshold, reference_group, favourable_label |
| `fairness.group_difference_tests` Omnibus tests of group differences (chi-square, Fisher) | Fairness | ml_classification, pd, aml, ews | **target**, **protected**, score, predicted, threshold, reference_group, favourable_label |
| `fairness.group_metrics` Confusion-matrix metrics by protected group | Fairness | ml_classification, pd, aml, ews | **target**, **protected**, score, predicted, threshold, reference_group, favourable_label |
| `fairness.intersectional` Intersectional groups (two protected attributes combined) | Fairness | ml_classification, pd, aml, ews | **target**, **protected**, score, predicted, threshold, reference_group, favourable_label, **protected_2**, alpha |
| `fairness.predictive_parity` Predictive parity (PPV and NPV gaps) | Fairness | ml_classification, pd, aml, ews | **target**, **protected**, score, predicted, threshold, reference_group, favourable_label |
| `fairness.score_distribution` Score distribution by group (SMD, KS, Mann–Whitney) | Fairness | ml_classification, pd, aml, ews | target, **protected**, **score**, reference_group |
| `fairness.treatment_equality` Treatment equality (FN / FP ratio by group) | Fairness | ml_classification, pd, aml, ews | **target**, **protected**, score, predicted, threshold, reference_group, favourable_label |

## `genai` (24)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `genai.answer_correctness` Answer correctness vs reference (LLM judge, graded 0–1) *(LLM judge)* | Answer quality | genai | **answer**, **reference**, question, max_rows |
| `genai.answer_relevance` Answer relevance to the question (LLM judge, 1–5) *(LLM judge)* | Answer quality | genai | **question**, **answer**, max_rows |
| `genai.atomic_facts` Atomic-fact precision / recall / hallucination (LLM judge, FActScore-style) *(LLM judge)* | Answer quality | genai | **answer**, **reference**, contexts, context_separator, max_rows |
| `genai.atomic_facts_lexical` Atomic-fact precision / recall (deterministic, lexical) | Answer quality | genai | **answer**, **reference**, threshold, min_clause_tokens, decimal, segment |
| `genai.atomic_facts_long` Atomic-fact precision / recall for long documents (LLM judge, chunked) *(LLM judge)* | Answer quality | genai | **answer**, **reference**, chunk_chars, fact_batch, source_chars, max_rows |
| `genai.bleu` BLEU (corpus and smoothed sentence-level) | Answer quality | genai | **answer**, **reference**, multi_reference, max_n, lowercase, epsilon, segment |
| `genai.chrf` chrF (character n-gram F-score) | Answer quality | genai | **answer**, **reference**, multi_reference, max_n, beta, segment |
| `genai.citations` Citation validity and lexical support | Groundedness | genai | **answer**, contexts, retrieved_ids, context_separator, citation_pattern, id_base, threshold |
| `genai.context_precision_recall` Context precision and context recall (LLM judge, RAGAS-style) *(LLM judge)* | Retrieval | genai | **question**, **contexts**, **reference**, context_separator, max_rows |
| `genai.exact_match` Exact match (SQuAD-normalised) | Answer quality | genai | **answer**, **reference**, multi_reference, segment |
| `genai.faithfulness` Faithfulness / groundedness to retrieved context (LLM judge) *(LLM judge)* | Groundedness | genai | **answer**, **contexts**, question, context_separator, max_rows |
| `genai.injection_judge` Prompt-injection response grading (LLM judge) *(LLM judge)* | Prompt injection | genai | **probe_id**, **answer**, canary, max_rows |
| `genai.injection_probes` Prompt-injection probe set (to run against the system) | Prompt injection | genai | canary, categories |
| `genai.injection_results` Prompt-injection results: attack success rate | Prompt injection | genai | **probe_id**, **answer**, canary, extra_patterns, segment |
| `genai.length_stats` Answer length statistics | Descriptive | genai | **answer**, reference, segment |
| `genai.lexical_groundedness` Lexical groundedness proxy (context token coverage) | Groundedness | genai | **answer**, **contexts**, context_separator, threshold, segment |
| `genai.numeric_consistency` Numeric consistency of answers vs reference / contexts | Groundedness | genai | **answer**, reference, contexts, context_separator, decimal, rel_tol, ignore_pattern, segment |
| `genai.pairwise_preference` Pairwise A/B preference with position swap (LLM judge) *(LLM judge)* | Answer quality | genai | **question**, **answer**, **answer_b**, reference, max_rows |
| `genai.pii_scan` PII leakage scan of answers | Safety & privacy | genai | **answer**, question, contexts, context_separator |
| `genai.refusal_rate` Refusal and abstention rate (regex library) | Safety & privacy | genai | **answer**, extra_patterns, segment |
| `genai.retrieval_metrics` Retrieval metrics: hit@k, recall@k, precision@k, MRR, nDCG@k, MAP | Retrieval | genai | **retrieved_ids**, **relevant_ids**, k, segment |
| `genai.rouge` ROUGE-1 / ROUGE-2 / ROUGE-L | Answer quality | genai | **answer**, **reference**, multi_reference, segment |
| `genai.self_consistency` Self-consistency across repeated runs of the same question | Consistency | genai | **question**, **answer**, segment |
| `genai.token_f1` Token-level precision / recall / F1 (SQuAD) | Answer quality | genai | **answer**, **reference**, multi_reference, segment |

## `ifrs9` (11)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `ifrs9.coverage_by_stage` Stage mix and ECL coverage ratios by stage / segment over time | Staging | ifrs9 | **stage**, **ead**, **ecl**, period, segment |
| `ifrs9.ecl_recompute` ECL recomputation (12-month / lifetime) and reconciliation | ECL measurement | ifrs9 | **id**, **ecl**, stage, horizon, pd_columns, pd_type, lgd, lgd_columns, ead, ead_columns, eir, eir_rate, term_structure, ts_id, ts_period, ts_pd, ts_lgd, ts_ead, period_length, discount_timing, tolerance, top_n |
| `ifrs9.km_lifetime_pd_backtest` Kaplan–Meier cumulative default curves vs predicted lifetime PD | Lifetime PD | ifrs9, pd | **duration**, **target**, segment, pd_columns, horizons, confidence |
| `ifrs9.lifetime_pd_markov` Lifetime PD term structure from a rating transition matrix (Markov chain) | Lifetime PD | ifrs9, pd | **transition**, from_column, default_state, horizon, grade, pd |
| `ifrs9.lifetime_pd_term_structure` Lifetime PD term structure from per-period PDs (consistency) | Lifetime PD | ifrs9, pd | **pd_columns**, pd_type, segment |
| `ifrs9.pit_calibration_backtest` Point-in-time calibration of 12-month PD per period | Calibration | ifrs9, pd | **target**, **pd**, **period** |
| `ifrs9.pit_macro_correlation` PIT-ness: correlation of the PD time series with a macro variable | PIT / TTC | ifrs9, pd | **pd**, **period**, **feature**, target, max_lag |
| `ifrs9.scenario_weight_sensitivity` Sensitivity of ECL to scenario weights | ECL measurement | ifrs9 | **id**, **scenario**, **ecl**, **weights**, alternative_weights, base_scenario, shift |
| `ifrs9.scenario_weighted_ecl` Probability-weighted ECL over macro scenarios vs reported | ECL measurement | ifrs9 | **id**, **scenario**, **ecl**, **weights**, reported, base_scenario, tolerance |
| `ifrs9.stage_migration` Stage migration matrix between two dates | Staging | ifrs9 | **id**, **stage**, period, reference_value, current_value, stage_to, ead |
| `ifrs9.staging_replication` Stage allocation replication from SICR rules | Staging | ifrs9 | **id**, **stage**, pd, pd_origination, relative_multiple, absolute_change, combine, low_credit_risk_pd, dpd, dpd_backstop, stage3_dpd, watchlist, forbearance, default_flag |

## `lgd` (15)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `lgd.backtest_ttest` LGD back-test t-test (realised vs estimated, per pool and portfolio) | Calibration | lgd | **actual**, **predicted**, grade, alternative |
| `lgd.clar` Cumulative LGD Accuracy Ratio (CLAR) | Discrimination | lgd | **actual**, **predicted** |
| `lgd.cure_rate` Cure rate by segment / period | Workout | lgd | **outcome**, cure_value, segment, period, predicted, actual, confidence |
| `lgd.distribution` Realised / estimated LGD distribution, mass at 0 and 1, bimodality | Distribution | lgd | **actual**, predicted, tol, bins |
| `lgd.downturn_comparison` Downturn vs long-run realised LGD by period | Calibration | lgd | **actual**, **period**, downturn_periods, downturn_flag, ead, predicted |
| `lgd.elbe_backtest` ELBE and LGD in-default vs realised LGD (defaulted exposures) | Calibration | lgd | **actual**, **predicted**, lgd_in_default, grade, alternative |
| `lgd.error_by_segment` LGD accuracy: bias, MAE, RMSE, R² by segment | Calibration | lgd | **actual**, **predicted**, segment, ead |
| `lgd.gauc` Generalised AUC (gAUC) of LGD estimates | Discrimination | lgd | **actual**, **predicted**, realised_bins, initial_gauc, confidence |
| `lgd.incomplete_workouts` Incomplete workouts: open cases and their effect on realised LGD | Workout | lgd | **actual**, **open_flag**, period, ead, predicted |
| `lgd.loss_shortfall` Loss shortfall and mean absolute deviation | Calibration | lgd | **actual**, **predicted**, ead, segment |
| `lgd.pool_homogeneity` Heterogeneity across and homogeneity within LGD pools | Discrimination | lgd | **actual**, **grade**, predicted |
| `lgd.rank_correlation` Pearson, Spearman and Kendall correlation of realised vs estimated LGD | Discrimination | lgd | **actual**, **predicted**, segment, confidence |
| `lgd.realised_from_cashflows` Realised LGD replicated from workout cash flows | Replication | lgd | **id**, **ead**, **date**, **cashflows**, **discount_rate**, cf_id, cf_date, cf_amount, cf_type, cost_values, day_count, actual, tolerance |
| `lgd.wilcoxon` Wilcoxon signed-rank and sign test of realised vs estimated LGD | Calibration | lgd | **actual**, **predicted**, grade, alternative |
| `lgd.workout_length` Recovery time / workout length statistics | Workout | lgd | **date**, **end_date**, as_of, segment, unit |

## `ml` (30)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `ml.classification_metrics` Binary classification performance at a threshold | Performance | ml_classification, pd, aml, ews | **target**, score, model, features, threshold, beta, n_boot |
| `ml.cluster_quality` Cluster validity indices (silhouette, Davies–Bouldin, Calinski–Harabasz, WCSS) | Clustering | ml_unsupervised | **features**, **segment**, standardize, silhouette_sample |
| `ml.cluster_sizes` Cluster size distribution and concentration (HHI) | Clustering | ml_unsupervised | **segment** |
| `ml.cluster_stability` Cluster stability under bootstrap re-clustering (ARI, Jaccard) | Clustering | ml_unsupervised | **features**, **segment**, model, n_boot, standardize |
| `ml.condition_index` Condition number and Belsley variance-decomposition proportions | Multicollinearity | ml_classification, ml_regression, pd, aml, ews, lgd | **features**, include_intercept |
| `ml.correlation_pairs` All pairwise feature correlations, sorted | Multicollinearity | ml_classification, ml_regression, pd, aml, ews, lgd, ml_unsupervised | **features** |
| `ml.cross_validation` k-fold cross-validation of the real model (refit) | Overfitting | ml_classification, ml_regression, pd, aml, ews, lgd | **model**, target, actual, features, k, metrics, threshold |
| `ml.explanation_stability` Stability of feature importance across bootstrap resamples | Explainability | ml_classification, ml_regression, pd, aml, ews, lgd | **model**, target, actual, features, metric, n_boot, n_repeats, top_k, threshold |
| `ml.extrapolation` Out-of-range extrapolation vs the training sample | Robustness | ml_classification, ml_regression, pd, aml, ews, lgd, ml_unsupervised | **features**, sample, reference_value, current_value, other, quantile |
| `ml.gains_lift` Cumulative gains, lift and KS by score band | Discrimination | ml_classification, pd, aml, ews | **target**, score, model, features, bins |
| `ml.leakage_screen` Target-leakage screen: single-feature predictive power | Data leakage | ml_classification, ml_regression, pd, aml, ews, lgd | **features**, target, actual |
| `ml.learning_curve` Learning curve of the real model (refit on growing training sizes) | Overfitting | ml_classification, ml_regression, pd, aml, ews, lgd | **model**, target, actual, features, metric, k, train_sizes, threshold |
| `ml.missing_value_robustness` Robustness to missing / imputed inputs | Robustness | ml_classification, ml_regression, pd, aml, ews, lgd | **model**, target, actual, features, vary, threshold, metric |
| `ml.monotonicity` Monotonicity of the model response vs expected sign | Explainability | ml_classification, ml_regression, pd, aml, ews, lgd | **model**, features, **expected_signs**, grid_points, max_rows, tolerance, grid_lower_quantile, grid_upper_quantile |
| `ml.multiclass_metrics` Multi-class classification performance (macro / micro / weighted) | Performance | ml_classification, pd | **target**, predicted, model, features |
| `ml.noise_robustness` Robustness to Gaussian input noise (real model) | Robustness | ml_classification, ml_regression, pd, aml, ews, lgd | **model**, target, actual, features, perturb, noise_levels, n_repeats, threshold, metric |
| `ml.partial_dependence` Partial dependence and ICE summary of the real model | Explainability | ml_classification, ml_regression, pd, aml, ews, lgd | **model**, features, explain, grid_points, max_rows, grid_lower_quantile, grid_upper_quantile |
| `ml.pca` PCA explained variance, loadings, Bartlett sphericity and KMO | Dimensionality | ml_unsupervised | **features**, standardize, n_components |
| `ml.permutation_importance` Permutation feature importance of the real model | Explainability | ml_classification, ml_regression, pd, aml, ews, lgd | **model**, target, actual, features, metric, n_repeats, threshold |
| `ml.probability_calibration` Probability calibration: reliability table, ECE, Brier decomposition | Calibration | ml_classification, pd, aml, ews | **target**, score, model, features, bins, strategy |
| `ml.regression_metrics` Regression performance (RMSE, MAE, MAPE, R², ...) | Performance | ml_regression, lgd | **actual**, predicted, model, features, n_features |
| `ml.residual_diagnostics` Residual diagnostics by decile of prediction | Performance | ml_regression, lgd | **actual**, predicted, model, features, bins |
| `ml.scenario_stress` Scenario stress test of the model (feature shocks) | Robustness | ml_classification, ml_regression, pd, aml, ews, lgd | **model**, features, **shocks**, shock_type, threshold, weight |
| `ml.sensitivity` One-at-a-time sensitivity of the model score | Robustness | ml_classification, ml_regression, pd, aml, ews, lgd | **model**, features, vary, shift_std, shift_pct, threshold |
| `ml.shap_importance` SHAP global importance and direction of the real model | Explainability | ml_classification, ml_regression, pd, aml, ews, lgd | **model**, features, max_rows, background_rows |
| `ml.surrogate_explainability` SURROGATE explainability (RandomForest stand-in — NOT the model under validation) | Explainability | ml_classification, ml_regression, pd, aml, ews, lgd | **features**, target, actual, score, test_size, max_depth, n_estimators, n_repeats, max_shap_rows |
| `ml.threshold_sweep` Metrics across classification thresholds (Youden, cost, F-beta optima) | Performance | ml_classification, pd, aml, ews | **target**, score, model, features, thresholds, cost_fp, cost_fn, beta |
| `ml.train_test_duplicates` Duplicate rows / IDs across train and test samples | Data leakage | ml_classification, ml_regression, pd, aml, ews, lgd, ml_unsupervised | **sample**, reference_value, current_value, features, id |
| `ml.train_test_gap` Overfitting: train vs test metric gap with bootstrap CI | Overfitting | ml_classification, ml_regression, pd, aml, ews, lgd | target, actual, score, predicted, model, features, **sample**, reference_value, current_value, metrics, threshold, n_boot |
| `ml.vif` Variance inflation factors | Multicollinearity | ml_classification, ml_regression, pd, aml, ews, lgd | **features** |

## `pd` (33)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `pd.adjacent_grade_dr_test` Default-rate heterogeneity between adjacent grades | Rating system | pd, ifrs9, ews | **target**, **grade**, pd, grade_order |
| `pd.auc` AUC (ROC) with DeLong variance and confidence interval | Discrimination | pd, ifrs9, ews, ml_classification, aml | **target**, **score**, higher_is_riskier, confidence |
| `pd.auc_by_period` AUC / Gini by period with confidence intervals | Discrimination | pd, ifrs9, ews, ml_classification, aml | **target**, **score**, higher_is_riskier, **period**, confidence |
| `pd.auc_by_segment` AUC / Gini by segment with confidence intervals | Discrimination | pd, ifrs9, ews, ml_classification, aml | **target**, **score**, higher_is_riskier, **segment**, confidence |
| `pd.auc_change` Change in AUC: current vs development / initial validation | Discrimination | pd, ifrs9, ews, ml_classification, aml | **target**, **score**, higher_is_riskier, sample, reference_value, current_value, other, reference_auc, reference_auc_se, period, se_method |
| `pd.binomial_test` Exact binomial test per grade and portfolio | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd**, grade, grade_order |
| `pd.brier` Brier score with Murphy decomposition and skill score | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd**, grade, bins |
| `pd.calibration_in_the_large` Calibration in the large (observed vs expected defaults) | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd**, weight |
| `pd.calibration_slope` Calibration intercept and slope (logistic recalibration) | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd** |
| `pd.cap_accuracy_ratio` CAP curve and Accuracy Ratio | Discrimination | pd, ifrs9, ews, ml_classification, aml | **target**, **score**, higher_is_riskier |
| `pd.chi_square_grades` Pearson chi-square calibration test across grades | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd**, **grade**, grade_order |
| `pd.default_definition_replication` Replicate the default flag from days past due | Default definition | pd, ifrs9 | **id**, **date**, **dpd**, **target**, past_due_amount, exposure, dpd_threshold, abs_threshold, rel_threshold, window_months |
| `pd.default_rate_series` Default rate time series by period | Default definition | pd, ifrs9, ews | **target**, **period**, pd, weight, confidence |
| `pd.delong_compare` DeLong test: two scores' AUCs on the same sample | Discrimination | pd, ifrs9, ews, ml_classification, aml | **target**, **score**, higher_is_riskier, **benchmark**, benchmark_higher_is_riskier, confidence |
| `pd.divergence` Divergence (separation of score means) | Discrimination | pd, ifrs9, ews, ml_classification, aml | **target**, **score**, higher_is_riskier |
| `pd.ece` Expected calibration error (ECE / MCE) with reliability diagram | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd**, bins, strategy |
| `pd.gini_bootstrap` Bootstrap confidence interval for Gini | Discrimination | pd, ifrs9, ews, ml_classification, aml | **target**, **score**, higher_is_riskier, n_boot, confidence |
| `pd.grade_concentration` Grade distribution and concentration (HHI, ECB Herfindahl test) | Rating system | pd, ifrs9, ews | **grade**, weight, grade_order, sample, reference_value, current_value, other, reference_cv |
| `pd.grade_homogeneity` Homogeneity of default rates within grades by sub-segment | Rating system | pd, ifrs9, ews | **target**, **grade**, **segment**, grade_order |
| `pd.hosmer_lemeshow` Hosmer–Lemeshow goodness-of-fit test | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd**, grade, groups, dof |
| `pd.information_value` Information Value and WoE of grades / score bands | Discrimination | pd, ifrs9, ews, ml_classification, aml | **target**, grade, score, bins |
| `pd.jeffreys_test` Jeffreys test per grade and portfolio (ECB) | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd**, grade, grade_order, weight, confidence |
| `pd.ks` Kolmogorov–Smirnov statistic (max separation) with location | Discrimination | pd, ifrs9, ews, ml_classification, aml | **target**, **score**, higher_is_riskier, bins |
| `pd.long_run_default_rate` Long-run average default rate vs average PD | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd**, **period**, weight |
| `pd.migration_matrix` Migration (transition) matrix and stability metrics | Rating system | pd, ifrs9, ews | **grade**, grade_to, id, period, from_period, to_period, grade_order |
| `pd.multi_period_normal_test` Multi-period normal test of PD calibration (Tasche) | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd**, **period** |
| `pd.normal_test` Normal-approximation (z) test of defaults per grade | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd**, grade, grade_order |
| `pd.overrides` Override analysis (model grade vs final grade) | Rating system | pd, ifrs9, ews | **grade**, **final_grade**, grade_order, target |
| `pd.pr_auc` Precision–recall AUC and average precision | Discrimination | pd, ifrs9, ews, ml_classification, aml | **target**, **score**, higher_is_riskier |
| `pd.rank_correlation` Somers' D and Kendall tau-b between score and default | Discrimination | pd, ifrs9, ews, ml_classification, aml | **target**, **score**, higher_is_riskier |
| `pd.rank_ordering` Rank ordering of default rates across grades | Discrimination | pd, ifrs9, ews | **target**, **grade**, pd, grade_order |
| `pd.spiegelhalter` Spiegelhalter z-test of calibration | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd** |
| `pd.vasicek_test` Vasicek / ASRF correlation-adjusted binomial test | Calibration | pd, ifrs9, ews, ml_classification, aml | **target**, **pd**, grade, grade_order, rho |

## `pricing` (15)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `pricing.bachelier` Bachelier (normal) model benchmark price and Greeks | Benchmarking | pricing | **forward**, **strike**, **maturity**, **vol**, option_type, default_type, rate, rate_value, model_price, model_delta, model_gamma, model_vega |
| `pricing.black76` Black-76 benchmark price and Greeks (options on forwards/futures) | Benchmarking | pricing | **forward**, **strike**, **maturity**, **vol**, option_type, default_type, rate, rate_value, model_price, model_delta, model_gamma, model_vega |
| `pricing.black_scholes` Black–Scholes–Merton benchmark price and Greeks | Benchmarking | pricing | **spot**, **strike**, **maturity**, **vol**, option_type, default_type, rate, rate_value, dividend, dividend_value, model_price, model_delta, model_gamma, model_vega |
| `pricing.bond_analytics` Fixed-coupon bond price, yield, duration and convexity | Curves | pricing | **coupon**, **maturity**, ytm, price, frequency, face, compounding, model_price, model_duration |
| `pricing.calendar_arbitrage` Calendar no-arbitrage: total implied variance increasing in maturity | No-arbitrage | pricing | **strike**, **maturity**, **vol**, forward, spot, rate, rate_value, dividend, dividend_value, tolerance |
| `pricing.curve_diagnostics` Yield-curve diagnostics: discount factors, forwards, consistency | Curves | pricing | **maturity**, discount_factor, zero_rate, compounding |
| `pricing.curve_repricing` Repricing of input bonds from a discount curve | Curves | pricing | **coupon**, **maturity**, **market_price**, **curve**, curve_time, curve_value, curve_value_type, frequency, face, price_type |
| `pricing.fd_greeks` Finite-difference sensitivities vs reported Greeks | Sensitivities | pricing | **price**, **price_up**, **price_down**, **bump**, level, reported_first, reported_second |
| `pricing.implied_vol` Implied volatility solver (Brent) and round-trip check | Benchmarking | pricing | **price**, **strike**, **maturity**, **underlying**, model, option_type, default_type, rate, rate_value, dividend, dividend_value, reported_vol |
| `pricing.mc_convergence` Monte Carlo convergence diagnostics from simulated payoffs | Monte Carlo | pricing | **payoff**, benchmark |
| `pricing.mc_gbm_check` Monte Carlo GBM European option vs Black–Scholes | Monte Carlo | pricing | **spot**, **strike**, **maturity**, **vol**, rate, dividend, option_type, n_paths, antithetic, model_price |
| `pricing.pnl_explain` P&L explain: risk-based vs full-revaluation P&L | P&L explain | pricing | **actual**, **predicted** |
| `pricing.put_call_parity` Put–call parity check across an option chain | No-arbitrage | pricing | **strike**, **maturity**, **call_price**, **put_price**, spot, forward, rate, rate_value, dividend, dividend_value |
| `pricing.stress_repricing` Stress / scenario repricing of a Black–Scholes option book | Stress testing | pricing | **spot**, **strike**, **maturity**, **vol**, option_type, default_type, rate, rate_value, dividend, dividend_value, quantity, spot_shocks, vol_shocks |
| `pricing.strike_arbitrage` Strike no-arbitrage checks: monotonicity, slope, convexity (butterfly) | No-arbitrage | pricing | **strike**, **maturity**, price, vol, forward, option_type, default_type, rate, rate_value, tolerance |

## `stability` (4)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `stability.csi` Characteristic Stability Index (CSI) for many variables | Stability | pd, lgd, ead, ifrs9, ews, aml, ml_classification, ml_regression, ml_unsupervised, satellite, general | features, sample, reference_value, current_value, other, bins |
| `stability.distribution_tests` Two-sample distribution tests (KS, AD, Wasserstein, JS) | Stability | pd, lgd, ead, ifrs9, ews, aml, ml_classification, ml_regression, ml_unsupervised, satellite, general | **column**, sample, reference_value, current_value, other, bins |
| `stability.psi` Population Stability Index (PSI) | Stability | pd, lgd, ead, ifrs9, ews, aml, ml_classification, ml_regression, ml_unsupervised, satellite, general | **column**, sample, reference_value, current_value, other, bins, categorical |
| `stability.psi_over_time` PSI of each period against a reference period | Stability | pd, lgd, ead, ifrs9, ews, aml, ml_classification, ml_regression, ml_unsupervised, satellite, general | **column**, **period**, reference_value, bins |

## `var` (20)

| Test | Area | Model types | Inputs |
|---|---|---|---|
| `var.berkowitz` Berkowitz likelihood-ratio test on PIT values | Back-testing | var | **pit**, date |
| `var.christoffersen` Christoffersen independence and conditional coverage tests | Back-testing | var | **pnl**, **var**, date, confidence, loss_positive, var_sign |
| `var.duration_weibull` Christoffersen–Pelletier Weibull duration test | Back-testing | var | **pnl**, **var**, date, confidence, loss_positive, var_sign |
| `var.dynamic_quantile` Engle–Manganelli Dynamic Quantile (DQ) test | Back-testing | var | **pnl**, **var**, date, confidence, loss_positive, var_sign, lags, include_var |
| `var.es_acerbi_szekely` Acerbi–Szekely ES back-tests Z1 and Z2 | Back-testing | var | **pnl**, **var**, **es**, date, confidence, loss_positive, var_sign, distribution, df, n_sims |
| `var.es_mcneil_frey` McNeil–Frey exceedance-residual test for ES | Back-testing | var | **pnl**, **var**, **es**, date, confidence, loss_positive, var_sign, volatility, n_boot |
| `var.exceptions` VaR exception series and count | Back-testing | var | **pnl**, **var**, date, confidence, loss_positive, var_sign |
| `var.haas_mixed_kupiec` Haas mixed Kupiec test | Back-testing | var | **pnl**, **var**, date, confidence, loss_positive, var_sign |
| `var.hs_rolling_replication` Rolling historical-simulation VaR vs reported VaR | Replication | var | **pnl**, **var**, date, confidence, loss_positive, var_sign, window, method |
| `var.hs_var` Historical-simulation VaR/ES replication from a P&L vector | Replication | var | scenarios, pnl, layout, confidence, loss_positive, window, method, reported_var, reported_es |
| `var.kupiec_pof` Kupiec proportion-of-failures (POF) test | Back-testing | var | **pnl**, **var**, date, confidence, loss_positive, var_sign |
| `var.kupiec_tuff` Kupiec time-until-first-failure (TUFF) test | Back-testing | var | **pnl**, **var**, date, confidence, loss_positive, var_sign |
| `var.loss_functions` VaR / ES scoring functions (quantile loss, Lopez, FZ0) | Back-testing | var | **pnl**, **var**, date, confidence, loss_positive, var_sign, es, benchmark_var |
| `var.parametric_var` Parametric (variance–covariance) VaR replication | Replication | var | **returns**, **weights**, confidence, horizon_days, ewma_lambda, include_mean, reported_var |
| `var.pit_uniformity` Uniformity tests of PIT values (KS, Anderson–Darling, CvM, chi-square) | Back-testing | var | **pit**, date, n_sims, bins |
| `var.pla_test` FRTB P&L attribution test (Spearman correlation and KS distance) | P&L attribution | var | **hpl**, **rtpl**, date |
| `var.rolling_exceptions` Rolling exception count (e.g. 250-day window) | Back-testing | var | **pnl**, **var**, date, confidence, loss_positive, var_sign, window |
| `var.scaling_check` Square-root-of-time scaling check (1-day vs h-day) | Replication | var | **pnl**, date, confidence, loss_positive, horizon, method |
| `var.stressed_period` Stressed-period identification (window with maximum VaR/ES) | Replication | var | **pnl**, date, confidence, loss_positive, window, measure |
| `var.traffic_light` Basel traffic-light test | Back-testing | var | **pnl**, **var**, date, confidence, loss_positive, var_sign |
