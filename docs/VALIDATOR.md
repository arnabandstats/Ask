# The validator agent: capability map, tools and build plan

Ask's chat agent works as a senior model validator in a bank's model-risk-management function.
It reviews a model's documentation, code and data and tests the model quantitatively. It writes
findings with evidence and can draft a validation report.

One rule shapes the design: **tools compute, the language model interprets**. Every number in an
answer comes from a deterministic Python function. The LLM chooses which functions to run, maps
the data's columns to their inputs, and explains the results. The one exception is the LLM-judge
GenAI tests, which are labelled as such everywhere they appear.

## 1. Capability map

**Who does what:** `Tool` = a deterministic Python function; `LLM` = the language model;
`Both` = the tool produces the evidence and the LLM judges it.

### Documentation review

| Task | Who | How |
|---|---|---|
| Completeness against regulation (ECB Guide, CRR, EBA GL, IFRS 9, EU AI Act, SR 11-7) | Both | `check_documentation` finds the document lines that match each checklist requirement (13 standards, 332 requirements). The LLM reads the cited lines and judges whether each requirement is actually met. |
| Consistency between documentation, code and data | Both | `cross_check` matches stated parameters (e.g. "PD floor 0.03%") to values in code, and documented variables to data columns. The LLM traces each methodology step into the code with `search`, `grep` and `read_file`. |
| Assumptions, limitations and use: stated and justified? | LLM | Judgement. Every statement must cite `[file:Lx-y]`. |

### Data review

| Task | Who | How |
|---|---|---|
| Quality: missing values, outliers, duplicates, ranges, types | Tool | `data.*` tests; every issue lists sample rows. Big Unity Catalog tables: `profile_table` runs in SQL on the full population. |
| Representativeness and stability | Tool | `data.representativeness`, `stability.*` (PSI, CSI, KS, Anderson–Darling, JS, PSI over time). |
| Default / target definition | Tool | `pd.default_definition_replication` (days past due plus materiality), `data.target_sanity`, `ifrs9.staging_replication`. |
| Lineage and reconciliation | Tool | `data.reconciliation`, `data.referential_integrity`. |

### Code review

| Task | Who | How |
|---|---|---|
| Logic errors, hard-coding, leakage, non-reproducibility, dependencies | Both | `scan_code` runs 44 rules (Python checks parse the code; SQL/R/SAS/config checks use regex), each finding citable. The LLM confirms each one in context and looks for errors no rule can see (e.g. a formula that differs from the documented one). |
| Implementation matches documented methodology | Both | `cross_check`, then the LLM reads the code. |
| Replicate key steps | Tool | Independent recomputation: default flag, IFRS 9 ECL, staging, realised LGD from cash flows, CCF, historical-simulation VaR/ES, scorecard points, benchmark pricers. The developer's own code is never executed. |

### Conceptual soundness

| Task | Who | How |
|---|---|---|
| Methodology, variable selection, segmentation, assumptions | Both | Evidence from `econ.*` (refit, diagnostics, breaks, bootstrap selection stability), `econ.woe_iv`, `pd.grade_homogeneity` and `aml.segmentation_quality`. The LLM writes the assessment. |
| Benchmark / challenger comparison | Tool | `pd.delong_compare`, `econ.diebold_mariano`, `var.loss_functions`, `genai.pairwise_preference` (LLM judge). |

### Quantitative testing

| Task | Who | How |
|---|---|---|
| Discrimination: AUC/Gini, KS, CAP | Tool | `pd.auc` (DeLong CI), `pd.ks`, `pd.cap_accuracy_ratio`, `pd.auc_change` (ECB), `lgd.gauc`. |
| Calibration: binomial, Jeffreys, Hosmer–Lemeshow, Brier | Tool | `pd.binomial_test`, `pd.jeffreys_test`, `pd.hosmer_lemeshow`, `pd.brier`, `pd.vasicek_test`, `pd.spiegelhalter`, and more. |
| Stability / back-testing: PSI, CSI, Kupiec, Christoffersen | Tool | `stability.*`, and in `var.*`: Kupiec POF/TUFF, Christoffersen, traffic light, Weibull duration, DQ, Acerbi–Szekely, McNeil–Frey, PLA. |
| Sensitivity, stress, scenarios | Tool | `ml.sensitivity`, `ml.scenario_stress`, `econ.macro_sensitivity`, `ifrs9.scenario_weight_sensitivity`, `pricing.stress_repricing`. |
| ML / AI: overfitting, SHAP, fairness, robustness | Tool | `ml.*` and `fairness.*` run on the real model loaded with `load_model`. A stand-in (surrogate) model is used only when explicitly requested and is labelled SURROGATE. |
| GenAI: hallucination, groundedness, injection, retrieval | Both | Deterministic: retrieval metrics, ROUGE/BLEU/chrF, numeric consistency, lexical atomic facts, 40 injection probes, PII scan. LLM judge: atomic facts (FActScore-style precision/recall), faithfulness, correctness, context precision/recall. |
| AML / EWS: alert rates, below-the-line testing, threshold tuning | Tool | `aml.*` (BTL with Clopper–Pearson CIs, ATL threshold sweep, sample size, Benford, screening). `ews.*` covers hit rate, lead time and lift. |

### Monitoring, findings and report

| Task | Who | How |
|---|---|---|
| Outcome analysis and monitoring (KPIs over time) | Tool | Tests by period: `pd.auc_by_period`, `stability.psi_over_time`, `var.rolling_exceptions`, and others. Traffic lights are configured per bank; none are set yet. |
| Findings: ID, area, evidence, root cause, impact, severity, remediation, owner | LLM | Two checks enforce the format: a finding with no evidence (`[file:Lx-y]`, `[test:<run_id>]` or table + column) goes back for repair, and so does a "positive" finding. |
| Validation report | LLM | Structure fixed by the system prompt. Everything in it must already be backed by tool output. |

## 2. Agent tools

These are the existing tools: `load_path`, `overview`, `search`, `grep`, `read_file`,
`list_files`, `data_overview`, `query_data`, `make_chart`, `compare`, `run_data_quality`,
`list_tests` and `run_tests`. The last two run the legacy battery with Excel/HTML export.

New tools:

| Tool | Purpose | Inputs | Output | Model types |
|---|---|---|---|---|
| `list_validation_tests` | Catalog of the 243 tests | model_type, area, query | Tests grouped by area, each with the inputs it needs | all |
| `describe_test` | One test in full | test_id | Description, H0, parameters, references | all |
| `run_validation_test` | Run one test | test_id, params (parameter → column/value), source | Statistics, tables, charts, provenance, run_id | all |
| `run_validation_suite` | Run every test whose inputs are covered | model_type, columns, areas, include_judge | One line per test with its run_id, plus a not-run list | all |
| `get_test_run` | Recall a saved run | run_id | All tables and provenance | all |
| `load_model` | Load a model artefact for model-based tests | .pkl/.joblib path, `runs:/`, `models:/` | Model name, type, SHA-256, features | ML, PD, AML, … |
| `load_table` | Unity Catalog table as a data source | table, columns, where, key, max_rows | Rows; above the limit, a hash sample that is identical on every run | all |
| `profile_table` | Full-population data-quality profile in SQL | table, columns, key, where | Missing, distinct, min/max/mean/std, duplicate keys | all |
| `query_table` | One read-only SELECT (e.g. GROUP BY to grade level) | sql, name | Result loaded as a data source | all |
| `scan_code` | Static code review | source, rules, path_glob | Citable findings `[file:Lx-y]` | all |
| `list_standards` | Available checklists | model_type | 13 standards | all |
| `check_documentation` | Documentation against a standard | standard, source | Per requirement: matching lines, or none | all |
| `cross_check` | Documentation vs code vs data | doc_source, code_source, data_source | Stated values and their code matches; variables vs columns | all |

The full test list, with inputs, is in [TEST_CATALOG.md](TEST_CATALOG.md).

### Reproducibility and traceability

- **What the run_id covers.** Every run records the test id, the parameters, a SHA-256 of the
  exact columns read, the seed, the library versions, and a hash of the test's source code. It
  also records the model artefact's SHA-256 and the judge model where they apply.
- **What the run_id is.** It is derived from all of the above. The same data, code and
  parameters always give the same run_id and the same numbers. Change any one of them and the
  run_id changes.
- **Where runs are saved.** Each run is a JSON file in `<data dir>/validation_runs/`.
- **How citations are checked.** Answers cite runs as `[test:<run_id>]`. An answer that cites
  a run that never happened, or reports test results without any citation, gets one repair
  round.
- **Randomness.** All randomness (bootstraps, noise, Monte Carlo) uses a fixed seed
  (`ASK_VALIDATION_SEED`).
- **Big tables.** They are sampled with `pmod(xxhash64(key), 1e6) < k` and ordered by the same
  hash, so a rerun reads the same rows in the same order.
- **LLM-judge tests.** They run at temperature 0 with a pinned model, and replies are cached by
  prompt hash in `<data dir>/judge_cache.jsonl`, so a rerun replays the same judgements. Their
  results are always labelled as judge-based.

## 3. System prompt

The prompt is in [`ask/agent/prompts.py`](../ask/agent/prompts.py). It covers:

- the validator role;
- the strict split between computing and interpreting;
- no house thresholds, and no unsupported qualitative labels;
- a seven-step validation workflow mapped to the tools;
- the findings format and the report structure;
- the faithfulness rules: `[path:Lx-y]` for code and documents, `[test:<run_id>]` for tests,
  and "General knowledge:" for anything else.

## 4. Build plan

**Phase 1 (this branch): test library and agent integration**

- **Test library:** 243 tests across PD, LGD, EAD, IFRS 9, EWS, satellite, VaR/ES, pricing,
  CCR/CVA, AML, ML, fairness, GenAI and data quality. Each has known-answer unit tests.
- **Review tools:** the static code scanner (44 rules) and 13 regulatory checklists.
- **Data access:** Unity Catalog tables (Spark or SQL warehouse) and model artefacts
  (pickle/joblib/MLflow).
- **Checks on answers:** provenance and run IDs; checks on `[test:]` citations and on
  findings.
- **Legacy test engine fixes:** seeded randomness, image-noise wrap-around fixed, robustness
  scored on hold-out data, and every surrogate result labelled.

**Phase 2: thresholds and monitoring**

- A versioned `thresholds.yaml` editable in Settings: PSI bands, AUC drop, Jeffreys p-value,
  Basel zones already defined.
- Traffic-light columns on each result, stamped with the threshold version.
- Monitoring packs: a fixed list of tests per model type, run per period, with trend charts.

**Phase 3: findings register and report export**

- A findings register (Delta table on Databricks, SQLite locally) with status tracking.
- DOCX validation report on the bank's template, assembled from saved runs and findings.
- Appendix of runs (run_id, data fingerprint, code hash) for audit.

**Phase 4: deeper replication and coverage**

- Execution of the developer's training pipeline in an isolated Databricks Job (not inside
  the app), to compare reproduced results with documented ones.
- SA-CCR for further asset classes, full IMM exposure simulation benchmarks, and more pricing
  models (local vol, SABR/Hull–White).
- Internal MRM policy checklists in the same YAML format as the external standards.
- GenAI: run the injection suite directly against a serving endpoint; red-team datasets.

## Limits to know

- **Checklists:** they paraphrase public texts and cite article or paragraph numbers only
  where certain. Review them against the current regulation before relying on them.
- **Documentation evidence:** `check_documentation` finds matching text, not compliance. The
  agent is instructed to read the lines and judge adequacy.
- **Model files:** loading a pickle runs code from the file, so load artefacts only from the
  model owner's controlled location.
- **Unity Catalog on a cluster:** when Ask runs as a separate process on the driver, there is
  no Spark session, so set a SQL warehouse ID in Settings → Validation.
