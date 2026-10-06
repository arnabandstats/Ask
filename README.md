# Ask

**Chat with a code repository, documents or a dataset, from one chat box.**
Load material by pasting its path, ask questions in plain language, run built-in
deterministic model-validation tests, and compare files, folders and tables. Answers about
code and documents carry line-level citations that are checked automatically against the
lines the assistant actually read.

Built for model-risk and model-validation work: the assistant acts as a **senior model
validator**. It has a library of **243 statistical tests**: 236 deterministic, plus 7 LLM-judge
tests for GenAI that are labelled as such. They cover PD, LGD, EAD, IFRS 9, early-warning,
VaR/ES, pricing, CCR/CVA, AML, ML, fairness, GenAI and data quality. It also has a static code
scanner, checklists for 13 regulations and standards, and Unity Catalog and model-artefact
access. Every number it reports comes from a test run with a citable, reproducible run ID.
See [docs/VALIDATOR.md](docs/VALIDATOR.md) for the capability map and
[docs/TEST_CATALOG.md](docs/TEST_CATALOG.md) for every test.

---

## Contents

- [What you can do](#what-you-can-do)
- [Quick start (local)](#quick-start-local)
- [Using the app](#using-the-app)
- [Model validation](#model-validation)
- [Built-in deterministic tests (legacy battery)](#built-in-deterministic-tests-legacy-battery)
- [How it works](#how-it-works)
- [Faithfulness: how answers are checked](#faithfulness-how-answers-are-checked)
- [Configuration](#configuration)
- [Running on Databricks](#running-on-databricks)
- [Security and privacy](#security-and-privacy)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

---

## What you can do

| Goal | Example message |
|---|---|
| Understand a repository | `load C:/projects/pd_model` → `how is the PD model calibrated?` |
| Get an overview of a folder of several projects | `load C:/work/repos` → `what's happening in these repos?` |
| Read documents (PDF, Word, Markdown, text) | `load "C:/policies/model_policy.pdf"` → `what does it say about annual validation?` |
| Ask about data | `read the data from D:/data/portfolio.csv` → `default rate by segment, as a chart` |
| Validate a model | `validate the calibration and discrimination of this PD model` |
| Back-test a market-risk model | `back-test the VaR in pnl_var.csv at 99%` |
| Review code and documentation | `scan the code for leakage and hard-coded values`, `check the documentation against the CRR IRB requirements` |
| Test a model artefact | `load the model /Volumes/risk/models/pd.pkl` → `SHAP and noise robustness on the oot sample` |
| Evaluate a RAG system | `atomic-fact precision and recall of the answers against the ground truth` |
| Read a Unity Catalog table | `load main.risk.pd_snapshot (columns pd, default_flag, rating, year)` |
| Run the legacy test battery | `run the classification tests on this data` |
| Check data quality | `check the data quality` |
| Compare two files, folders, documents or tables | `compare v1.csv with v2.csv`, `compare the two policies` |
| Check code against a document | `does the code implement section 4 of the methodology?` |
| Load from Databricks | `load /Workspace/Users/you@company.com/MLOps`, `read /Volumes/cat/schema/vol/data.csv` |

There are no modes, tabs or tags: type what you want and the assistant decides which
tools to use. Several sources can be loaded at once and referred to by name.

---

## Quick start (local)

Requirements: **Python 3.10+** and an **OpenAI API key** (or an Azure OpenAI deployment).

```bash
git clone https://github.com/arnabandstats/Ask.git
cd Ask
python -m venv .venv
.venv\Scripts\activate          # Windows;  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Windows;  macOS/Linux: cp .env.example .env
```

Put your key in `.env` (this file is git-ignored and never committed):

```ini
OPENAI_API_KEY=sk-...
```

Start the app:

```bash
streamlit run app.py
```

It opens at <http://localhost:8501>. Startup takes a few seconds: nothing heavy is loaded
until you use it (no embeddings, no index build; the statistics libraries are imported
only when a test runs).

### Try it with the bundled demo data

```
read the data from demo data/Data - cat.xlsx
run the classification tests, observed Y, predicted Y_Pred, split Test_Train_Flag
```

`demo data/Data - num.xlsx` is a regression example (`Y`, `Y_pred`, `Test_Train_Flag`).
The assistant usually infers these columns itself; naming them just removes any guesswork.

---

## Using the app

### The screen

- **Chat** in the middle: your messages on the right in a grey bubble, answers on the
  left. While it works you see three animated dots and the current step
  ("Searching for …", "Reading train.py", "Running tests", "Checking citations").
- **Sidebar**: **＋** starts a new chat; **Recents** lists saved chats (click to reopen);
  **⚙ Settings** at the bottom.
- **Settings** (a dialog, hidden until opened):
  - **General**: tool name, *Think deeper*, default and think-deeper models, *Verify
    citations*.
  - **Loaded sources**: what this chat has loaded, with **Unload**.
  - **Built-in tests**: the full list of deterministic tests.
  - **This chat**: rename or delete it.

### Loading material

Paste a path in any form; quotes are optional unless a path contains spaces and is
followed by more words.

```
load C:\Users\me\projects\pd_model
load "C:/My Projects/model x"
read the data from /home/me/data/portfolio.parquet
load /Workspace/Users/you@company.com/MLOps          (Databricks)
```

A message that only loads something is handled instantly, without calling the model, and
confirmed in one line ("Loaded repo **pd_model**: 214 files, 18,302 lines (.py 120, …)").
A message that loads *and* asks (`load X and explain the training loop`) goes straight to
the assistant.

| Kind | What is read |
|---|---|
| **Repository** (a folder with code) | Code and text files (`.py`, `.ipynb`, `.sql`, `.r`, `.scala`, `.yaml`, `.json`, `.toml`, …) plus the docs inside it (README, PDF, DOCX). Caches, virtual environments, `.git` and binaries are skipped. |
| **Workspace of several projects** | A folder whose sub-folders are separate projects (each with a README, `.git`, `pyproject.toml`, `requirements.txt`, …) is detected and described project by project. |
| **Documents** | PDF (page markers kept), Word (headings and tables), Markdown, text, RTF/TeX, HTML. |
| **Data** | CSV, TSV, Excel (all sheets), Parquet, Feather, pickle. |
| **Databricks** | `/Workspace/...` folders, files and notebooks; `/Volumes/...` files and folders. See [Running on Databricks](#running-on-databricks). |

Reloading the same path replaces it (no duplicates) and only re-reads files that changed.
Extracted PDF/Word text is cached on disk, so large documents reopen instantly.

### Asking questions

- **Code and documents**: answers cite exact lines, shown compactly as
  `train.py · L40–58`. A **Sources** panel under the answer lists each citation with the
  quoted lines and whether it was verified.
- **Data**: numbers are computed with pandas, never estimated. Ask for a table and it is
  shown under the answer. Ask for a chart in plain words and you get an interactive one:

  | Ask for | You get |
  |---|---|
  | "histograms of all numerical variables" | a grid with one histogram per numeric column (also box / violin plots) |
  | "distribution of revenue", "box plot of profit by store" | a single histogram / box / violin |
  | "how many rows per store" | category counts |
  | "revenue by store", "profit over time" | bar / line / area (repeated values aggregated) |
  | "monthly revenue trend" | a chart of a computed result (e.g. a monthly sum) |
  | "Y vs Y_pred with a trend line" | scatter (large tables sampled) with an OLS line |
  | "share of train vs test" | pie |
  | "correlation heatmap" | correlations of the numeric columns (or a count table of two categories) |

  Column names don't need exact case; typos get a "did you mean…".
- **General knowledge** (e.g. "what is PSI?") is answered too, labelled
  *General knowledge* so it is never confused with facts about your material.
- **Follow-ups** remember earlier results in the chat (e.g. "explain the AUC you got").

### Comparing

| Compare | Result |
|---|---|
| Two files | Line similarity and an exact line diff |
| Two folders / repos | Identical, changed, added and removed files, most-changed first |
| A file or document with a folder | The closest matching files, and an exact diff with the best match when it is a near-copy |
| Two tables | Schema differences, dtype changes, per-column statistics (mean, std, min, max, nulls, new categories), cell-level differences for aligned tables |
| Document vs code ("does the code implement…") | Each requirement marked implemented / partly / not found, citing both sides |

### Chats are saved

Every chat (messages, charts, tables and the list of loaded sources) is stored in a local
SQLite file, `ask_data/chats.db`. Reopening a chat restores its messages and reloads its
sources. Chats are titled after their first question (or "Chat · <source>" when they start
with a load).

### Renaming the tool

Settings → General → **Tool name** (default **Ask**) → **Save**. The new name appears in
the browser tab and the assistant uses it to refer to itself. It is stored in
`ask_data/preferences.json`.

---

## Model validation

Ask the way you would brief a colleague: "validate this PD model", "back-test the VaR",
"is the documentation complete for the AI Act?", "write the findings". The agent works
through documentation, consistency, code, data, conceptual soundness, quantitative tests
and monitoring. It picks tests from the library and maps your columns to their inputs.

**What it can use:**
- **Test library** (`ask/validation/t_*.py`): 243 registered tests, each a plain Python
  function with known-answer unit tests.
  - Areas: discrimination (AUC with DeLong CI, KS, CAP, ECB AUC-change test), calibration
    (binomial, Jeffreys, Hosmer–Lemeshow, Spiegelhalter, Vasicek, Brier, Tasche multi-period),
    rating systems (HHI, migration matrices, MWB), LGD and CCF back-tests (gAUC, loss
    shortfall, t-tests, CLAR), IFRS 9 (ECL and staging replication, Kaplan–Meier lifetime PD),
    VaR/ES back-tests (Kupiec, Christoffersen, traffic light, duration, DQ, Acerbi–Szekely,
    PLA), pricing benchmarks and arbitrage checks, CCR exposure profiles and CVA,
    econometric diagnostics, ML performance and robustness on the real model, fairness,
    AML (BTL/ATL, Benford, sample sizes), EWS, GenAI and data quality.
  - See [docs/TEST_CATALOG.md](docs/TEST_CATALOG.md).
- **Code review:** `scan_code` runs 44 rules (unseeded randomness, leakage, hard-coding,
  silent errors, secrets, unpinned dependencies, notebook order, SQL), each with a citable
  line.
- **Documentation review:** `check_documentation` checks against 13 checklists: ECB Guide
  to internal models, CRR IRB, EBA GL 2017/16, 2016/07 and 2019/03, IFRS 9, CRR market-risk
  IMA/FRTB, CCR IMM, EU AI Act, AML transaction monitoring, SR 11-7, plus generic and GenAI
  documentation lists.
  - The tool finds matching lines; the agent reads them and judges adequacy.
  - `cross_check` compares stated parameters and variables with the code and the data.
- **Model artefacts:** `load_model` takes .pkl/.joblib files or MLflow `runs:/` and
  `models:/` URIs (mlflow needed). ML, fairness and robustness tests score the real model.
  A surrogate is used only when asked for, and labelled SURROGATE.
- **Unity Catalog:** `load_table`, `profile_table` and `query_table` work through Spark,
  or through a SQL warehouse (Settings → Validation).
  - Tables above `ASK_MAX_TABLE_ROWS` are sampled by hash, so every run reads the same rows.
  - Profiles and aggregations run on the full population.

**Reproducible and traceable:**
- **What's recorded:** every run saves its parameters, a SHA-256 of the data it read, the
  seed, the library versions, a hash of the test's code and the run ID to
  `ask_data/validation_runs/<run_id>.json`. The same inputs and code always give the same
  run ID and numbers.
- **Citations:** answers cite results as `[test:<run_id>]`. The *Test runs* panel under an
  answer shows each run's provenance.
- **Thresholds:** none are configured yet. Results are reported as statistics, p-values and
  confidence intervals; zones appear only where a test defines them (Basel traffic light,
  FRTB PLA).
- **LLM-judge tests** (GenAI atomic facts, faithfulness, correctness, …) use a pinned model
  at temperature 0. Replies are cached by prompt hash, so a rerun replays the same
  judgements. They are always labelled as judge-based.

**Findings and reports:**
- **Finding format:** findings come as ID, area, description, evidence, root cause, impact,
  severity, remediation and owner.
- **Repair checks:** a finding without evidence, a "positive" finding, or a test result
  cited by an unknown run ID goes back for repair.

---

## Built-in deterministic tests (legacy battery)

The original battery is still available as `run_tests`, with its Excel and HTML export.
Its stand-in-model results (cross-validation, SHAP, robustness) are labelled SURROGATE,
because they come from a RandomForest fitted on the validation data, not from the model
under validation. Its randomness is now seeded. These run real statistics (scikit-learn, statsmodels, SHAP, OpenCV), not the language
model. Ask in plain words; the assistant picks the model type and the observed /
predicted / split columns from the data (or asks if they are genuinely unclear).

| Model type | Tests |
|---|---|
| **Supervised: Classification** | Performance metrics (accuracy, precision, recall, F1, ROC/AUC, PR-AUC, confusion matrix) · Class imbalance (SMOTE) · Ranking (Gini) · Statistical diagnostics (VIF) · Cross-validation (5-fold) · Explainability (SHAP, feature importance) · Bias–variance (learning curves) · Robustness (noise perturbation) · Drift (PSI) |
| **Supervised: Regression** | Performance metrics (RMSE, MAE, R², MAD) · VIF · Cross-validation · Explainability · Learning curves · Robustness · Drift (PSI) |
| **Unsupervised: Clustering** | Silhouette, Davies–Bouldin, Calinski–Harabasz, WCSS · Granularity (HHI) |
| **Unsupervised: Dimensionality reduction** | PCA explained variance · Correlation diagnostics |
| **Computer vision** | Image quality (Laplacian variance, edge density) · Noise robustness |
| **Data quality** (any table) | Missing values, validity checks (duplicates, constant and ID-like columns), IQR outliers, descriptive statistics, distribution and frequency plots |

Results appear under the answer in a collapsible group: metric tables, interactive
charts, static figures, plus a downloadable **Excel report** and an **HTML file** of all
charts (saved under `ask_data/outputs/`). Tunable parameters (classification threshold,
surrogate depth, SMOTE neighbours, PSI bins, blur threshold) can be set in the message,
e.g. "use a 0.3 threshold".

---

## How it works

```
 your message
     │
     ├─ only a path? ──────────► load it directly (no model call)
     │
     ▼
 tool-calling agent (OpenAI Responses API, via client_create())
     │   tools: load_path · overview · search · grep · read_file · list_files
     │          data_overview · query_data · make_chart
     │          run_data_quality · list_tests · run_tests · compare
     │          list_validation_tests · describe_test · run_validation_test
     │          run_validation_suite · get_test_run · load_model · load_table
     │          profile_table · query_table · scan_code · list_standards
     │          check_documentation · cross_check
     ▼
 answer ──► deterministic checks ──► (one repair round if needed) ──► shown + saved
```

- **One agent, no modes.** The model sees what is loaded and chooses its tools for each
  question; there is no separate "code mode" or "data mode".
- **Retrieval without embeddings.** Files are kept in memory as text. `search` ranks
  40-line windows with a small BM25 keyword index (built on first use, in milliseconds) and
  gives extra weight to the file that *defines* a name over files that merely mention it;
  `grep` finds exact text or regexes; `read_file` returns exact line ranges. Every line
  shown to the model is recorded as evidence.
- **Data questions** run one pandas expression at a time in a restricted evaluator: no
  imports, no private attributes, no file or system access, only `df`, `dfs`, `pd`, `np`
  and a few safe built-ins.
- **Models.** `client_create()` in `ask/llm.py` is the only place an API client is created
  (OpenAI or Azure OpenAI). Calls use the **Responses API**, so reasoning models can use
  tools with reasoning switched on; if an endpoint doesn't offer it, the app falls back to
  Chat Completions automatically for the session.
- **Think deeper** switches to the think-deeper model (default `gpt-5.6-sol`) with the
  configured reasoning effort (default `medium`).

---

## Faithfulness: how answers are checked

After every answer, deterministic checks run (no second model acting as judge):

1. **Every citation** `[path:Lstart-end]` must point to a loaded file and a real line
   range, and the cited lines must have been shown to the model **in this turn** (via
   search, grep, read_file, overview or compare). Several ranges in one citation are
   checked separately.
2. **Quoted identifiers** (e.g. `` `passes_quality_gate` ``) near a citation must exist in
   the cited or read files.
3. **Uncited answers** about loaded code or documents are rejected, unless explicitly
   labelled *General knowledge*.
4. **Claims of charts or figures** that no tool produced are rejected.
5. **Test results** must cite `[test:<run_id>]` of a run that actually happened. An answer
   that reports test results without any run citation is sent back.
6. **Findings** must be deficiencies, each with evidence (`[path:Lx-y]`, `[test:<run_id>]`
   or table + column).

If anything fails, the model gets one repair round with the exact problems ("these lines
were never shown to you", "`magic_fn` does not appear in the cited files"). Under the
answer you see either *✓ N citations checked against the source* or a warning listing what
could not be verified. Never hidden. Turn this off in Settings → *Verify citations*.

---

## Configuration

Set in `.env` locally (see `.env.example`), or as environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | | OpenAI key (standard OpenAI) |
| `USE_AZURE_OPENAI` | `false` | `true` to use Azure OpenAI |
| `AZURE_OPENAI_KEY` / `AZURE_OPENAI_TOKEN` | | Azure key |
| `AZURE_OPENAI_BASE_URL` | | Azure endpoint |
| `AZURE_OPENAI_VERSION` | `2024-02-01` | Azure API version (use a recent one, e.g. `2025-03-01-preview`, for the Responses API) |
| `OPENAI_DEPLOYMENT_NAME` | `gpt-5.6-luna` | Default model (Azure: your deployment name) |
| `ASK_DEEP_MODEL` | `gpt-5.6-sol` | Model used when *Think deeper* is on |
| `ASK_REASONING_EFFORT` | `medium` | Reasoning effort for *Think deeper* (`low` / `medium` / `high`) |
| `ASK_DATA_DIR` | `./ask_data` | Where chats, outputs and caches are stored |
| `ASK_VALIDATION_SEED` | `20240601` | Seed for every random step in a test (bootstrap, noise, Monte Carlo) |
| `ASK_JUDGE_MODEL` | default model | Model for LLM-judge tests (also in Settings → Validation) |
| `ASK_READ_IMAGES` | `1` | Read images with the vision model: PDF pages that are scanned or show figures, and image files in folders (`0` = off) |
| `ASK_VISION_MODEL` | default model | Model used to read images |
| `ASK_SQL_WAREHOUSE_ID` | | Databricks SQL warehouse for Unity Catalog tables when there's no Spark session (also in Settings → Validation) |
| `ASK_MAX_TABLE_ROWS` | `2000000` | Tables above this are hash-sampled deterministically |
| `ASK_LIVE_TESTS` | | `1` to run the optional tests that call the real API |

Models can also be changed for the current session in Settings → General.

Limits (in `ask/config.py`): 20 tool rounds per answer, 60 tests per suite call, 400 lines per `read_file` call,
5 MB per text file, 6,000 files per repository, the last 12 turns of chat history sent
with each question.

---

## Running on Databricks

### 1. Store the Azure OpenAI key as a secret (once)

The launcher reads the key from a Databricks secret (default scope `OneLab-SecretScope`).
If it isn't there yet, from a terminal with the
[Databricks CLI](https://docs.databricks.com/dev-tools/cli/) signed in:

```bash
databricks secrets put-secret OneLab-SecretScope <key-name>      # prompts for the value
```

Never paste the key into a notebook cell: notebooks keep a revision history.

### 2. Run the launcher notebook

Clone this repository as a **Git folder** and open `databricks_launcher_cluster.ipynb`
in the repo root, next to `app.py`. At the top of the second cell set:

| Setting | Meaning |
|---|---|
| `AZURE_KEY_SECRET` | Name of the key in `OneLab-SecretScope` holding the Azure OpenAI key |
| `AZURE_ENDPOINT`, `AZURE_API_VERSION` | Your Azure OpenAI endpoint and API version |
| `AZURE_DEPLOYMENT` | Azure deployment name (blank = `gpt-5.6-luna`) |
| `PORT` | Port on the driver (default `8502`) |

Then **Run all**. The notebook installs `requirements.txt`, starts the app on the cluster's
driver and shows an **Open Ask** link through the driver proxy
(`https://<workspace>/driver-proxy/o/<workspace-id>/<cluster-id>/<port>/`). The cluster
must stay up while the app is used. The last cell shows whether the app is running and
the end of its log (`ask-streamlit.log` in the repo folder).

The driver-proxy link needs a classic cluster; Databricks blocks it on serverless and on
some Shared/Standard clusters (*Traffic on this port is not permitted*).

**Deploying as a Databricks App instead:** Compute → Apps → Create app → custom app; add an
App resource of type **Secret** (permission *Can read*, resource key `openai-api-key`);
deploy this folder. `app.yaml` holds the app's command and environment.

### 3. Load Databricks folders and files in the chat

```
load /Workspace/Users/you@company.com/MLOps
read the data from /Volumes/catalog/schema/volume/portfolio.csv
```

Use ⋮ → **Copy path** in Databricks; browser links (`…/browse/folders/<id>`) contain an
ID, not the path. Inside a Databricks App (and on a laptop), these paths are read through
the Databricks API and mirrored into the app's cache, notebooks included as source code.
Only changed files are downloaded again on reload.

> **Grant the app access once.** Inside a Databricks App the files are read as the
> **app's service principal**, not as you. Share the folder with it: folder ⋮ → **Share** →
> add the app's service principal (named after the app) → **Can Read**. For a Volume,
> grant it `READ VOLUME`. Without this the app tells you exactly that.

On a laptop, the same paths work once the Databricks CLI is signed in.

### 4. Unity Catalog tables and model artefacts

```
load main.risk.pd_snapshot (columns pd_12m, default_flag, rating, year)
profile main.risk.loans with key loan_id
load the model models:/main.risk.pd_model/3
```

- **How tables are read:** through the active Spark session if there is one. Otherwise
  through a SQL warehouse: set its ID in Settings → Validation or as `ASK_SQL_WAREHOUSE_ID`.
- **Launcher notebook:** it runs Ask as a separate process on the driver, which has no Spark
  session, so set the warehouse ID there too.
- **Permissions:** whoever runs the query (you, or the app's service principal) needs `SELECT`
  on the table and `CAN USE` on the warehouse.
- **Model artefacts:** MLflow URIs need `mlflow`, which is preinstalled on Databricks ML
  runtimes; elsewhere run `pip install mlflow`. Pickle and joblib files load from local,
  `/Workspace` or `/Volumes` paths.

### Where chats live on Databricks

- **Databricks App:** `/tmp/ask_data` inside the app; reset when the app is redeployed.
- **Launcher notebook:** `/tmp/ask_data` on the driver, wiped when the cluster terminates.

---

## Security and privacy

- **The API key is never in the repository.** Locally it lives in `.env` (git-ignored);
  on Databricks in a secret scope, passed only to the app process and never printed.
  `app.yaml` references the secret by name only.
- **Your material stays local.** Chats, cached document text, Databricks mirrors and test
  outputs live in `ask_data/`, which is git-ignored. Only the snippets the assistant reads
  to answer a question are sent to the model API.
- **Data questions are sandboxed:** the pandas evaluator cannot import modules, touch
  files or reach private attributes.
- **Guards in the test suite:** `tests/test_no_secrets.py` fails if any committable file
  contains an OpenAI key, Databricks/GitHub/AWS token, private key, or a secret-looking
  assignment. Run it before pushing:

  ```bash
  pytest tests/test_no_secrets.py
  ```

---

## Project layout

```
app.py                       Streamlit entry point (layout, chat loop)
app.yaml                     Databricks Apps configuration
databricks_launcher_cluster.ipynb  Databricks notebook: run the app on a cluster
requirements.txt
.env.example                 settings template (copy to .env)
ask/
  config.py                  paths, models, limits
  llm.py                     client_create(): the only place an API client is made
  preferences.py             persisted preferences (tool name)
  agent/
    router.py                one turn: fast load, agent loop, checks, repair
    tools.py                 the agent's tools and their schemas
    validation_tools.py      validator tools: tests, suites, models, tables, code, docs
    llm_call.py              Responses API conversation (+ Chat Completions fallback)
    faithfulness.py          deterministic citation / claim checks
    prompts.py               system and repair prompts
  sources/
    paths.py                 finding paths in chat messages
    loaders.py               reading repos, documents and tables
    databricks.py            /Workspace and /Volumes through the Databricks API
    registry.py              the sources (and loaded models) in a chat
    tables.py                Unity Catalog tables: load (hash-sampled), profile, SELECT
  validation/
    core.py                  test registry, parameter checks, provenance, run_id
    t_*.py                   the tests: pd, lgd, ead, ifrs9, ews, econometrics, market,
                             pricing, ccr, aml, ml, fairness, genai, data, stability
    code_scan.py             static code-review rules
    doc_review.py, standards/  regulatory checklists (YAML) and the evidence search
    models.py                loading and scoring model artefacts
    judge.py                 LLM judge with a prompt-hash cache
    store.py                 saved runs (audit trail) and [test:] citation check
    genai_probes.py          versioned prompt-injection probes
    catalog_doc.py           writes docs/TEST_CATALOG.md
  retrieval/search.py        BM25 search, grep, read_file, overview (+ evidence)
  analysis/
    test_engine.py           the deterministic test engine
    data_quality.py          data-quality checks
    runner.py                runs the two above and summarises results
    data_query.py            safe pandas expressions and charts
    compare.py               file / folder / table comparison
  memory/store.py            SQLite chat storage
  ui/                        sidebar and Settings, message rendering, styles
tests/                       pytest suite (see below)
demo data/                   sample regression and classification workbooks
```

---

## Tests

```bash
pytest                               # everything (~1 minute)
pytest -m "not slow"                 # skip the real statistics-engine runs (~30 s)
ASK_LIVE_TESTS=1 pytest -m live      # optional: a few checks against the real model API
```

About 840 tests: unit, integration and end-to-end. `tests/validation/` checks every
statistical test against known answers: textbook values, published examples, scipy,
statsmodels or scikit-learn, or a hand calculation. No test calls the model API unless
`ASK_LIVE_TESTS=1`: the agent is driven by a scripted fake model, Databricks by a fake
client, and every test gets its own temporary data folder.

| File | Covers |
|---|---|
| `test_app.py` | The real Streamlit app end to end: load → ask → citations → reopen chat, errors, Think deeper, nothing heavy at startup |
| `test_router.py`, `test_tools.py`, `test_llm_call.py` | Agent loop, repair rounds, every tool, Responses API and fallback |
| `test_faithfulness.py` | Citation formats, unread / out-of-range / unknown citations, invented identifiers, chart claims |
| `test_search.py`, `test_paths.py`, `test_loaders.py`, `test_registry.py` | Retrieval and ranking, path parsing, every file format, workspaces, caching |
| `test_data_query.py`, `test_compare.py` | Pandas results and 20+ blocked unsafe patterns; every comparison type |
| `test_builtin_tests.py` | **All 22 built-in tests** run on their own with exact expected outputs; every statistic checked against scikit-learn / statsmodels or a hand-computed answer |
| `test_runner.py` | Test catalogue, input validation, data quality |
| `test_databricks_paths.py`, `test_databricks_launcher.py` | Databricks paths and permissions; the launcher notebook's settings, app environment and link |
| `test_store.py`, `test_preferences.py`, `test_render.py` | Chat storage, tool name, message rendering |
| `tests/validation/` | Every validation test (known answers, edge cases, determinism), the code scanner, the checklists, the validator tools, `[test:]` citation and findings checks, and the catalog doc being up to date |
| `test_verbatim.py` | `client_create()`, the test engine and the data-quality checks are pinned by SHA-256 (the engine's hash was updated deliberately for the seeding and SURROGATE-label fixes) |
| `test_no_secrets.py` | No secrets in any committable file |

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `OPENAI_API_KEY not found` | No `.env` (locally) or secret (Databricks). Create `.env` from `.env.example`, or check `AZURE_KEY_SECRET` in the launcher. |
| "Path not found" for a local path | Check the spelling; put paths with spaces in quotes. |
| "That is a browser link…" | Paste the path (⋮ → Copy path), not the Databricks URL. |
| "…does not exist, or the app's service principal can't see it" | Share the folder with the app's service principal (**Can Read**), or `READ VOLUME` on a Volume. |
| Databricks link says *Traffic on this port is not permitted* | The notebook is on serverless or a Shared/Standard cluster, where the driver proxy is blocked. Use a Dedicated (single user) cluster, or deploy as a Databricks App. |
| "Function tools with reasoning_effort are not supported…" | Fixed by the Responses API; on the Chat Completions fallback the app retries with reasoning off automatically. |
| Answer shows *⚠ … could not be verified* | The model cited lines it hadn't read, or quoted something not in the file. Ask it to re-check, or read the Sources panel. |

---

## Limitations

- Images inside documents are not read (no OCR / vision); scanned PDFs without a text
  layer can't be loaded.
- Search is keyword-based (BM25 + exact grep), not semantic: phrasing questions with
  the terms used in the code or document helps.
- Very large repositories are capped at 6,000 files and 5 MB per text file.
- Inside a Databricks App, chats reset on redeploy.
- Databricks notebooks saved in DBC/HTML-only formats, dashboards and libraries are not
  read; Python/SQL/Scala/R notebooks are exported as source.
