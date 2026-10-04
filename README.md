# Ask

Chat with a code repository, documents, or a dataset — and run the built-in
deterministic model-validation tests — from one chat box.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Put `OPENAI_API_KEY=...` in `.env` (or the Azure variables read by
`ask/llm.py::client_create`).

## Using it

Type what you want; there are no modes or tags.

| You type | What happens |
|---|---|
| `load C:/projects/model_x` | Loads the folder as a repo (code + its docs). One-line confirmation, no LLM call. |
| `read the data from D:/data/portfolio.csv` | Loads a table (CSV, Excel, Parquet, Feather, pickle). |
| `load "C:/policies/model_policy.pdf"` | Loads a document (PDF, DOCX, Markdown, text). |
| `how is the PD model calibrated?` | Searches and reads the code, answers with `[file:L10-40]` citations. |
| `default rate by segment, as a chart` | Runs a pandas query and draws a chart. |
| `run the assessment on this data` | Runs the data-quality checks and the model-validation tests. |
| `compare v1.csv with v2.csv` / `compare the two policies` | Exact diffs, then an explanation. |
| `does the code implement section 4 of the methodology?` | Cites both the document and the code. |

Several sources can be loaded at once. Chats (messages + loaded sources) are
saved in `ask_data/chats.db`; reopening a chat reloads its sources.
Settings (model, Think deeper, citation check, loaded sources, test list) sit
behind **⚙ Settings** in the sidebar.

## Faithfulness

Repo/document answers must cite `[path:Lstart-end]`. After every answer a
deterministic check (no LLM judge) confirms each citation points to a real
file, a real line range, and lines the model actually read in that turn;
quoted identifiers must exist in the cited file; uncited answers about loaded
material and claims of charts that were never produced are rejected. Failures
trigger one repair round; the result and the cited lines are shown under the
answer.

## Keeping the API key out of git

- Locally the key lives only in `.env`, which `.gitignore` excludes. Copy
  `.env.example` to `.env` to set it up on a new machine.
- On Databricks the key lives in a **secret scope**; `databricks_launcher.py` reads it
  with `dbutils.secrets.get` and passes it only to the app process. No `.env` there.
- `ask_data/` (chats, cached document text), `validator_cache/` and
  `validator_output/` are git-ignored too: they hold your documents' contents.
- `tests/test_no_secrets.py` fails if anything that git would commit contains a key,
  token or private key. Run `pytest tests/test_no_secrets.py` before pushing.

## Running on Databricks

1. Store the API key once, from a terminal:
   `databricks secrets create-scope ask`, then `databricks secrets put-secret ask openai-api-key`.
2. Clone this repo as a Git folder, open `databricks_launcher.py` (repo root, next to
   `app.py`) and **Run all**. It checks the secret, then picks the way that works on the
   compute it is attached to:

| Attached to | What the notebook does |
|---|---|
| **Serverless**, or a cluster in **Shared/Standard** access mode | Creates/updates a **Databricks App** (`ask-<your user>`), attaches the secret as the app resource `openai-api-key`, deploys this folder with `app.yaml`, and prints the app URL. Needs Databricks Apps enabled in the workspace. |
| A cluster in **Dedicated (single user)** access mode | Installs the requirements and runs the app on the driver, with a link through the driver proxy. |

Set the **Run mode** widget to force `databricks_app` or `cluster`. Redeploying after a
**Pull** is just **Run all** again.

**Databricks App notes:** share it from Compute → Apps → your app → Permissions (**Can
use**). Chats are kept in `/tmp/ask_data` inside the app and reset on redeploy. To set it up
by hand instead: create a custom app, add a Secret resource (scope `ask`, key
`openai-api-key`, **Can read**, resource key `openai-api-key`) and deploy this folder.

**Cluster mode notes:** the driver link does not work on serverless or Shared/Standard
clusters ("Traffic on this port is not permitted"), which is why the notebook uses an App
there. Chats live on the driver disk; set the backup folder to a Volume path to keep them
across cluster restarts. Later cells show the log, back up chats and stop the app.

## Tests

```bash
pytest                      # full suite (~25 s): unit, integration and end-to-end app tests
pytest -m "not slow"        # skip the runs of the real sklearn/shap test engine
ASK_LIVE_TESTS=1 pytest -m live   # optional: a few checks against the real model API
```

No test touches the real API unless `ASK_LIVE_TESTS=1` is set — the agent is
driven by a scripted fake model, and every test gets its own temporary chat DB and
output folder. `tests/test_verbatim.py` fails if `client_create()`, the test engine
or the data-quality checks ever drift from the original code in `legacy/`.

`tests/test_builtin_tests.py` covers every built-in deterministic test (all 22, across
the five model types): each is run on its own and must produce exactly its own
metrics, sheets, charts and files; every statistical function is checked against
sklearn / statsmodels or a hand-computed answer. Adding a test to the engine without
adding it there makes the suite fail.

## Layout

```
app.py                       Streamlit entry point
ask/
  config.py                  paths, models, limits
  llm.py                     client_create() (verbatim from the original app)
  sources/                   path parsing, file readers, the loaded-source registry
  retrieval/search.py        keyword (BM25) search, grep, read_file + evidence tracking
  agent/                     prompts, tools, router (agent loop), faithfulness check
  analysis/
    test_engine.py           deterministic tests (verbatim from the original app)
    data_quality.py          data-quality checks (verbatim from the original app)
    runner.py                calls the two above and summarises results
    data_query.py            safe pandas expressions + charts
    compare.py               file / source / table comparison
  memory/store.py            SQLite chat store
  ui/                        sidebar, settings dialog, message rendering, styles
legacy/                      the previous single-file app, kept for reference
```
