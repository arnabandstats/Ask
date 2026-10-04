# Databricks notebook source
# MAGIC %md
# MAGIC # Launch the chat app on Databricks
# MAGIC
# MAGIC Runs the Streamlit app on this cluster's **driver** and gives you a link to open it
# MAGIC (through the Databricks driver proxy — only people with access to this cluster can open it).
# MAGIC
# MAGIC **How to use**
# MAGIC 1. Put this repo in your workspace (Git folder / Repos, or import the folder) — keep this
# MAGIC    notebook in the repo root, next to `app.py`.
# MAGIC 2. The API key is read from the Databricks secret **`ask` / `openai-api-key`** (the widget
# MAGIC    defaults). It was stored once from a terminal with:
# MAGIC    `databricks secrets create-scope ask` then `databricks secrets put-secret ask openai-api-key`.
# MAGIC    If you used other names, change the two secret widgets.
# MAGIC 3. **Run all**. Cell 2 checks the secret before anything is installed; the last cells print
# MAGIC    the link, show the log, back up chats and stop the app.
# MAGIC
# MAGIC The app keeps running while this cluster is up (you can detach the notebook). It stops when
# MAGIC the cluster terminates or when you run **Stop the app**.
# MAGIC
# MAGIC **Chat history** lives on the driver's local disk (SQLite needs a normal disk; Volumes and
# MAGIC workspace files don't support its writes), which is wiped when the cluster terminates. Set
# MAGIC **Backup folder** to a Volume path to keep it: it is restored on start and saved on
# MAGIC **Stop the app** (or any time with **Back up chats**).
# MAGIC
# MAGIC Inside the app, load files with Databricks paths, e.g.
# MAGIC `load /Workspace/Users/you@company.com/my_repo` or
# MAGIC `read the data from /Volumes/catalog/schema/volume/portfolio.csv`.

# COMMAND ----------

# MAGIC %md ## 1 · Settings

# COMMAND ----------

dbutils.widgets.text("port", "8501", "Port")
dbutils.widgets.text("secret_scope", "ask", "Secret scope (API key)")
dbutils.widgets.text("secret_key", "openai-api-key", "Secret key name")
dbutils.widgets.dropdown("use_azure", "false", ["false", "true"], "Use Azure OpenAI")
dbutils.widgets.text("azure_endpoint", "", "Azure endpoint (if Azure)")
dbutils.widgets.text("azure_api_version", "2025-03-01-preview", "Azure API version (if Azure)")
dbutils.widgets.text("default_model", "", "Default model (blank = app default)")
dbutils.widgets.text("deep_model", "", "Think-deeper model (blank = app default)")
dbutils.widgets.text("data_dir", "/local_disk0/ask_data", "Chat history folder (driver disk)")
dbutils.widgets.text("backup_dir", "", "Backup folder, e.g. /Volumes/cat/schema/vol/ask (optional)")

# COMMAND ----------

# MAGIC %md ## 2 · Check the API key secret
# MAGIC Fails fast, before the install, if the secret is missing or you can't read it. The value is
# MAGIC never printed (Databricks would show it as `[REDACTED]` anyway).

# COMMAND ----------


def check_api_key_secret():
    """Return (scope, key) after confirming the secret exists and is readable; raise a clear error otherwise."""
    scope = dbutils.widgets.get("secret_scope").strip()
    key = dbutils.widgets.get("secret_key").strip()
    if not scope:
        print("No secret scope set: the app will use an API key defined on the cluster, if any.")
        return None, None
    scopes = [s.name for s in dbutils.secrets.listScopes()]
    if scope not in scopes:
        raise ValueError(f"Secret scope '{scope}' not found (or you have no access). "
                         f"Scopes you can see: {', '.join(sorted(scopes)) or 'none'}. "
                         f"Create it with: databricks secrets create-scope {scope}")
    keys = [k.key for k in dbutils.secrets.list(scope)]
    if key not in keys:
        raise ValueError(f"Key '{key}' not found in scope '{scope}'. Keys there: "
                         f"{', '.join(sorted(keys)) or 'none'}. "
                         f"Add it with: databricks secrets put-secret {scope} {key}")
    value = dbutils.secrets.get(scope, key)
    if not value or not value.strip():
        raise ValueError(f"Secret '{scope}/{key}' is empty. Store the key again with put-secret.")
    if value != value.strip():
        print("Note: the stored key has leading/trailing whitespace; it will be trimmed.")
    print(f"API key found in secret '{scope}/{key}' ({len(value.strip())} characters).")
    return scope, key


check_api_key_secret()

# COMMAND ----------

# MAGIC %md ## 3 · Install dependencies
# MAGIC Uses the repo's `requirements.txt`, swapping `opencv-python` for `opencv-python-headless`
# MAGIC (clusters have no display libraries, and the GUI build fails to import there).

# COMMAND ----------

import os

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
APP_DIR = "/Workspace" + os.path.dirname(_nb_path)
assert os.path.exists(os.path.join(APP_DIR, "app.py")), (
    f"app.py not found in {APP_DIR}. Keep this notebook in the repo root, next to app.py.")

_reqs = []
for line in open(os.path.join(APP_DIR, "requirements.txt"), encoding="utf-8"):
    req = line.split("#", 1)[0].strip()
    if not req:
        continue
    if req.lower().startswith("opencv-python"):
        req = "opencv-python-headless"
    _reqs.append(req)
with open("/tmp/ask_requirements.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(_reqs) + "\n")
print(f"App folder: {APP_DIR}")
print("Installing:", ", ".join(_reqs))

# COMMAND ----------

# MAGIC %pip install -q -r /tmp/ask_requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## 4 · Start the app

# COMMAND ----------

import os
import signal
import subprocess
import sys
import time
import urllib.request

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
APP_DIR = "/Workspace" + os.path.dirname(_nb_path)
PORT = int(dbutils.widgets.get("port"))
DATA_DIR = dbutils.widgets.get("data_dir").strip() or "/local_disk0/ask_data"
os.makedirs(DATA_DIR, exist_ok=True)
PID_FILE = os.path.join(DATA_DIR, f"streamlit_{PORT}.pid")
LOG_FILE = os.path.join(DATA_DIR, f"streamlit_{PORT}.log")

# ── environment for the app (secrets go only into the child process, never printed) ──
env = dict(os.environ)
env["ASK_DATA_DIR"] = DATA_DIR
scope, key = dbutils.widgets.get("secret_scope").strip(), dbutils.widgets.get("secret_key").strip()
if dbutils.widgets.get("use_azure") == "true":
    env["USE_AZURE_OPENAI"] = "true"
    env["AZURE_OPENAI_BASE_URL"] = dbutils.widgets.get("azure_endpoint").strip()
    env["AZURE_OPENAI_VERSION"] = dbutils.widgets.get("azure_api_version").strip()
    if scope:
        env["AZURE_OPENAI_KEY"] = dbutils.secrets.get(scope, key).strip()
    assert env["AZURE_OPENAI_BASE_URL"], "Set the Azure endpoint widget."
else:
    env["USE_AZURE_OPENAI"] = "false"
    if scope:
        env["OPENAI_API_KEY"] = dbutils.secrets.get(scope, key).strip()
assert env.get("OPENAI_API_KEY") or env.get("AZURE_OPENAI_KEY") or env.get("AZURE_OPENAI_TOKEN"), (
    "No API key: set 'Secret scope' and 'Secret key name' (or define the key on the cluster).")
for widget, var in (("default_model", "OPENAI_DEPLOYMENT_NAME"), ("deep_model", "ASK_DEEP_MODEL")):
    if dbutils.widgets.get(widget).strip():
        env[var] = dbutils.widgets.get(widget).strip()


def _stop_previous():
    """Stop an app this notebook started earlier on the same port."""
    if not os.path.exists(PID_FILE):
        return
    try:
        pid = int(open(PID_FILE).read().strip())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(2)
        print(f"Stopped the previous app (pid {pid}).")
    except (ProcessLookupError, ValueError, PermissionError):
        pass
    finally:
        os.remove(PID_FILE)


_stop_previous()

# restore chat history from the backup folder, if there is one and nothing newer locally
import shutil
BACKUP_DIR = dbutils.widgets.get("backup_dir").strip()
if BACKUP_DIR:
    for name in ("chats.db", "preferences.json"):
        src, dst = os.path.join(BACKUP_DIR, name), os.path.join(DATA_DIR, name)
        if os.path.exists(src) and (not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst)):
            shutil.copy2(src, dst)
            print(f"Restored {name} from {BACKUP_DIR}")

cmd = [sys.executable, "-m", "streamlit", "run", "app.py",
       "--server.port", str(PORT), "--server.address", "0.0.0.0",
       "--server.headless", "true", "--browser.gatherUsageStats", "false",
       # The driver proxy sits in front of the app on another origin:
       "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
log = open(LOG_FILE, "w")
proc = subprocess.Popen(cmd, cwd=APP_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True)          # survives notebook detach / Python restarts
open(PID_FILE, "w").write(str(proc.pid))

# wait until Streamlit answers its health check
for _ in range(90):
    if proc.poll() is not None:
        raise RuntimeError("The app exited during startup. Log:\n" + open(LOG_FILE).read()[-4000:])
    try:
        if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/_stcore/health", timeout=2).status == 200:
            break
    except Exception:
        time.sleep(1)
else:
    raise TimeoutError("The app did not become healthy within 90 s. Log:\n" + open(LOG_FILE).read()[-4000:])
print(f"App running (pid {proc.pid}) on port {PORT}. Chats are saved in {DATA_DIR}.")

# COMMAND ----------

# MAGIC %md ## 5 · Open it

# COMMAND ----------

_host = spark.conf.get("spark.databricks.workspaceUrl")
_org = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")
_cluster = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
APP_URL = f"https://{_host}/driver-proxy/o/{_org}/{_cluster}/{PORT}/"

displayHTML(f"""
<div style="font-family: system-ui, sans-serif; font-size: 15px; padding: 8px 0">
  <a href="{APP_URL}" target="_blank" rel="noopener"
     style="background:#16191f;color:#fff;padding:10px 18px;border-radius:10px;text-decoration:none">
     Open the app ↗</a>
  <div style="margin-top:12px;color:#6b7079">{APP_URL}</div>
</div>""")

# COMMAND ----------

# MAGIC %md ## 6 · Log (run any time)

# COMMAND ----------

PORT = int(dbutils.widgets.get("port"))
_data_dir = dbutils.widgets.get("data_dir").strip() or "/local_disk0/ask_data"
print(open(f"{_data_dir}/streamlit_{PORT}.log").read()[-6000:])

# COMMAND ----------

# MAGIC %md ## 7 · Back up chats (run any time; also done by Stop)

# COMMAND ----------

import os
import shutil
import sqlite3


def backup_chats():
    data_dir = dbutils.widgets.get("data_dir").strip() or "/local_disk0/ask_data"
    backup_dir = dbutils.widgets.get("backup_dir").strip()
    if not backup_dir:
        print("No backup folder set; chats stay on the driver disk only.")
        return
    os.makedirs(backup_dir, exist_ok=True)
    db = os.path.join(data_dir, "chats.db")
    if os.path.exists(db):
        # sqlite's backup API gives a consistent copy even while the app is writing
        snapshot = "/tmp/chats_backup.db"
        with sqlite3.connect(db) as src, sqlite3.connect(snapshot) as dst:
            src.backup(dst)
        shutil.copyfile(snapshot, os.path.join(backup_dir, "chats.db"))
    prefs = os.path.join(data_dir, "preferences.json")
    if os.path.exists(prefs):
        shutil.copyfile(prefs, os.path.join(backup_dir, "preferences.json"))
    print(f"Chats backed up to {backup_dir}")


backup_chats()

# COMMAND ----------

# MAGIC %md ## 8 · Stop the app

# COMMAND ----------

import os
import signal

if "backup_chats" in globals():
    backup_chats()
else:
    print("Not backed up: run the 'Back up chats' cell first if you set a backup folder.")
PORT = int(dbutils.widgets.get("port"))
_data_dir = dbutils.widgets.get("data_dir").strip() or "/local_disk0/ask_data"
_pid_file = f"{_data_dir}/streamlit_{PORT}.pid"
if os.path.exists(_pid_file):
    _pid = int(open(_pid_file).read().strip())
    try:
        os.killpg(os.getpgid(_pid), signal.SIGTERM)
        print(f"Stopped the app (pid {_pid}).")
    except ProcessLookupError:
        print("The app was not running.")
    os.remove(_pid_file)
else:
    print(f"No app started from this notebook on port {PORT}.")
