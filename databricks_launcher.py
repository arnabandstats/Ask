# Databricks notebook source
# MAGIC %md
# MAGIC # Launch Ask on this cluster
# MAGIC
# MAGIC Runs `app.py` on the attached cluster's driver and shows an **Open Ask** link through the
# MAGIC cluster's driver proxy.
# MAGIC
# MAGIC 1. Keep this notebook in the repo root, next to `app.py` and `requirements.txt` (a Git folder).
# MAGIC 2. Fill in **Settings** (secret scope / key of your Azure OpenAI key, endpoint, deployment name).
# MAGIC 3. **Run all.** The cluster must stay up while the app is used; **Stop** ends it.
# MAGIC
# MAGIC Chats live on the driver's local disk and are wiped when the cluster terminates; set
# MAGIC `BACKUP_DIR` to a Volume path to restore them on start and save them on **Stop**.
# MAGIC Inside the app, load files with Databricks paths, e.g. `load /Workspace/Users/you@company.com/my_repo`.

# COMMAND ----------

# MAGIC %pip install --upgrade --force-reinstall --no-cache-dir -r requirements.txt

# COMMAND ----------

# MAGIC %md ## Settings

# COMMAND ----------

PORT = 8502
APP_FILE = "app.py"

# Azure OpenAI. The key is read from a Databricks secret and passed only to the app process.
SECRET_SCOPE = "OneLab-SecretScope"
AZURE_KEY_SECRET = ""                    # name of the key inside SECRET_SCOPE that holds the Azure OpenAI key
AZURE_ENDPOINT = ""                      # e.g. https://<resource>.openai.azure.com/
AZURE_API_VERSION = "2025-03-01-preview"
AZURE_DEPLOYMENT = ""                    # your Azure deployment name (blank = gpt-4.1)
AZURE_DEEP_DEPLOYMENT = ""               # deployment for "Think deeper" (blank = app default)

DATA_DIR = "/local_disk0/ask_data"       # chat history (needs a normal disk; falls back to /tmp)
BACKUP_DIR = ""                          # optional, e.g. /Volumes/catalog/schema/volume/ask

# COMMAND ----------

# MAGIC %md ## Start Ask

# COMMAND ----------

import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

from IPython import get_ipython
from IPython.display import HTML, display

_ipython = get_ipython()
if _ipython is None or "dbutils" not in _ipython.user_ns:
    raise RuntimeError("Attach and run this notebook on a Databricks cluster.")
dbutils = _ipython.user_ns["dbutils"]

repo_dir = pathlib.Path.cwd()
app_path = repo_dir / APP_FILE
if not app_path.exists():
    raise FileNotFoundError(
        f"{APP_FILE} was not found in {repo_dir}. Run this notebook from the repo root, next to app.py."
    )


def read_secret(scope, key):
    """The secret's value, trimmed; a clear error (never the value) if it can't be read."""
    if not key:
        names = sorted(k.key for k in dbutils.secrets.list(scope))
        raise ValueError(f"Set AZURE_KEY_SECRET in Settings. Keys in '{scope}': {', '.join(names) or 'none'}.")
    try:
        value = dbutils.secrets.get(scope=scope, key=key)
    except Exception as exc:
        raise ValueError(f"Could not read secret '{scope}/{key}': {exc}") from exc
    if not value or not value.strip():
        raise ValueError(f"Secret '{scope}/{key}' is empty.")
    return value.strip()


def first_writable(candidates):
    """First folder we can create and write to (/local_disk0 is read-only on some clusters)."""
    tried = []
    for folder in candidates:
        if not folder or folder in tried:
            continue
        tried.append(folder)
        try:
            os.makedirs(folder, exist_ok=True)
            probe = os.path.join(folder, ".write_test")
            with open(probe, "w") as fh:
                fh.write("ok")
            os.remove(probe)
            return folder
        except OSError:
            continue
    raise PermissionError("No writable folder for chat history. Tried: " + ", ".join(tried))


def context_text(value):
    """Read Databricks context values without converting Java objects to integers."""
    text = str(value).strip()
    if text.startswith("Some(") and text.endswith(")"):
        return text[5:-1]
    return "" if text in {"None", "null"} else text


def spark_conf(key):
    try:
        return spark.conf.get(key, "") or ""
    except Exception:
        return ""


if not AZURE_ENDPOINT.strip():
    raise ValueError("Set AZURE_ENDPOINT in Settings (your Azure OpenAI endpoint URL).")

# ── environment for the app: secrets go only into the child process, never printed ──
app_env = os.environ.copy()
app_env["USE_AZURE_OPENAI"] = "true"
app_env["AZURE_OPENAI_KEY"] = read_secret(SECRET_SCOPE, AZURE_KEY_SECRET.strip())
app_env["AZURE_OPENAI_BASE_URL"] = AZURE_ENDPOINT.strip()
app_env["AZURE_OPENAI_VERSION"] = AZURE_API_VERSION.strip()
if AZURE_DEPLOYMENT.strip():
    app_env["OPENAI_DEPLOYMENT_NAME"] = AZURE_DEPLOYMENT.strip()
if AZURE_DEEP_DEPLOYMENT.strip():
    app_env["ASK_DEEP_MODEL"] = AZURE_DEEP_DEPLOYMENT.strip()

data_dir = first_writable([DATA_DIR, "/local_disk0/ask_data", "/tmp/ask_data",
                           os.path.join(tempfile.gettempdir(), "ask_data")])
if data_dir != DATA_DIR:
    print(f"Note: {DATA_DIR} is not writable here, so chat history goes to {data_dir}.")
app_env["ASK_DATA_DIR"] = data_dir

pid_path = os.path.join(data_dir, f"streamlit_{PORT}.pid")
log_path = os.path.join(data_dir, f"streamlit_{PORT}.log")
# Where the Log / Stop cells find this run, even after %pip restarted Python.
state_path = os.path.join(tempfile.gettempdir(), f"ask_launcher_{PORT}.json")

# ── stop an app started earlier on this port (a rerun would otherwise find the port taken) ──
existing_process = globals().get("ask_process")
if existing_process is not None and existing_process.poll() is None:
    existing_process.terminate()
    try:
        existing_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        existing_process.kill()
        existing_process.wait(timeout=5)
if os.path.exists(pid_path):                     # started before a Python restart
    try:
        old_pid = int(open(pid_path).read().strip())
        os.killpg(os.getpgid(old_pid), signal.SIGTERM)
        time.sleep(2)
        print(f"Stopped the previous app (pid {old_pid}).")
    except (OSError, ValueError):
        pass
    os.remove(pid_path)
existing_log_handle = globals().get("ask_log_handle")
if existing_log_handle is not None and not existing_log_handle.closed:
    existing_log_handle.close()

# ── restore chats from the backup folder when it is newer ──
if BACKUP_DIR.strip():
    for name in ("chats.db", "preferences.json"):
        src, dst = os.path.join(BACKUP_DIR.strip(), name), os.path.join(data_dir, name)
        if os.path.exists(src) and (not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst)):
            shutil.copy2(src, dst)
            print(f"Restored {name} from {BACKUP_DIR.strip()}")

# ── workspace, host and cluster for the driver-proxy link ──
context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
cluster_id = context_text(context.clusterId())
browser_host = context_text(context.browserHostName())
workspace_id = context_text(context.workspaceId())
if not cluster_id:
    cluster_id = spark_conf("spark.databricks.clusterUsageTags.clusterId")
if not browser_host:
    browser_host = spark_conf("spark.databricks.workspaceUrl")
if not workspace_id:
    workspace_id = spark_conf("spark.databricks.clusterUsageTags.orgId")
browser_host = browser_host.replace("https://", "").strip("/")
cluster_id = cluster_id.strip()
workspace_id = workspace_id.strip()
if not cluster_id or not browser_host or not workspace_id:
    raise RuntimeError("Could not resolve the Databricks workspace, browser host, or cluster ID. "
                       "Re-attach to an active cluster and rerun this cell.")

# ── start Streamlit ──
ask_log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
ask_process = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address", "0.0.0.0",
        "--server.port", str(PORT),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        # the driver proxy serves the app from another origin
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ],
    cwd=repo_dir,
    env=app_env,
    stdout=ask_log_handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open(pid_path, "w") as fh:
    fh.write(str(ask_process.pid))
with open(state_path, "w") as fh:
    json.dump({"data_dir": data_dir, "pid_file": pid_path, "log_file": log_path}, fh)

# wait until Streamlit answers its health check
for _ in range(120):
    if ask_process.poll() is not None:
        ask_log_handle.flush()
        raise RuntimeError("Ask stopped during startup. Log:\n" + open(log_path).read()[-4000:])
    try:
        if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/_stcore/health", timeout=2).status == 200:
            break
    except Exception:
        time.sleep(1)
else:
    raise TimeoutError("Ask did not start within 2 minutes. Log:\n" + open(log_path).read()[-4000:])
print(f"Ask is running (pid {ask_process.pid}) on port {PORT}. Chats are saved in {data_dir}.")

proxy_url = f"https://{browser_host}/driver-proxy/o/{workspace_id}/{cluster_id}/{PORT}/"
display(HTML(f"""
<div style="max-width:760px;padding:28px;border:1px solid #dfe3e8;border-radius:16px;background:linear-gradient(135deg,#f7fbff,#ffffff);font-family:Arial,sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.08)">
  <div style="font-size:30px;margin-bottom:8px">💬 Ask</div>
  <div style="color:#475569;margin-bottom:20px;line-height:1.5">The Streamlit application is running on the attached cluster. The cluster must remain active while the application is used.</div>
  <a href="{proxy_url}" target="_blank" rel="noopener noreferrer" style="display:inline-block;padding:12px 20px;border-radius:9px;background:#2C3696;color:white;text-decoration:none;font-weight:700">Open Ask</a>
  <div style="margin-top:14px;color:#64748b;font-size:13px">{proxy_url}</div>
</div>
"""))

# COMMAND ----------

# MAGIC %md ## Log (run any time)

# COMMAND ----------

import json
import os
import tempfile


def launcher_state(port):
    """What the Start cell recorded for this port: data_dir, pid_file, log_file."""
    try:
        with open(os.path.join(tempfile.gettempdir(), f"ask_launcher_{port}.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


_log = launcher_state(PORT).get("log_file", "")
print(open(_log).read()[-6000:] if _log and os.path.exists(_log) else "No log yet: run 'Start Ask' first.")

# COMMAND ----------

# MAGIC %md ## Stop Ask (backs up chats first when `BACKUP_DIR` is set)

# COMMAND ----------

import os
import shutil
import signal
import sqlite3
import tempfile


def backup_chats(data_dir, backup_dir):
    if not backup_dir:
        print("No BACKUP_DIR set; chats stay on the driver disk only.")
        return
    os.makedirs(backup_dir, exist_ok=True)
    db = os.path.join(data_dir, "chats.db")
    if os.path.exists(db):
        # sqlite's backup API gives a consistent copy even while the app is writing
        snapshot = os.path.join(tempfile.gettempdir(), "chats_backup.db")
        with sqlite3.connect(db) as src, sqlite3.connect(snapshot) as dst:
            src.backup(dst)
        shutil.copyfile(snapshot, os.path.join(backup_dir, "chats.db"))
    prefs = os.path.join(data_dir, "preferences.json")
    if os.path.exists(prefs):
        shutil.copyfile(prefs, os.path.join(backup_dir, "preferences.json"))
    print(f"Chats backed up to {backup_dir}")


_state = launcher_state(PORT)
if not _state:
    print(f"No app started from this notebook on port {PORT}.")
else:
    backup_chats(_state["data_dir"], BACKUP_DIR.strip())
    _pid_file = _state["pid_file"]
    if os.path.exists(_pid_file):
        _pid = int(open(_pid_file).read().strip())
        try:
            os.killpg(os.getpgid(_pid), signal.SIGTERM)
            print(f"Stopped Ask (pid {_pid}).")
        except ProcessLookupError:
            print("Ask was not running.")
        os.remove(_pid_file)
    else:
        print("Ask was not running.")
