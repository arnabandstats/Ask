# Databricks notebook source
# MAGIC %md
# MAGIC # Launch the chat app on Databricks
# MAGIC
# MAGIC **Run all** — the notebook picks the right way for the compute it is attached to:
# MAGIC
# MAGIC | Attached to | What happens |
# MAGIC |---|---|
# MAGIC | **Serverless**, or a cluster in **Shared/Standard** access mode | Deploys the app as a **Databricks App** (created/updated for you, your API-key secret attached) and prints its URL. |
# MAGIC | A cluster in **Dedicated (single user)** access mode | Runs the app on the cluster's driver and prints a link (driver proxy). |
# MAGIC
# MAGIC Change **Run mode** to force one or the other.
# MAGIC
# MAGIC **Before the first run**
# MAGIC 1. This notebook must sit in the repo root, next to `app.py` (a Git folder of the repo).
# MAGIC 2. The API key is read from the Databricks secret **`ask` / `openai-api-key`** (the widget
# MAGIC    defaults), stored once from a terminal with
# MAGIC    `databricks secrets create-scope ask` then `databricks secrets put-secret ask openai-api-key`.
# MAGIC 3. Databricks Apps must be enabled in the workspace (ask an admin if creating an app fails).
# MAGIC
# MAGIC Step 2 checks the secret and the compute before anything is installed.
# MAGIC
# MAGIC *Cluster mode only:* the app runs while the cluster is up; **Stop the app** stops it.
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

dbutils.widgets.dropdown("run_mode", "auto", ["auto", "databricks_app", "cluster"], "Run mode")
dbutils.widgets.text("app_name", "", "App name (blank = ask-<your user>)")
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


SHARED_MODES = {"USER_ISOLATION", "DATA_SECURITY_MODE_STANDARD", "STANDARD"}


def spark_conf(key):
    """A Spark setting, or None. Serverless compute raises CONFIG_NOT_AVAILABLE instead of
    returning a default, so every lookup is guarded."""
    try:
        return spark.conf.get(key)
    except Exception:
        return None


def compute_kind():
    """'serverless', 'shared' or 'dedicated'."""
    mode = spark_conf("spark.databricks.clusterUsageTags.dataSecurityMode")
    if mode is None or spark_conf("spark.databricks.clusterUsageTags.clusterId") is None:
        return "serverless"
    return "shared" if mode.upper() in SHARED_MODES else "dedicated"


MODE_FILE = "/tmp/ask_launcher_mode.txt"      # survives the Python restart after %pip


def decide_run_mode():
    """'app' (deploy a Databricks App) or 'cluster' (run on this cluster's driver).
    The driver link needs the driver proxy, which serverless does not have and Shared/Standard
    clusters block ('Traffic on this port is not permitted') — so those use Databricks Apps."""
    choice = dbutils.widgets.get("run_mode")
    kind = compute_kind()
    if choice == "cluster" and kind != "dedicated":
        raise RuntimeError(
            f"Run mode 'cluster' needs a cluster in Dedicated (single user) access mode, but this "
            f"notebook is on {kind} compute, where Databricks blocks the app link. Set Run mode to "
            "'auto' or 'databricks_app'.")
    mode = "cluster" if choice == "cluster" or (choice == "auto" and kind == "dedicated") else "app"
    where = {"serverless": "serverless compute", "shared": "a Shared/Standard cluster",
             "dedicated": "a Dedicated cluster"}[kind]
    print(f"Attached to {where}: " + ("running the app on this cluster's driver."
                                       if mode == "cluster" else "deploying it as a Databricks App."))
    with open(MODE_FILE, "w") as fh:
        fh.write(mode)
    return mode


check_api_key_secret()
RUN_MODE = decide_run_mode()

# COMMAND ----------

# MAGIC %md ## 3 · Install dependencies
# MAGIC App mode needs only the Databricks SDK (Databricks Apps installs the app's own
# MAGIC requirements). Cluster mode installs the repo's `requirements.txt` here.

# COMMAND ----------

import os

_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
APP_DIR = "/Workspace" + os.path.dirname(_nb_path)
assert os.path.exists(os.path.join(APP_DIR, "app.py")), (
    f"app.py not found in {APP_DIR}. Keep this notebook in the repo root, next to app.py.")

_mode = open("/tmp/ask_launcher_mode.txt").read().strip()
_reqs = ["databricks-sdk>=0.50"]                 # app deployment API
if _mode == "cluster":
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

# MAGIC %md ## 4 · Deploy as a Databricks App (serverless / Shared clusters)
# MAGIC Creates the app the first time (or updates it), attaches your API-key secret as the
# MAGIC app resource `openai-api-key` (read by `app.yaml`), deploys this folder, and prints the
# MAGIC URL. The first deployment installs the requirements and takes a few minutes. In cluster
# MAGIC mode this step is skipped.

# COMMAND ----------

import os
import re
from datetime import timedelta


def deploy_databricks_app(w, app_name, source_path, scope, key, log=print):
    """Create or update the app, make sure its compute runs, deploy `source_path`.
    Returns the app URL. `w` is a databricks.sdk.WorkspaceClient."""
    from databricks.sdk.service import apps as A

    secret = A.AppResource(name="openai-api-key", description="OpenAI API key",
                           secret=A.AppResourceSecret(scope=scope, key=key,
                                                      permission=A.AppResourceSecretSecretPermission.READ))
    spec = A.App(name=app_name, description="Ask: chat with repositories, documents and data",
                 resources=[secret])
    from databricks.sdk.errors import NotFound

    try:
        w.apps.get(app_name)
        exists = True
    except NotFound:                             # anything else (e.g. no permission) is raised as-is
        exists = False

    if exists:
        log(f"Updating app '{app_name}' (secret resource '{scope}/{key}')…")
        w.apps.update(name=app_name, app=spec)
    else:
        log(f"Creating app '{app_name}' with secret resource '{scope}/{key}' (takes a minute or two)…")
        w.apps.create_and_wait(app=spec, timeout=timedelta(minutes=20))

    app = w.apps.get(app_name)
    state = getattr(getattr(app, "compute_status", None), "state", None)
    if state is not None and str(getattr(state, "value", state)) not in ("ACTIVE", "STARTING", "UPDATING"):
        log("Starting the app's compute…")
        w.apps.start_and_wait(app_name, timeout=timedelta(minutes=20))

    log("Deploying (the first time installs requirements; a few minutes)…")

    def _fail(detail):
        url = w.apps.get(app_name).url or ""
        return RuntimeError(f"Deployment of '{app_name}' failed: {detail}. Open Compute → Apps → "
                            f"{app_name} → Logs{(' (' + url + '/logz)') if url else ''} for the details.")

    try:
        dep = w.apps.deploy_and_wait(
            app_name=app_name,
            app_deployment=A.AppDeployment(source_code_path=source_path, mode=A.AppDeploymentMode.SNAPSHOT),
            timeout=timedelta(minutes=30))
    except Exception as exc:                     # the SDK raises OperationFailed on a failed deploy
        raise _fail(exc) from exc
    status = getattr(dep, "status", None)
    result = str(getattr(getattr(status, "state", None), "value", getattr(status, "state", "")))
    if result != "SUCCEEDED":
        raise _fail(f"{result or 'unknown state'}: {getattr(status, 'message', '') or 'no message'}")
    return w.apps.get(app_name).url


def default_app_name(user_name):
    """'ask-<user>' in the form Apps accepts: lowercase letters, digits, dashes, max 30 chars."""
    user = re.sub(r"[^a-z0-9]+", "-", user_name.split("@")[0].lower()).strip("-")
    return ("ask-" + user)[:30].rstrip("-") or "ask"


if open("/tmp/ask_launcher_mode.txt").read().strip() == "app":
    from databricks.sdk import WorkspaceClient

    _w = WorkspaceClient()
    _nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    _source = "/Workspace" + os.path.dirname(_nb_path)
    _name = dbutils.widgets.get("app_name").strip() or default_app_name(_w.current_user.me().user_name)
    APP_URL = deploy_databricks_app(_w, _name, _source, dbutils.widgets.get("secret_scope").strip(),
                                    dbutils.widgets.get("secret_key").strip())
    displayHTML(f"""
<div style="font-family: system-ui, sans-serif; font-size: 15px; padding: 8px 0">
  <a href="{APP_URL}" target="_blank" rel="noopener"
     style="background:#16191f;color:#fff;padding:10px 18px;border-radius:10px;text-decoration:none">
     Open the app ↗</a>
  <div style="margin-top:12px;color:#6b7079">{APP_URL} · Databricks App “{_name}”.
  Share it from Compute → Apps → {_name} → Permissions (Can use).</div>
</div>""")
    dbutils.notebook.exit(APP_URL)               # done: the cluster-mode cells below are skipped
else:
    print("Cluster mode: skipping the Databricks App deployment.")

# COMMAND ----------

# MAGIC %md ## 5 · Start the app (cluster mode)

# COMMAND ----------

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request


def _first_writable(candidates):
    """First folder we can create and write to. /local_disk0 is not writable on clusters in
    Shared / Standard access mode, where notebook code runs as a restricted user."""
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


_nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
APP_DIR = "/Workspace" + os.path.dirname(_nb_path)
PORT = int(dbutils.widgets.get("port"))
_requested = dbutils.widgets.get("data_dir").strip()
DATA_DIR = _first_writable([_requested, "/local_disk0/ask_data", "/tmp/ask_data",
                            os.path.join(tempfile.gettempdir(), "ask_data"),
                            os.path.expanduser("~/ask_data")])
if _requested and DATA_DIR != _requested:
    print(f"Note: {_requested} is not writable on this cluster (typical in Shared/Standard access "
          f"mode), so chat history goes to {DATA_DIR} instead.")
PID_FILE = os.path.join(DATA_DIR, f"streamlit_{PORT}.pid")
LOG_FILE = os.path.join(DATA_DIR, f"streamlit_{PORT}.log")
# Where the later cells (log / back up / stop) find this run, even after a detach.
STATE_FILE = os.path.join(tempfile.gettempdir(), f"ask_launcher_{PORT}.json")

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
    except (OSError, ValueError):
        pass
    finally:
        try:
            os.remove(PID_FILE)
        except OSError:
            pass


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
with open(STATE_FILE, "w") as fh:
    json.dump({"data_dir": DATA_DIR, "pid": proc.pid, "log_file": LOG_FILE, "pid_file": PID_FILE}, fh)

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

# MAGIC %md ## 6 · Open it (cluster mode)

# COMMAND ----------

def _conf(key):
    try:
        return spark.conf.get(key)
    except Exception:          # serverless raises CONFIG_NOT_AVAILABLE
        return None


_host = _conf("spark.databricks.workspaceUrl")
_org = _conf("spark.databricks.clusterUsageTags.clusterOwnerOrgId")
_cluster = _conf("spark.databricks.clusterUsageTags.clusterId")
_mode = _conf("spark.databricks.clusterUsageTags.dataSecurityMode") or ""
if not (_host and _org and _cluster):
    raise RuntimeError("No app link on this compute (serverless has no driver proxy). Attach a "
                       "classic cluster in Dedicated access mode, or deploy as a Databricks App.")
APP_URL = f"https://{_host}/driver-proxy/o/{_org}/{_cluster}/{PORT}/"
_blocked = _mode.upper() in {"USER_ISOLATION", "DATA_SECURITY_MODE_STANDARD", "STANDARD"}
_warning = (f"""<div style="margin:0 0 14px;padding:10px 14px;border-radius:10px;background:#fff4e5;
  color:#7a4a00">This cluster is in <b>Shared/Standard</b> access mode ({_mode}). Databricks blocks
  this link on such clusters (“Traffic on this port is not permitted”). Switch the cluster to
  <b>Dedicated</b> access mode and rerun, or deploy as a <b>Databricks App</b> (see the README).</div>"""
            if _blocked else "")

displayHTML(_warning + f"""
<div style="font-family: system-ui, sans-serif; font-size: 15px; padding: 8px 0">
  <a href="{APP_URL}" target="_blank" rel="noopener"
     style="background:#16191f;color:#fff;padding:10px 18px;border-radius:10px;text-decoration:none">
     Open the app ↗</a>
  <div style="margin-top:12px;color:#6b7079">{APP_URL}</div>
</div>""")

# COMMAND ----------

# MAGIC %md ## 7 · Log (cluster mode, run any time)

# COMMAND ----------

import json
import os
import tempfile


def launcher_state():
    """What the Start cell recorded for this port: data_dir, pid, log_file, pid_file."""
    port = int(dbutils.widgets.get("port"))
    try:
        with open(os.path.join(tempfile.gettempdir(), f"ask_launcher_{port}.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        data_dir = dbutils.widgets.get("data_dir").strip() or "/local_disk0/ask_data"
        return {"data_dir": data_dir, "log_file": os.path.join(data_dir, f"streamlit_{port}.log"),
                "pid_file": os.path.join(data_dir, f"streamlit_{port}.pid")}


_log = launcher_state()["log_file"]
print(open(_log).read()[-6000:] if os.path.exists(_log) else "No log yet: run 'Start the app' first.")

# COMMAND ----------

# MAGIC %md ## 8 · Back up chats (run any time; also done by Stop)

# COMMAND ----------

import json
import os
import shutil
import sqlite3
import tempfile


def launcher_state():
    """What the Start cell recorded for this port: data_dir, pid, log_file, pid_file."""
    port = int(dbutils.widgets.get("port"))
    try:
        with open(os.path.join(tempfile.gettempdir(), f"ask_launcher_{port}.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        data_dir = dbutils.widgets.get("data_dir").strip() or "/local_disk0/ask_data"
        return {"data_dir": data_dir, "log_file": os.path.join(data_dir, f"streamlit_{port}.log"),
                "pid_file": os.path.join(data_dir, f"streamlit_{port}.pid")}


def backup_chats():
    data_dir = launcher_state()["data_dir"]
    backup_dir = dbutils.widgets.get("backup_dir").strip()
    if not backup_dir:
        print("No backup folder set; chats stay on the driver disk only.")
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


backup_chats()

# COMMAND ----------

# MAGIC %md ## 9 · Stop the app (cluster mode)

# COMMAND ----------

import os
import signal

if "backup_chats" in globals():
    backup_chats()
else:
    print("Not backed up: run the 'Back up chats' cell first if you set a backup folder.")
PORT = int(dbutils.widgets.get("port"))
_pid_file = launcher_state()["pid_file"] if "launcher_state" in globals() else ""
if _pid_file and os.path.exists(_pid_file):
    _pid = int(open(_pid_file).read().strip())
    try:
        os.killpg(os.getpgid(_pid), signal.SIGTERM)
        print(f"Stopped the app (pid {_pid}).")
    except ProcessLookupError:
        print("The app was not running.")
    os.remove(_pid_file)
else:
    print(f"No app started from this notebook on port {PORT} "
          "(if you just reattached, run the 'Back up chats' cell first, then this one).")
