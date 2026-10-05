# Databricks notebook source
# MAGIC %pip install --upgrade --force-reinstall --no-cache-dir -r requirements.txt

# COMMAND ----------

import os
import pathlib
import subprocess
import sys
import time

from IPython import get_ipython
from IPython.display import HTML, display

PORT = 8502
APP_FILE = "app.py"

# Azure OpenAI used by Ask. The key is read from SECRET_SCOPE below.
AZURE_KEY_SECRET = ""                      # name of the key in OneLab-SecretScope holding the Azure OpenAI key
AZURE_ENDPOINT = ""                        # e.g. https://<resource>.openai.azure.com/
AZURE_API_VERSION = "2025-03-01-preview"
AZURE_DEPLOYMENT = ""                      # Azure deployment name (blank = gpt-4.1)

_ipython = get_ipython()
if _ipython is None or "dbutils" not in _ipython.user_ns:
    raise RuntimeError("Attach and run this notebook on a Databricks cluster.")
dbutils = _ipython.user_ns["dbutils"]

existing_process = globals().get("ask_process")
if existing_process is not None and existing_process.poll() is None:
    existing_process.terminate()
    try:
        existing_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        existing_process.kill()
        existing_process.wait(timeout=5)

existing_log_handle = globals().get("ask_log_handle")
if existing_log_handle is not None and not existing_log_handle.closed:
    existing_log_handle.close()

repo_dir = pathlib.Path.cwd()
app_path = repo_dir / APP_FILE
if not app_path.exists():
    raise FileNotFoundError(
        f"{APP_FILE} was not found in {repo_dir}. Run this notebook from the same Databricks Repo folder as the application."
    )
if not AZURE_KEY_SECRET or not AZURE_ENDPOINT:
    raise ValueError("Set AZURE_KEY_SECRET and AZURE_ENDPOINT at the top of this cell.")

app_env = os.environ.copy()
app_env["DataServicePrincipalClientId"] = dbutils.secrets.get(
    scope="OneLab-SecretScope",
    key="DataServicePrincipalClientId",
)
app_env["DataServicePrincipalClientSecret"] = dbutils.secrets.get(
    scope="OneLab-SecretScope",
    key="DataServicePrincipalClientSecret",
)
app_env["USE_AZURE_OPENAI"] = "true"
app_env["AZURE_OPENAI_KEY"] = dbutils.secrets.get(
    scope="OneLab-SecretScope",
    key=AZURE_KEY_SECRET,
).strip()
app_env["AZURE_OPENAI_BASE_URL"] = AZURE_ENDPOINT
app_env["AZURE_OPENAI_VERSION"] = AZURE_API_VERSION
if AZURE_DEPLOYMENT:
    app_env["OPENAI_DEPLOYMENT_NAME"] = AZURE_DEPLOYMENT
# Ask keeps its chat history in SQLite, which needs a normal disk (not the repo folder).
app_env["ASK_DATA_DIR"] = "/tmp/ask_data"

context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()


def context_text(value):
    """Read Databricks context values without converting Java objects to integers."""
    text = str(value).strip()
    if text.startswith("Some(") and text.endswith(")"):
        return text[5:-1]
    return "" if text in {"None", "null"} else text


cluster_id = context_text(context.clusterId())
browser_host = context_text(context.browserHostName())
workspace_id = context_text(context.workspaceId())

if not cluster_id and "spark" in globals():
    cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId", "")
if not browser_host and "spark" in globals():
    browser_host = spark.conf.get("spark.databricks.workspaceUrl", "")
if not workspace_id and "spark" in globals():
    workspace_id = spark.conf.get("spark.databricks.clusterUsageTags.orgId", "")

browser_host = browser_host.replace("https://", "").strip("/")
cluster_id = cluster_id.strip()
workspace_id = workspace_id.strip()
if not cluster_id or not browser_host or not workspace_id:
    raise RuntimeError("Could not resolve the Databricks workspace, browser host, or cluster ID. Re-attach to an active cluster and rerun this cell.")

log_path = repo_dir / "ask-streamlit.log"
ask_log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
ask_process = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address", "0.0.0.0",
        "--server.port", str(PORT),
        "--server.headless", "true",
    ],
    cwd=repo_dir,
    env=app_env,
    stdout=ask_log_handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

for _ in range(40):
    if ask_process.poll() is not None:
        ask_log_handle.flush()
        #raise RuntimeError("Ask stopped during startup. Run the diagnostics cell below.")
    time.sleep(0.25)

proxy_url = f"https://{browser_host}/driver-proxy/o/{workspace_id}/{cluster_id}/{PORT}/"
display(HTML(f"""
<div style="max-width:760px;padding:28px;border:1px solid #dfe3e8;border-radius:16px;background:linear-gradient(135deg,#f7fbff,#ffffff);font-family:Arial,sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.08)">
  <div style="font-size:30px;margin-bottom:8px">💬 Ask</div>
  <div style="color:#475569;margin-bottom:20px;line-height:1.5">The Streamlit application is running on the attached cluster. The cluster must remain active while the application is used.</div>
  <a href="{proxy_url}" target="_blank" rel="noopener noreferrer" style="display:inline-block;padding:12px 20px;border-radius:9px;background:#2C3696;color:white;text-decoration:none;font-weight:700">Open Ask</a>
</div>
"""))

# COMMAND ----------

# Diagnostics: is the app running, and the end of its log.
print("Running" if ask_process.poll() is None else f"Stopped (exit code {ask_process.returncode})")
print(open(log_path, encoding="utf-8").read()[-6000:])
