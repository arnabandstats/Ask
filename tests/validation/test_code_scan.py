"""Static code scanner (ask/validation/code_scan.py): exact rule/file/line expectations on a fixture repository,
plus compliant counterparts proving the seeded / safe version of every pattern raises nothing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ask.validation.code_scan import INFO_RULES, RULES, CodeFinding, scan, summarise

TRAIN = '''\
import random
import time
import warnings
import pickle
import subprocess
import os
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

warnings.filterwarnings("ignore")
DATA = "C:\\\\data\\\\loans.csv"
password = "Sup3rS3cret!"
START = "2019-01-01"
CUTOFF = 0.35

df = pd.read_csv(DATA)
df = df[df["snapshot"] <= datetime(2023, 12, 31)]
noise = np.random.normal(0, 1, len(df))
idx = random.choice(range(10))
rng = np.random.default_rng()
gen = np.random.default_rng(seed=int(time.time()))
df["fut_bal"] = df.groupby("id")["balance"].shift(-1)
df["te"] = df.groupby("region")["default"].transform("mean")
scaler = StandardScaler()
df[["income", "age", "ltv"]] = scaler.fit_transform(df[["income", "age", "ltv"]])
X = df.drop(columns=["id"])
y = df["default"]
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.3)
X_test_s = scaler.fit_transform(X_test)
rf = RandomForestClassifier(n_estimators=300)
lr = LogisticRegression(solver="saga")
kf = KFold(n_splits=5, shuffle=True)
model = lgb.LGBMClassifier()
proba = model.predict_proba(X_test)[:, 1]
y_hat = (proba > 0.5).astype(int)
features = list(set(X.columns) - {"id"})
try:
    risky = 1 / 0
except:
    pass
try:
    risky = 2
except Exception:
    pass
obj = pickle.load(open("m.pkl", "rb"))
subprocess.run("ls -l", shell=True)
os.system("echo hi")
table = "loans"
q = f"SELECT * FROM {table} WHERE dt > '2023-01-01'"
eval("1+1")
df[df["age"] > 30]["flag"] = 1
df["age"].fillna(0, inplace=True)
df.fillna(0)
assert len(df) > 0
# TODO: tune thresholds
from os.path import *
num_cols = ["ltv", "income", "age"]


def build(cols=[]):
    alpha = 0.05
    return cols
'''

PIPELINE = '''\
"""Pipeline module."""
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import SMOTE
import helpers


def select_then_cv(X, y, model):
    """Selection on full data."""
    sel = SelectKBest(k=10)
    X_sel = sel.fit_transform(X, y)
    return cross_val_score(model, X_sel, y, cv=5)


def smote_then_split(X, y):
    """Oversampling before split."""
    X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
    return train_test_split(X_res, y_res, random_state=1)


def good(X, y, model):
    """Compliant: split first, fit on the training part only."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    sel = SelectKBest(k=10)
    X_train_s = sel.fit_transform(X_train, y_train)
    X_test_s = sel.transform(X_test)
    return cross_val_score(model, X_train_s, y_train, cv=5), X_test_s, helpers
'''

FAKE_AWS_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"        # assembled so the repo's own secret test does not trip
CREDS = f'''\
"""Connection settings."""
KEY = "{FAKE_AWS_KEY}"
URL = "postgresql://admin:pa55word@db.internal:5432/risk"
'''

BROKEN = "def f(:\n    x = 'C:\\\\tmp\\\\a.csv'\n"

DBX = '''\
# Databricks notebook source
x = 1

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM loans

# COMMAND ----------

# MAGIC %pip install xgboost
'''

SQL = '''\
-- extract the development sample
SELECT * FROM loans WHERE snapshot_dt = '2023-12-31';
DELETE FROM staging;
UPDATE scores SET flag = 1;
SELECT id, COUNT(*) FROM loans GROUP BY id;
'''

R_CODE = '''\
df <- read.csv("C:/data/x.csv")
s <- sample(1:10)
suppressWarnings(fit <- glm(y ~ x, data = df))
start <- "2020-01-31"
'''

SAS = '''\
libname src 'C:\\sas\\data';
data a; set src.loans; u = ranuni(0); if snap <= '31DEC2023'd; run;
proc surveyselect data=a out=b method=srs n=100; run;
'''

CONFIG = '''\
db:
  host: risk-db
  db_password: hunter2pass
  api_key: ${API_KEY}
'''

REQS = '''\
pandas>=2.0
numpy==1.26.4
scikit-learn==1.4.2
numpy==1.26.3
'''

ENV = '''\
name: model
channels:
  - conda-forge
dependencies:
  - python=3.11
  - scikit-learn=1.3.0
  - lightgbm
  - pip:
    - shap>=0.40
'''


def _nb(cells):
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


def _code(src, count, outputs=()):
    return {"cell_type": "code", "source": src, "execution_count": count, "outputs": list(outputs), "metadata": {}}


NOTEBOOK = _nb([
    {"cell_type": "markdown", "source": "# Title\nTODO in markdown is not code", "metadata": {}},
    _code("import numpy as np\nx = np.random.rand(3)", 2,
          [{"output_type": "stream", "name": "stdout", "text": ["[0.1 0.2]\n", "# TODO not code\n"]}]),
    _code("y = 1", 1),
    _code("z = 2", None),
    _code("raise ValueError('bad')", 3,
          [{"output_type": "error", "ename": "ValueError", "evalue": "bad", "traceback": []}]),
    _code("%pip install pandas\nprint(1)", 4),
    _code("%sql\nDELETE FROM loans", 5),
])

LONG = '"""Long."""\n\n\ndef long_fn():\n    """Doc."""\n' + "    a = 0\n" * 100 + "    return a\n"

BAD = {
    "src/train.py": TRAIN,
    "src/pipeline.py": PIPELINE,
    "src/helpers.py": '"""Helpers."""\n',
    "src/creds.py": CREDS,
    "src/broken.py": BROKEN,
    "src/long.py": LONG,
    "notebooks/dbx_job.py": DBX,
    "notebooks/explore.ipynb": NOTEBOOK,
    "sql/extract.sql": SQL,
    "r/model.R": R_CODE,
    "sas/sample.sas": SAS,
    "conf/config.yml": CONFIG,
    "requirements.txt": REQS,
    "environment.yml": ENV,
}


def L(text: str, marker: str, nth: int = 1) -> int:
    """1-based line number of the nth line containing marker."""
    hits = [i for i, ln in enumerate(text.split("\n"), 1) if marker in ln]
    assert len(hits) >= nth, f"marker {marker!r} not found"
    return hits[nth - 1]


GOOD_PY = '''\
"""Compliant model script: the safe version of every flagged pattern."""
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


def build(path, cfg_path, spark, run_date, cols=None):
    """Load, split and fit with every random component seeded."""
    with open(cfg_path) as fh:
        conf = yaml.safe_load(fh)
    df = pd.read_parquet(path)
    noise = np.random.normal(0, 1, len(df))
    pick = random.choice([3, 4, 5])
    rng = np.random.default_rng(SEED)
    sub = df.sample(frac=0.1, random_state=SEED)
    df["lag_bal"] = df.groupby("id")["balance"].shift(1)
    df["target"] = df.groupby("id")["dpd"].shift(-12)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, random_state=SEED)
    pipe = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=SEED))
    pipe.fit(X_fit, y_fit)
    cal = CalibratedClassifierCV(pipe, cv="prefit")
    cal.fit(X_val, y_val)
    lr = LogisticRegression(solver="lbfgs")
    kf = KFold(n_splits=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cal.predict_proba(X_test)[:, 1]
    labels = (proba >= conf["threshold"]).astype(int)
    feats = sorted(set(X.columns))
    try:
        n_bad = int(conf["n_bad"])
    except (KeyError, ValueError):
        n_bad = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df.loc[df["lag_bal"].isna(), "lag_bal"] = df["balance"]
    query = "SELECT id, score FROM loans WHERE snapshot_dt = :run_date"
    rows = spark.sql(query, args={"run_date": run_date})
    password = os.environ["DB_PASSWORD"]
    api_key = os.getenv("API_KEY", "")
    cols = cols or []
    return df, labels, feats, pick, noise, rng, sub, lr, kf, skf, n_bad, rows, password, api_key, cols
'''

GOOD_NB = _nb([
    {"cell_type": "markdown", "source": "# Model", "metadata": {}},
    _code("import numpy as np\nrng = np.random.default_rng(7)\nx = rng.normal(size=3)", 1),
    _code("%pip install pandas==2.2.2\ny = 1", 2),
    _code("", None),
    _code("%sql\nSELECT id, score FROM loans WHERE snapshot_dt = :ref_date", 3),
])

GOOD = {
    "src/model.py": GOOD_PY,
    "notebooks/model.ipynb": GOOD_NB,
    "sql/extract.sql": "SELECT id, score FROM loans WHERE snapshot_dt = :ref_date;\n"
                       "DELETE FROM staging WHERE run_id = 3;\nUPDATE scores SET flag = 1 WHERE id = 7;\n",
    "r/model.R": 'set.seed(20240101)\ns <- sample(1:10)\nstart <- as.Date(params$start)\n',
    "sas/sample.sas": "data a; call streaminit(123); u = rand('uniform'); run;\n"
                      "proc surveyselect data=a out=b method=srs n=100 seed=42; run;\n",
    "conf/config.yml": "db:\n  host: risk-db\n  db_password: ${DB_PASSWORD}\n  max_token: 4000\n",
    "requirements.txt": "numpy==1.26.4\npandas==2.2.2\nscikit-learn==1.4.2\ntorch==2.3.0\npyyaml==6.0.1\n",
}


def _expected_bad():
    t, p, s = TRAIN, PIPELINE, "src/train.py"
    exp = {
        ("CS005", s, L(t, "import torch")), ("CS503", s, L(t, "import torch")),
        ("CS303", s, L(t, "filterwarnings")), ("CS103", s, L(t, "DATA =")), ("CS401", s, L(t, "password =")),
        ("CS104", s, L(t, "START =")), ("CS101", s, L(t, "CUTOFF =")), ("CS104", s, L(t, "datetime(2023")),
        ("CS001", s, L(t, "np.random.normal")), ("CS001", s, L(t, "random.choice")),
        ("CS002", s, L(t, "default_rng()")), ("CS004", s, L(t, "time.time()")), ("CS207", s, L(t, "shift(-1)")),
        ("CS203", s, L(t, '["default"].transform')), ("CS201", s, L(t, "scaler.fit_transform(df")),
        ("CS204", s, L(t, "X = df.drop")), ("CS003", s, L(t, "= train_test_split")),
        ("CS202", s, L(t, "fit_transform(X_test)")), ("CS003", s, L(t, "RandomForestClassifier(n_")),
        ("CS003", s, L(t, 'solver="saga"')), ("CS003", s, L(t, "shuffle=True")),
        ("CS003", s, L(t, "LGBMClassifier()")),
        ("CS101", s, L(t, "proba > 0.5")), ("CS006", s, L(t, "list(set(")), ("CS301", s, L(t, "except:")),
        ("CS302", s, L(t, "except Exception:")), ("CS403", s, L(t, "pickle.load")),
        ("CS404", s, L(t, "shell=True")), ("CS404", s, L(t, "os.system")), ("CS405", s, L(t, 'f"SELECT')),
        ("CS701", s, L(t, 'f"SELECT')), ("CS703", s, L(t, 'f"SELECT')), ("CS402", s, L(t, "eval(")),
        ("CS102", s, L(t, '["age"] > 30')), ("CS305", s, L(t, '["age"] > 30')),
        ("CS305", s, L(t, "inplace=True")), ("CS304", s, L(t, "df.fillna(0)")), ("CS306", s, L(t, "assert len")),
        ("CS801", s, L(t, "# TODO")), ("CS805", s, L(t, "import *")), ("CS105", s, L(t, "num_cols =")),
        ("CS803", s, L(t, "def build")), ("CS804", s, L(t, "def build")), ("CS102", s, L(t, "alpha = 0.05")),
        ("CS503", "src/pipeline.py", L(p, "from imblearn")),
        ("CS206", "src/pipeline.py", L(p, "sel.fit_transform(X, y)")),
        ("CS205", "src/pipeline.py", L(p, "fit_resample")),
        ("CS401", "src/creds.py", 2), ("CS401", "src/creds.py", 3),
        ("CS802", "src/long.py", 4), ("CS307", "src/broken.py", 1), ("CS103", "src/broken.py", 2),
        ("CS701", "notebooks/dbx_job.py", L(DBX, "MAGIC SELECT")),
        ("CS501", "notebooks/dbx_job.py", L(DBX, "%pip install xgboost")),
        ("CS701", "sql/extract.sql", 2), ("CS703", "sql/extract.sql", 2), ("CS702", "sql/extract.sql", 3),
        ("CS702", "sql/extract.sql", 4),
        ("CS103", "r/model.R", 1), ("CS001", "r/model.R", 2), ("CS303", "r/model.R", 3), ("CS104", "r/model.R", 4),
        ("CS103", "sas/sample.sas", 1), ("CS004", "sas/sample.sas", 2), ("CS104", "sas/sample.sas", 2),
        ("CS001", "sas/sample.sas", 3),
        ("CS401", "conf/config.yml", 3),
        ("CS501", "requirements.txt", 1), ("CS502", "requirements.txt", 3), ("CS502", "requirements.txt", 4),
        ("CS501", "environment.yml", 7), ("CS501", "environment.yml", 9),
    }
    # notebook lines refer to the loaders-style rendering ("# %% [cell N] kind" headers)
    nb = "notebooks/explore.ipynb"
    exp |= {("CS001", nb, 6), ("CS601", nb, 10), ("CS603", nb, 12), ("CS602", nb, 14), ("CS501", nb, 17),
            ("CS702", nb, 21)}
    return exp


def test_bad_repository_exact_findings():
    got = {(f.rule, f.file, f.line) for f in scan(BAD, include_info=True)}
    exp = _expected_bad()
    assert got - exp == set(), "unexpected findings (false positives)"
    assert exp - got == set(), "missed findings"


def test_every_rule_is_exercised_by_the_fixture():
    assert {r for r, _, _ in _expected_bad()} == set(RULES)


def test_compliant_repository_has_no_findings():
    assert scan(GOOD, include_info=True) == []


def test_info_rules_hidden_by_default_and_rule_filter():
    default = scan(BAD)
    assert default and not any(f.rule in INFO_RULES for f in default)
    assert {f.rule for f in scan(BAD, include_info=True)} >= INFO_RULES
    only = scan(BAD, rules=["CS003", "CS801"])          # an explicitly requested info rule is reported
    assert {f.rule for f in only} == {"CS003", "CS801"}
    assert sum(f.rule == "CS003" for f in only) == 5
    with pytest.raises(ValueError, match="CS999"):
        scan(BAD, rules=["CS999"])


def test_finding_fields_sorted_and_deterministic():
    a, b = scan(BAD, include_info=True), scan(dict(reversed(list(BAD.items()))), include_info=True)
    assert a == b
    assert a == sorted(a, key=lambda f: (f.file, f.line, f.rule, f.end_line, f.message))
    cats = {"reproducibility", "hard-coding", "leakage", "robustness", "security", "dependencies",
            "documentation", "maintainability", "sql"}
    for f in a:
        assert isinstance(f, CodeFinding)
        assert (f.title, f.category) == RULES[f.rule][:2] and f.category in cats
        assert f.end_line >= f.line >= 1 and len(f.snippet) <= 200 and f.message


def test_messages_are_specific():
    by = {(f.rule, f.line): f for f in scan({"src/train.py": TRAIN}, include_info=True)}
    cut = by[("CS101", L(TRAIN, "proba > 0.5"))].message
    assert "0.5" in cut and "proba" in cut
    assert "'default'" in by[("CS204", L(TRAIN, "X = df.drop"))].message
    leak = by[("CS201", L(TRAIN, "scaler.fit_transform(df"))].message
    assert "StandardScaler" in leak and "train_test_split" in leak
    assert f"line {L(TRAIN, '= train_test_split')}" in leak
    assert "X_test" in by[("CS202", L(TRAIN, "fit_transform(X_test)"))].message
    assert "time.time" in by[("CS004", L(TRAIN, "time.time()"))].message
    assert "table" in by[("CS405", L(TRAIN, 'f"SELECT'))].message
    assert "`cols`" in by[("CS804", L(TRAIN, "def build"))].message


def test_secrets_are_redacted():
    secrets = ["Sup3rS3cret!", FAKE_AWS_KEY, "pa55word", "hunter2pass"]
    found = scan(BAD, rules=["CS401"])
    assert len(found) == 4
    for f in found:
        assert not any(s in f.snippet or s in f.message for s in secrets), f
        assert "<redacted>" in f.snippet


def test_notebook_raw_json_matches_loader_rendering(tmp_path: Path):
    from ask.sources.loaders import _read_ipynb
    p = tmp_path / "explore.ipynb"
    p.write_text(NOTEBOOK, encoding="utf-8")
    rendered = _read_ipynb(p)
    raw = {(f.rule, f.line) for f in scan({"nb.ipynb": NOTEBOOK}, include_info=True)}
    ren = {(f.rule, f.line) for f in scan({"nb.ipynb": rendered}, include_info=True)}
    # execution counts / error outputs exist only in the raw JSON; everything else lines up exactly
    exec_only = {("CS601", 10), ("CS603", 12), ("CS602", 14)}
    assert raw - ren == exec_only and ren == raw - exec_only
    assert rendered.split("\n")[5] == "x = np.random.rand(3)"
    f = next(f for f in scan({"nb.ipynb": rendered}) if f.rule == "CS001")
    assert f.message.endswith("[notebook cell 2]") and f.snippet == "x = np.random.rand(3)"


def test_traceback_in_rendered_output_is_reported():
    text = "# %% [cell 1] code\nimport os\n# [output]\n# Traceback (most recent call last)\n# KeyError: 'a'"
    assert [(f.rule, f.line) for f in scan({"a.ipynb": text})] == [("CS602", 4)]


def test_seeded_and_scoped_variants_are_not_flagged():
    code = '''\
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
np.random.seed(0)
random.seed(1)
a = np.random.rand(3)
b = random.random()
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
sc = StandardScaler().fit(X_train)
Z = sc.transform(X_test)
config = {"a": {"b": 1}}
config["a"]["b"] = 2
risk_grade = df["score"].rank()
interval = model.fit(train_interval)
try:
    import xgboost
except ImportError:
    xgboost = None
'''
    assert scan({"m.py": code, "requirements.txt": "numpy==1\nscikit-learn==1\n"}) == []


def test_unseeded_generator_seed_call():
    code = "import numpy as np\nnp.random.seed()\nx = np.random.rand(2)\n"
    assert [(f.rule, f.line) for f in scan({"m.py": code})] == [("CS002", 2), ("CS001", 3)]


def test_dependency_rules_need_a_requirements_file_and_respect_lock_files():
    assert scan({"m.py": "import lightgbm\n"}) == []
    py = '[project]\nname = "x"\ndependencies = ["pandas>=2", "numpy==1.26.4"]\n'
    assert [(f.rule, f.line) for f in scan({"pyproject.toml": py})] == [("CS501", 3)]
    assert scan({"pyproject.toml": py, "uv.lock": ""}) == []
    poetry = '[tool.poetry.dependencies]\npython = "^3.11"\npandas = "^2.0"\nnumpy = "1.26.4"\n'
    assert [(f.rule, f.line) for f in scan({"pyproject.toml": poetry})] == [("CS501", 3)]


def test_summarise_counts():
    findings = scan(BAD, include_info=True)
    s = summarise(findings)
    assert list(s.columns) == ["rule", "title", "category", "level", "file", "findings", "first_line"]
    assert s["findings"].sum() == len(findings)
    row = s[(s["rule"] == "CS003") & (s["file"] == "src/train.py")].iloc[0]
    assert row["findings"] == 5 and row["first_line"] == L(TRAIN, "= train_test_split")
    assert row["level"] == "finding"
    assert set(s.loc[s["rule"] == "CS801", "level"]) == {"info"}
    assert summarise([]).empty


def test_degenerate_inputs_do_not_crash():
    assert scan({}) == []
    odd = {"a.py": "", "b.ipynb": "{not json", "c.sql": "", "d.R": "\x00\x01", "e.yml": ":", "f.bin": "�",
           "g.ipynb": json.dumps({"cells": []}), "h.py": "x = 1\r\ny = np.random.rand(2)\r\nimport numpy as np\r\n"}
    found = scan(odd, include_info=True)
    assert [(f.rule, f.file, f.line) for f in found] == [("CS001", "h.py", 2)]


def test_rule_catalog_well_formed():
    assert INFO_RULES <= set(RULES)
    for rid, (title, category, desc) in RULES.items():
        assert rid.startswith("CS") and len(rid) == 5 and title and desc
