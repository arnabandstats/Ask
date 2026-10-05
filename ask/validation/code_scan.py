"""Deterministic static code scanner used by the agent's code-review tool.

`scan(files)` takes a loaded repository (relpath -> text, as held in `Source.files`) and returns `CodeFinding`s for
patterns a model validator cares about: unreproducible randomness, hard-coded cut-offs/paths/dates, data leakage,
silent error handling, secrets, unpinned dependencies, notebook execution hygiene and risky SQL.

Analysis of Python is AST-based (exact line numbers, import aliases resolved); regex is used only for SQL, R, SAS,
config/requirements files and for Python that does not parse. Notebooks may be given either as raw .ipynb JSON or
already rendered by `ask.sources.loaders` ("# %% [cell N] code" headers); both are analysed on that rendering, so line
numbers always refer to the rendered text and the message names the notebook cell. Execution-order and error-output
checks need the raw JSON (the rendering drops execution counts and error outputs).

Rules are listed in `RULES` (id -> title, category, description). Rules in `INFO_RULES` are informational and are only
reported with `include_info=True` or when requested explicitly through `rules=[...]`. Everything is deterministic:
output is sorted by (file, line, rule) and depends on nothing but the input text.
"""
from __future__ import annotations

import ast
import datetime as _dt
import io
import json
import re
import sys
import tokenize
import tomllib
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CodeFinding:
    rule: str
    title: str
    category: str
    file: str
    line: int
    end_line: int
    snippet: str
    message: str


RULES: dict[str, tuple[str, str, str]] = {
    # reproducibility
    "CS001": ("Unseeded global random number generator", "reproducibility",
              "np.random.* / random.* legacy global-state functions (or R/SAS random functions) used while no seed is "
              "set anywhere in the same module, so results change on every run."),
    "CS002": ("Random generator created without a seed", "reproducibility",
              "np.random.default_rng() / RandomState() / random.Random() / seed() called with no seed (or None): "
              "the generator is seeded from OS entropy."),
    "CS003": ("Randomised estimator or splitter without random_state", "reproducibility",
              "sklearn / XGBoost / LightGBM / CatBoost / imblearn estimators, splitters and DataFrame.sample that "
              "use randomness but are called without random_state/seed (or with None)."),
    "CS004": ("Time- or entropy-based seed", "reproducibility",
              "A seed derived from the clock, the process id or uuid/os.urandom, which makes the run unrepeatable."),
    "CS005": ("Deep-learning framework without global seed", "reproducibility",
              "torch / tensorflow imported but torch.manual_seed / tf.random.set_seed never called (informational)."),
    "CS006": ("Order taken from a set", "reproducibility",
              "A list/tuple or loop built from a set; string-set order depends on PYTHONHASHSEED, so feature order "
              "can change between runs (informational)."),
    # hard-coding
    "CS101": ("Hard-coded decision cut-off", "hard-coding",
              "A score/probability/prediction compared with a literal, or a threshold/cut-off defined as a literal."),
    "CS102": ("Magic number", "hard-coding",
              "Unexplained numeric literal in a comparison, or assigned inside a function (informational)."),
    "CS103": ("Hard-coded file path", "hard-coding",
              "Absolute local, network, DBFS/Volumes/Workspace or cloud-storage path written into code."),
    "CS104": ("Hard-coded date", "hard-coding",
              "Date literal (string or datetime(...)) fixing a sample window, snapshot or reference date in code."),
    "CS105": ("Repeated hard-coded column list", "hard-coding",
              "The same literal list of column names appears in several places (informational)."),
    # leakage
    "CS201": ("Preprocessing fitted before the split", "leakage",
              "Scaler / encoder / imputer / PCA fitted on data that is split or cross-validated afterwards."),
    "CS202": ("Fit on test / validation / OOT data", "leakage",
              "fit / fit_transform / fit_resample called on a variable whose name marks it as hold-out data."),
    "CS203": ("Target encoding on the full data", "leakage",
              "Target encoder fitted, or target mean per category computed, before any train/test split (heuristic)."),
    "CS204": ("Target column left in the feature matrix", "leakage",
              "X is taken from the same frame as y without dropping the target column (heuristic)."),
    "CS205": ("Resampling before the split", "leakage",
              "SMOTE / over- / under-sampling applied to data that is split or cross-validated afterwards."),
    "CS206": ("Feature selection on the full data", "leakage",
              "Supervised feature selection fitted on data that is split or cross-validated afterwards."),
    "CS207": ("Look-ahead feature", "leakage",
              "shift with a negative period or centred rolling window: values from the future enter a feature."),
    # robustness
    "CS301": ("Bare except", "robustness",
              "`except:` catches everything, including KeyboardInterrupt/SystemExit, and hides the real error."),
    "CS302": ("Exception silently swallowed", "robustness",
              "`except Exception` whose body is only pass/continue/return: failures disappear without trace."),
    "CS303": ("Warnings suppressed globally", "robustness",
              "warnings.filterwarnings('ignore'), np.seterr(... 'ignore'), chained-assignment warning disabled, "
              "R suppressWarnings / options(warn=-1)."),
    "CS304": ("Whole-frame fillna / dropna", "robustness",
              "Missing values filled or rows dropped across an entire frame without a column subset "
              "(informational)."),
    "CS305": ("Chained assignment", "robustness",
              "df[mask]['col'] = v, df['col'].loc[...] = v or df['col'].method(inplace=True): with pandas "
              "copy-on-write the original frame is not modified."),
    "CS306": ("assert used for data validation", "robustness",
              "assert statements in non-test code are removed under `python -O` (informational)."),
    "CS307": ("Python code does not parse", "robustness",
              "Syntax error: the code cannot run as written and the AST checks could not analyse it."),
    # security
    "CS401": ("Hard-coded secret", "security",
              "Password/token/API key literal or a recognised credential pattern (AWS, Azure, Databricks, GitHub, "
              "bearer token, credentials in URL). The snippet is redacted."),
    "CS402": ("eval / exec", "security", "Dynamic code execution with eval/exec (or R eval(parse()))."),
    "CS403": ("Unsafe deserialisation", "security",
              "pickle / joblib / dill / torch.load / yaml.load / read_pickle execute code from the file they read "
              "(informational)."),
    "CS404": ("Shell command execution", "security", "subprocess with shell=True, os.system or os.popen."),
    "CS405": ("SQL built by string formatting", "security",
              "SQL assembled with f-strings, .format, % or + concatenation instead of bound parameters."),
    # dependencies
    "CS501": ("Unpinned dependency", "dependencies",
              "requirements / environment.yml / pyproject / %pip install entry without an exact version."),
    "CS502": ("Duplicate or conflicting requirement", "dependencies",
              "The same package listed twice in one file, or pinned to incompatible versions in two files."),
    "CS503": ("Import missing from requirements", "dependencies",
              "A third-party package is imported but not declared in any requirements file of the repository."),
    # notebooks
    "CS601": ("Notebook cells executed out of order", "reproducibility",
              "Execution counts are not increasing in cell order: the saved outputs came from a different order."),
    "CS602": ("Notebook cell saved with an error", "robustness",
              "A cell output contains an exception, so the saved notebook did not run end to end."),
    "CS603": ("Notebook cell never executed", "reproducibility",
              "Code cell without an execution count in a notebook whose other cells were executed."),
    # SQL
    "CS701": ("SELECT *", "sql", "SELECT * makes the extracted feature set depend on the current table schema."),
    "CS702": ("DELETE / UPDATE without WHERE", "sql", "Statement modifies every row of the table."),
    "CS703": ("Hard-coded date in SQL filter", "sql",
              "Date literal in a WHERE/HAVING/BETWEEN condition fixes the sample window inside the query."),
    # maintainability / documentation
    "CS801": ("TODO / FIXME / HACK / XXX comment", "maintainability",
              "Open work-item marker left in code (informational)."),
    "CS802": ("Function longer than 100 lines", "maintainability", "Long function (informational)."),
    "CS803": ("Public function without docstring", "documentation",
              "Public function or method without a docstring (informational)."),
    "CS804": ("Mutable default argument", "maintainability",
              "A list/dict/set default is created once and shared across calls: state leaks between calls."),
    "CS805": ("Wildcard import", "maintainability",
              "`from x import *` hides where names come from (informational)."),
}

INFO_RULES = frozenset({"CS005", "CS006", "CS102", "CS105", "CS304", "CS306", "CS403",
                        "CS801", "CS802", "CS803", "CS805"})
# test code legitimately hard-codes fixtures, paths, dates and cut-offs: these rules are not applied to test files
_TEST_EXEMPT = frozenset({"CS101", "CS102", "CS103", "CS104", "CS105", "CS203", "CS204", "CS207", "CS306",
                          "CS701", "CS702", "CS703", "CS802", "CS803"})

# --------------------------------------------------------------------------------------------- vocabularies

_NP_LEGACY = {
    "rand", "randn", "randint", "random", "random_sample", "ranf", "sample", "choice", "shuffle", "permutation",
    "normal", "uniform", "binomial", "poisson", "exponential", "beta", "gamma", "multinomial", "multivariate_normal",
    "lognormal", "standard_normal", "random_integers", "geometric", "chisquare", "dirichlet", "laplace", "logistic",
    "negative_binomial", "pareto", "triangular", "weibull", "standard_t", "gumbel", "hypergeometric", "rayleigh",
    "f", "noncentral_chisquare", "noncentral_f", "power", "vonmises", "wald", "zipf", "logseries", "bytes",
    "standard_cauchy", "standard_exponential", "standard_gamma",
}
_PY_RANDOM = {"random", "randint", "randrange", "choice", "choices", "shuffle", "sample", "uniform", "gauss",
              "normalvariate", "triangular", "betavariate", "expovariate", "getrandbits", "lognormvariate",
              "gammavariate", "vonmisesvariate", "paretovariate", "weibullvariate", "randbytes", "binomialvariate"}
_UNSEEDED_CTORS = {"numpy.random.default_rng", "numpy.random.RandomState", "numpy.random.SeedSequence",
                   "numpy.random.PCG64", "numpy.random.PCG64DXSM", "numpy.random.MT19937", "numpy.random.Philox",
                   "numpy.random.SFC64", "random.Random", "numpy.random.seed", "random.seed"}
_SEED_FUNCS = {"numpy.random.seed", "random.seed", "numpy.random.default_rng", "numpy.random.RandomState",
               "torch.manual_seed", "tensorflow.random.set_seed", "tensorflow.keras.utils.set_random_seed",
               "keras.utils.set_random_seed", "random.Random"}
_SEED_KWS = {"random_state", "seed", "random_seed"}
_RANDOM_PKGS = ("sklearn.", "xgboost.", "lightgbm.", "catboost.", "imblearn.")
_ALWAYS_RANDOM = {
    "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor",
    "ExtraTreeClassifier", "ExtraTreeRegressor", "DecisionTreeClassifier", "DecisionTreeRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor", "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor", "AdaBoostClassifier", "AdaBoostRegressor", "BaggingClassifier",
    "BaggingRegressor", "IsolationForest", "RandomTreesEmbedding", "KMeans", "MiniBatchKMeans", "BisectingKMeans",
    "GaussianMixture", "BayesianGaussianMixture", "MLPClassifier", "MLPRegressor", "SGDClassifier", "SGDRegressor",
    "Perceptron", "PassiveAggressiveClassifier", "TruncatedSVD", "TSNE", "ShuffleSplit", "StratifiedShuffleSplit",
    "GroupShuffleSplit", "RepeatedKFold", "RepeatedStratifiedKFold", "RandomizedSearchCV", "permutation_importance",
    "XGBClassifier", "XGBRegressor", "XGBRanker", "XGBRFClassifier", "XGBRFRegressor", "LGBMClassifier",
    "LGBMRegressor", "LGBMRanker", "CatBoostClassifier", "CatBoostRegressor", "CatBoostRanker",
    "SMOTE", "SMOTENC", "SMOTEN", "ADASYN", "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE", "RandomOverSampler",
    "RandomUnderSampler", "SMOTEENN", "SMOTETomek", "BalancedRandomForestClassifier", "EasyEnsembleClassifier",
    "RUSBoostClassifier", "BalancedBaggingClassifier",
}
_SHUFFLE_SPLITTERS = {"KFold", "StratifiedKFold", "StratifiedGroupKFold", "GroupKFold"}
_LIB_DEFAULT_SEED = ("xgboost.", "lightgbm.", "catboost.")

_PREPROC = {"StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "QuantileTransformer",
            "PowerTransformer", "OneHotEncoder", "OrdinalEncoder", "SimpleImputer", "KNNImputer",
            "IterativeImputer", "KBinsDiscretizer", "PCA", "TruncatedSVD", "FactorAnalysis", "SplineTransformer"}
_TARGET_ENC = {"TargetEncoder", "WOEEncoder", "LeaveOneOutEncoder", "CatBoostEncoder", "JamesSteinEncoder",
               "MEstimateEncoder", "GLMMEncoder"}
_RESAMPLERS = {"SMOTE", "SMOTENC", "SMOTEN", "ADASYN", "BorderlineSMOTE", "SVMSMOTE", "KMeansSMOTE",
               "RandomOverSampler", "RandomUnderSampler", "SMOTEENN", "SMOTETomek", "NearMiss", "TomekLinks"}
_SELECTORS = {"SelectKBest", "SelectPercentile", "SelectFpr", "SelectFdr", "SelectFwe", "GenericUnivariateSelect",
              "RFE", "RFECV", "SelectFromModel", "SequentialFeatureSelector", "BorutaPy"}
_FIT_RULE = {**{c: ("CS201", "preprocessing") for c in _PREPROC},
             **{c: ("CS203", "target encoding") for c in _TARGET_ENC},
             **{c: ("CS205", "resampling") for c in _RESAMPLERS},
             **{c: ("CS206", "feature selection") for c in _SELECTORS}}
_SPLIT_FUNCS = {"train_test_split", "cross_val_score", "cross_validate", "cross_val_predict", "learning_curve",
                "validation_curve", "permutation_test_score"}
_CV_SEARCH = {"GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV"}
_SPLITTERS = {"KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold", "ShuffleSplit",
              "StratifiedShuffleSplit", "GroupShuffleSplit", "RepeatedKFold", "RepeatedStratifiedKFold",
              "TimeSeriesSplit", "LeaveOneOut", "LeavePOut", "LeaveOneGroupOut"}

_HOLDOUT_TOKENS = {"test", "tst", "val", "valid", "validation", "oot", "oos", "holdout", "heldout", "hold"}
_TARGET_TOKENS = {"target", "label", "labels", "y", "default", "dflt", "bad", "outcome", "response", "event",
                  "churn", "fraud", "tgt"}
_SCORE_TOKENS = {"prob", "probs", "proba", "probas", "probability", "probabilities", "score", "scores", "pred",
                 "preds", "prediction", "predictions", "predict", "yhat", "phat", "pd", "logit", "logits"}
_THRESH_TOKENS = {"threshold", "thresh", "cutoff", "cut"}
_SECRET_NAME = re.compile(r"(?:^|_)(?:password|passwd|pwd|secret|token|api_?key|apikey|access_?key|secret_?key|"
                          r"client_?secret|private_?key|auth_?token|sas_?token|account_?key|"
                          r"conn(?:ection)?_?str(?:ing)?)(?:$|_)", re.I)
_PLACEHOLDER = re.compile(r"your|example|xxx|\*\*\*|dummy|placeholder|redacted|changeme|<|>|\{|\}|^\$|^%|"
                          r"^none$|^null$|^true$|^false$|^test$|^secret$|^password$|^token$", re.I)
_SECRET_PATTERNS = [
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----")),
    ("Azure storage account key", re.compile(r"AccountKey=[A-Za-z0-9+/]{30,}={0,2}")),
    ("Azure SAS signature", re.compile(r"[?&]sig=[A-Za-z0-9%+/]{20,}")),
    ("bearer token", re.compile(r"\b[Bb]earer\s+[A-Za-z0-9\-._~+/]{20,}=*")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Databricks personal access token", re.compile(r"\bdapi[0-9a-f]{32}(?:-\d)?\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("OpenAI-style API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("credentials embedded in a URL", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^/\s:@'\"{}$<>]+:[^/\s@'\"{}$<>]{3,}@")),
]
_KV_SECRET = re.compile(r"(?i)(?<![A-Za-z0-9])(?:[A-Za-z0-9]+[_-])*(password|passwd|pwd|secret|token|api[_-]?key|"
                        r"access[_-]?key|secret[_-]?key|client[_-]?secret|private[_-]?key|sas[_-]?token|"
                        r"account[_-]?key)(?![A-Za-z0-9])[\"']?\s*(?:=|:|<-)\s*[\"']?([^\s\"',;#}]+)")
_PATH_PATTERNS = [
    ("Windows drive path", re.compile(r"^[A-Za-z]:[\\/]")),
    ("UNC network path", re.compile(r"^\\\\[^\\\s]+\\")),
    ("DBFS path", re.compile(r"(?:^|[\s'\"(=])(?:dbfs:/|/dbfs/)")),
    ("Unity Catalog volume path", re.compile(r"(?:^|[\s'\"(=])/Volumes/")),
    ("Databricks workspace path", re.compile(r"(?:^|[\s'\"(=])/Workspace/")),
    ("DBFS mount path", re.compile(r"^/mnt/")),
    ("cloud storage URI", re.compile(r"\b(?:s3a?|s3n|abfss?|wasbs?|hdfs|gs|adl)://")),
    ("user home directory path", re.compile(r"^(?:/home/|/Users/)[^/\s]+/")),
]
_DATE_STR = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?|(\d{4})/(\d{2})/(\d{2})|"
                       r"(\d{2})[./](\d{2})[./](\d{4})")
# a string is treated as SQL only when it starts like a statement ("select ... from <table>", not English prose)
_SQL_START = re.compile(
    r"^\s*\(?\s*(?:\{\}\s*)?(?:"
    r"select\b[\s\S]*?\bfrom\s+(?!(?:the|a|an|this|that|these|those|each|every|your|our|its|it|them|which|where|"
    r"one|all|any|there|here)\b)[\w{(\[`\"]"
    r"|with\s+\w+\s+as\s*\("
    r"|insert\s+(?:into|overwrite)\b|update\s+[\w.{}`\"\[\]]+\s+set\b|delete\s+from\b|merge\s+into\b"
    r"|(?:create|drop|alter|truncate)\s+(?:or\s+replace\s+)?(?:temp(?:orary)?\s+)?(?:table|view)\b)", re.I)
_SQL_LIKE = _SQL_START
_TODO = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b")
_TODO_COMMENT = re.compile(r"(?:#|--|//|/\*|^\s*\*).*?\b(TODO|FIXME|HACK|XXX)\b")
_CELL_HDR = re.compile(r"^# %% \[cell (\d+)\] (\w[\w-]*)\s*$")
_PIP_LINE = re.compile(r"^\s*(?:# MAGIC\s+)?[%!]\s*pip3?\s+install\s+(.+)$")
_LOCK_FILES = {"poetry.lock", "uv.lock", "pdm.lock", "pipfile.lock", "conda-lock.yml"}
_IMPORT_TO_DIST = {
    "sklearn": "scikit-learn", "skimage": "scikit-image", "yaml": "pyyaml", "cv2": "opencv-python",
    "PIL": "pillow", "bs4": "beautifulsoup4", "dateutil": "python-dateutil", "dotenv": "python-dotenv",
    "jwt": "pyjwt", "Crypto": "pycryptodome", "OpenSSL": "pyopenssl", "attr": "attrs", "docx": "python-docx",
    "pptx": "python-pptx", "fitz": "pymupdf", "mpl_toolkits": "matplotlib", "pkg_resources": "setuptools",
    "win32api": "pywin32", "win32com": "pywin32", "IPython": "ipython", "umap": "umap-learn",
    "category_encoders": "category-encoders", "imblearn": "imbalanced-learn", "git": "gitpython",
    "serial": "pyserial", "magic": "python-magic", "MySQLdb": "mysqlclient", "delta": "delta-spark",
    "sentence_transformers": "sentence-transformers", "google": "google", "azure": "azure",
}
_RUNTIME_PROVIDED = {"pyspark", "py4j", "dbutils", "__main__"}
_NON_PY_MAGIC = {"%md", "%md-sandbox", "%r", "%scala", "%sh", "%fs", "%%bash", "%%sh", "%%html", "%%markdown",
                 "%%javascript", "%%js", "%%latex", "%%writefile", "%%R", "%%capture"}


def _norm_pkg(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _tokens(name: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name)]


def _valid_date(y: str, m: str, d: str) -> bool:
    try:
        return 1900 <= int(y) <= 2100 and bool(_dt.date(int(y), int(m), int(d)))
    except ValueError:
        return False


def _date_literal(s: str) -> bool:
    m = _DATE_STR.fullmatch(s.strip())
    if not m:
        return False
    g = m.groups()
    if g[0]:
        return _valid_date(g[0], g[1], g[2])
    if g[3]:
        return _valid_date(g[3], g[4], g[5])
    return _valid_date(g[8], g[7], g[6]) or _valid_date(g[8], g[6], g[7])


def _path_kind(s: str) -> str | None:
    for label, rx in _PATH_PATTERNS:
        m = rx.search(s)
        if m and s[m.end():].strip("/\\ \t"):        # a bare prefix such as "/Volumes/" is not a path
            return label
    return None


def _num(node) -> float | int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _num(node.operand)
        if v is not None:
            return -v if isinstance(node.op, ast.USub) else v
    return None


def _last(func) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _const(node, default=None):
    return node.value if isinstance(node, ast.Constant) else default


def _short(node, n: int = 70) -> str:
    try:
        s = ast.unparse(node)
    except Exception:  # pragma: no cover - unparse handles every parsed node
        s = "?"
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def _is_test_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    base = p.rsplit("/", 1)[-1]
    return (base.startswith("test_") or base.endswith("_test.py") or base == "conftest.py"
            or "/tests/" in "/" + p or "/test/" in "/" + p)


def _kind(path: str) -> str:
    p = path.replace("\\", "/").lower()
    base = p.rsplit("/", 1)[-1]
    ext = "." + base.rsplit(".", 1)[-1] if "." in base else ""
    if ext == ".ipynb":
        return "notebook"
    if ext == ".py":
        return "python"
    if ext in {".sql", ".hql"}:
        return "sql"
    if ext in {".r", ".rmd"}:
        return "r"
    if ext == ".sas":
        return "sas"
    if base.startswith("requirements") and ext in {".txt", ".in"}:
        return "requirements"
    if base.startswith("environment") and ext in {".yml", ".yaml"}:
        return "conda"
    if base == "pyproject.toml":
        return "pyproject"
    if ext in {".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".conf", ".env", ".properties"} or base == ".env":
        return "config"
    if ext in {".scala", ".java", ".js", ".ts", ".tsx", ".sh", ".ps1", ".jl", ".go", ".cs", ".m"}:
        return "code"
    return "text"


# --------------------------------------------------------------------------------------------- per-file emitter

class _File:
    def __init__(self, path: str, lines: list[str]):
        self.path, self.lines = path, lines
        self.found: list[tuple] = []
        self.cells: list[_Cell] = []
        self.secret_lines: set[int] = set()

    def snippet(self, line: int, end: int) -> str:
        seg = [ln.strip() for ln in self.lines[line - 1:min(end, line + 2)] if ln.strip()]
        s = " ".join(seg)
        return s if len(s) <= 200 else s[:197] + "..."

    def add(self, rule: str, line: int, message: str, end: int | None = None, snippet: str | None = None):
        line = max(1, min(line, max(len(self.lines), 1)))
        end = max(line, min(end or line, max(len(self.lines), 1)))
        self.found.append((rule, line, end, self.snippet(line, end) if snippet is None else snippet, message))


@dataclass
class _Cell:
    number: int
    kind: str
    header: int
    src_start: int
    src_end: int
    out_start: int | None = None
    exec_count: int | None = None
    error: str | None = None
    has_source: bool = False


def _render_notebook(raw: str) -> tuple[str, list[_Cell], bool] | None:
    """Raw .ipynb JSON -> loaders-style rendering plus cell metadata; None when it is not JSON."""
    try:
        nb = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        return None
    out: list[str] = []
    cells: list[_Cell] = []
    for i, cell in enumerate(nb["cells"], 1):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        kind = cell.get("cell_type", "code")
        out.append(f"# %% [cell {i}] {kind}")
        header = len(out)
        out.extend(src.rstrip().split("\n"))
        c = _Cell(i, kind, header, header + 1, len(out), has_source=bool(src.strip()))
        texts, error = [], None
        for o in cell.get("outputs", []) or []:
            t = o.get("text") or (o.get("data") or {}).get("text/plain")
            if t:
                texts.append("".join(t) if isinstance(t, list) else str(t))
            if o.get("output_type") == "error" and error is None:
                error = f"{o.get('ename', 'Error')}: {o.get('evalue', '')}".strip().rstrip(":")
        if texts:
            joined = "\n".join(texts).strip()
            if len(joined) > 2000:
                joined = joined[:2000] + "\n…(output truncated)"
            c.out_start = len(out) + 1
            out.append("# [output]")
            out.extend("# " + ln for ln in joined.splitlines())
        if kind == "code":
            c.exec_count = cell.get("execution_count")
        c.error = error
        cells.append(c)
    return "\n".join(out), cells, True


def _parse_rendered(lines: list[str]) -> list[_Cell]:
    heads = [(i + 1, int(m[1]), m[2]) for i, ln in enumerate(lines) if (m := _CELL_HDR.match(ln))]
    cells = []
    for k, (h, num, kind) in enumerate(heads):
        nxt = heads[k + 1][0] if k + 1 < len(heads) else len(lines) + 1
        out_start = None
        j = nxt - 1
        while j > h and lines[j - 1].startswith("#"):
            if lines[j - 1] == "# [output]":
                out_start = j
            j -= 1
        src_end = (out_start - 1) if out_start else nxt - 1
        has_src = any(ln.strip() for ln in lines[h:src_end])
        cells.append(_Cell(num, kind, h, h + 1, src_end, out_start, has_source=has_src))
    return cells


# --------------------------------------------------------------------------------------------- SQL / regex helpers

def _strip_sql_comments(text: str) -> str:
    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))
    text = re.sub(r"/\*[\s\S]*?\*/", blank, text)
    return re.sub(r"--[^\n]*", blank, text)


def _sql_checks(f: _File, text: str, base_line: int, last_line: int | None = None, where: str = ""):
    clean = _strip_sql_comments(text)

    def ln(pos: int) -> int:
        line = base_line + clean.count("\n", 0, pos)
        return min(line, last_line) if last_line else line

    pos = 0
    for stmt in clean.split(";"):
        start = pos
        pos += len(stmt) + 1
        if not stmt.strip():
            continue
        for m in re.finditer(r"\bselect\s+(?:distinct\s+|top\s+\d+\s+)?(?:\w+\.)?\*", stmt, re.I):
            f.add("CS701", ln(start + m.start()),
                  f"SELECT * {where}pulls every column of the source table: the extracted data (and any feature "
                  "list derived from it) silently changes when the table schema changes; list the columns used.")
        head = stmt.lstrip()
        m = re.match(r"(delete\s+from|update)\s+([\w.\[\]`\"]+)", head, re.I)
        if m and not re.search(r"\bwhere\b", stmt, re.I) and (m.group(1).lower() != "update"
                                                                or re.search(r"\bset\b", stmt, re.I)):
            verb = m.group(1).split()[0].upper()
            f.add("CS702", ln(start + len(stmt) - len(head)),
                  f"{verb} on {m.group(2)} {where}has no WHERE clause and affects every row of the table.")
        w = re.search(r"\b(where|having)\b", stmt, re.I)
        if w:
            for d in re.finditer(r"'(\d{4}-\d{2}-\d{2}|\d{2}[./]\d{2}[./]\d{4}|\d{4}/\d{2}/\d{2})"
                                 r"(?:[ T][\d:.]+)?'", stmt[w.start():]):
                if _date_literal(d.group(1)):
                    f.add("CS703", ln(start + w.start() + d.start()),
                          f"date {d.group(0)} is hard-coded in a {w.group(1).upper()} condition {where}- the sample "
                          "window is fixed inside the query; pass it as a documented parameter.")


def _quoted_strings(line: str):
    for m in re.finditer(r"'([^'\n]*)'|\"([^\"\n]*)\"", line):
        yield m.group(1) if m.group(1) is not None else m.group(2)


def _strip_hash_comment(line: str) -> str:
    q = None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
        elif ch == "#":
            return line[:i]
    return line


def _string_rules(f: _File, line: int, s: str, dates: bool = True):
    kind = _path_kind(s)
    if kind:
        f.add("CS103", line, f"{kind} '{s[:80]}' is hard-coded: the code only runs in one environment and the "
                             "data actually used cannot be traced from configuration; take it from a parameter "
                             "or config file.")
    if dates and _date_literal(s) and not s.startswith("1970-01-01"):     # the Unix epoch is not a sample date
        f.add("CS104", line, f"date '{s}' is hard-coded: sample windows / reference dates should be explicit, "
                             "documented parameters so the validator can confirm the period used.")


def _secret_scan(f: _File, kv: bool):
    for i, line in enumerate(f.lines, 1):
        hit = False
        for label, rx in _SECRET_PATTERNS:
            m = rx.search(line)
            if m:
                f.add("CS401", i, f"{label} found in plain text; rotate it and load it from a secret store "
                                  "(e.g. dbutils.secrets / Key Vault / environment variable).",
                      snippet=rx.sub("<redacted>", line.strip())[:200])
                f.secret_lines.add(i)
                hit = True
                break
        if hit or not kv or i in f.secret_lines:
            continue
        m = _KV_SECRET.search(line)
        if m and len(m.group(2)) >= 4 and not _PLACEHOLDER.search(m.group(2)) and not m.group(2).isdigit() \
                and not re.fullmatch(r"[A-Z0-9_]+", m.group(2)) and not m.group(2).startswith(("os.", "env", "!")):
            f.add("CS401", i, f"'{m.group(1)}' is set to a literal value; secrets must come from a secret store, "
                              "never from code or committed config.",
                  snippet=line.strip().replace(m.group(2), "<redacted>")[:200])
            f.secret_lines.add(i)


def _todo_regex(f: _File):
    for i, line in enumerate(f.lines, 1):
        m = _TODO_COMMENT.search(line)
        if m:
            f.add("CS801", i, f"{m.group(1)} comment left in code: '{line.strip()[:100]}': confirm the open "
                              "item does not affect the validated model.")


# --------------------------------------------------------------------------------------------- requirements

@dataclass
class _Req:
    file: str
    line: int
    name: str
    display: str
    spec: str
    version: str | None


def _parse_req_token(s: str) -> tuple[str, str, str | None] | None:
    """'pandas>=2.0' -> (name, spec, exact version or None). URLs / paths / options -> None."""
    s = s.split(" #", 1)[0].strip()
    if not s or s.startswith(("#", "-")):
        return None
    s = s.split(";", 1)[0].strip()
    if "@" in s and "://" in s:
        name = s.split("@", 1)[0].strip()
        pinned = bool(re.search(r"\.git@[\w.\-]+", s) or re.search(r"\.(whl|tar\.gz|zip)\b", s))
        return (name, s.split("@", 1)[1].strip(), "url" if pinned else None) if name else None
    if s.startswith(("git+", "http", ".", "/")) or re.match(r"^[A-Za-z]:\\", s):
        return None
    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$", s)
    if not m:
        return None
    spec = m.group(3).strip()
    v = re.fullmatch(r"===?\s*([^,*\s]+)", spec)
    return m.group(1), spec, v.group(1) if v else None


def _requirements(f: _File) -> list[_Req]:
    reqs = []
    for i, line in enumerate(f.lines, 1):
        p = _parse_req_token(line)
        if p:
            reqs.append(_Req(f.path, i, _norm_pkg(p[0]), p[0], p[1], p[2]))
    return reqs


def _conda(f: _File) -> list[_Req]:
    reqs, section, in_pip, pip_indent = [], None, False, 0
    for i, line in enumerate(f.lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            section, in_pip = line.split(":", 1)[0].strip(), False
            continue
        if section != "dependencies":
            continue
        m = re.match(r"^\s*-\s*(.+?)\s*$", line)
        if not m:
            continue
        item = m.group(1).split(" #", 1)[0].strip().strip("'\"")
        if item.rstrip(":") == "pip" and item.endswith(":"):
            in_pip, pip_indent = True, indent
            continue
        if in_pip and indent > pip_indent:
            p = _parse_req_token(item)
            if p:
                reqs.append(_Req(f.path, i, _norm_pkg(p[0]), p[0], p[1], p[2]))
            continue
        in_pip = False
        if item.startswith("{"):
            continue
        cm = re.match(r"^(?:[\w.-]+::)?([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*)$", item)
        if not cm or cm.group(1).lower() == "pip":
            continue
        spec = cm.group(2).strip()
        v = re.fullmatch(r"==?\s*([0-9][^=,*\s]*)(?:=\S+)?", spec)
        reqs.append(_Req(f.path, i, _norm_pkg(cm.group(1)), cm.group(1), spec, v.group(1) if v else None))
    return reqs


def _pyproject(f: _File, text: str) -> list[_Req]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []

    def line_of(needle: str, key: bool = False) -> int:
        rx = re.compile(r"^\s*[\"']?" + re.escape(needle) + r"[\"']?\s*=") if key else None
        for i, ln in enumerate(f.lines, 1):
            if (rx.search(ln) if key else (f'"{needle}"' in ln or f"'{needle}'" in ln)):
                return i
        return 1

    reqs = []
    proj = data.get("project", {}) or {}
    items = list(proj.get("dependencies", []) or [])
    for extra in sorted((proj.get("optional-dependencies", {}) or {}).items()):
        items.extend(extra[1])
    for it in items:
        p = _parse_req_token(str(it))
        if p:
            reqs.append(_Req(f.path, line_of(str(it)), _norm_pkg(p[0]), p[0], p[1], p[2]))
    poetry = ((data.get("tool", {}) or {}).get("poetry", {}) or {}).get("dependencies", {}) or {}
    for name, spec in poetry.items():
        if name.lower() == "python":
            continue
        s = spec.get("version", "") if isinstance(spec, dict) else str(spec)
        v = re.fullmatch(r"(?:==?)?\s*(\d[\w.]*)", s.strip())
        reqs.append(_Req(f.path, line_of(name, key=True), _norm_pkg(name), name, s, v.group(1) if v else None))
    return reqs


def _pip_lines(f: _File, line_numbers) -> list[_Req]:
    reqs = []
    for i in line_numbers:
        m = _PIP_LINE.match(f.lines[i - 1])
        if not m:
            continue
        toks, skip = m.group(1).split(), False
        for t in toks:
            if skip:
                skip = False
                continue
            if t in {"-r", "-c", "-e", "-i", "--index-url", "--extra-index-url", "--target", "-t", "--find-links",
                     "-f"}:
                skip = True
                continue
            if t.startswith("-") or "$" in t or "{" in t or t.endswith((".txt", ".whl")):
                continue
            p = _parse_req_token(t.strip("'\""))
            if p:
                reqs.append(_Req(f.path, i, _norm_pkg(p[0]), p[0], p[1], p[2]))
    return reqs


# --------------------------------------------------------------------------------------------- Python analysis

class _Py:
    def __init__(self, repo: _Repo, f: _File, tree: ast.Module):
        self.repo, self.f, self.tree = repo, f, tree
        self.nodes = list(ast.walk(tree))          # one traversal, reused by every check
        self.parent: dict[int, ast.AST] = {}
        for n in self.nodes:
            for c in ast.iter_child_nodes(n):
                self.parent[id(c)] = n
        self.aliases: dict[str, str] = {}
        self.star: set[str] = set()
        self.imports: list[tuple[int, str, ast.AST]] = []
        for n in self.nodes:
            if isinstance(n, ast.Import):
                for a in n.names:
                    self.aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                    self.imports.append((n.lineno, a.name.split(".")[0], n))
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                self.imports.append((n.lineno, n.module.split(".")[0], n))
                for a in n.names:
                    if a.name == "*":
                        self.star.add(n.module)
                    else:
                        self.aliases[a.asname or a.name] = f"{n.module}.{a.name}"
        self.funcs = sorted((n for n in self.nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))),
                            key=lambda n: (n.lineno, n.col_offset))
        self.defined = {n.name for n in self.nodes
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        self.docstrings: set[int] = set()
        for n in self.nodes:
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
                b = n.body[0]
                if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str):
                    self.docstrings.add(id(b.value))
        self.instances: dict[str, set[str]] = {}
        for n in self.nodes:
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and _last(n.value.func):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        self.instances.setdefault(t.id, set()).add(_last(n.value.func))
        self.calls = sorted((n for n in self.nodes if isinstance(n, ast.Call)),
                            key=lambda n: (n.lineno, n.col_offset))
        self.is_test = _is_test_path(f.path)
        self.uses_pandas = any(v.split(".")[0] in {"pandas", "pyspark"} for v in self.aliases.values())
        self.cut_compares: set[int] = set()

    # ---- helpers
    def qual(self, node) -> str | None:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name) and node.id in self.aliases:
            return ".".join([self.aliases[node.id], *reversed(parts)])
        return None

    def add(self, rule: str, node, message: str):
        line = node.lineno
        end = getattr(node, "end_lineno", line) or line
        self.f.add(rule, line, message, min(end, line + 2))

    def scope_of(self, node) -> int:
        p = self.parent.get(id(node))
        while p is not None:
            if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return id(p)
            p = self.parent.get(id(p))
        return 0

    def in_function(self, node) -> bool:
        return self.scope_of(node) != 0

    def stmt_of(self, node):
        while node is not None and not isinstance(node, ast.stmt):
            node = self.parent.get(id(node))
        return node

    def names(self, node) -> set[str]:
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id not in self.aliases:
                p = self.parent.get(id(n))
                if isinstance(p, ast.Call) and p.func is n:
                    continue
                out.add(n.id)
        return out

    @staticmethod
    def target_names(stmt) -> set[str]:
        out = set()
        stack = list(stmt.targets) if isinstance(stmt, ast.Assign) else \
            [stmt.target] if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) else []
        while stack:
            t = stack.pop()
            if isinstance(t, (ast.Tuple, ast.List)):
                stack.extend(t.elts)
            elif isinstance(t, ast.Starred):
                stack.append(t.value)
            else:
                r = _Py.root(t)
                if r:
                    out.add(r)
        return out

    @staticmethod
    def root(node) -> str | None:
        while True:
            if isinstance(node, (ast.Subscript, ast.Attribute)):
                node = node.value
            elif isinstance(node, ast.Call):
                node = node.func
            elif isinstance(node, ast.Name):
                return node.id
            else:
                return None

    def receiver_classes(self, call: ast.Call) -> set[str]:
        if not isinstance(call.func, ast.Attribute):
            return set()
        recv = call.func.value
        if isinstance(recv, ast.Call):
            return {_last(recv.func)} - {None}
        if isinstance(recv, ast.Name):
            return self.instances.get(recv.id, set())
        return set()

    def is_random_lib(self, call: ast.Call) -> str | None:
        q = self.qual(call.func)
        if q and q.startswith(_RANDOM_PKGS):
            return q
        if q is None and isinstance(call.func, ast.Name) and any(s.startswith(_RANDOM_PKGS) or s.split(".")[0] in
                                                                  {"sklearn", "xgboost", "lightgbm", "catboost",
                                                                   "imblearn"} for s in self.star):
            return call.func.id
        return None

    # ---- run all checks
    def run(self):
        self.reproducibility()
        self.hard_coding()
        self.strings()
        self.leakage()
        self.robustness()
        self.security()
        self.maintainability()
        for line, top, node in self.imports:
            p, in_try = self.parent.get(id(node)), False
            while p is not None:
                if isinstance(p, ast.Try) or (isinstance(p, ast.If) and "TYPE_CHECKING" in _short(p.test)):
                    in_try = True
                p = self.parent.get(id(p))
            self.repo.imports.append((self.f.path, line, top, in_try))

    # ---- reproducibility
    def reproducibility(self):
        np_seeded = py_seeded = False
        for c in self.calls:
            q = self.qual(c.func)
            if q in {"numpy.random.seed", "random.seed"} and (c.args or c.keywords) and \
                    not (c.args and _const(c.args[0], 0) is None and isinstance(c.args[0], ast.Constant)):
                if q == "random.seed":
                    py_seeded = True
                else:
                    np_seeded = True
        torch_seed = tf_seed = False
        for c in self.calls:
            q = self.qual(c.func) or ""
            last = _last(c.func)
            if q.startswith(("torch.manual_seed", "torch.cuda.manual_seed")) or last == "seed_everything" or \
                    q in {"transformers.set_seed"}:
                torch_seed = True
            if q in {"tensorflow.random.set_seed", "tensorflow.set_random_seed", "tensorflow.keras.utils."
                     "set_random_seed", "keras.utils.set_random_seed"} or last == "seed_everything":
                tf_seed = True

        for c in self.calls:
            q = self.qual(c.func)
            last = _last(c.func)
            # CS004: time/entropy seeds
            seed_exprs = [k.value for k in c.keywords if k.arg in _SEED_KWS]
            if q in _SEED_FUNCS and c.args:
                seed_exprs.append(c.args[0])
            for e in seed_exprs:
                src = self.entropy_source(e)
                if src:
                    self.add("CS004", c, f"seed is derived from {src}() in `{_short(c)}`: every run uses a "
                                         "different seed, so results cannot be reproduced; use a fixed, "
                                         "documented integer seed.")
                    break
            if q is None:
                pass
            elif q.startswith("numpy.random.") and q.count(".") == 2 and last in _NP_LEGACY and not np_seeded:
                self.add("CS001", c, f"`{_short(c.func)}()` draws from NumPy's global random state and no "
                                     "np.random.seed(...) is set in this module: results change on every run. Use "
                                     "rng = np.random.default_rng(<seed>) and draw from rng.")
            elif q.startswith("random.") and q.count(".") == 1 and last in _PY_RANDOM and not py_seeded:
                self.add("CS001", c, f"`{_short(c.func)}()` uses Python's global random state and random.seed(...) "
                                     "is never called in this module: results change on every run.")
            if q in _UNSEEDED_CTORS:
                arg = c.args[0] if c.args else _kw(c, "seed")
                if arg is None or (isinstance(arg, ast.Constant) and arg.value is None):
                    what = "seeds the global generator from OS entropy" if q.endswith(".seed") else \
                        "creates a generator seeded from OS entropy"
                    self.add("CS002", c, f"`{_short(c)}` {what}: draws differ on every run. Pass a fixed integer "
                                         "seed.")
            self.check_random_state(c, np_seeded)

        if not torch_seed:
            first = next((ln for ln, top, _ in self.imports if top == "torch"), None)
            if first:
                self.f.add("CS005", first, "torch is imported but torch.manual_seed(...) is never called: weight "
                                           "initialisation, dropout and data-loader shuffling are not reproducible.")
        if not tf_seed:
            first = next((ln for ln, top, _ in self.imports if top in {"tensorflow", "keras"}), None)
            if first:
                self.f.add("CS005", first, "tensorflow/keras is imported but tf.random.set_seed(...) / "
                                           "keras.utils.set_random_seed(...) is never called: training is not "
                                           "reproducible.")
        self.set_order()

    def entropy_source(self, e) -> str | None:
        for n in ast.walk(e):
            if isinstance(n, ast.Call):
                q = self.qual(n.func) or ""
                last = _last(n.func)
                if last in {"time", "time_ns", "perf_counter", "monotonic", "now", "utcnow", "today", "getpid",
                            "uuid4", "urandom", "randbits", "token_hex"} and \
                        q.split(".")[0] in {"time", "datetime", "pandas", "os", "uuid", "secrets", "numpy"}:
                    return q or last
        return None

    def check_random_state(self, c: ast.Call, np_seeded: bool):
        last = _last(c.func)
        if any(k.arg is None for k in c.keywords):
            return
        seed = next((k for k in c.keywords if k.arg in _SEED_KWS), None)
        # DataFrame.sample(frac=..)/(n=..) (pandas) or sample(fraction=..) (Spark) without a seed
        if last == "sample" and isinstance(c.func, ast.Attribute) and self.qual(c.func) is None and \
                any(_kw(c, k) is not None for k in ("frac", "n", "fraction")):
            if seed is None and not np_seeded:
                self.add("CS003", c, f"`{_short(c)}` samples rows without random_state: a different subset is "
                                     "drawn on every run. Add random_state=<seed>.")
            elif seed is not None and _const(seed.value, 0) is None and isinstance(seed.value, ast.Constant):
                self.add("CS003", c, f"`{_short(c)}` passes random_state=None: a different subset is drawn on "
                                     "every run.")
            return
        lib = self.is_random_lib(c)
        if not lib or last is None:
            return
        if last in _SHUFFLE_SPLITTERS:
            if _const(_kw(c, "shuffle")) is not True:
                return
        elif last == "train_test_split":
            if _const(_kw(c, "shuffle")) is False:
                return
        elif last == "LogisticRegression":
            if _const(_kw(c, "solver")) not in {"sag", "saga", "liblinear"}:
                return
        elif last in {"SVC", "NuSVC"}:
            if _const(_kw(c, "probability")) is not True:
                return
        elif last == "PCA":
            if _const(_kw(c, "svd_solver")) not in {"randomized", "arpack"}:
                return
        elif last not in _ALWAYS_RANDOM:
            return
        if seed is not None and not (isinstance(seed.value, ast.Constant) and seed.value.value is None):
            return
        how = "with random_state=None" if seed is not None else "without random_state"
        if lib.startswith(_LIB_DEFAULT_SEED):
            why = ("it falls back to the library's implicit default seed, which is not part of the documented "
                   "model configuration and can change between versions")
        elif last == "train_test_split":
            why = "the train/test partition (and every metric computed on it) changes on every run"
        elif last in _SHUFFLE_SPLITTERS or last.endswith(("Split", "KFold")):
            why = "the fold assignment changes on every run"
        else:
            why = "fitted parameters and results change on every run"
        self.add("CS003", c, f"{last} is called {how}: {why}. Pass random_state=<fixed seed>.")

    def set_valued(self, node) -> bool:
        if isinstance(node, (ast.Set, ast.SetComp)):
            return not all(_num(e) is not None for e in getattr(node, "elts", [])) or isinstance(node, ast.SetComp)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"set", "frozenset"} \
                and node.func.id not in self.defined:
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in \
                {"union", "intersection", "difference", "symmetric_difference"}:
            return self.set_valued(node.func.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitOr, ast.BitAnd, ast.Sub, ast.BitXor)):
            return self.set_valued(node.left) or self.set_valued(node.right)
        return False

    def set_order(self):
        for n in self.nodes:
            it = None
            # only constructs whose output keeps the iteration order (lists, appends, yields) are order-sensitive
            if isinstance(n, ast.For) and any(
                    isinstance(x, (ast.Yield, ast.AugAssign)) or
                    (isinstance(x, ast.Call) and _last(x.func) in {"append", "extend", "insert"})
                    for s in n.body for x in ast.walk(s)):
                it, what = n.iter, "loop iterates"
            elif isinstance(n, ast.comprehension) and isinstance(self.parent.get(id(n)), ast.ListComp):
                it, what = n.iter, "list comprehension iterates"
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"list", "tuple"} \
                    and len(n.args) == 1:
                it, what = n.args[0], f"{n.func.id}() takes its order"
            if it is not None and self.set_valued(it):
                node = n if not isinstance(n, ast.comprehension) else n.iter
                self.add("CS006", node, f"{what} from the set `{_short(it, 50)}`: the order of string elements "
                                        "depends on PYTHONHASHSEED, so column/feature order can differ between "
                                        "runs; use sorted(...).")

    # ---- hard-coding
    def score_like(self, node) -> str | None:
        for n in ast.walk(node):
            ident = None
            if isinstance(n, ast.Name) and n.id not in self.aliases:
                ident = n.id
            elif isinstance(n, ast.Attribute):
                ident = n.attr
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and isinstance(
                    self.parent.get(id(n)), ast.Subscript):
                ident = n.value
            if ident and ident != "pd" and _SCORE_TOKENS & set(_tokens(ident)):
                return ident
        return None

    def hard_coding(self):
        for n in self.nodes:
            if isinstance(n, ast.Compare):
                operands = [n.left, *n.comparators]
                for op, a, b in zip(n.ops, operands, operands[1:]):
                    if not isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                        continue
                    for lit, other in ((b, a), (a, b)):
                        v = _num(lit)
                        if v is None or v in (0, 1):
                            continue
                        ident = self.score_like(other)
                        if ident:
                            self.cut_compares.add(id(n))
                            self.add("CS101", n, f"cut-off {v} is hard-coded in `{_short(n)}` (applied to "
                                                 f"{ident}): the decision threshold is a model parameter that must "
                                                 "be documented, justified and configurable, not buried in code.")
                        break
                # size guards (len(x) < 10, df.shape[0] > 50) and float tolerances are not modelling choices
                guard = any(isinstance(o, ast.Call) and _last(o.func) in {"len", "nunique", "count"} or
                            (isinstance(o, (ast.Attribute, ast.Subscript)) and "shape" in _short(o)) or
                            (isinstance(o, ast.Attribute) and o.attr in {"size", "st_size"}) for o in operands)
                if id(n) not in self.cut_compares and not self.is_test and not guard:
                    for lit in operands:
                        v = _num(lit)
                        if v is not None and v not in (0, 1, -1, 2, 100) and abs(v) >= 1e-4:
                            self.add("CS102", n, f"magic number {v} in `{_short(n)}`: name it and document why "
                                                 "this value was chosen.")
                            break
            elif isinstance(n, ast.Call):
                for k in n.keywords:
                    v = _num(k.value)
                    lib = (self.qual(n.func) or "").split(".")[0]
                    if k.arg and v is not None and 0 < v < 1 and _THRESH_TOKENS & set(_tokens(k.arg)) and \
                            lib not in sys.stdlib_module_names and lib not in {"numpy", "pandas", "scipy"}:
                        self.add("CS101", k.value, f"{k.arg}={v} is hard-coded in the call to "
                                                   f"{_last(n.func)}(): document and parameterise this cut-off.")
                if self.qual(n.func) in {"datetime.datetime", "datetime.date"} and len(n.args) >= 2 and \
                        all(isinstance(_num(a), int) for a in n.args[:2]) and 1900 <= _num(n.args[0]) <= 2100:
                    self.add("CS104", n, f"date `{_short(n)}` is hard-coded: sample windows / reference dates "
                                         "should be explicit, documented parameters.")
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                v = _num(n.value) if n.value is not None else None
                if v is None or len(targets) != 1:
                    continue
                t = targets[0]
                ident = t.id if isinstance(t, ast.Name) else t.attr if isinstance(t, ast.Attribute) else None
                if ident and v not in (0, 1) and _THRESH_TOKENS & set(_tokens(ident)):
                    self.add("CS101", n, f"cut-off {ident} = {v} is a hard-coded literal: document its derivation "
                                         "and take it from the model configuration.")
                elif ident and isinstance(t, ast.Name) and self.in_function(n) and not ident.isupper() and \
                        v not in (0, 1, -1, 2, 100) and not self.is_test:
                    self.add("CS102", n, f"magic number {v} assigned to {ident} inside a function: name it as a "
                                         "documented constant or parameter.")
            elif isinstance(n, ast.List) and len(n.elts) >= 3 and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value.strip()
                    and not any(ch.isspace() for ch in e.value) for e in n.elts):
                self.repo.lists.append((self.f.path, n.lineno, tuple(sorted({e.value for e in n.elts}))))

    def strings(self):
        for n in self.nodes:
            p = self.parent.get(id(n))
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                if isinstance(p, ast.JoinedStr) or id(n) in self.docstrings:
                    continue
                text = n.value
                # a configurable default (os.getenv("X", "2024-01-31"), cfg.get(...)) is not a hard-coded date
                configurable = isinstance(p, ast.Call) and _last(p.func) in {"getenv", "get", "setdefault"}
                if "\n" in text:
                    _string_rules(self.f, n.lineno, text.strip(), dates=False)
                else:
                    _string_rules(self.f, n.lineno, text, dates=not configurable)
            elif isinstance(n, ast.JoinedStr) and not isinstance(p, ast.FormattedValue):
                text = "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in n.values)
                _string_rules(self.f, n.lineno, text, dates=False)
            else:
                continue
            if _SQL_LIKE.search(text):
                _sql_checks(self.f, text, n.lineno, n.end_lineno or n.lineno, where="in this embedded query ")

    # ---- leakage
    def leakage(self):
        by_scope: dict[int, list[ast.Call]] = {}
        for c in self.calls:
            by_scope.setdefault(self.scope_of(c), []).append(c)
        stmts_by_scope: dict[int, list] = {}
        for n in self.nodes:
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                stmts_by_scope.setdefault(self.scope_of(n), []).append(n)
        for sid, calls in by_scope.items():
            stmts = sorted(stmts_by_scope.get(sid, []), key=lambda s: (s.lineno, s.col_offset))
            splits, split_outputs = [], set()
            for c in calls:
                last = _last(c.func)
                label = None
                if last in _SPLIT_FUNCS:
                    label = last
                elif isinstance(c.func, ast.Attribute):
                    cls = self.receiver_classes(c)
                    if last == "fit" and cls & _CV_SEARCH:
                        label = f"{sorted(cls & _CV_SEARCH)[0]}.fit"
                    elif last == "split" and cls & _SPLITTERS:
                        label = f"{sorted(cls & _SPLITTERS)[0]}.split"
                if label:
                    splits.append((c.lineno, label, self.names(c)))
                    if last == "train_test_split":
                        st = self.stmt_of(c)
                        if st is not None:
                            split_outputs |= self.target_names(st)
            for c in calls:
                if not (isinstance(c.func, ast.Attribute) and c.func.attr in
                        {"fit", "fit_transform", "fit_resample", "fit_sample"}):
                    continue
                classes = sorted(self.receiver_classes(c) & set(_FIT_RULE))
                data = c.args[0] if c.args else _kw(c, "X")
                if not classes or data is None:
                    continue
                roots = self.names(data)
                if roots & split_outputs or any(_HOLDOUT_TOKENS & set(_tokens(r)) for r in roots):
                    continue
                st = self.stmt_of(c)
                tainted = roots | (self.target_names(st) if st is not None else set())
                for s in stmts:
                    if s.lineno > c.lineno and s.value is not None and self.names(s.value) & tainted:
                        tainted |= self.target_names(s)
                hit = next((sp for sp in sorted(splits) if sp[0] > c.lineno and sp[2] & tainted), None)
                if hit:
                    rule, what = _FIT_RULE[classes[0]]
                    self.add(rule, c, f"{classes[0]} is fitted on `{_short(data, 50)}` (line {c.lineno}) before "
                                      f"the data is split / cross-validated by {hit[1]} (line {hit[0]}): the "
                                      f"{what} step learns from rows that later serve as hold-out data, so "
                                      "performance is overstated. Fit it on the training fold only (e.g. inside "
                                      "a Pipeline).")
            self.target_mean(calls, splits)
        self.holdout_fits()
        self.target_in_features()
        self.lookahead()

    def target_mean(self, calls, splits):
        for c in calls:
            if not isinstance(c.func, ast.Attribute):
                continue
            is_mean = c.func.attr == "mean" or (c.func.attr == "transform" and c.args and
                                                (_const(c.args[0]) == "mean" or _last(c.args[0]) == "mean"))
            sub = c.func.value
            if not (is_mean and isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Call)
                    and _last(sub.value.func) == "groupby"):
                continue
            col = _const(sub.slice)
            if not isinstance(col, str) or not (_TARGET_TOKENS & set(_tokens(col))):
                continue
            if any(sp[0] < c.lineno for sp in splits):
                continue
            self.add("CS203", c, f"mean of target column '{col}' per category is computed in `{_short(c, 60)}` "
                                 "before any train/test split in this scope (heuristic): each row's encoding "
                                 "includes its own and the hold-out outcomes. Compute it on the training data only, "
                                 "out-of-fold.")

    def holdout_fits(self):
        for c in self.calls:
            if not (isinstance(c.func, ast.Attribute) and c.func.attr in
                    {"fit", "fit_transform", "fit_resample", "fit_predict", "fit_sample"}):
                continue
            data = c.args[0] if c.args else _kw(c, "X")
            r = self.root(data) if data is not None else None
            if not r or r in self.aliases or not (_HOLDOUT_TOKENS & set(_tokens(r))):
                continue
            recv = self.root(c.func.value) or ""
            classes = self.receiver_classes(c)
            if {"calib", "calibrator", "cal", "calibration"} & set(_tokens(recv)) or \
                    classes & {"CalibratedClassifierCV", "IsotonicRegression"}:
                continue
            what = f"{sorted(classes)[0]} " if classes else f"`{recv}` " if recv else ""
            self.add("CS202", c, f"{what}is fitted with .{c.func.attr}() on `{_short(data, 40)}`, which is "
                                 "hold-out (test/validation/OOT) data: parameters estimated on it make every metric "
                                 "computed on that sample optimistic. Fit on training data, then only "
                                 "transform/predict the hold-out sample.")

    def target_in_features(self):
        assigns: dict[int, list[ast.Assign]] = {}
        for n in self.nodes:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                assigns.setdefault(self.scope_of(n), []).append(n)
        for sid in sorted(assigns):
            ys: dict[str, set[str]] = {}
            for a in assigns[sid]:
                toks = set(_tokens(a.targets[0].id))
                v = a.value
                if toks & {"y", "target", "label", "labels", "tgt"}:
                    if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name) and \
                            isinstance(_const(v.slice), str):
                        ys.setdefault(v.value.id, set()).add(v.slice.value)
                    elif isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and \
                            v.value.id not in self.aliases:
                        ys.setdefault(v.value.id, set()).add(v.attr)
            if not ys:
                continue
            for a in sorted(assigns[sid], key=lambda s: s.lineno):
                if not ({"x", "features", "feature", "feats", "predictors", "inputs"} & set(_tokens(a.targets[0].id))):
                    continue
                v, frame, kept = a.value, None, None
                if isinstance(v, ast.Name):
                    frame = v.id
                elif isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and \
                        isinstance(v.func.value, ast.Name) and v.func.attr in {"copy", "drop"}:
                    frame = v.func.value.id
                    if v.func.attr == "drop":
                        consts, unknown = set(), False
                        for e in [*v.args, *[k.value for k in v.keywords if k.arg in {"columns", "labels"}]]:
                            elts = e.elts if isinstance(e, (ast.List, ast.Tuple)) else [e]
                            for x in elts:
                                if isinstance(x, ast.Constant) and isinstance(x.value, str):
                                    consts.add(x.value)
                                else:
                                    unknown = True
                        if unknown or frame not in ys or ys[frame] & consts:
                            continue
                elif isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name) and isinstance(v.slice, ast.List):
                    frame = v.value.id
                    kept = {_const(e) for e in v.slice.elts}
                if frame not in ys:
                    continue
                cols = sorted(ys[frame]) if kept is None else sorted(ys[frame] & kept)
                if not cols:
                    continue
                self.add("CS204", a, f"{a.targets[0].id} is taken from `{frame}` without removing the target "
                                     f"column {', '.join(repr(c) for c in cols)} that y is read from (heuristic): "
                                     "the label would be a feature. Drop it explicitly.")

    def lookahead(self):
        for c in self.calls:
            if not isinstance(c.func, ast.Attribute):
                continue
            if c.func.attr == "shift":
                per = c.args[0] if c.args else _kw(c, "periods")
                v = _num(per) if per is not None else None
                if v is None or v >= 0:
                    continue
                st = self.stmt_of(c)
                tnames = set()
                if isinstance(st, (ast.Assign, ast.AnnAssign)):
                    for t in (st.targets if isinstance(st, ast.Assign) else [st.target]):
                        if isinstance(t, ast.Name):
                            tnames.add(t.id)
                        elif isinstance(t, ast.Subscript) and isinstance(_const(t.slice), str):
                            tnames.add(t.slice.value)
                if any(_TARGET_TOKENS & set(_tokens(t)) for t in tnames):
                    continue
                self.add("CS207", c, f"`{_short(c, 60)}` shifts by {v}, pulling values from later periods into "
                                     "this row: if it feeds a feature (not the target) it is look-ahead leakage.")
            elif c.func.attr == "rolling" and _const(_kw(c, "center")) is True:
                self.add("CS207", c, f"`{_short(c, 60)}` uses a centred window, so each value averages future "
                                     "observations: look-ahead leakage if used as a feature.")

    # ---- robustness
    def robustness(self):
        for n in self.nodes:
            if isinstance(n, ast.ExceptHandler):
                silent = all(isinstance(s, (ast.Pass, ast.Continue, ast.Break)) or
                             (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)) or
                             (isinstance(s, ast.Return) and (s.value is None or _const(s.value, 0) is None))
                             for s in n.body)
                body = _short(n.body[0], 20) if n.body else ""
                if n.type is None:
                    extra = f" and the body (`{body}`) discards the error silently" if silent else ""
                    self.f.add("CS301", n.lineno, f"bare `except:` catches every exception including "
                                                  f"KeyboardInterrupt/SystemExit{extra}; catch the specific "
                                                  "exception and log or re-raise it.")
                elif silent:
                    names = {_last(e) for e in (n.type.elts if isinstance(n.type, ast.Tuple) else [n.type])}
                    if names & {"Exception", "BaseException"}:
                        self.f.add("CS302", n.lineno, f"`except {_short(n.type)}` with body `{body}` swallows "
                                                      "every error: failed steps (e.g. a feature not computed) go "
                                                      "unnoticed and the output is silently incomplete.")
            elif isinstance(n, ast.Call):
                self.robust_call(n)
            elif isinstance(n, (ast.Assign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    if self.uses_pandas and self.chained(t):
                        self.add("CS305", n, f"chained assignment `{_short(t, 50)} = ...`: the write goes to a "
                                             "temporary copy (pandas copy-on-write), so the original frame is "
                                             "unchanged. Use a single .loc[row_selector, column] assignment.")
                if isinstance(n, ast.Assign) and _const(n.value, 0) is None and isinstance(n.value, ast.Constant) \
                        and any(_short(t).endswith("mode.chained_assignment") for t in targets):
                    self.add("CS303", n, "pandas chained-assignment warnings are switched off: writes that do "
                                         "not reach the original frame will no longer be reported.")
            elif isinstance(n, ast.Assert) and not self.is_test:
                self.add("CS306", n, f"`assert {_short(n.test, 50)}` is used as a runtime check; asserts are "
                                     "stripped under python -O, so the check silently disappears. Raise an "
                                     "explicit exception instead.")

    def chained(self, t) -> bool:
        if not isinstance(t, ast.Subscript):
            return False
        inner = t.value
        if isinstance(inner, ast.Attribute) and inner.attr in {"loc", "iloc", "at", "iat"} and \
                isinstance(inner.value, ast.Subscript):
            return True
        if isinstance(inner, ast.Subscript):
            sl = inner.slice
            return isinstance(sl, (ast.Compare, ast.BoolOp, ast.List)) or \
                (isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.Invert)) or \
                (isinstance(sl, ast.BinOp) and isinstance(sl.op, (ast.BitAnd, ast.BitOr)))
        return False

    def robust_call(self, c: ast.Call):
        q = self.qual(c.func) or ""
        last = _last(c.func)
        if q in {"warnings.filterwarnings", "warnings.simplefilter"}:
            action = c.args[0] if c.args else _kw(c, "action")
            # simplefilter(action, category, ...) / filterwarnings(action, message, category, ...)
            pos_cat = 1 if q == "warnings.simplefilter" else 2
            cat = _kw(c, "category") or (c.args[pos_cat] if len(c.args) > pos_cat else None)
            msg = _kw(c, "message") or (c.args[1] if q == "warnings.filterwarnings" and len(c.args) > 1 else None)
            scoped = False
            p = self.parent.get(id(c))
            while p is not None and not scoped:
                scoped = isinstance(p, ast.With) and any(_last(getattr(i.context_expr, "func", None)) ==
                                                         "catch_warnings" for i in p.items)
                p = self.parent.get(id(p))
            if _const(action) == "ignore" and (msg is None or _const(msg) == "") and not scoped and \
                    (cat is None or _last(cat) in {"Warning", "Exception"}):
                self.add("CS303", c, f"`{_short(c)}` hides every warning (convergence, deprecation, "
                                     "SettingWithCopy, ...) that would reveal a modelling problem; filter only "
                                     "the specific warning category/message.")
        elif q in {"numpy.seterr"} and any(_const(k.value) == "ignore" for k in c.keywords):
            self.add("CS303", c, f"`{_short(c)}` silences floating-point errors (division by zero, overflow, "
                                 "invalid values): NaN/inf results propagate unnoticed.")
        elif q in {"pandas.set_option"} and c.args and _const(c.args[0]) == "mode.chained_assignment" and \
                len(c.args) > 1 and _const(c.args[1], 0) is None:
            self.add("CS303", c, "pandas chained-assignment warnings are switched off.")
        if self.uses_pandas and isinstance(c.func, ast.Attribute):
            # only whole-frame replacement (`df = df.dropna()`, `df.fillna(0, inplace=True)`), not s.dropna().unique()
            st = self.parent.get(id(c))
            whole = (isinstance(st, ast.Assign) and st.value is c and isinstance(c.func.value, ast.Name) and
                     any(isinstance(t, ast.Name) and t.id == c.func.value.id for t in st.targets)) or \
                (isinstance(st, ast.Expr) and st.value is c)
            if last in {"fillna", "dropna"} and whole and isinstance(c.func.value, ast.Name) and \
                    c.func.value.id not in self.aliases and _kw(c, "subset") is None:
                frame = c.func.value.id
                if last == "fillna" and (c.args or _kw(c, "value") is not None):
                    val = _short(c.args[0] if c.args else _kw(c, "value"), 30)
                    self.add("CS304", c, f"`{frame}.fillna({val})` imputes every missing value of `{frame}` with "
                                         "one constant, across all columns, without recording how many were "
                                         "filled: document the imputation per variable.")
                elif last == "dropna" and not c.args and _kw(c, "axis") is None:
                    self.add("CS304", c, f"`{frame}.dropna()` drops every row with a missing value in any "
                                         "column, without reporting how many rows were removed: the modelling "
                                         "sample may no longer represent the population.")
            if _const(_kw(c, "inplace")) is True and isinstance(c.func.value, ast.Subscript):
                self.add("CS305", c, f"`{_short(c, 60)}` modifies a column selection in place: under pandas "
                                     "copy-on-write the original frame is not changed. Assign the result back "
                                     "instead.")

    # ---- security
    def security(self):
        for n in self.nodes:
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                v = n.value
                if not (isinstance(v, ast.Constant) and isinstance(v.value, str)):
                    continue
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    ident = t.id if isinstance(t, ast.Name) else t.attr if isinstance(t, ast.Attribute) else \
                        _const(t.slice) if isinstance(t, ast.Subscript) else None
                    if isinstance(ident, str):
                        self.secret(n, ident, v.value)
            elif isinstance(n, ast.Call):
                for k in n.keywords:
                    if k.arg and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str):
                        self.secret(k.value, k.arg, k.value.value)
                self.security_call(n)
            elif isinstance(n, ast.Dict):
                for k, v in zip(n.keys, n.values):
                    if isinstance(_const(k), str) and isinstance(v, ast.Constant) and isinstance(v.value, str):
                        self.secret(v, k.value, v.value)
        self.sql_building()

    def secret(self, node, ident: str, value: str):
        if not _SECRET_NAME.search(ident) or len(value) < 4 or " " in value or _PLACEHOLDER.search(value) \
                or re.fullmatch(r"[A-Z0-9_]+", value) or node.lineno in self.f.secret_lines:
            return
        self.f.secret_lines.add(node.lineno)
        line = self.f.lines[node.lineno - 1].strip().replace(value, "<redacted>")
        self.f.add("CS401", node.lineno, f"{ident} is set to a hard-coded string literal: credentials in code end "
                                         "up in version control and notebooks; read it from a secret store "
                                         "(dbutils.secrets, Key Vault, environment).", snippet=line[:200])

    def security_call(self, c: ast.Call):
        q = self.qual(c.func) or ""
        if isinstance(c.func, ast.Name) and c.func.id in {"eval", "exec"} and c.func.id not in self.aliases \
                and c.func.id not in self.defined:
            self.add("CS402", c, f"`{_short(c, 60)}` executes dynamically built code: behaviour cannot be "
                                 "reviewed statically and arbitrary code can run if the input is not trusted.")
        unsafe = {"pickle.load", "pickle.loads", "_pickle.load", "dill.load", "dill.loads", "cloudpickle.load",
                  "cloudpickle.loads", "joblib.load", "pandas.read_pickle", "shelve.open", "yaml.unsafe_load"}
        if q in unsafe or (q == "torch.load" and _const(_kw(c, "weights_only")) is not True) or \
                (q == "numpy.load" and _const(_kw(c, "allow_pickle")) is True):
            self.add("CS403", c, f"`{_short(c, 60)}` deserialises with pickle, which executes code embedded in "
                                 "the file: only load artefacts from a controlled, checksummed location.")
        elif q == "yaml.load":
            loader = _kw(c, "Loader") or (c.args[1] if len(c.args) > 1 else None)
            if loader is None or (_last(loader) or "") not in {"SafeLoader", "CSafeLoader", "BaseLoader"}:
                self.add("CS403", c, f"`{_short(c, 60)}` is not using yaml.SafeLoader: crafted YAML can construct "
                                     "arbitrary Python objects; use yaml.safe_load.")
        if q.startswith("subprocess.") and _const(_kw(c, "shell")) is True:
            self.add("CS404", c, f"`{_short(c, 60)}` runs through the shell (shell=True): any variable part of "
                                 "the command is a shell-injection risk; pass an argument list without shell=True.")
        elif q in {"os.system", "os.popen"}:
            self.add("CS404", c, f"`{_short(c, 60)}` runs a shell command: shell-injection risk and the step is "
                                 "invisible to the Python environment; use subprocess.run([...]).")

    def sql_building(self):
        done: set[int] = set()

        def flag(node, how, names):
            nm = ", ".join(sorted(names)[:4]) or "variables"
            self.add("CS405", node, f"SQL statement is assembled with {how} from {nm}: values are spliced into "
                                    "the query text (SQL-injection risk; the exact query run is not "
                                    "reproducible from code alone). Use bound parameters, e.g. "
                                    "spark.sql(query, args={...}) or cursor.execute(query, params).")

        for n in self.nodes:
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add) and \
                    not (isinstance(self.parent.get(id(n)), ast.BinOp) and isinstance(self.parent[id(n)].op, ast.Add)):
                parts, stack = [], [n]
                while stack:
                    x = stack.pop()
                    if isinstance(x, ast.BinOp) and isinstance(x.op, ast.Add):
                        stack.extend([x.right, x.left])
                    else:
                        parts.append(x)
                text = "".join(p.value if isinstance(p, ast.Constant) and isinstance(p.value, str) else "{}"
                               for p in parts)
                if any(not (isinstance(p, ast.Constant)) for p in parts) and \
                        any(isinstance(p, (ast.Constant, ast.JoinedStr)) for p in parts) and _SQL_START.search(text):
                    flag(n, "string concatenation", set().union(*(self.names(p) for p in parts)))
                    done |= {id(p) for p in ast.walk(n)}
            elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod) and isinstance(_const(n.left), str) and \
                    _SQL_START.search(n.left.value) and not isinstance(n.right, ast.Constant):
                flag(n, "%-formatting", self.names(n.right))
                done |= {id(p) for p in ast.walk(n)}
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "format" and \
                    isinstance(_const(n.func.value), str) and _SQL_START.search(n.func.value.value):
                flag(n, "str.format()", self.names(n) or {"arguments"})
        for n in self.nodes:
            if isinstance(n, ast.JoinedStr) and id(n) not in done and \
                    any(isinstance(v, ast.FormattedValue) for v in n.values):
                text = "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in n.values)
                if _SQL_START.search(text):
                    flag(n, "an f-string", self.names(n))

    # ---- maintainability / documentation
    def maintainability(self):
        top_level = {id(n) for n in self.tree.body}
        for n in self.tree.body:
            if isinstance(n, ast.ClassDef) and not n.name.startswith("_"):
                top_level |= {id(m) for m in n.body}
        for fn in self.funcs:
            length = (fn.end_lineno or fn.lineno) - fn.lineno + 1
            if length > 100:
                self.f.add("CS802", fn.lineno, f"function {fn.name}() is {length} lines long: hard to review and "
                                               "test; split it into smaller, separately testable steps.")
            if id(fn) in top_level and not fn.name.startswith("_") and not self.is_test and \
                    ast.get_docstring(fn) is None:
                self.f.add("CS803", fn.lineno, f"public function {fn.name}() has no docstring describing its "
                                               "inputs, outputs and assumptions.")
            defaults = [*fn.args.defaults, *[d for d in fn.args.kw_defaults if d is not None]]
            params = [*fn.args.posonlyargs, *fn.args.args][-len(fn.args.defaults):] if fn.args.defaults else []
            params += [a for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is not None]
            for p, d in zip(params, defaults):
                mutable = isinstance(d, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)) or \
                    (isinstance(d, ast.Call) and _last(d.func) in {"list", "dict", "set", "defaultdict",
                                                                   "OrderedDict", "Counter", "DataFrame"})
                if mutable:
                    self.f.add("CS804", d.lineno, f"parameter `{p.arg}` of {fn.name}() defaults to the mutable "
                                                  f"`{_short(d, 30)}`, created once and shared by every call: "
                                                  "changes made in one call leak into the next. Default to None.")
        for n in self.nodes:
            if isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names):
                self.add("CS805", n, f"`from {n.module} import *` hides where names come from and can silently "
                                     "shadow other names.")


# --------------------------------------------------------------------------------------------- repository driver

class _Repo:
    def __init__(self, files: dict[str, str]):
        self.files = files
        self.out: list[_File] = []
        self.imports: list[tuple[str, int, str, bool]] = []
        self.lists: list[tuple[str, int, tuple]] = []
        self.reqs: list[_Req] = []
        self.declared_files = False
        self.pip_reqs: list[_Req] = []
        names = {p.replace("\\", "/").rsplit("/", 1)[-1].lower() for p in files}
        self.locked = bool(names & _LOCK_FILES)
        self.local: set[str] = set()
        for p in files:
            parts = p.replace("\\", "/").split("/")
            self.local.update(parts[:-1])
            if parts[-1].endswith(".py"):
                self.local.add(parts[-1][:-3])

    def run(self) -> list[tuple[str, tuple]]:
        for path in sorted(self.files):
            self.file(path, self.files[path] if isinstance(self.files[path], str) else str(self.files[path]))
        self.repo_rules()
        return [(f.path, r) for f in self.out for r in f.found]

    def file(self, path: str, text: str):
        kind = _kind(path)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        cells: list[_Cell] = []
        raw_nb = False
        if kind == "notebook":
            rendered = _render_notebook(text)
            if rendered:
                text, cells, raw_nb = rendered
        lines = text.split("\n")
        f = _File(path, lines)
        self.out.append(f)
        if kind == "notebook" and not cells:
            cells = _parse_rendered(lines)
        f.cells = cells
        if kind in {"python", "notebook"}:
            self.python(f, kind, cells, raw_nb)
        elif kind == "sql":
            _secret_scan(f, kv=True)
            for i, ln in enumerate(lines, 1):
                for s in _quoted_strings(ln):
                    if _path_kind(s):
                        _string_rules(f, i, s, dates=False)
            _sql_checks(f, text, 1)
            _todo_regex(f)
        elif kind == "r":
            self.r_file(f)
        elif kind == "sas":
            self.sas_file(f, text)
        elif kind == "requirements":
            self.declared_files = True
            _secret_scan(f, kv=False)
            self.reqs.extend(_requirements(f))
        elif kind == "conda":
            self.declared_files = True
            _secret_scan(f, kv=False)
            self.reqs.extend(_conda(f))
        elif kind == "pyproject":
            self.declared_files = True
            _secret_scan(f, kv=True)
            self.reqs.extend(_pyproject(f, text))
        elif kind == "config":
            _secret_scan(f, kv=True)
        elif kind == "code":
            _secret_scan(f, kv=True)
            _todo_regex(f)
        else:
            _secret_scan(f, kv=False)

    # ---- Python and notebooks
    def python(self, f: _File, kind: str, cells: list[_Cell], raw_nb: bool):
        lines = f.lines
        view = list(lines)
        units: list[tuple[int, int]] = []
        pip_line_nos: list[int] = []
        sql_units: list[tuple[int, int]] = []
        if kind == "notebook":
            view = [""] * len(lines)
            for c in cells:
                if c.kind != "code" or c.src_end < c.src_start:
                    continue
                src = list(range(c.src_start, c.src_end + 1))
                first = next((lines[i - 1].strip() for i in src if lines[i - 1].strip()), "")
                magic = first.split()[0] if first else ""
                if magic in {"%sql", "%%sql"}:
                    start = next(i for i in src if lines[i - 1].strip()) + 1
                    sql_units.append((start, c.src_end))
                    continue
                if magic in _NON_PY_MAGIC or (magic.startswith("%%") and magic not in {"%%time", "%%timeit"}):
                    continue
                for i in src:
                    if lines[i - 1].lstrip().startswith(("%", "!")):
                        pip_line_nos.append(i)      # IPython magic / shell line: not Python
                    else:
                        view[i - 1] = lines[i - 1]
                units.append((c.src_start, c.src_end))
            self.notebook_rules(f, cells, raw_nb)
        else:
            units.append((1, len(lines)))
            pip_line_nos = [i for i, ln in enumerate(lines, 1) if _PIP_LINE.match(ln)]
            sql_units = self.databricks_sql_cells(lines)
        self.pip_reqs.extend(_pip_lines(f, pip_line_nos))
        for r in _pip_lines(f, pip_line_nos):
            if r.version is None:
                f.add("CS501", r.line, f"`{r.display}` is installed in the notebook without an exact version "
                                       f"({r.spec or 'no version'}): re-running it can install a different release; "
                                       f"pin it, e.g. {r.display}==<tested version>.")
        for s, e in sql_units:
            body = "\n".join(re.sub(r"^\s*# MAGIC ?", "", ln) for ln in lines[s - 1:e])
            _sql_checks(f, body, s, where="in this SQL cell ")

        body: list[ast.stmt] = []
        bad_lines: set[int] = set()
        for s, e in units:
            code = "\n".join(view[s - 1:e])
            try:
                tree = ast.parse(code)
            except SyntaxError as err:
                ln = s + (err.lineno or 1) - 1
                f.add("CS307", ln, f"Python code does not parse ({err.msg}); none of the AST-based checks could "
                                   "run on this code: it cannot have produced the reported results as written.")
                bad_lines.update(range(s, e + 1))
                for i in range(s, e + 1):
                    for q in _quoted_strings(_strip_hash_comment(view[i - 1])):
                        _string_rules(f, i, q)
                continue
            if s > 1:
                ast.increment_lineno(tree, s - 1)
            body.extend(tree.body)
        _secret_scan(f, kv=False)
        for i in sorted(bad_lines):
            m = _KV_SECRET.search(view[i - 1])
            if m and i not in f.secret_lines and not _PLACEHOLDER.search(m.group(2)) and len(m.group(2)) >= 4:
                f.add("CS401", i, f"'{m.group(1)}' is set to a literal value in code that does not parse.",
                      snippet=view[i - 1].strip().replace(m.group(2), "<redacted>")[:200])
        if body or not bad_lines:
            _Py(self, f, ast.Module(body=body, type_ignores=[])).run()
        self.python_comments(f, view)

    def python_comments(self, f: _File, view: list[str]):
        text = "\n".join(view)
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
            comments = [(t.start[0], t.string) for t in toks if t.type == tokenize.COMMENT]
        except (tokenize.TokenError, IndentationError, SyntaxError):
            comments = [(i, ln[ln.index("#"):]) for i, ln in enumerate(view, 1) if "#" in ln]
        for line, s in comments:
            if s.startswith("# MAGIC"):
                continue
            m = _TODO.search(s)
            if m:
                f.add("CS801", line, f"{m.group(1)} comment left in code: '{s.strip()[:100]}': confirm the open "
                                     "item does not affect the validated model.")

    @staticmethod
    def databricks_sql_cells(lines: list[str]) -> list[tuple[int, int]]:
        if not lines or not lines[0].startswith("# Databricks notebook source"):
            return []
        cells, start = [], 2
        for i, ln in enumerate(lines + ["# COMMAND ----------"], 1):
            if ln.strip() == "# COMMAND ----------":
                body = [j for j in range(start, min(i, len(lines) + 1)) if lines[j - 1].strip()]
                if body and lines[body[0] - 1].strip() == "# MAGIC %sql":
                    cells.append((body[0] + 1, body[-1]))
                start = i + 1
        return cells

    def notebook_rules(self, f: _File, cells: list[_Cell], raw_nb: bool):
        if raw_nb:
            code = [c for c in cells if c.kind == "code"]
            prev = None
            for c in code:
                if c.exec_count is None:
                    continue
                if prev is not None and c.exec_count <= prev.exec_count:
                    f.add("CS601", c.header, f"cell {c.number} has execution count [{c.exec_count}] but the earlier "
                                             f"cell {prev.number} has [{prev.exec_count}]: the notebook was run out "
                                             "of order, so its saved outputs may not be reproducible by running "
                                             "top to bottom.")
                if prev is None or c.exec_count > prev.exec_count:
                    prev = c
            if any(c.exec_count is not None for c in code):
                for c in code:
                    if c.exec_count is None and c.has_source:
                        f.add("CS603", c.header, f"code cell {c.number} was never executed although other cells "
                                                 "were: the saved results were produced without it (or with a "
                                                 "different version of it).")
            for c in cells:
                if c.error:
                    f.add("CS602", c.header, f"cell {c.number} was saved with an error output ({c.error[:120]}): "
                                             "the notebook did not run end to end, so later results may come from "
                                             "stale state.")
        else:
            for c in cells:
                if c.out_start is None:
                    continue
                nxt = next((d.header for d in cells if d.header > c.header), len(f.lines) + 1)
                for i in range(c.out_start, nxt):
                    if f.lines[i - 1].startswith("# Traceback (most recent call last)"):
                        f.add("CS602", i, f"cell {c.number} output contains a traceback: the notebook did not run "
                                          "end to end.")
                        break

    # ---- R and SAS
    def r_file(self, f: _File):
        _secret_scan(f, kv=True)
        code = [_strip_hash_comment(ln) for ln in f.lines]
        seeded = any(re.search(r"\bset\.seed\s*\(", ln) for ln in code)
        rx = re.compile(r"(?<![\w.])(sample|sample_n|sample_frac|slice_sample|runif|rnorm|rbinom|rpois|rexp|rbeta|"
                        r"rgamma|createDataPartition|createFolds|initial_split|vfold_cv|randomForest|ranger|"
                        r"kmeans)\s*\(")
        for i, ln in enumerate(code, 1):
            if not seeded:
                for m in rx.finditer(ln):
                    f.add("CS001", i, f"{m.group(1)}() uses R's random number generator and set.seed() is never "
                                      "called in this file: results change on every run.")
            m = re.search(r"\bset\.seed\s*\(\s*(.*?)\)", ln)
            if m and re.search(r"Sys\.time|proc\.time|Sys\.getpid|as\.numeric\(\s*Sys", m.group(1)):
                f.add("CS004", i, f"set.seed({m.group(1)}) derives the seed from the clock/process id: runs are "
                                  "not reproducible.")
            if re.search(r"\bsuppressWarnings\s*\(|\boptions\s*\(\s*warn\s*=\s*-1", ln):
                f.add("CS303", i, "warnings are suppressed: convergence or coercion warnings that reveal "
                                  "modelling problems are hidden.")
            if re.search(r"\beval\s*\(\s*parse\s*\(", ln):
                f.add("CS402", i, "eval(parse(...)) executes dynamically built code.")
            for s in _quoted_strings(ln):
                _string_rules(f, i, s)
        _todo_regex(f)

    def sas_file(self, f: _File, text: str):
        _secret_scan(f, kv=True)
        clean = re.sub(r"/\*[\s\S]*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
        lines = clean.split("\n")
        lines = [("" if re.match(r"^\s*\*", ln) else ln) for ln in lines]
        joined = "\n".join(lines)
        streaminit = re.search(r"\bcall\s+streaminit\s*\(", joined, re.I)
        for i, ln in enumerate(lines, 1):
            for m in re.finditer(r"\b(ranuni|rannor|ranbin|ranexp|rangam|ranpoi|rantbl|rantri|rancau|uniform|"
                                 r"normal)\s*\(\s*(-?\d+)?", ln, re.I):
                if m.group(2) is not None and int(m.group(2)) <= 0:
                    f.add("CS004", i, f"{m.group(1)}({m.group(2)}) seeds from the system clock: runs are not "
                                      "reproducible; use a positive fixed seed.")
            if not streaminit and re.search(r"\brand\s*\(\s*['\"]", ln, re.I):
                f.add("CS001", i, "RAND() is used without CALL STREAMINIT(<seed>): results change on every run.")
            for s in _quoted_strings(ln):
                _string_rules(f, i, s, dates=False)
            for m in re.finditer(r"(['\"])(\d{1,2}[A-Za-z]{3}\d{2,4})\1d\b", ln, re.I):
                f.add("CS104", i, f"SAS date literal {m.group(0)} is hard-coded: sample windows / reference dates "
                                  "should be documented macro parameters.")
        for m in re.finditer(r"\bproc\s+surveyselect\b[^;]*;", joined, re.I):
            if not re.search(r"\bseed\s*=", m.group(0), re.I):
                f.add("CS001", joined.count("\n", 0, m.start()) + 1,
                      "PROC SURVEYSELECT without SEED=: the selected sample changes on every run.")
        _sql_checks(f, joined, 1, where="")
        _todo_regex(f)

    # ---- repository-level rules
    def repo_rules(self):
        by_path = {f.path: f for f in self.out}
        # CS501 / CS502
        seen_in_file: dict[tuple[str, str], _Req] = {}
        first_pin: dict[str, _Req] = {}
        for r in sorted(self.reqs, key=lambda r: (r.file, r.line)):
            f = by_path[r.file]
            is_pyproject = r.file.replace("\\", "/").rsplit("/", 1)[-1].lower() == "pyproject.toml"
            if r.version is None and not (is_pyproject and self.locked):
                f.add("CS501", r.line, f"`{r.display}` is not pinned to an exact version "
                                       f"({r.spec or 'no version constraint'}): a fresh install can resolve a "
                                       "different release and change model outputs; pin it (==) or ship a lock "
                                       "file.")
            key = (r.file, r.name)
            if key in seen_in_file:
                p = seen_in_file[key]
                kind = "conflicts with" if (p.spec != r.spec) else "duplicates"
                f.add("CS502", r.line, f"`{r.display}{r.spec}` {kind} `{p.display}{p.spec}` on line {p.line} of "
                                       "the same file: which one is installed depends on the resolver.")
                continue
            seen_in_file[key] = r
            if r.version and r.version != "url":
                if r.name in first_pin and first_pin[r.name].file != r.file:
                    p = first_pin[r.name]
                    a, b = p.version.split("."), r.version.split(".")
                    if a[:len(b)] != b[:len(a)]:
                        f.add("CS502", r.line, f"`{r.display}` is pinned to {r.version} here but to {p.version} in "
                                               f"{p.file} (line {p.line}): the environments disagree on which "
                                               "version the model was built with.")
                else:
                    first_pin.setdefault(r.name, r)
        # CS503
        if self.declared_files:
            declared = {r.name for r in self.reqs} | {r.name for r in self.pip_reqs}
            stdlib = set(sys.stdlib_module_names)
            reported: set[tuple[str, str]] = set()
            for path, line, top, in_try in sorted(self.imports):
                if in_try or top in stdlib or top in self.local or top in _RUNTIME_PROVIDED or \
                        (path, top) in reported:
                    continue
                cands = {_norm_pkg(top), _norm_pkg(_IMPORT_TO_DIST.get(top, top))}
                if cands & declared or any(d.startswith(c + "-") for d in declared for c in cands):
                    continue
                reported.add((path, top))
                by_path[path].add("CS503", line, f"`{top}` is imported but no requirements file of the repository "
                                                 f"declares it (expected '{_IMPORT_TO_DIST.get(top, top)}'): the "
                                                 "environment cannot be rebuilt from the declared dependencies.")
        # CS105
        groups: dict[tuple, list[tuple[str, int]]] = {}
        for path, line, key in sorted(set(self.lists)):
            if (path, line) not in groups.setdefault(key, []):
                groups[key].append((path, line))
        for key, occ in groups.items():
            if len(occ) < 2:
                continue
            fp, fl = occ[0]
            for path, line in occ[1:]:
                by_path[path].add("CS105", line, f"the literal list {list(key)[:5]}{'...' if len(key) > 5 else ''} "
                                                 f"is repeated (first at {fp}:{fl}): define it once so the copies "
                                                 "cannot drift apart.")


# --------------------------------------------------------------------------------------------- public API

def scan(files: dict[str, str], rules: list[str] | None = None, include_info: bool = False) -> list[CodeFinding]:
    """Scan a repository (relpath -> text) and return findings sorted by (file, line, rule).

    `rules` restricts the scan to those rule ids (info rules listed there are always reported); otherwise every
    rule runs and informational rules (INFO_RULES) are reported only when `include_info` is True.
    """
    if rules is not None:
        unknown = sorted(set(rules) - set(RULES))
        if unknown:
            raise ValueError(f"unknown rule id(s): {', '.join(unknown)}; valid ids are {', '.join(sorted(RULES))}")
        wanted = set(rules)
    else:
        wanted = set(RULES) if include_info else set(RULES) - INFO_RULES
    repo = _Repo(files)
    raw = repo.run()
    cells_by_path = {f.path: f.cells for f in repo.out if f.cells}
    seen, out = set(), []
    for path, (rule, line, end, snippet, message) in raw:
        if rule not in wanted or (rule in _TEST_EXEMPT and _is_test_path(path)):
            continue
        cells = cells_by_path.get(path)
        if cells:
            cell = next((c for c in reversed(cells) if c.header <= line), None)
            if cell is not None and f"cell {cell.number}" not in message:
                message = f"{message} [notebook cell {cell.number}]"
        key = (path, line, rule, message)
        if key in seen:
            continue
        seen.add(key)
        title, category, _ = RULES[rule]
        out.append(CodeFinding(rule, title, category, path, line, end, snippet, message))
    out.sort(key=lambda x: (x.file, x.line, x.rule, x.end_line, x.message))
    return out


def summarise(findings: list[CodeFinding]) -> pd.DataFrame:
    """Counts of findings by rule, category and file (with the first line and whether the rule is informational)."""
    cols = ["rule", "title", "category", "level", "file", "findings", "first_line"]
    if not findings:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({"rule": [f.rule for f in findings], "category": [f.category for f in findings],
                       "file": [f.file for f in findings], "line": [f.line for f in findings]})
    out = (df.groupby(["rule", "category", "file"], sort=True)
             .agg(findings=("line", "size"), first_line=("line", "min")).reset_index())
    out.insert(1, "title", out["rule"].map(lambda r: RULES[r][0]))
    out.insert(3, "level", out["rule"].map(lambda r: "info" if r in INFO_RULES else "finding"))
    return out[cols].reset_index(drop=True)
