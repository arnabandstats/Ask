"""Loading model artefacts and scoring them, for tests that need the real model.

Supported: pickle / joblib / cloudpickle files, statsmodels results pickles, and
MLflow models (local model folder, runs:/..., models:/..., Unity Catalog
models:/catalog.schema.model/1) when mlflow is installed.

Unpickling runs code from the file, so only load artefacts from the model
owner's controlled location (the agent tells the user this when it loads one).
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LoadedModel:
    name: str
    location: str
    obj: Any
    flavour: str                     # sklearn-like | statsmodels | mlflow-pyfunc | callable
    sha256: str = ""
    feature_names: list[str] = field(default_factory=list)

    def describe(self) -> str:
        feats = f"; {len(self.feature_names)} features: {', '.join(self.feature_names[:25])}" \
            if self.feature_names else ""
        return (f"{self.name}: {type(self.obj).__name__} ({self.flavour}) from {self.location}"
                f"{f'; sha256 {self.sha256[:16]}' if self.sha256 else ''}{feats}")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _features(obj: Any) -> list[str]:
    for attr in ("feature_names_in_", "feature_name_", "exog_names"):
        v = getattr(obj, attr, None)
        if v is not None:
            return [str(x) for x in list(v)]
    model = getattr(obj, "model", None)
    if model is not None and getattr(model, "exog_names", None):
        return [str(x) for x in model.exog_names]
    try:   # mlflow signature
        sig = obj.metadata.get_input_schema()
        return [c.name for c in sig.inputs] if sig else []
    except Exception:
        return []


def load_model(location: str, name: str | None = None) -> LoadedModel:
    loc = location.strip().strip("\"'`")
    name = name or Path(loc.rstrip("/")).stem or "model"
    if loc.startswith(("runs:/", "models:/")) or (Path(loc).is_dir() and (Path(loc) / "MLmodel").exists()):
        try:
            import mlflow
        except ImportError as exc:
            raise ValueError("Loading MLflow models needs the mlflow package (pip install mlflow).") from exc
        if loc.startswith("models:/") and loc.count(".") >= 2:
            mlflow.set_registry_uri("databricks-uc")
        obj = mlflow.pyfunc.load_model(loc)
        return LoadedModel(name, loc, obj, "mlflow-pyfunc", "", _features(obj))
    p = Path(loc)
    if not p.exists():
        raise ValueError(f"Model file not found: {loc}")
    sha = _sha(p)
    obj = None
    if p.suffix.lower() in {".joblib", ".jbl"}:
        import joblib
        obj = joblib.load(p)
    else:
        try:
            with open(p, "rb") as fh:
                obj = pickle.load(fh)
        except Exception:
            import joblib
            obj = joblib.load(p)
    flavour = "statsmodels" if type(obj).__module__.startswith("statsmodels") else (
        "sklearn-like" if hasattr(obj, "predict") else "callable" if callable(obj) else "unknown")
    if flavour == "unknown":
        raise ValueError(f"{p.name} holds a {type(obj).__name__}, which has no predict method.")
    return LoadedModel(name, str(p), obj, flavour, sha, _features(obj))


def predict_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Risk scores from a model: P(class 1) for classifiers, predictions otherwise.

    `model` may be a LoadedModel or the raw object."""
    obj = model.obj if isinstance(model, LoadedModel) else model
    if hasattr(obj, "predict_proba"):
        p = np.asarray(obj.predict_proba(X))
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else (p if p.ndim == 1 else p.max(axis=1))
    if type(obj).__module__.startswith("statsmodels"):
        Xs = X
        names = _features(obj)
        if names and "const" in names and "const" not in X.columns:
            Xs = X.assign(const=1.0)[names]
        return np.asarray(obj.predict(Xs), dtype=float)
    if hasattr(obj, "predict"):
        return np.asarray(obj.predict(X), dtype=float).ravel()
    if callable(obj):
        return np.asarray(obj(X), dtype=float).ravel()
    raise TypeError(f"Cannot score with {type(obj).__name__}")


def predict_labels(model: Any, X: pd.DataFrame) -> np.ndarray:
    obj = model.obj if isinstance(model, LoadedModel) else model
    return np.asarray(obj.predict(X)).ravel()


def model_features(model: Any, df: pd.DataFrame, features: list[str] | None) -> list[str]:
    """Feature list for scoring: explicit list, else the model's own, else error."""
    if features:
        return list(features)
    names = model.feature_names if isinstance(model, LoadedModel) else _features(model)
    names = [n for n in names if n != "const"]
    if names and all(n in df.columns for n in names):
        return names
    raise ValueError("Pass `features`: the model does not record its feature names "
                     "(or they are not all in the table).")
