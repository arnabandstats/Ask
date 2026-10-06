"""Simulated (synthetic) data: draw columns from named distributions into a table.

Seeded and reproducible: column k uses its own generator np.random.default_rng([seed, k]),
so the same spec always gives the same table, and adding a column never changes the others.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_ROWS = 1_000_000
MAX_COLUMNS = 50


def _normal(rng, n, mean=0.0, sd=None, var=None):
    if sd is not None and var is not None:
        raise ValueError("give sd or var, not both")
    sd = np.sqrt(var) if var is not None else (1.0 if sd is None else sd)
    return rng.normal(mean, sd, n)


def _exponential(rng, n, scale=None, rate=None):
    if scale is not None and rate is not None:
        raise ValueError("give scale or rate, not both")
    return rng.exponential(1 / rate if rate is not None else (1.0 if scale is None else scale), n)


def _categorical(rng, n, categories, probs=None):
    if not isinstance(categories, (list, tuple)) or not categories:
        raise ValueError("categories must be a non-empty list")
    if probs is not None:
        probs = np.asarray(probs, float)
        if len(probs) != len(categories) or not np.isclose(probs.sum(), 1):
            raise ValueError("probs must have one value per category and sum to 1")
    return rng.choice(np.asarray(categories, dtype=object), n, p=probs)


# name -> (sampler(rng, n, **params), parameter help)
DISTRIBUTIONS = {
    "normal": (_normal, "mean (0), sd (1) or var"),
    "uniform": (lambda rng, n, low=0.0, high=1.0: rng.uniform(low, high, n), "low (0), high (1)"),
    "lognormal": (lambda rng, n, mean=0.0, sigma=1.0: rng.lognormal(mean, sigma, n),
                  "mean (0), sigma (1) of the underlying normal"),
    "exponential": (_exponential, "scale (1) or rate"),
    "gamma": (lambda rng, n, shape, scale=1.0: rng.gamma(shape, scale, n), "shape, scale (1)"),
    "beta": (lambda rng, n, a, b: rng.beta(a, b, n), "a, b"),
    "t": (lambda rng, n, df: rng.standard_t(df, n), "df"),
    "chi2": (lambda rng, n, df: rng.chisquare(df, n), "df"),
    "poisson": (lambda rng, n, lam: rng.poisson(lam, n), "lam"),
    "binomial": (lambda rng, n, trials, p: rng.binomial(trials, p, n), "trials, p"),
    "bernoulli": (lambda rng, n, p: rng.binomial(1, p, n), "p (0/1 outcome)"),
    "categorical": (_categorical, "categories (list), probs (optional, sum to 1)"),
}


def help_text() -> str:
    return "; ".join(f"{k}: {h}" for k, (_, h) in DISTRIBUTIONS.items())


def generate(n: int, columns: list[dict], seed: int) -> pd.DataFrame:
    """columns: [{"name": "x", "distribution": "normal", "params": {"mean": 0, "var": 1}}, ...]"""
    n = int(n)
    if not 1 <= n <= MAX_ROWS:
        raise ValueError(f"n must be between 1 and {MAX_ROWS:,}")
    if not columns or len(columns) > MAX_COLUMNS:
        raise ValueError(f"give 1 to {MAX_COLUMNS} columns")
    out = {}
    for k, col in enumerate(columns):
        if not isinstance(col, dict):
            raise ValueError("each column must be an object with name, distribution and params")
        name = str(col.get("name") or f"x{k + 1}")
        dist = str(col.get("distribution", "")).lower().strip()
        if dist not in DISTRIBUTIONS:
            raise ValueError(f"unknown distribution '{dist}' for column '{name}'. Use one of: {help_text()}")
        if name in out:
            raise ValueError(f"duplicate column name '{name}'")
        params = col.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"params for '{name}' must be an object")
        fn, hint = DISTRIBUTIONS[dist]
        try:
            out[name] = fn(np.random.default_rng([int(seed), k]), n, **params)
        except TypeError:
            raise ValueError(f"bad parameters {sorted(params)} for {dist} column '{name}'; it takes: {hint}") \
                from None
        except ValueError as exc:
            raise ValueError(f"{dist} column '{name}': {exc}") from None
    return pd.DataFrame(out)
