"""Econometric diagnostics for regression-based models.

Covers the regressions that sit inside credit-risk and stress-testing models:
application/behavioural scorecards (logit), macro / satellite models used for
stress testing and IFRS 9 forward-looking information (OLS on time series), and
LGD / CCF regressions (OLS, fractional logit).

Most tests REFIT the documented specification with statsmodels from the target
and feature columns (a constant is always added), so the validator can replicate
the developer's numbers and then probe them: significance, robust errors, fit,
residual assumptions, specification, structural breaks, stationarity and
cointegration of the series, influence, multicollinearity, forecast accuracy,
sensitivities and coefficient stability. Scorecard-specific tests (WoE / IV,
points recomputation) are at the end.

Time order: give `date` and rows are sorted by it (stable sort) before fitting;
without it the table's row order is taken as the time order. Every test reports
how many rows it dropped for missing values.
"""
from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

from ask.validation.core import (NotApplicable, Outcome, P, RunContext, binary, dropped_note, num,
                                 register, require_two_classes)

_REG = ("satellite", "pd", "lgd", "ead", "ifrs9", "ml_regression")
_TS = ("satellite", "ifrs9")
_SCORE = ("pd",)
_KINDS = ("auto", "ols", "logit", "probit", "fractional_logit")
_COVS = ("nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC")
_INTERCEPT = {"const", "intercept", "(intercept)", "constant", "alpha", "b0", "beta0"}

_SPEC_REFS = ("Greene (2018), Econometric Analysis, 8th ed.",
              "Wooldridge (2010), Econometric Analysis of Cross Section and Panel Data, 2nd ed.",
              "statsmodels documentation (OLS, Logit, Probit, GLM)")


def _spec(required: bool = True):
    return (
        P("target", help="Dependent variable: continuous (OLS), 0/1 event (logit/probit) or a rate in "
                         "[0,1] (fractional logit)", required=required),
        P("features", "columns", help="Explanatory variables of the documented specification "
                                      "(a constant is added)", required=required),
        P("model_kind", "string", default="auto", choices=_KINDS,
          help="auto = logit for a 0/1 target, otherwise OLS"),
        P("date", required=False, help="Time-order column; rows are sorted by it before fitting "
                                       "(default: table row order)"),
    )


# ── data preparation and fitting ───────────────────────────────────────────

def _order_key(s: pd.Series) -> pd.Series:
    """A sortable version of a date / period column."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return s
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.to_datetime(s, errors="coerce", format="mixed")
    if s.notna().any() and d.notna().sum() >= 0.99 * s.notna().sum():
        return d
    return s.where(s.isna(), s.astype(str))


def _prepare(ctx: RunContext, cols: list[str], date=None) -> tuple[pd.DataFrame, int]:
    """Numeric copy of `cols` with complete rows only, sorted by `date` if given."""
    df = ctx.df
    cols = list(dict.fromkeys(c for c in cols if c))
    work = pd.DataFrame({c: num(df, c) for c in cols}, index=df.index)
    for c in cols:
        if work[c].notna().sum() == 0 and df[c].notna().any():
            raise ValueError(f"Column '{c}' is not numeric.")
    if date:
        work["__order__"] = _order_key(df[date])
    sub = work.dropna()
    dropped = len(work) - len(sub)
    if date:
        sub = sub.sort_values("__order__", kind="mergesort")
    return sub, dropped


def _labels(ctx: RunContext, sub: pd.DataFrame, date=None) -> pd.Series:
    """Row identifiers for reports: the date value when given, else the row label."""
    if date:
        return ctx.df.loc[sub.index, date].astype(str)
    return pd.Series(sub.index.astype(str), index=sub.index)


def _resolve_kind(y: pd.Series, kind: str) -> str:
    vals = set(np.unique(y))
    if kind == "auto":
        return "logit" if vals <= {0.0, 1.0} and len(vals) == 2 else "ols"
    if kind in ("logit", "probit"):
        if not vals <= {0.0, 1.0}:
            raise ValueError(f"{kind} needs a 0/1 target.")
        require_two_classes(y)
    if kind == "fractional_logit" and (y.min() < 0 or y.max() > 1):
        raise ValueError("fractional_logit needs a target in [0, 1].")
    return kind


def _fit_model(y, X, kind: str, cov_type: str = "nonrobust", hac_lags=None):
    import statsmodels.api as sm
    kw = {"cov_type": cov_type}
    if cov_type == "HAC":
        kw["cov_kwds"] = {"maxlags": int(hac_lags)}
    if kind == "ols":
        return sm.OLS(y, X).fit(**kw)
    if kind != "ols" and cov_type in ("HC1", "HC2", "HC3"):
        kw["cov_type"] = "HC0"          # statsmodels has only the sandwich (HC0) form for ML models
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind in ("logit", "probit"):
            m = sm.Logit(y, X) if kind == "logit" else sm.Probit(y, X)
            return m.fit(disp=0, maxiter=200, **kw)
        if kind == "fractional_logit":
            if kw["cov_type"] == "nonrobust":
                kw["cov_type"] = "HC0"   # quasi-likelihood: sandwich errors are required
            return sm.GLM(y, X, family=sm.families.Binomial()).fit(**kw)
    raise ValueError(f"Unknown model_kind {kind}")


def _default_hac_lags(n: int) -> int:
    """Newey–West (1994) plug-in rule floor(4·(n/100)^(2/9))."""
    return int(math.floor(4 * (n / 100) ** (2 / 9)))


def _fit(ctx: RunContext, target, features, model_kind="auto", date=None, cov_type="nonrobust",
         hac_lags=None, extra=(), allowed=None, min_extra_obs: int = 2):
    import statsmodels.api as sm
    if not features:
        raise ValueError("features: give at least one explanatory variable.")
    if target in features:
        raise ValueError("target cannot also be a feature.")
    sub, dropped = _prepare(ctx, [target, *features, *extra], date)
    y = sub[target]
    X = sm.add_constant(sub[list(features)], has_constant="add")
    k = X.shape[1]
    if len(y) < k + min_extra_obs:
        raise NotApplicable(f"{len(y)} complete observations for {k} parameters.")
    if np.linalg.matrix_rank(X.to_numpy()) < k:
        raise NotApplicable("The design matrix is rank deficient (a feature is constant or an exact "
                            "linear combination of others).")
    kind = _resolve_kind(y, model_kind)
    if allowed and kind not in allowed:
        raise NotApplicable(f"This test is defined for {', '.join(allowed)} models; the model is {kind}.")
    notes = dropped_note(dropped)
    if cov_type == "HAC":
        if hac_lags is None:
            hac_lags = _default_hac_lags(len(y))
            notes.append(f"HAC lags set by the Newey–West rule floor(4(n/100)^(2/9)) = {hac_lags}.")
        if not date:
            notes.append("HAC errors assume the table's row order is the time order (no date given).")
    if kind != "ols" and cov_type in ("HC1", "HC2", "HC3"):
        notes.append(f"{cov_type} is defined for OLS only; the sandwich (HC0) estimator was used.")
    if kind == "fractional_logit":
        notes.append("Fractional logit (Papke & Wooldridge 1996): Bernoulli quasi-likelihood GLM with "
                     "sandwich standard errors.")
    try:
        res = _fit_model(y, X, kind, cov_type, hac_lags)
    except np.linalg.LinAlgError as exc:
        raise NotApplicable(f"Model could not be estimated: {exc}")
    if not np.all(np.isfinite(res.params)):
        raise NotApplicable("Estimation produced non-finite coefficients (perfect separation?).")
    conv = getattr(res, "mle_retvals", None) or {}
    if conv and not conv.get("converged", True):
        notes.append("The maximum-likelihood optimiser did not report convergence; treat estimates with care.")
    return SimpleNamespace(y=y, X=X, kind=kind, res=res, sub=sub, dropped=dropped, notes=notes,
                           n=len(y), k=k, hac_lags=hac_lags)


def _coef_table(res, alpha: float = 0.05) -> pd.DataFrame:
    ci = np.asarray(res.conf_int(alpha))
    stat = "t_stat" if getattr(res, "use_t", False) else "z_stat"
    return pd.DataFrame({"variable": list(res.params.index), "estimate": res.params.to_numpy(),
                         "std_error": res.bse.to_numpy(), stat: res.tvalues.to_numpy(),
                         "p_value": res.pvalues.to_numpy(),
                         f"ci_low_{1 - alpha:.0%}": ci[:, 0], f"ci_high_{1 - alpha:.0%}": ci[:, 1]})


def _canon(name: str, index) -> str:
    """Map a user coefficient name onto the fitted parameter names."""
    if name in index:
        return name
    if str(name).strip().lower() in _INTERCEPT:
        return "const"
    low = {str(c).lower(): c for c in index}
    if str(name).lower() in low:
        return low[str(name).lower()]
    raise ValueError(f"Coefficient '{name}' is not in the model. Parameters: {', '.join(map(str, index))}")


def _sign(v) -> float:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"+", "positive", "pos", "+1", "1"}:
            return 1.0
        if s in {"-", "negative", "neg", "-1"}:
            return -1.0
        raise ValueError(f"Expected sign '{v}' must be '+' or '-'.")
    return float(np.sign(float(v)))


# ── coefficients ───────────────────────────────────────────────────────────

@register("econ.coefficients", "Coefficient estimates, significance and replication of documented values",
          "Model specification", _REG,
          params=(*_spec(),
                  P("cov_type", "string", default="nonrobust", choices=_COVS,
                    help="Covariance estimator for standard errors"),
                  P("hac_lags", "integer", required=False, help="Lags for HAC (Newey–West) errors"),
                  P("alpha", "number", default=0.05, help="1 − confidence level of the intervals"),
                  P("expected_coefficients", "dict", required=False,
                    help="Documented coefficients {variable: value}; 'const'/'intercept' for the constant"),
                  P("expected_signs", "dict", required=False,
                    help="Expected signs {variable: '+' or '-'}; default: sign of the documented coefficient")),
          description="""Refits the documented specification (OLS, logit, probit or fractional logit, constant
added) and reports each coefficient with its standard error, t or z statistic, two-sided p-value (H0: β = 0) and
confidence interval, under the chosen covariance estimator (classical, White HC0–HC3, or Newey–West HAC).
When documented coefficients are given, each is compared with the re-estimate: difference, relative difference,
difference in standard errors ((β̂ − β_doc)/SE), whether β_doc lies inside the CI and whether the estimated sign
equals the expected sign. A joint Wald test of H0: β = β_doc over the documented coefficients is reported,
W = (β̂ − β_doc)' V⁻¹ (β̂ − β_doc) ~ χ²(m). Large differences point to a different sample, variable
transformation or estimator than documented — replication should be exact on the development data.""",
          references=(*_SPEC_REFS, "White (1980), Econometrica 48(4)",
                      "Newey & West (1987), Econometrica 55(3)"))
def coefficients(ctx: RunContext, target, features, model_kind="auto", date=None, cov_type="nonrobust",
                 hac_lags=None, alpha=0.05, expected_coefficients=None, expected_signs=None) -> Outcome:
    f = _fit(ctx, target, features, model_kind, date, cov_type, hac_lags)
    res = f.res
    tab = _coef_table(res, alpha)
    tables = {"Coefficients": tab}
    summary = {"model_kind": f.kind, "cov_type": cov_type, "n": f.n, "parameters": f.k}
    exp = {_canon(k, res.params.index): float(v) for k, v in (expected_coefficients or {}).items()}
    signs = {_canon(k, res.params.index): _sign(v) for k, v in (expected_signs or {}).items()}
    for k, v in exp.items():
        signs.setdefault(k, float(np.sign(v)))
    ci = np.asarray(res.conf_int(alpha))
    pos = {v: i for i, v in enumerate(res.params.index)}
    if exp or signs:
        rows = []
        for v in res.params.index:
            if v not in exp and v not in signs:
                continue
            i, est, se = pos[v], float(res.params[v]), float(res.bse[v])
            doc = exp.get(v, np.nan)
            rows.append({"variable": v, "estimate": est, "documented": doc, "difference": est - doc,
                         "relative_difference": (est - doc) / abs(doc) if np.isfinite(doc) and doc != 0
                         else np.nan,
                         "difference_in_SE": (est - doc) / se if se > 0 else np.nan,
                         "documented_inside_CI": bool(ci[i, 0] <= doc <= ci[i, 1]) if np.isfinite(doc) else None,
                         "expected_sign": {1.0: "+", -1.0: "-", 0.0: "0"}.get(signs.get(v), None),
                         "sign_holds": bool(np.sign(est) == signs[v]) if v in signs else None,
                         "p_value": float(res.pvalues[v])})
        comp = pd.DataFrame(rows)
        tables["Comparison with documented coefficients"] = comp
        summary["sign_mismatches"] = int((comp["sign_holds"] == False).sum())  # noqa: E712
        if exp:
            names = [v for v in res.params.index if v in exp]
            d = np.array([res.params[v] - exp[v] for v in names])
            V = np.asarray(res.cov_params().loc[names, names])
            W = float(d @ np.linalg.pinv(V) @ d)
            summary |= {"max_abs_difference": float(np.max(np.abs(d))),
                        "wald_equal_documented": W, "wald_df": len(names),
                        "wald_p_value": float(stats.chi2.sf(W, len(names)))}
    return Outcome(summary, tables, notes=f.notes, rows_used=f.n)


@register("econ.robust_se", "Classical vs robust standard errors (HC0–HC3, Newey–West HAC)",
          "Model specification", _REG,
          params=(*_spec(), P("hac_lags", "integer", required=False, help="Lags for HAC errors")),
          description="""Re-estimates the standard errors of every coefficient under the classical (i.i.d.)
assumption, White heteroskedasticity-consistent estimators HC0–HC3 (OLS; ML models only have the HC0 sandwich)
and the Newey–West HAC estimator (Bartlett kernel). Reports each SE, the ratio robust/classical and the p-value
of H0: β = 0 under each. Ratios far from 1 mean the classical inference in the documentation is not reliable;
coefficients whose significance depends on the estimator deserve attention. HC3 is preferred in small samples
(Long & Ervin 2000); HAC is the relevant one for time-series satellite models.""",
          references=("White (1980), Econometrica 48(4)", "MacKinnon & White (1985), J. Econometrics 29(3)",
                      "Newey & West (1987), Econometrica 55(3)",
                      "Long & Ervin (2000), The American Statistician 54(3)"))
def robust_se(ctx: RunContext, target, features, model_kind="auto", date=None, hac_lags=None) -> Outcome:
    base = _fit(ctx, target, features, model_kind, date)
    lags = hac_lags if hac_lags is not None else _default_hac_lags(base.n)
    covs = ["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"] if base.kind == "ols" else ["nonrobust", "HC0", "HAC"]
    rows = []
    for c in covs:
        r = _fit_model(base.y, base.X, base.kind, c, lags)
        for v in r.params.index:
            rows.append({"variable": v, "cov_type": c, "estimate": float(r.params[v]),
                         "std_error": float(r.bse[v]), "ratio_to_classical": float(r.bse[v] / base.res.bse[v]),
                         "p_value": float(r.pvalues[v])})
    long = pd.DataFrame(rows)
    wide = long.pivot(index="variable", columns="cov_type", values="std_error").reindex(
        index=list(base.res.params.index), columns=covs).reset_index()
    wide.columns = ["variable"] + [f"SE_{c}" for c in covs]
    notes = base.notes + [f"HAC with {lags} lags (Bartlett kernel)" + ("" if date else
                          "; row order taken as time order") + "."]
    if base.kind != "ols":
        notes.append("HC1–HC3 are OLS corrections and are not reported for ML models.")
    ratio = long[long["cov_type"] != "nonrobust"]["ratio_to_classical"]
    return Outcome({"model_kind": base.kind, "n": base.n, "hac_lags": lags,
                    "max_ratio_robust_to_classical": float(ratio.max()),
                    "min_ratio_robust_to_classical": float(ratio.min())},
                   {"Standard errors": wide, "Detail": long}, notes=notes, rows_used=base.n)


# ── goodness of fit ────────────────────────────────────────────────────────

@register("econ.goodness_of_fit", "Goodness of fit (R², AIC/BIC, log-likelihood, pseudo-R², LR test)",
          "Goodness of fit", _REG, params=_spec(),
          description="""OLS: R², adjusted R², overall F-test (H0: all slopes are zero), RMSE, log-likelihood,
AIC, BIC. Logit / probit: log-likelihood of the model and of the intercept-only model, likelihood-ratio test
LR = 2(LL − LL₀) ~ χ²(k−1) (H0: all slopes are zero), McFadden pseudo-R² = 1 − LL/LL₀, adjusted McFadden
1 − (LL − K)/LL₀, Cox–Snell and Nagelkerke R², AIC, BIC. Fractional logit: quasi-log-likelihood, deviance,
Pearson χ², and the squared correlation of fitted and actual values. Information criteria are comparable only
between models fitted on the same observations.""",
          references=(*_SPEC_REFS, "McFadden (1974), Conditional logit analysis of qualitative choice behavior",
                      "Nagelkerke (1991), Biometrika 78(3)", "Akaike (1974), IEEE TAC 19(6)",
                      "Schwarz (1978), Annals of Statistics 6(2)"))
def goodness_of_fit(ctx: RunContext, target, features, model_kind="auto", date=None) -> Outcome:
    f = _fit(ctx, target, features, model_kind, date)
    r, n, k = f.res, f.n, f.k
    if f.kind == "ols":
        s = {"R2": r.rsquared, "adj_R2": r.rsquared_adj, "F_stat": r.fvalue, "F_p_value": r.f_pvalue,
             "RMSE": float(np.sqrt(r.ssr / n)), "residual_SE": float(np.sqrt(r.scale)),
             "log_likelihood": r.llf, "AIC": r.aic, "BIC": r.bic}
    elif f.kind in ("logit", "probit"):
        ll, ll0 = float(r.llf), float(r.llnull)
        cs = 1 - math.exp(2 * (ll0 - ll) / n)
        s = {"log_likelihood": ll, "log_likelihood_null": ll0, "LR_stat": float(r.llr),
             "LR_df": int(r.df_model), "LR_p_value": float(r.llr_pvalue),
             "McFadden_R2": 1 - ll / ll0, "McFadden_adj_R2": 1 - (ll - k) / ll0,
             "Cox_Snell_R2": cs, "Nagelkerke_R2": cs / (1 - math.exp(2 * ll0 / n)),
             "AIC": r.aic, "BIC": float(-2 * ll + k * math.log(n))}
    else:
        fitted = np.asarray(r.fittedvalues)
        s = {"quasi_log_likelihood": float(r.llf), "deviance": float(r.deviance),
             "null_deviance": float(r.null_deviance), "deviance_R2": 1 - r.deviance / r.null_deviance,
             "pearson_chi2": float(r.pearson_chi2),
             "corr2_fitted_actual": float(np.corrcoef(fitted, f.y)[0, 1] ** 2),
             "RMSE": float(np.sqrt(np.mean((f.y - fitted) ** 2)))}
    s = {"model_kind": f.kind, "n": n, "parameters": k} | {a: float(b) for a, b in s.items()}
    tab = pd.DataFrame({"statistic": list(s), "value": [str(v) if isinstance(v, str) else v for v in s.values()]})
    notes = list(f.notes)
    if f.kind in ("logit", "probit"):
        notes.append("BIC = −2LL + K·ln(n) with K parameters including the constant.")
    return Outcome(s, {"Fit statistics": tab}, notes=notes, rows_used=n)


# ── residual diagnostics ───────────────────────────────────────────────────

def _residuals(ctx: RunContext, target, features, model_kind, date, residuals):
    """(residual series in time order, fit or None, notes). Supplied residuals take precedence."""
    if residuals:
        sub, dropped = _prepare(ctx, [residuals], date)
        return sub[residuals], None, dropped_note(dropped)
    if not (target and features):
        raise ValueError("Give `residuals` or both `target` and `features`.")
    f = _fit(ctx, target, features, model_kind, date, allowed=("ols",))
    return f.res.resid, f, list(f.notes)


_RESID = (*_spec(required=False),
          P("residuals", required=False, help="Column of model residuals (instead of refitting)"))


@register("econ.residual_normality", "Residual normality (Jarque–Bera, Shapiro–Wilk, Anderson–Darling, D'Agostino)",
          "Residual diagnostics", _REG, params=_RESID,
          description="""H0: the residuals are normally distributed. Jarque–Bera JB = n/6·(S² + (K−3)²/4) ~ χ²(2);
Shapiro–Wilk W (scipy; exact algorithm up to n = 5000, approximate p beyond); Anderson–Darling A² with the
Stephens adjustment for estimated mean and variance (statsmodels normal_ad); D'Agostino–Pearson K² (skewness and
kurtosis combined). Residuals come from an OLS refit of the specification or from a supplied residual column.
Normality matters for exact small-sample t/F inference and for prediction intervals of satellite models; with
large n tiny departures are 'significant', so read skewness and excess kurtosis alongside the p-values.""",
          references=("Jarque & Bera (1987), International Statistical Review 55(2)",
                      "Shapiro & Wilk (1965), Biometrika 52(3-4)",
                      "Anderson & Darling (1954), JASA 49(268); Stephens (1974), JASA 69(347)",
                      "D'Agostino & Pearson (1973), Biometrika 60(3)"))
def residual_normality(ctx: RunContext, target=None, features=None, model_kind="auto", date=None,
                       residuals=None) -> Outcome:
    from statsmodels.stats.diagnostic import normal_ad
    from statsmodels.stats.stattools import jarque_bera
    e, f, notes = _residuals(ctx, target, features, model_kind, date, residuals)
    e = np.asarray(e, float)
    n = len(e)
    if n < 8:
        raise NotApplicable(f"{n} residuals; normality tests need at least 8.")
    if np.std(e) == 0:
        raise NotApplicable("Residuals have no variance.")
    jb, jbp, skew, kurt = jarque_bera(e)
    sw = stats.shapiro(e)
    ad, adp = normal_ad(e)
    k2 = stats.normaltest(e)
    rows = [{"test": "Jarque–Bera", "statistic": float(jb), "p_value": float(jbp)},
            {"test": "Shapiro–Wilk", "statistic": float(sw.statistic), "p_value": float(sw.pvalue)},
            {"test": "Anderson–Darling (Stephens)", "statistic": float(ad), "p_value": float(adp)},
            {"test": "D'Agostino–Pearson K²", "statistic": float(k2.statistic), "p_value": float(k2.pvalue)}]
    if n > 5000:
        notes.append("Shapiro–Wilk p-value is approximate for n > 5000.")
    return Outcome({"n": n, "skewness": float(skew), "excess_kurtosis": float(kurt) - 3,
                    "jarque_bera": float(jb), "jarque_bera_p": float(jbp),
                    "shapiro_wilk_p": float(sw.pvalue), "anderson_darling_p": float(adp)},
                   {"Normality tests": pd.DataFrame(rows)}, notes=notes, rows_used=n)


@register("econ.heteroskedasticity", "Heteroskedasticity (Breusch–Pagan, White, Goldfeld–Quandt, ARCH-LM)",
          "Residual diagnostics", _REG,
          params=(*_spec(), P("sort_by", required=False,
                              help="Variable ordering the Goldfeld–Quandt split (default: fitted values)"),
                  P("drop_fraction", "number", default=0.2, help="Central share dropped in Goldfeld–Quandt"),
                  P("arch_lags", "integer", default=4, help="Lags of the ARCH-LM test")),
          description="""H0 for all tests: the OLS error variance is constant. Breusch–Pagan regresses squared
residuals on the regressors (Koenker's studentised LM = n·R², robust to non-normality, plus the original
Breusch–Pagan version); White adds squares and cross-products (detects any variance form, low power with many
regressors); Goldfeld–Quandt orders observations (by `sort_by` or the fitted values), drops a central share and
compares the residual variances of the two ends with an F test (two-sided); Engle's ARCH-LM regresses e²_t on its
own lags (time series: volatility clustering). Rejection means classical standard errors are unreliable —
compare with econ.robust_se.""",
          references=("Breusch & Pagan (1979), Econometrica 47(5)", "Koenker (1981), J. Econometrics 17(1)",
                      "White (1980), Econometrica 48(4)", "Goldfeld & Quandt (1965), JASA 60(310)",
                      "Engle (1982), Econometrica 50(4)"))
def heteroskedasticity(ctx: RunContext, target, features, model_kind="auto", date=None, sort_by=None,
                       drop_fraction=0.2, arch_lags=4) -> Outcome:
    from statsmodels.stats.diagnostic import het_arch, het_breuschpagan, het_goldfeldquandt, het_white
    f = _fit(ctx, target, features, model_kind, date, allowed=("ols",), extra=(sort_by,) if sort_by else ())
    e, X = f.res.resid.to_numpy(), f.X.to_numpy()
    rows = []
    lm, lmp, fv, fp = het_breuschpagan(e, X, robust=True)
    rows.append({"test": "Breusch–Pagan (Koenker studentised)", "statistic": lm, "df": f.k - 1, "p_value": lmp})
    lm0, lmp0, _, _ = het_breuschpagan(e, X, robust=False)
    rows.append({"test": "Breusch–Pagan (original)", "statistic": lm0, "df": f.k - 1, "p_value": lmp0})
    try:
        w, wp, _, _ = het_white(e, X)
        rows.append({"test": "White", "statistic": w, "df": np.nan, "p_value": wp})
    except Exception as exc:  # too many terms for n
        f.notes.append(f"White test not computed: {exc}")
    key = f.sub[sort_by].to_numpy() if sort_by else f.res.fittedvalues.to_numpy()
    order = np.argsort(key, kind="mergesort")
    gq = het_goldfeldquandt(f.y.to_numpy()[order], X[order], drop=drop_fraction, alternative="two-sided",
                            result_object=True)
    rows.append({"test": f"Goldfeld–Quandt (ordered by {sort_by or 'fitted values'}, two-sided)",
                 "statistic": float(gq.fval), "df": np.nan, "p_value": float(gq.pval)})
    if f.n > arch_lags + 5:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            a, ap, _, _ = het_arch(e, nlags=arch_lags)
        rows.append({"test": f"ARCH-LM ({arch_lags} lags)", "statistic": a, "df": arch_lags, "p_value": ap})
    t = pd.DataFrame(rows)
    t[["statistic", "p_value"]] = t[["statistic", "p_value"]].astype(float)
    notes = list(f.notes)
    if not date:
        notes.append("ARCH-LM uses the table's row order as the time order.")
    return Outcome({"n": f.n, "breusch_pagan_p": float(lmp), "white_p": float(t.loc[t["test"] == "White",
                    "p_value"].iloc[0]) if (t["test"] == "White").any() else float("nan"),
                    "goldfeld_quandt_p": float(gq.pval)},
                   {"Heteroskedasticity tests": t}, notes=notes, rows_used=f.n)


@register("econ.autocorrelation", "Residual autocorrelation (Durbin–Watson, Breusch–Godfrey, Ljung–Box)",
          "Residual diagnostics", _REG,
          params=(*_RESID, P("lags", "integer", default=4, help="Lag order for Breusch–Godfrey / Ljung–Box")),
          description="""H0: no serial correlation in the residuals (in time order). Durbin–Watson
d = Σ(e_t − e_{t−1})²/Σe_t² ≈ 2(1 − ρ̂₁) (no p-value: exact bounds depend on the design; invalid with a lagged
dependent variable); Breusch–Godfrey LM test of order p (regress e_t on X and p lagged residuals; valid with
lagged dependent variables; needs a refit); Ljung–Box Q and Box–Pierce at each lag 1..p (on supplied or refitted
residuals). Also reports the residual ACF. Autocorrelation biases classical standard errors (use HAC) and,
with a lagged dependent variable, the coefficients themselves.""",
          references=("Durbin & Watson (1950, 1951), Biometrika 37, 38", "Breusch (1978), Australian Economic Papers 17",
                      "Godfrey (1978), Econometrica 46(6)", "Ljung & Box (1978), Biometrika 65(2)",
                      "Box & Pierce (1970), JASA 65(332)"))
def autocorrelation(ctx: RunContext, target=None, features=None, model_kind="auto", date=None,
                    residuals=None, lags=4) -> Outcome:
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.tsa.stattools import acf
    e, f, notes = _residuals(ctx, target, features, model_kind, date, residuals)
    e = np.asarray(e, float)
    n = len(e)
    if lags < 1:
        raise ValueError("lags must be >= 1.")
    if n < lags + 5:
        raise NotApplicable(f"{n} residuals for {lags} lags.")
    dw = float(durbin_watson(e))
    lb = acorr_ljungbox(e, lags=list(range(1, lags + 1)), boxpierce=True,
                        model_df=0, return_df=True)
    ac = acf(e, nlags=lags, fft=False)[1:]
    lagtab = pd.DataFrame({"lag": range(1, lags + 1), "acf": ac, "ljung_box_Q": lb["lb_stat"].to_numpy(),
                           "ljung_box_p": lb["lb_pvalue"].to_numpy(), "box_pierce_Q": lb["bp_stat"].to_numpy(),
                           "box_pierce_p": lb["bp_pvalue"].to_numpy()})
    summary = {"n": n, "durbin_watson": dw, "rho1": float(ac[0]),
               f"ljung_box_Q_{lags}": float(lb["lb_stat"].iloc[-1]), f"ljung_box_p_{lags}": float(lb["lb_pvalue"].iloc[-1])}
    if f is not None:
        bg = acorr_breusch_godfrey(f.res, nlags=lags, result_object=True)
        summary |= {"breusch_godfrey_LM": float(bg.lm), "breusch_godfrey_p": float(bg.lmpval),
                    "breusch_godfrey_F": float(bg.fval), "breusch_godfrey_F_p": float(bg.fpval)}
    else:
        notes.append("Breusch–Godfrey needs the regressors: give target and features to include it.")
    if not date:
        notes.append("Row order taken as the time order (no date given).")
    return Outcome(summary, {"Autocorrelation by lag": lagtab}, notes=notes, rows_used=n)


@register("econ.specification", "Functional-form tests (Ramsey RESET, Harvey–Collier, Rainbow, link test)",
          "Model specification", _REG,
          params=(*_spec(), P("reset_power", "integer", default=3,
                              help="RESET adds fitted values to powers 2..reset_power")),
          description="""OLS: Ramsey RESET (H0: no omitted non-linearity; F test on ŷ², …, ŷ^p added to the
regression); Harvey–Collier (H0: linear; t test that the mean of recursive residuals is zero, in data/time order);
Utts' Rainbow test (H0: the fit on the central half of the data equals the full fit; F test). Logit / probit:
Pregibon's link test (refit y on the linear predictor and its square; H0: coefficient of the square is zero,
i.e. the link and linear index are correctly specified). Rejection suggests missing transformations,
interactions or variables.""",
          references=("Ramsey (1969), JRSS B 31(2)", "Harvey & Collier (1977), J. Econometrics 6(1)",
                      "Utts (1982), Communications in Statistics A 11(24)",
                      "Pregibon (1980), Applied Statistics 29(1)"))
def specification(ctx: RunContext, target, features, model_kind="auto", date=None, reset_power=3) -> Outcome:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import linear_harvey_collier, linear_rainbow, linear_reset
    f = _fit(ctx, target, features, model_kind, date)
    rows, notes = [], list(f.notes)
    if f.kind == "ols":
        if reset_power < 2:
            raise ValueError("reset_power must be >= 2.")
        rr = linear_reset(f.res, power=reset_power, test_type="fitted", use_f=True)
        rows.append({"test": f"Ramsey RESET (ŷ^2..ŷ^{reset_power})", "statistic": float(rr.fvalue),
                     "p_value": float(rr.pvalue), "df": f"({int(rr.df_num)}, {int(rr.df_denom)})"})
        try:
            hc = linear_harvey_collier(f.res)
            rows.append({"test": "Harvey–Collier", "statistic": float(hc.statistic), "p_value": float(hc.pvalue),
                         "df": str(int(hc.df))})
        except Exception as exc:
            notes.append(f"Harvey–Collier not computed: {exc}")
        rb, rbp = linear_rainbow(f.res)
        rows.append({"test": "Rainbow (central 50%)", "statistic": float(rb), "p_value": float(rbp), "df": ""})
    elif f.kind in ("logit", "probit"):
        eta = f.res.fittedvalues.to_numpy() if hasattr(f.res.fittedvalues, "to_numpy") else f.res.fittedvalues
        Z = sm.add_constant(np.column_stack([eta, eta ** 2]))
        M = sm.Logit if f.kind == "logit" else sm.Probit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = M(f.y.to_numpy(), Z).fit(disp=0, maxiter=200)
        rows.append({"test": "Pregibon link test (ŷ² term)", "statistic": float(lr.tvalues[2]),
                     "p_value": float(lr.pvalues[2]), "df": ""})
        rows.append({"test": "Pregibon link test (ŷ term, should be significant)", "statistic": float(lr.tvalues[1]),
                     "p_value": float(lr.pvalues[1]), "df": ""})
    else:
        raise NotApplicable("Specification tests are implemented for OLS, logit and probit.")
    if f.kind == "ols" and not date:
        notes.append("Harvey–Collier uses the table's row order (no date given).")
    t = pd.DataFrame(rows)
    return Outcome({"model_kind": f.kind, "n": f.n} | {r["test"]: r["p_value"] for r in rows},
                   {"Specification tests": t}, notes=notes, rows_used=f.n)


# ── structural breaks ──────────────────────────────────────────────────────

def _split_position(f, ctx, date, break_at, break_index) -> int:
    if break_index is not None:
        return int(break_index)
    if break_at is None:
        raise ValueError("Give break_at (a date/period value) or break_index (row position after sorting).")
    if not date:
        raise ValueError("break_at needs the `date` column; otherwise use break_index.")
    key = f.sub["__order__"]
    if pd.api.types.is_datetime64_any_dtype(key):
        b = pd.Timestamp(break_at)
    elif pd.api.types.is_numeric_dtype(key):
        b = float(break_at)
    else:
        b = str(break_at)
    return int((key < b).sum())


@register("econ.chow_test", "Chow test for a structural break at a known date",
          "Structural stability", _REG,
          params=(*_spec(), P("break_at", "string", required=False,
                              help="First date/period of the second regime (needs `date`)"),
                  P("break_index", "integer", required=False,
                    help="Row position (0-based, after sorting) where the second regime starts")),
          description="""H0: the coefficients are the same before and after the break. OLS:
F = [(SSR_pooled − SSR₁ − SSR₂)/k] / [(SSR₁ + SSR₂)/(n − 2k)] ~ F(k, n − 2k), assuming equal error variances
in both regimes. When the second regime has fewer observations than parameters, the Chow predictive (forecast)
test F = [(SSR_pooled − SSR₁)/n₂] / [SSR₁/(n₁ − k)] ~ F(n₂, n₁ − k) is used. Logit / probit: likelihood-ratio
version LR = 2(LL₁ + LL₂ − LL_pooled) ~ χ²(k). The break date must be chosen without looking at the data
(e.g. a crisis or definition change); for an unknown break use econ.sup_f_break.""",
          references=("Chow (1960), Econometrica 28(3)", "Greene (2018), Econometric Analysis, 8th ed., ch. 6"))
def chow_test(ctx: RunContext, target, features, model_kind="auto", date=None, break_at=None,
              break_index=None) -> Outcome:
    f = _fit(ctx, target, features, model_kind, date)
    if f.kind not in ("ols", "logit", "probit"):
        raise NotApplicable("Chow test implemented for OLS, logit and probit.")
    pos = _split_position(f, ctx, date, break_at, break_index)
    n, k = f.n, f.k
    y, X = f.y.to_numpy(), f.X.to_numpy()
    n1, n2 = pos, n - pos
    if n1 < k + 1 or n2 < 1:
        raise NotApplicable(f"Split gives {n1} and {n2} observations; the first regime needs > {k}.")
    labels = _labels(ctx, f.sub, date)
    first2 = labels.iloc[pos]
    if f.kind == "ols":
        ssr_p = float(f.res.ssr)
        r1 = _fit_model(y[:pos], X[:pos], "ols")
        ssr1 = float(r1.ssr)
        if n2 > k:
            r2 = _fit_model(y[pos:], X[pos:], "ols")
            ssr2 = float(r2.ssr)
            F = ((ssr_p - ssr1 - ssr2) / k) / ((ssr1 + ssr2) / (n - 2 * k))
            p, df1, df2, test = stats.f.sf(F, k, n - 2 * k), k, n - 2 * k, "Chow breakpoint F"
        else:
            ssr2 = float("nan")
            F = ((ssr_p - ssr1) / n2) / (ssr1 / (n1 - k))
            p, df1, df2, test = stats.f.sf(F, n2, n1 - k), n2, n1 - k, "Chow predictive F"
        rows = [{"sample": "pooled", "n": n, "SSR": ssr_p}, {"sample": "regime 1", "n": n1, "SSR": ssr1},
                {"sample": "regime 2", "n": n2, "SSR": ssr2}]
        summary = {"test": test, "F_stat": float(F), "df_num": df1, "df_denom": df2, "p_value": float(p)}
    else:
        if min(n1, n2) < k + 1:
            raise NotApplicable("Both regimes need more observations than parameters for the LR Chow test.")
        try:
            r1 = _fit_model(y[:pos], X[:pos], f.kind)
            r2 = _fit_model(y[pos:], X[pos:], f.kind)
        except Exception as exc:
            raise NotApplicable(f"Regime model could not be fitted: {exc}")
        LR = 2 * (r1.llf + r2.llf - f.res.llf)
        rows = [{"sample": "pooled", "n": n, "log_likelihood": float(f.res.llf)},
                {"sample": "regime 1", "n": n1, "log_likelihood": float(r1.llf)},
                {"sample": "regime 2", "n": n2, "log_likelihood": float(r2.llf)}]
        summary = {"test": "Likelihood-ratio Chow", "LR_stat": float(LR), "df": k,
                   "p_value": float(stats.chi2.sf(LR, k))}
    summary |= {"n_regime1": n1, "n_regime2": n2, "second_regime_starts": str(first2)}
    return Outcome(summary, {"Regimes": pd.DataFrame(rows)}, notes=f.notes, rows_used=n)


def _segment_ssr(X: np.ndarray, Y: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """SSR₁ + SSR₂ for every split point in `cand` (first τ obs vs rest) and every column of Y."""
    XX = np.cumsum(X[:, :, None] * X[:, None, :], axis=0)
    XY = np.cumsum(X[:, :, None] * Y[:, None, :], axis=0)
    YY = np.cumsum(Y ** 2, axis=0)
    out = np.empty((len(cand), Y.shape[1]))
    for i, t in enumerate(cand):
        a1, a2 = XX[t - 1], XX[-1] - XX[t - 1]
        b1, b2 = XY[t - 1], XY[-1] - XY[t - 1]
        q1 = np.einsum("kb,kb->b", b1, np.linalg.pinv(a1) @ b1)
        q2 = np.einsum("kb,kb->b", b2, np.linalg.pinv(a2) @ b2)
        out[i] = (YY[t - 1] - q1) + (YY[-1] - YY[t - 1] - q2)
    return out


@register("econ.sup_f_break", "Unknown-date structural break (Quandt–Andrews sup-F, bootstrap p-value)",
          "Structural stability", _TS,
          params=(*_spec(), P("trim", "number", default=0.15, help="Share trimmed at each end of the sample"),
                  P("n_boot", "integer", default=499, help="Fixed-regressor bootstrap replications")),
          description="""H0: no structural break in any OLS coefficient over the central (1 − 2·trim) part of the
sample. Computes the Chow F statistic at every candidate break date and takes the supremum (Quandt likelihood
ratio / Andrews sup-F). Because the break date is estimated, the pointwise F distribution does not apply; the
p-value is from Hansen's (2000) fixed-regressor bootstrap (y* = ê·η, η ~ N(0,1), regressors held fixed),
which is valid under heteroskedasticity. Reports the most likely break date and the F path.""",
          references=("Quandt (1960), JASA 55(290)", "Andrews (1993), Econometrica 61(4)",
                      "Hansen (2000), J. Econometrics 97(1)"))
def sup_f_break(ctx: RunContext, target, features, model_kind="auto", date=None, trim=0.15,
                n_boot=499) -> Outcome:
    f = _fit(ctx, target, features, model_kind, date, allowed=("ols",))
    if not 0 < trim < 0.5:
        raise ValueError("trim must be in (0, 0.5).")
    n, k = f.n, f.k
    X, y = f.X.to_numpy(), f.y.to_numpy()
    lo, hi = max(int(math.ceil(trim * n)), k + 1), min(int(math.floor((1 - trim) * n)), n - k - 1)
    cand = np.arange(lo, hi + 1)
    if len(cand) < 1:
        raise NotApplicable(f"No admissible break dates with n = {n}, k = {k}, trim = {trim}.")
    ssr_p = float(f.res.ssr)
    ssr_u = _segment_ssr(X, y[:, None], cand)[:, 0]
    F = ((ssr_p - ssr_u) / k) / (ssr_u / (n - 2 * k))
    best = int(np.argmax(F))
    e = f.res.resid.to_numpy()
    Ys = e[:, None] * ctx.rng(11).standard_normal((n, n_boot))
    bp = np.linalg.pinv(X) @ Ys
    ssr_pb = ((Ys - X @ bp) ** 2).sum(axis=0)
    ssr_ub = _segment_ssr(X, Ys, cand)
    Fb = ((ssr_pb[None, :] - ssr_ub) / k) / (ssr_ub / (n - 2 * k))
    supb = Fb.max(axis=0)
    p = (1 + int((supb >= F[best]).sum())) / (n_boot + 1)
    labels = _labels(ctx, f.sub, date).to_numpy()
    path = pd.DataFrame({"break_position": cand, "second_regime_starts": labels[cand], "F_stat": F,
                         "pointwise_p_value": stats.f.sf(F, k, n - 2 * k)})
    return Outcome({"sup_F": float(F[best]), "bootstrap_p_value": float(p), "n_boot": n_boot,
                    "break_position": int(cand[best]), "second_regime_starts": str(labels[cand[best]]),
                    "candidates": len(cand), "n": n},
                   {"F statistic by candidate break": path},
                   notes=f.notes + ["Pointwise p-values in the table ignore the search over dates; use the "
                                    "bootstrap p-value for the sup-F test."], rows_used=n)


def _recursive_residuals(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Standardised one-step-ahead recursive residuals w_t, t = k..n-1 (Brown, Durbin & Evans 1975)."""
    n, k = X.shape
    w = []
    for t in range(k, n):
        Xt, yt = X[:t], y[:t]
        A = np.linalg.pinv(Xt.T @ Xt)
        b = A @ Xt.T @ yt
        x = X[t]
        w.append((y[t] - x @ b) / math.sqrt(1 + x @ A @ x))
    return np.asarray(w)


def _bde_pvalue(a: float) -> float:
    """Asymptotic P(CUSUM crosses the a-lines) = 2[1 − Φ(3a) + exp(−4a²)Φ(a)] (Brown, Durbin & Evans 1975)."""
    return float(min(1.0, 2 * (stats.norm.sf(3 * a) + math.exp(-4 * a * a) * stats.norm.cdf(a))))


@register("econ.cusum", "CUSUM stability tests (recursive and OLS-residual CUSUM, CUSUM of squares)",
          "Structural stability", _TS + ("lgd", "ead", "pd"), params=_spec(),
          description="""H0: coefficients (and variance) are stable over time. Brown–Durbin–Evans CUSUM of
standardised recursive residuals W_r = Σ w_t/σ̂ with the 5% boundary ±[0.948·√(n−k) + 2·0.948·(r−k)/√(n−k)];
reported as the maximal scaled excursion a* and its asymptotic p-value 2[1 − Φ(3a*) + e^(−4a*²)Φ(a*)].
Ploberger–Krämer OLS-residual CUSUM (sup of the scaled cumulated OLS residuals, Brownian-bridge p-value,
statsmodels breaks_cusumolsresid). CUSUM of squares S_r = Σw²_t/Σw² against its mean line (r−k)/(n−k):
reported as the maximal deviation (critical values: Durbin 1969 tables). Rows are in time order.""",
          references=("Brown, Durbin & Evans (1975), JRSS B 37(2)", "Ploberger & Krämer (1992), Econometrica 60(2)",
                      "Durbin (1969), Biometrika 56(1)"))
def cusum(ctx: RunContext, target, features, model_kind="auto", date=None) -> Outcome:
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    f = _fit(ctx, target, features, model_kind, date, allowed=("ols",), min_extra_obs=5)
    n, k = f.n, f.k
    y, X = f.y.to_numpy(), f.X.to_numpy()
    w = _recursive_residuals(y, X)
    sig = float(np.std(w, ddof=1))
    if sig == 0:
        raise NotApplicable("Recursive residuals have no variance.")
    W = np.cumsum(w) / sig
    r = np.arange(1, len(w) + 1)                      # r − k
    scale = math.sqrt(n - k) + 2 * r / math.sqrt(n - k)
    a_star = float(np.max(np.abs(W) / scale))
    S = np.cumsum(w ** 2) / np.sum(w ** 2)
    dev = S - r / (n - k)
    ols_stat, ols_p, _ = breaks_cusumolsresid(f.res.resid.to_numpy(), ddof=k)
    labels = _labels(ctx, f.sub, date).to_numpy()[k:]
    path = pd.DataFrame({"date": labels, "recursive_residual": w, "cusum": W, "bound_5pct": 0.948 * scale,
                         "cusum_sq": S, "cusum_sq_mean_line": r / (n - k)})
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_scatter(x=labels, y=W, name="CUSUM")
    fig.add_scatter(x=labels, y=0.948 * scale, name="5% bound", line={"dash": "dash"})
    fig.add_scatter(x=labels, y=-0.948 * scale, name="−5% bound", line={"dash": "dash"})
    fig.update_layout(title="CUSUM of recursive residuals", xaxis_title="date", yaxis_title="W_r")
    notes = list(f.notes) + ([] if date else ["Row order taken as the time order (no date given)."])
    return Outcome({"n": n, "cusum_max_scaled_excursion": a_star, "cusum_p_value": _bde_pvalue(a_star),
                    "ols_cusum_stat": float(ols_stat), "ols_cusum_p_value": float(ols_p),
                    "cusum_sq_max_deviation": float(np.max(np.abs(dev)))},
                   {"CUSUM path": path}, figures=[fig], notes=notes, rows_used=n)


@register("econ.recursive_coefficients", "Recursive (expanding-window) coefficient estimates",
          "Structural stability", _REG,
          params=(*_spec(), P("min_obs", "integer", required=False,
                              help="First window size (default max(k + 10, 30% of n))"),
                  P("step", "integer", default=1, help="Re-estimate every `step` observations")),
          description="""Re-estimates the specification on expanding windows (first min_obs observations, then
adding `step` at a time, in time order) and reports each coefficient's path. For each coefficient: the range of
estimates, the share of windows where the sign equals the full-sample sign, and the share of windows where the
estimate lies inside the full-sample 95% CI. Drifting or sign-switching coefficients indicate parameter
instability that a single full-sample fit hides.""",
          references=("Brown, Durbin & Evans (1975), JRSS B 37(2)", "Greene (2018), Econometric Analysis, 8th ed."))
def recursive_coefficients(ctx: RunContext, target, features, model_kind="auto", date=None, min_obs=None,
                           step=1) -> Outcome:
    f = _fit(ctx, target, features, model_kind, date)
    n, k = f.n, f.k
    m0 = int(min_obs) if min_obs else max(k + 10, int(math.ceil(0.3 * n)))
    if m0 >= n or m0 <= k:
        raise NotApplicable(f"min_obs = {m0} leaves no recursive windows (n = {n}, k = {k}).")
    ends = list(range(m0, n + 1, max(int(step), 1)))
    if ends[-1] != n:
        ends.append(n)
    labels = _labels(ctx, f.sub, date).to_numpy()
    y, X = f.y, f.X
    rows, failed = [], 0
    for e in ends:
        try:
            r = _fit_model(y.iloc[:e], X.iloc[:e], f.kind)
            if not np.all(np.isfinite(r.params)):
                raise ValueError
        except Exception:
            failed += 1
            continue
        for v in r.params.index:
            rows.append({"window_end": labels[e - 1], "n_obs": e, "variable": v,
                         "estimate": float(r.params[v]), "std_error": float(r.bse[v])})
    path = pd.DataFrame(rows)
    if path.empty:
        raise NotApplicable("No recursive window could be estimated.")
    ci = f.res.conf_int(0.05)
    summ = []
    for v in f.res.params.index:
        p = path.loc[path["variable"] == v, "estimate"]
        full = float(f.res.params[v])
        summ.append({"variable": v, "full_sample": full, "min": float(p.min()), "max": float(p.max()),
                     "share_same_sign": float((np.sign(p) == np.sign(full)).mean()),
                     "share_inside_full_CI": float(((p >= ci.loc[v, 0]) & (p <= ci.loc[v, 1])).mean())})
    import plotly.graph_objects as go
    fig = go.Figure()
    for v in features:
        p = path[path["variable"] == v]
        fig.add_scatter(x=p["window_end"], y=p["estimate"], name=str(v))
    fig.update_layout(title="Recursive coefficient estimates", xaxis_title="window end", yaxis_title="estimate")
    notes = list(f.notes) + ([f"{failed} windows could not be estimated."] if failed else [])
    return Outcome({"windows": len(ends) - failed, "first_window_obs": m0, "n": n,
                    "min_share_same_sign": float(min(s["share_same_sign"] for s in summ))},
                   {"Coefficient stability": pd.DataFrame(summ), "Recursive estimates": path},
                   figures=[fig], notes=notes, rows_used=n)


# ── stationarity and cointegration ─────────────────────────────────────────

def phillips_perron(x: np.ndarray, regression: str = "c", lags: int | None = None) -> dict:
    """Phillips–Perron Z_tau and Z_alpha, following the arch package's formulas.

    Regression y_t = ρ y_{t−1} + deterministic + u_t; long-run variance by Newey–West (Bartlett) with
    `lags` (default ceil(12·(T/100)^(1/4))). The Z_tau p-value uses MacKinnon's (1994/2010) DF surface.
    """
    import statsmodels.api as sm
    from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
    x = np.asarray(x, float)
    T = len(x)
    if lags is None:
        lags = int(math.ceil(12 * (T / 100) ** 0.25))
    y, ylag = x[1:], x[:-1]
    n = len(y)
    cols = [ylag]
    if regression in ("c", "ct"):
        cols.append(np.ones(n))
    if regression == "ct":
        cols.append(np.arange(1, n + 1, dtype=float))
    rhs = np.column_stack(cols)
    res = sm.OLS(y, rhs).fit()
    k = rhs.shape[1]
    u = res.resid
    lags = min(int(lags), n - 1)
    lam2 = u @ u / n
    for j in range(1, lags + 1):
        lam2 += 2 * (1 - j / (lags + 1)) * (u[j:] @ u[:-j]) / n
    s2 = u @ u / (n - k)
    gamma0 = s2 * (n - k) / n
    sigma, rho = res.bse[0], res.params[0]
    t = (rho - 1) / sigma
    lam = math.sqrt(lam2)
    z_tau = math.sqrt(gamma0 / lam2) * t - 0.5 * ((lam2 - gamma0) / lam) * (n * sigma / math.sqrt(s2))
    z_alpha = n * (rho - 1) - 0.5 * (n ** 2 * sigma ** 2 / s2) * (lam2 - gamma0)
    crit = mackinnoncrit(N=1, regression=regression, nobs=n)
    return {"z_tau": float(z_tau), "p_value": float(mackinnonp(z_tau, regression=regression, N=1)),
            "z_alpha": float(z_alpha), "lags": lags, "crit_1pct": float(crit[0]), "crit_5pct": float(crit[1]),
            "crit_10pct": float(crit[2])}


@register("econ.unit_root", "Stationarity of each series (ADF, Phillips–Perron, KPSS)", "Stationarity", _TS,
          params=(P("columns", "columns", help="Series to test (target and macro drivers)"),
                  P("date", required=False, help="Time-order column"),
                  P("regression", "string", default="c", choices=("c", "ct", "n"),
                    help="Deterministic terms: constant, constant+trend, none (KPSS uses c for n)"),
                  P("differences", "boolean", default=False, help="Also test first differences")),
          description="""For every series: Augmented Dickey–Fuller (H0: unit root; lag length by AIC; MacKinnon
p-values); Phillips–Perron Z_tau (H0: unit root; Newey–West long-run variance, Bartlett lags ceil(12(T/100)^¼);
same MacKinnon distribution as ADF) and Z_alpha (statistic only); KPSS (H0: stationary around a level or trend;
automatic Hobijn et al. bandwidth). Reading them jointly: ADF/PP reject and KPSS does not → stationary;
the reverse → unit root; both reject or neither → inconclusive. Regressions of non-stationary levels on each
other are spurious unless the series cointegrate (econ.engle_granger, econ.johansen).""",
          references=("Dickey & Fuller (1979), JASA 74(366); Said & Dickey (1984), Biometrika 71(3)",
                      "Phillips & Perron (1988), Biometrika 75(2)",
                      "Kwiatkowski, Phillips, Schmidt & Shin (1992), J. Econometrics 54(1-3)",
                      "MacKinnon (2010), Critical values for cointegration tests, Queen's Economics WP 1227"))
def unit_root(ctx: RunContext, columns, date=None, regression="c", differences=False) -> Outcome:
    from statsmodels.tsa.stattools import adfuller, kpss
    sub, dropped = _prepare(ctx, columns, date)
    rows, notes = [], dropped_note(dropped)
    for c in columns:
        series = {"level": sub[c].to_numpy()}
        if differences:
            series["first difference"] = np.diff(sub[c].to_numpy())
        for form, x in series.items():
            if len(x) < 20 or np.std(x) == 0:
                rows.append({"series": c, "form": form, "n": len(x), "note": "too short or constant"})
                continue
            adf = adfuller(x, regression=regression, autolag="AIC", result_object=True)
            pp = phillips_perron(x, regression)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kp = kpss(x, regression="ct" if regression == "ct" else "c", nlags="auto", result_object=True)
            rows.append({"series": c, "form": form, "n": len(x), "ADF_stat": float(adf.statistic),
                         "ADF_p": float(adf.pvalue), "ADF_lags": int(adf.lags), "PP_Z_tau": pp["z_tau"],
                         "PP_p": pp["p_value"], "PP_Z_alpha": pp["z_alpha"], "PP_lags": pp["lags"],
                         "KPSS_stat": float(kp.statistic), "KPSS_p": float(kp.pvalue), "KPSS_lags": int(kp.lags),
                         "note": ""})
    t = pd.DataFrame(rows)
    if "ADF_stat" not in t:
        raise NotApplicable("All series are too short (< 20 observations) or constant.")
    notes += ["KPSS p-values are interpolated from the KPSS (1992) table and truncated to [0.01, 0.10].",
              "Phillips–Perron implemented in-house (the arch package is not available); Z_alpha has no "
              "p-value here."]
    if not date:
        notes.append("Row order taken as the time order (no date given).")
    ok = t.dropna(subset=["ADF_stat"])
    return Outcome({"series_tested": int(len(ok)), "n": len(sub), "regression": regression,
                    "max_ADF_p": float(ok["ADF_p"].max()), "min_KPSS_p": float(ok["KPSS_p"].min())},
                   {"Unit-root and stationarity tests": t}, notes=notes, rows_used=len(sub))


@register("econ.zivot_andrews", "Zivot–Andrews unit-root test with an endogenous break", "Stationarity", _TS,
          params=(P("columns", "columns", help="Series to test"), P("date", required=False),
                  P("regression", "string", default="c", choices=("c", "t", "ct"),
                    help="Break in intercept (c), trend (t) or both (ct)"),
                  P("trim", "number", default=0.15, help="Share trimmed at each end")),
          description="""H0: unit root without a break; H1: stationary with one structural break at an unknown
date (intercept, trend or both). The break date minimises the ADF t-statistic over the trimmed sample; p-values
and critical values from the Zivot & Andrews (1992) distribution (statsmodels). Use when ADF does not reject but
the series plausibly has a level shift (e.g. a crisis or a definition change), which biases ADF towards the unit
root.""",
          references=("Zivot & Andrews (1992), JBES 10(3)", "statsmodels.tsa.stattools.zivot_andrews"))
def zivot_andrews_test(ctx: RunContext, columns, date=None, regression="c", trim=0.15) -> Outcome:
    from statsmodels.tsa.stattools import zivot_andrews
    sub, dropped = _prepare(ctx, columns, date)
    labels = _labels(ctx, sub, date).to_numpy()
    rows = []
    for c in columns:
        x = sub[c].to_numpy()
        if len(x) < 30 or np.std(x) == 0:
            rows.append({"series": c, "n": len(x), "note": "too short (< 30) or constant"})
            continue
        stat, p, crit, lag, bp = zivot_andrews(x, trim=trim, regression=regression, autolag="AIC")
        rows.append({"series": c, "n": len(x), "ZA_stat": float(stat), "p_value": float(p),
                     "crit_1pct": float(crit["1%"]), "crit_5pct": float(crit["5%"]),
                     "crit_10pct": float(crit["10%"]), "lags": int(lag), "break_position": int(bp),
                     "break_date": str(labels[int(bp)]), "note": ""})
    t = pd.DataFrame(rows)
    if "ZA_stat" not in t:
        raise NotApplicable("All series are too short (< 30 observations) or constant.")
    ok = t.dropna(subset=["ZA_stat"])
    return Outcome({"series_tested": len(ok), "n": len(sub), "min_p_value": float(ok["p_value"].min())},
                   {"Zivot–Andrews": t}, notes=dropped_note(dropped), rows_used=len(sub))


@register("econ.engle_granger", "Engle–Granger cointegration test", "Stationarity", _TS,
          params=(P("target", help="Dependent series y"), P("features", "columns", help="Regressor series"),
                  P("date", required=False),
                  P("trend", "string", default="c", choices=("c", "ct", "n"),
                    help="Deterministic terms in the cointegrating regression")),
          description="""H0: no cointegration. Step 1: static OLS of y on the regressors (long-run relation);
step 2: ADF test on its residuals with lag length by AIC. The statistic is compared with MacKinnon (1994, 2010)
Engle–Granger critical values for N = 1 + number of regressors (not the plain DF tables). Rejection means a
levels regression of integrated series is a valid long-run relation; otherwise the model should be in
differences or an error-correction form.""",
          references=("Engle & Granger (1987), Econometrica 55(2)", "MacKinnon (2010), Queen's Economics WP 1227",
                      "statsmodels.tsa.stattools.coint"))
def engle_granger(ctx: RunContext, target, features, date=None, trend="c") -> Outcome:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint
    sub, dropped = _prepare(ctx, [target, *features], date)
    if len(sub) < 20:
        raise NotApplicable(f"{len(sub)} observations; need at least 20.")
    y, X = sub[target].to_numpy(), sub[list(features)].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, crit = coint(y, X, trend=trend, autolag="aic", return_results=False)
    Xc = X if trend == "n" else sm.add_constant(X, has_constant="add")
    if trend == "ct":
        Xc = np.column_stack([Xc, np.arange(1, len(y) + 1)])
    lr = sm.OLS(y, Xc).fit()
    names = (([] if trend == "n" else ["const"]) + list(features) + (["trend"] if trend == "ct" else []))
    tab = pd.DataFrame({"variable": names, "long_run_coefficient": lr.params})
    return Outcome({"EG_stat": float(stat), "p_value": float(p), "crit_1pct": float(crit[0]),
                    "crit_5pct": float(crit[1]), "crit_10pct": float(crit[2]), "n": len(sub),
                    "regressors": len(features)},
                   {"Cointegrating regression": tab},
                   notes=dropped_note(dropped) + ["Standard errors of the static long-run regression are not "
                                                  "valid for inference and are not reported."],
                   rows_used=len(sub))


@register("econ.johansen", "Johansen cointegration rank test (trace and maximum eigenvalue)", "Stationarity", _TS,
          params=(P("columns", "columns", help="Series of the system (two or more)"), P("date", required=False),
                  P("det_order", "integer", default=0, choices=(-1, 0, 1),
                    help="-1 none, 0 constant, 1 linear trend"),
                  P("k_ar_diff", "integer", default=1, help="Lagged differences in the VECM"),
                  P("confidence", "number", default=0.95, choices=(0.90, 0.95, 0.99),
                    help="Level used for the sequential rank determination")),
          description="""Johansen's reduced-rank VECM procedure. For each r = 0..m−1: trace statistic
(H0: rank ≤ r vs rank = m) and maximum-eigenvalue statistic (H0: rank = r vs r + 1), with Osterwald-Lenum /
MacKinnon–Haug–Michelis critical values at 90/95/99% from statsmodels. The rank is the first r whose H0 is not
rejected at the chosen level in the sequential procedure (part of the test definition). Assumes Gaussian VAR
errors; sensitive to lag length and deterministic terms, and over-rejects in short samples.""",
          references=("Johansen (1988), J. Economic Dynamics and Control 12(2-3)",
                      "Johansen (1991), Econometrica 59(6)", "Osterwald-Lenum (1992), OBES 54(3)",
                      "statsmodels.tsa.vector_ar.vecm.coint_johansen"))
def johansen(ctx: RunContext, columns, date=None, det_order=0, k_ar_diff=1, confidence=0.95) -> Outcome:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    if len(columns) < 2:
        raise ValueError("Give at least two series.")
    sub, dropped = _prepare(ctx, columns, date)
    m = len(columns)
    if len(sub) < 10 * m + k_ar_diff:
        raise NotApplicable(f"{len(sub)} observations are too few for a {m}-variable VECM.")
    with warnings.catch_warnings():      # statsmodels casts eigenvalues with ~0 imaginary parts
        warnings.simplefilter("ignore")
        res = coint_johansen(sub[list(columns)].to_numpy(), int(det_order), int(k_ar_diff))
    eig = np.real(res.eig)
    ci ={0.90: 0, 0.95: 1, 0.99: 2}[round(float(confidence), 2)]
    rows = []
    for r in range(m):
        rows.append({"H0_rank_le": r, "eigenvalue": float(eig[r]), "trace_stat": float(np.real(res.lr1[r])),
                     "trace_crit_90": float(res.cvt[r, 0]), "trace_crit_95": float(res.cvt[r, 1]),
                     "trace_crit_99": float(res.cvt[r, 2]), "max_eig_stat": float(np.real(res.lr2[r])),
                     "max_eig_crit_90": float(res.cvm[r, 0]), "max_eig_crit_95": float(res.cvm[r, 1]),
                     "max_eig_crit_99": float(res.cvm[r, 2])})
    t = pd.DataFrame(rows)

    def _rank(stat, crit):
        for r in range(m):
            if stat[r] <= crit[r, ci]:
                return r
        return m
    return Outcome({"series": m, "n": len(sub), "rank_trace": _rank(res.lr1, res.cvt),
                    "rank_max_eig": _rank(res.lr2, res.cvm), "confidence": confidence},
                   {"Johansen rank tests": t},
                   notes=dropped_note(dropped) + ["No p-values: statsmodels provides tabulated critical values only."],
                   rows_used=len(sub))


@register("econ.granger_causality", "Granger causality of each driver for the target", "Stationarity", _TS,
          params=(P("target"), P("features", "columns"), P("date", required=False),
                  P("max_lag", "integer", default=4, help="Test lags 1..max_lag"),
                  P("both_directions", "boolean", default=False, help="Also test target → feature")),
          description="""H0: lags of x add no predictive power for y given lags of y (x does not Granger-cause
y). For each lag order p = 1..max_lag: SSR-based F test and likelihood-ratio χ² from the restricted (own lags)
vs unrestricted (own and x lags) autoregressions. Both series must be stationary (difference them first,
see econ.unit_root). Granger causality is predictive precedence, not structural causality; for satellite
models it supports (or questions) the lead/lag structure of the macro drivers.""",
          references=("Granger (1969), Econometrica 37(3)", "statsmodels.tsa.stattools.grangercausalitytests"))
def granger_causality(ctx: RunContext, target, features, date=None, max_lag=4, both_directions=False) -> Outcome:
    from statsmodels.tsa.stattools import grangercausalitytests
    sub, dropped = _prepare(ctx, [target, *features], date)
    if len(sub) < 3 * max_lag + 10:
        raise NotApplicable(f"{len(sub)} observations are too few for {max_lag} lags.")
    pairs = [(x, target) for x in features] + ([(target, x) for x in features] if both_directions else [])
    rows = []
    for cause, effect in pairs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = grangercausalitytests(sub[[effect, cause]].to_numpy(), int(max_lag))
        for lag in range(1, max_lag + 1):
            ft, lrt = out[lag][0]["ssr_ftest"], out[lag][0]["lrtest"]
            rows.append({"cause": cause, "effect": effect, "lag": lag, "F_stat": float(ft[0]), "F_p": float(ft[1]),
                         "df_num": int(ft[3]), "df_denom": int(ft[2]), "LR_stat": float(lrt[0]), "LR_p": float(lrt[1])})
    t = pd.DataFrame(rows)
    best = t.loc[t.groupby(["cause", "effect"])["F_p"].idxmin()].sort_values(["effect", "cause"])
    return Outcome({"pairs": len(pairs), "max_lag": max_lag, "n": len(sub),
                    "min_F_p": float(t["F_p"].min())},
                   {"Granger causality": t, "Smallest p-value per pair": best.reset_index(drop=True)},
                   notes=dropped_note(dropped) + ["Smallest p over several lags is not a valid test on its own "
                                                  "(multiple testing); fix the lag in advance."],
                   rows_used=len(sub))


# ── influence and collinearity ─────────────────────────────────────────────

@register("econ.influence", "Influential observations (Cook's distance, leverage, DFFITS, DFBETAS)",
          "Model specification", _REG,
          params=(*_spec(), P("top_n", "integer", default=10, help="Observations to list")),
          description="""Regression influence diagnostics per observation: leverage h_ii (diagonal of the hat
matrix; mean k/n), externally studentised residual, Cook's distance D_i (change in all fitted values when i is
dropped), DFFITS (scaled change in its own fit) and DFBETAS (scaled change in each coefficient). Logit / probit /
fractional logit use the GLM (Pregibon 1981) one-step versions. Lists the top observations by Cook's distance
and the largest |DFBETAS| per coefficient; a few points that drive a satellite model's coefficients (e.g.
crisis quarters) should be justified in the documentation.""",
          references=("Cook (1977), Technometrics 19(1)", "Belsley, Kuh & Welsch (1980), Regression Diagnostics",
                      "Pregibon (1981), Annals of Statistics 9(4)"))
def influence(ctx: RunContext, target, features, model_kind="auto", date=None, top_n=10) -> Outcome:
    import statsmodels.api as sm
    f = _fit(ctx, target, features, model_kind, date)
    if f.kind == "ols":
        inf = f.res.get_influence()
        cooks, lev = inf.cooks_distance[0], inf.hat_matrix_diag
        stud, dffits, dfb = inf.resid_studentized_external, inf.dffits[0], inf.dfbetas
    else:
        link = sm.families.links.Probit() if f.kind == "probit" else sm.families.links.Logit()
        g = sm.GLM(f.y, f.X, family=sm.families.Binomial(link=link)).fit()
        inf = g.get_influence()
        cooks, lev, stud = inf.cooks_distance[0], inf.hat_matrix_diag, inf.resid_studentized
        dffits = stud * np.sqrt(lev / (1 - lev))
        dfb = inf.dfbetas
    labels = _labels(ctx, f.sub, date)
    t = pd.DataFrame({"row": f.sub.index.astype(str), "label": labels.to_numpy(), "cooks_distance": cooks,
                      "leverage": lev, "studentized_residual": stud, "dffits": dffits})
    top = t.sort_values("cooks_distance", ascending=False, kind="mergesort").head(top_n).reset_index(drop=True)
    dfb = np.asarray(dfb)
    bi = np.argmax(np.abs(dfb), axis=0)
    dtab = pd.DataFrame({"variable": list(f.X.columns), "max_abs_dfbetas": np.abs(dfb).max(axis=0),
                         "row": f.sub.index.astype(str)[bi], "label": labels.to_numpy()[bi]})
    i = int(np.argmax(cooks))
    return Outcome({"n": f.n, "max_cooks_distance": float(cooks[i]), "max_cooks_row": str(f.sub.index[i]),
                    "max_leverage": float(np.max(lev)), "mean_leverage": float(np.mean(lev)),
                    "max_abs_dffits": float(np.max(np.abs(dffits)))},
                   {"Top observations by Cook's distance": top, "Largest DFBETAS per coefficient": dtab},
                   notes=f.notes, rows_used=f.n)


@register("econ.vif", "Multicollinearity (VIF, condition number, correlations)", "Model specification",
          _REG, params=(P("features", "columns", help="Explanatory variables of the model"),
                        P("target", required=False, help="Optional: restrict to rows where the target is present")),
          description="""Variance inflation factor VIF_j = 1/(1 − R²_j), R²_j from regressing feature j on the
other features and a constant (computed on the model's design matrix, complete rows only); tolerance = 1/VIF.
Belsley–Kuh–Welsch condition number of the design with columns scaled to unit length (constant included), and
the condition indices with variance-decomposition proportions for the smallest eigenvalue. Pairwise Pearson
correlations are listed by absolute size. High collinearity inflates standard errors and makes coefficient signs
fragile (see econ.bootstrap_stability).""",
          references=("Belsley, Kuh & Welsch (1980), Regression Diagnostics, ch. 3",
                      "O'Brien (2007), Quality & Quantity 41(5)"))
def vif(ctx: RunContext, features, target=None) -> Outcome:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    if len(features) < 2:
        raise NotApplicable("VIF needs at least two features.")
    sub, dropped = _prepare(ctx, [*features, *([target] if target else [])])
    X = sm.add_constant(sub[list(features)], has_constant="add").to_numpy()
    if len(sub) <= X.shape[1]:
        raise NotApplicable("Too few observations.")
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise NotApplicable("Design is rank deficient (perfect collinearity): VIF is infinite.")
    v = [float(variance_inflation_factor(X, j)) for j in range(1, X.shape[1])]
    vt = pd.DataFrame({"variable": list(features), "VIF": v, "tolerance": 1 / np.asarray(v)}
                      ).sort_values("VIF", ascending=False, kind="mergesort")
    Z = X / np.linalg.norm(X, axis=0)
    _, s, vh = np.linalg.svd(Z, full_matrices=False)
    cidx = s[0] / s
    phi = (vh.T ** 2) / s ** 2
    prop = phi / phi.sum(axis=1, keepdims=True)
    ci = pd.DataFrame({"condition_index": cidx} | {f"prop_{n}": prop[i] for i, n in
                                                   enumerate(["const", *features])})
    corr = sub[list(features)].corr()
    pairs = [{"variable_1": a, "variable_2": b, "correlation": float(corr.loc[a, b])}
             for i, a in enumerate(features) for b in features[i + 1:]]
    pt = pd.DataFrame(pairs)
    pt = pt.reindex(pt["correlation"].abs().sort_values(ascending=False, kind="mergesort").index)
    return Outcome({"n": len(sub), "max_VIF": float(max(v)), "max_VIF_variable": vt.iloc[0]["variable"],
                    "condition_number": float(cidx[-1]),
                    "max_abs_correlation": float(pt["correlation"].abs().max())},
                   {"VIF": vt, "Condition indices": ci, "Pairwise correlations": pt.reset_index(drop=True)},
                   notes=dropped_note(dropped), rows_used=len(sub))


# ── forecasting ────────────────────────────────────────────────────────────

def _forecast_frame(ctx, actual, predicted, benchmark, date, sample, current_value):
    cols = [actual, predicted] + ([benchmark] if benchmark else [])
    df = ctx.df
    if sample:
        if current_value is None:
            raise ValueError("With `sample`, give current_value (the out-of-time sample label).")
        mask = df[sample].astype(str).str.lower() == str(current_value).lower()
        if not mask.any():
            raise ValueError(f"No rows with {sample} = {current_value}.")
        ctx = RunContext(df=df[mask], tables=ctx.tables, seed=ctx.seed)
    return _prepare(ctx, cols, date)


def _accuracy(a: np.ndarray, f: np.ndarray) -> dict:
    e = a - f
    n = len(a)
    mse = float(np.mean(e ** 2))
    nz = a != 0
    sa, sf = np.std(a), np.std(f)
    r = np.corrcoef(a, f)[0, 1] if sa > 0 and sf > 0 else np.nan
    out = {"n": n, "mean_error": float(np.mean(e)), "RMSE": math.sqrt(mse), "MAE": float(np.mean(np.abs(e))),
           "MAPE": float(np.mean(np.abs(e[nz] / a[nz]))) if nz.any() else np.nan,
           "sMAPE": float(np.mean(2 * np.abs(e) / (np.abs(a) + np.abs(f)))) if np.all(np.abs(a) + np.abs(f) > 0)
           else np.nan,
           "theil_U1": math.sqrt(mse) / (math.sqrt(np.mean(a ** 2)) + math.sqrt(np.mean(f ** 2)))}
    if n > 1:
        naive = a[:-1]
        rmse_naive = math.sqrt(np.mean((a[1:] - naive) ** 2))
        out["theil_U2_vs_no_change"] = (math.sqrt(np.mean((a[1:] - f[1:]) ** 2)) / rmse_naive
                                        if rmse_naive > 0 else np.nan)
    if mse > 0:
        out["bias_proportion_UM"] = float((np.mean(f) - np.mean(a)) ** 2 / mse)
        out["variance_proportion_US"] = float((sf - sa) ** 2 / mse)
        out["covariance_proportion_UC"] = float(2 * (1 - r) * sf * sa / mse) if np.isfinite(r) else np.nan
    return out


_FC = (P("actual"), P("predicted", help="Model forecast"), P("date", required=False),
       P("sample", required=False, help="Sample column to select the out-of-time rows"),
       P("current_value", "string", required=False, help="Value of `sample` marking the out-of-time rows"))


@register("econ.forecast_accuracy", "Out-of-time forecast accuracy (RMSE, MAE, MAPE, Theil's U)",
          "Predictive accuracy", _TS + ("lgd", "ead", "ml_regression"),
          params=(*_FC, P("benchmark", required=False, help="Benchmark forecast column (e.g. naive, prior model)")),
          description="""Accuracy of the forecast against realised values on the selected (out-of-time) rows:
mean error (bias, actual − forecast), RMSE, MAE, MAPE (rows with actual = 0 excluded), symmetric MAPE, Theil's
U1 = RMSE/(√mean a² + √mean f²) ∈ [0,1], Theil's U2 = RMSE of the forecast / RMSE of the no-change forecast
a_{t−1} (< 1 beats a random walk; needs time order), and Theil's MSE decomposition into bias (UM), variance (US)
and covariance (UC) proportions (UM + US + UC = 1). With a benchmark column the same metrics are shown for it
and their ratios; test the difference with econ.diebold_mariano.""",
          references=("Theil (1966), Applied Economic Forecasting", "Hyndman & Koehler (2006), IJF 22(4)",
                      "Bliemel (1973), J. Marketing Research 10(4) (U1/U2 definitions)"))
def forecast_accuracy(ctx: RunContext, actual, predicted, date=None, sample=None, current_value=None,
                      benchmark=None) -> Outcome:
    sub, dropped = _forecast_frame(ctx, actual, predicted, benchmark, date, sample, current_value)
    if len(sub) < 3:
        raise NotApplicable(f"{len(sub)} forecast observations.")
    a = sub[actual].to_numpy()
    m = _accuracy(a, sub[predicted].to_numpy())
    rows = [{"forecast": predicted} | m]
    if benchmark:
        b = _accuracy(a, sub[benchmark].to_numpy())
        rows.append({"forecast": benchmark} | b)
    t = pd.DataFrame(rows)
    summary = {k: m[k] for k in ("n", "mean_error", "RMSE", "MAE", "MAPE", "theil_U1") if k in m}
    if "theil_U2_vs_no_change" in m:
        summary["theil_U2_vs_no_change"] = m["theil_U2_vs_no_change"]
    if benchmark:
        summary |= {"RMSE_ratio_to_benchmark": m["RMSE"] / b["RMSE"] if b["RMSE"] else np.nan,
                    "MAE_ratio_to_benchmark": m["MAE"] / b["MAE"] if b["MAE"] else np.nan}
    notes = dropped_note(dropped)
    nz = int((a == 0).sum())
    if nz:
        notes.append(f"MAPE excludes {nz} rows with actual = 0.")
    if not date:
        notes.append("Theil's U2 uses the table's row order as the time order (no date given).")
    return Outcome(summary, {"Forecast accuracy": t}, notes=notes, rows_used=len(sub))


def diebold_mariano_stat(e1: np.ndarray, e2: np.ndarray, h: int = 1, loss: str = "squared") -> dict:
    """DM statistic with Harvey–Leybourne–Newbold correction. d = L(e1) − L(e2)."""
    L = (lambda e: e ** 2) if loss == "squared" else (lambda e: np.abs(e))
    d = L(e1) - L(e2)
    n = len(d)
    dbar = float(np.mean(d))
    dc = d - dbar
    gam = [float(dc @ dc) / n] + [float(dc[k:] @ dc[:-k]) / n for k in range(1, h)]
    V = gam[0] + 2 * sum(gam[1:])
    kernel = "rectangular (h−1 autocovariances)"
    if V <= 0:
        V = gam[0] + 2 * sum((1 - k / h) * g for k, g in enumerate(gam[1:], start=1))
        kernel = "Bartlett (rectangular estimate was non-positive)"
    if V <= 0:
        raise NotApplicable("Loss differential has zero variance (identical forecast losses).")
    dm = dbar / math.sqrt(V / n)
    hln = dm * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"n": n, "mean_loss_differential": dbar, "DM_stat": dm,
            "DM_p_value_normal": float(2 * stats.norm.sf(abs(dm))), "HLN_stat": hln,
            "HLN_p_value": float(2 * stats.t.sf(abs(hln), n - 1)),
            "HLN_p_value_model_better": float(stats.t.cdf(hln, n - 1)), "variance_kernel": kernel}


@register("econ.diebold_mariano", "Diebold–Mariano test of equal forecast accuracy (HLN-corrected)",
          "Predictive accuracy", _TS + ("lgd", "ead", "ml_regression"),
          params=(*_FC, P("benchmark", help="Benchmark forecast column"),
                  P("horizon", "integer", default=1, help="Forecast horizon h (autocovariances up to h−1)"),
                  P("loss", "string", default="squared", choices=("squared", "absolute"))),
          description="""H0: the model and the benchmark forecasts have equal expected loss. Loss differential
d_t = L(a_t − f_t) − L(a_t − b_t); DM = d̄ / √(V̂/n) with V̂ = γ₀ + 2Σ_{k=1}^{h−1} γ_k (h-step forecasts are
MA(h−1)). Harvey, Leybourne & Newbold (1997) small-sample correction DM* = DM·√[(n + 1 − 2h + h(h−1)/n)/n]
compared with Student t(n−1). Negative statistics favour the model. Also reported: the one-sided p-value for
H1: the model is more accurate. Not valid for comparing nested models' estimated forecasts in large samples
(Clark & West).""",
          references=("Diebold & Mariano (1995), JBES 13(3)", "Harvey, Leybourne & Newbold (1997), IJF 13(2)",
                      "Diebold (2015), JBES 33(1)"))
def diebold_mariano(ctx: RunContext, actual, predicted, benchmark, date=None, sample=None, current_value=None,
                    horizon=1, loss="squared") -> Outcome:
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    sub, dropped = _forecast_frame(ctx, actual, predicted, benchmark, date, sample, current_value)
    if len(sub) < max(5, 2 * horizon + 2):
        raise NotApplicable(f"{len(sub)} observations are too few.")
    a = sub[actual].to_numpy()
    r = diebold_mariano_stat(a - sub[predicted].to_numpy(), a - sub[benchmark].to_numpy(), horizon, loss)
    kernel = r.pop("variance_kernel")
    notes = dropped_note(dropped) + [f"Long-run variance kernel: {kernel}.", f"Loss: {loss} error."]
    return Outcome(r | {"horizon": horizon}, notes=notes, rows_used=len(sub))


# ── sensitivity and coefficient stability ──────────────────────────────────

@register("econ.macro_sensitivity", "Sensitivity of the prediction to each driver (±k standard deviations)",
          "Sensitivity analysis", _REG,
          params=(*_spec(), P("shock_sd", "number", default=1.0, help="Shock size in standard deviations")),
          description="""Using the refitted model, shifts one driver at a time by ±shock_sd standard deviations
(its sample SD) for every observation, keeps the others at their observed values, and reports the change in the
average prediction (on the response scale: the target for OLS, the probability / rate for logit, probit and
fractional logit). For OLS the effect is exactly β·shock·SD and symmetric; for non-linear links the asymmetry
between up and down shocks is shown. Use it to check that the economic direction and size of each macro
effect is plausible and to rank drivers. If a driver enters with several lags or transformations, each column
is shocked separately.""",
          references=("EBA (2023), Methodological note, EU-wide stress test (satellite model sensitivities)",
                      "Saltelli et al. (2008), Global Sensitivity Analysis: The Primer"))
def macro_sensitivity(ctx: RunContext, target, features, model_kind="auto", date=None, shock_sd=1.0) -> Outcome:
    f = _fit(ctx, target, features, model_kind, date)
    X = f.X
    base = float(np.mean(f.res.predict(X)))
    rows = []
    for v in features:
        sd = float(f.sub[v].std(ddof=1))
        up, dn = X.copy(), X.copy()
        up[v] = up[v] + shock_sd * sd
        dn[v] = dn[v] - shock_sd * sd
        pu, pd_ = float(np.mean(f.res.predict(up))), float(np.mean(f.res.predict(dn)))
        rows.append({"variable": v, "coefficient": float(f.res.params[v]), "sd": sd, "shock": shock_sd * sd,
                     "baseline_mean_prediction": base, "prediction_up": pu, "prediction_down": pd_,
                     "change_up": pu - base, "change_down": pd_ - base,
                     "relative_change_up": (pu - base) / base if base else np.nan})
    t = pd.DataFrame(rows)
    t = t.reindex(t["change_up"].abs().sort_values(ascending=False, kind="mergesort").index).reset_index(drop=True)
    return Outcome({"model_kind": f.kind, "baseline_mean_prediction": base, "shock_sd": shock_sd,
                    "most_sensitive_driver": t.iloc[0]["variable"],
                    "largest_change_up": float(t.iloc[0]["change_up"])},
                   {"Sensitivity by driver": t}, notes=f.notes, rows_used=f.n)


@register("econ.bootstrap_stability", "Coefficient and selection stability under bootstrap resampling",
          "Model specification", _REG,
          params=(*_spec(), P("n_boot", "integer", default=200, help="Bootstrap resamples"),
                  P("alpha", "number", default=0.05, help="Significance level counted in each refit"),
                  P("block_length", "integer", default=1,
                    help="Moving-block length (1 = i.i.d. rows; >1 for time series)"),
                  P("expected_signs", "dict", required=False,
                    help="Expected signs {variable: '+'/'-'}; default: full-sample sign")),
          description="""Refits the specification on n_boot bootstrap resamples (i.i.d. rows, or moving blocks
of length block_length for serially dependent data) and reports per coefficient: the share of resamples where
it is significant at alpha, the share where it keeps the expected (or full-sample) sign, the bootstrap mean,
SD and 2.5%/97.5% percentile interval. A variable that is significant in only part of the resamples, or flips
sign, is a fragile selection choice (Austin & Tu 2004). Resamples where the fit fails (e.g. separation) are
counted and excluded.""",
          references=("Efron & Tibshirani (1993), An Introduction to the Bootstrap",
                      "Künsch (1989), Annals of Statistics 17(3) (moving block bootstrap)",
                      "Austin & Tu (2004), J. Clinical Epidemiology 57(11)"))
def bootstrap_stability(ctx: RunContext, target, features, model_kind="auto", date=None, n_boot=200, alpha=0.05,
                        block_length=1, expected_signs=None) -> Outcome:
    f = _fit(ctx, target, features, model_kind, date)
    names = list(f.res.params.index)
    signs = {v: float(np.sign(f.res.params[v])) for v in names}
    for k, v in (expected_signs or {}).items():
        signs[_canon(k, f.res.params.index)] = _sign(v)
    rng = ctx.rng(7)
    n, L = f.n, max(int(block_length), 1)
    y, X = f.y.to_numpy(), f.X.to_numpy()
    est, sig, fails = [], [], 0
    for _ in range(int(n_boot)):
        if L == 1:
            idx = rng.integers(0, n, n)
        else:
            starts = rng.integers(0, n - L + 1, int(math.ceil(n / L)))
            idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
        try:
            yb = y[idx]
            if f.kind in ("logit", "probit") and len(np.unique(yb)) < 2:
                raise ValueError
            r = _fit_model(yb, X[idx], f.kind)
            p = np.asarray(r.params)
            conv = getattr(r, "mle_retvals", None) or {}
            if not np.all(np.isfinite(p)) or (conv and not conv.get("converged", True)):
                raise ValueError
            est.append(p)
            sig.append(np.asarray(r.pvalues) < alpha)
        except Exception:
            fails += 1
    if len(est) < 10:
        raise NotApplicable(f"Only {len(est)} bootstrap fits succeeded.")
    E, S = np.vstack(est), np.vstack(sig)
    sgn = np.array([signs[v] for v in names])
    t = pd.DataFrame({"variable": names, "full_sample_estimate": f.res.params.to_numpy(),
                      "full_sample_p": f.res.pvalues.to_numpy(), "expected_sign": np.where(sgn > 0, "+", "-"),
                      "share_significant": S.mean(axis=0), "share_expected_sign": (np.sign(E) == sgn).mean(axis=0),
                      "boot_mean": E.mean(axis=0), "boot_sd": E.std(axis=0, ddof=1),
                      "boot_p2_5": np.percentile(E, 2.5, axis=0), "boot_p97_5": np.percentile(E, 97.5, axis=0)})
    slopes = t[t["variable"] != "const"]
    notes = list(f.notes) + ([f"{fails} resamples failed to fit and were excluded."] if fails else [])
    if L > 1:
        notes.append(f"Moving-block bootstrap with block length {L}" + ("" if date else " (row order as time order)") + ".")
    return Outcome({"n_boot_used": len(est), "failed": fails,
                    "min_share_significant": float(slopes["share_significant"].min()),
                    "min_share_expected_sign": float(slopes["share_expected_sign"].min())},
                   {"Bootstrap stability": t}, notes=notes, rows_used=n)


# ── scorecard specifics ────────────────────────────────────────────────────

def woe_table(x: pd.Series, y: pd.Series, bins: int = 10, edges=None, max_levels: int = 20) -> tuple[pd.DataFrame, bool]:
    """WoE/IV per bin of x. WoE = ln(%good / %bad), target 1 = bad. Returns (table, numeric_bins)."""
    xn = pd.to_numeric(x, errors="coerce")
    numeric = (x.notna().sum() > 0 and xn.notna().sum() == x.notna().sum()
               and (edges is not None or xn.nunique() > max_levels))
    if numeric:
        if edges is None:
            e = np.unique(np.nanquantile(xn.dropna(), np.linspace(0, 1, bins + 1)))
            e[0], e[-1] = -np.inf, np.inf
        else:
            e = np.unique(np.r_[-np.inf, np.asarray(edges, float), np.inf])
        b = pd.cut(xn, e, include_lowest=True)
        lab = b.astype(str).where(b.notna(), "<missing>")
        order = [str(c) for c in b.cat.categories] + ["<missing>"]
    else:
        lab = x.astype("object").where(x.notna(), "<missing>").astype(str)
        order = sorted(lab.unique(), key=str)
    g = pd.DataFrame({"bin": lab, "bad": y.astype(float)}).groupby("bin", sort=False)["bad"].agg(["count", "sum"])
    g = g.reindex([o for o in order if o in g.index])
    bad, good = g["sum"].to_numpy(float), (g["count"] - g["sum"]).to_numpy(float)
    adj = (bad == 0) | (good == 0)
    b_a, g_a = bad + 0.5 * adj, good + 0.5 * adj
    db, dg = b_a / b_a.sum(), g_a / g_a.sum()
    woe = np.log(dg / db)
    t = pd.DataFrame({"bin": g.index.astype(str), "count": g["count"].to_numpy(int), "bads": bad.astype(int),
                      "goods": good.astype(int), "bad_rate": bad / g["count"].to_numpy(float),
                      "dist_good": dg, "dist_bad": db, "WoE": woe, "IV_contribution": (dg - db) * woe,
                      "zero_cell_adjusted": adj})
    return t, numeric


@register("econ.woe_iv", "Weight of Evidence, Information Value and WoE monotonicity", "Discrimination",
          _SCORE + ("ifrs9",),
          params=(P("target"), P("features", "columns", help="Characteristics (raw values)"),
                  P("bins", "integer", default=10, help="Quantile bins for numeric characteristics"),
                  P("bin_edges", "dict", required=False,
                    help="Documented bin edges {variable: [inner cut points]} (overrides quantile bins)")),
          description="""Per characteristic and bin (quantile bins, documented cut points, or one bin per level
for categorical / ≤ 20-level variables; missing values form their own bin): counts, bad rate, distribution of
goods and bads, WoE = ln(%good/%bad) (target 1 = bad) and IV = Σ(%good − %bad)·WoE. Monotonicity of WoE over
ordered numeric bins (missing bin excluded): whether it is monotone, the number of direction reversals and the
Spearman correlation of bin order with WoE. A non-monotone WoE in a characteristic documented as monotone, or
a WoE that disagrees with the documented one, indicates a binning or data issue.""",
          references=("Siddiqi (2006), Credit Risk Scorecards, ch. 6", "Thomas, Edelman & Crook (2002), "
                      "Credit Scoring and Its Applications", "Kullback (1959), Information Theory and Statistics"))
def woe_iv(ctx: RunContext, target, features, bins=10, bin_edges=None) -> Outcome:
    df = ctx.df
    y = binary(df, target)
    keep = y.notna()
    y = y[keep]
    require_two_classes(y)
    edges = bin_edges or {}
    unknown = set(edges) - set(features)
    if unknown:
        raise ValueError(f"bin_edges for variables not in features: {sorted(unknown)}")
    summ, details, notes = [], [], dropped_note(int((~keep).sum()), "rows with missing target")
    for v in features:
        t, numeric = woe_table(df.loc[keep, v], y, bins, edges.get(v))
        t.insert(0, "variable", v)
        details.append(t)
        w = t.loc[t["bin"] != "<missing>", "WoE"].to_numpy()
        if numeric and len(w) >= 2:
            d = np.sign(np.diff(w))
            d = d[d != 0]
            rev = int((np.diff(d) != 0).sum()) if len(d) > 1 else 0
            rho = float(stats.spearmanr(np.arange(len(w)), w).statistic) if len(w) > 2 else float(np.sign(w[-1] - w[0]))
            mono, rev_ = rev == 0, rev
        else:
            mono, rev_, rho = None, None, np.nan
        if t["zero_cell_adjusted"].any():
            notes.append(f"{v}: bins with zero goods or bads adjusted by +0.5.")
        summ.append({"variable": v, "IV": float(t["IV_contribution"].sum()), "bins": len(t),
                     "binning": "numeric" if numeric else "categorical", "WoE_monotone": mono,
                     "direction_reversals": rev_, "spearman_bin_order_vs_WoE": rho,
                     "missing_share": float(df.loc[keep, v].isna().mean())})
    s = pd.DataFrame(summ).sort_values("IV", ascending=False, kind="mergesort").reset_index(drop=True)
    return Outcome({"variables": len(s), "n": int(keep.sum()), "events": int(y.sum()),
                    "max_IV": float(s["IV"].max()), "max_IV_variable": s.iloc[0]["variable"],
                    "non_monotone_variables": int((s["WoE_monotone"] == False).sum())},  # noqa: E712
                   {"IV by variable": s, "WoE by bin": pd.concat(details, ignore_index=True)},
                   notes=notes, rows_used=int(keep.sum()))


@register("econ.scorecard_points", "Scorecard points recomputation (PDO scaling)", "Model implementation",
          _SCORE,
          params=(P("pdo", "number", default=20.0, help="Points to double the odds"),
                  P("base_score", "number", default=600.0, help="Score at the base odds"),
                  P("base_odds", "number", default=50.0, help="Good:bad odds at the base score"),
                  P("pd", required=False, help="Model PD column (score = offset + factor·ln((1−PD)/PD))"),
                  P("features", "columns", required=False, help="Model inputs as used in the logit (e.g. WoE)"),
                  P("coefficients", "dict", required=False,
                    help="Logit coefficients {variable: β, 'const': β0} for P(bad)"),
                  P("score", required=False, help="Implemented score column to compare against"),
                  P("round_points", "boolean", default=False, help="Round points per attribute before summing"),
                  P("tolerance", "number", default=0.5, help="Absolute difference counted as a match")),
          description="""Recomputes the score from the documented scaling: factor = PDO/ln 2,
offset = base_score − factor·ln(base_odds), score = offset + factor·ln(odds_good). From a PD column,
ln(odds_good) = ln((1 − PD)/PD); from a logit for P(bad) with coefficients β, ln(odds_good) = −(β₀ + Σβ_j x_j),
and points per attribute = −(β_j·x_j + β₀/m)·factor + offset/m for m characteristics (optionally rounded, as
implemented in most scorecards). Compares with the implemented score column: mean and maximum absolute
difference, share within tolerance and the largest mismatches. Exact agreement is expected up to rounding.""",
          references=("Siddiqi (2006), Credit Risk Scorecards, ch. 6 (scaling)",
                      "Refaat (2011), Credit Risk Scorecards: Development and Implementation using SAS"),
          suite=False)
def scorecard_points(ctx: RunContext, pdo=20.0, base_score=600.0, base_odds=50.0, pd=None, features=None,
                     coefficients=None, score=None, round_points=False, tolerance=0.5) -> Outcome:
    import pandas as _pd
    if pdo <= 0 or base_odds <= 0:
        raise ValueError("pdo and base_odds must be positive.")
    factor = pdo / math.log(2)
    offset = base_score - factor * math.log(base_odds)
    df = ctx.df
    tables, notes = {}, []
    if pd:
        sub, dropped = _prepare(ctx, [pd] + ([score] if score else []))
        p = sub[pd]
        bad = (p <= 0) | (p >= 1)
        if bad.any():
            notes.append(f"{int(bad.sum())} rows with PD outside (0,1) excluded.")
            sub, p = sub[~bad], p[~bad]
        calc = offset + factor * np.log((1 - p) / p)
    elif features and coefficients:
        coefs = {}
        for k, v in coefficients.items():
            kk = "const" if str(k).strip().lower() in _INTERCEPT else k
            if kk != "const" and kk not in features:
                raise ValueError(f"Coefficient '{k}' is not one of the features.")
            coefs[kk] = float(v)
        missing = [v for v in features if v not in coefs]
        if missing:
            raise ValueError(f"No coefficient for {missing}.")
        b0, m = coefs.get("const", 0.0), len(features)
        sub, dropped = _prepare(ctx, list(features) + ([score] if score else []))
        pts = _pd.DataFrame({v: -(coefs[v] * sub[v] + b0 / m) * factor + offset / m for v in features})
        if round_points:
            pts = pts.round(0)
        calc = pts.sum(axis=1)
        attr = []
        for v in features:
            u = _pd.DataFrame({"value": sub[v], "points": pts[v]}).drop_duplicates().sort_values("value")
            if len(u) <= 50:
                attr.append(u.assign(variable=v)[["variable", "value", "points"]])
            else:
                notes.append(f"{v}: {len(u)} distinct values, attribute points not listed.")
        if attr:
            tables["Points by attribute"] = _pd.concat(attr, ignore_index=True)
    else:
        raise ValueError("Give either `pd`, or `features` with `coefficients`.")
    notes = dropped_note(dropped) + notes
    summary = {"factor": factor, "offset": offset, "n": len(sub), "mean_recomputed_score": float(calc.mean())}
    if score:
        diff = sub[score] - calc
        summary |= {"mean_abs_difference": float(diff.abs().mean()), "max_abs_difference": float(diff.abs().max()),
                    "share_within_tolerance": float((diff.abs() <= tolerance).mean())}
        mm = _pd.DataFrame({"row": sub.index.astype(str), "implemented_score": sub[score].to_numpy(),
                            "recomputed_score": calc.to_numpy(), "difference": diff.to_numpy()})
        tables["Largest differences"] = mm.reindex(mm["difference"].abs().sort_values(
            ascending=False, kind="mergesort").index).head(20).reset_index(drop=True)
    else:
        tables["Recomputed scores (first 20)"] = _pd.DataFrame({"row": sub.index.astype(str)[:20],
                                                                 "recomputed_score": calc.to_numpy()[:20]})
    return Outcome(summary, tables, notes=notes, rows_used=len(sub))
