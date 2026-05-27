from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def fit_mass_univariate(
    features: pd.DataFrame,
    phenotype: pd.DataFrame,
    outcome: str,
    covariates: list[str],
    id_column: str = "subject_id",
) -> pd.DataFrame:
    """Fit one OLS model per feature."""
    missing = [column for column in [id_column, outcome, *covariates] if column not in phenotype.columns]
    if missing:
        raise KeyError(f"missing phenotype columns: {missing}")
    merged = phenotype[[id_column, outcome] + covariates].merge(features, on=id_column)
    rows = []
    feature_columns = [c for c in features.columns if c.startswith("node_") or c.startswith("edge_")]
    for feature in feature_columns:
        data = merged[[outcome] + covariates + [feature]].dropna()
        if data[feature].nunique() < 2 or data.shape[0] <= len(covariates) + 2:
            rows.append({"feature": feature, "beta": np.nan, "t": np.nan, "p": np.nan, "n": data.shape[0]})
            continue
        y = data[outcome].astype(float)
        x = sm.add_constant(data[covariates + [feature]].astype(float), has_constant="add")
        fit = sm.OLS(y, x).fit()
        rows.append(
            {
                "feature": feature,
                "beta": float(fit.params[feature]),
                "t": float(fit.tvalues[feature]),
                "p": float(fit.pvalues[feature]),
                "n": int(data.shape[0]),
            }
        )
    out = pd.DataFrame(rows)
    valid = out["p"].notna()
    out["q"] = np.nan
    if valid.any():
        out.loc[valid, "q"] = multipletests(out.loc[valid, "p"], method="fdr_bh")[1]
    return out


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Residualize columns by a design matrix."""
    q, _ = np.linalg.qr(design)
    return values - q @ (q.T @ values)


def fit_mass_univariate_fast(
    features: pd.DataFrame,
    phenotype: pd.DataFrame,
    outcome: str,
    covariates: list[str],
    id_column: str = "subject_id",
) -> pd.DataFrame:
    """Fit adjusted OLS for many node or edge features."""
    missing = [column for column in [id_column, outcome, *covariates] if column not in phenotype.columns]
    if missing:
        raise KeyError(f"missing phenotype columns: {missing}")
    feature_columns = [c for c in features.columns if c.startswith("node_") or c.startswith("edge_")]
    merged = phenotype[[id_column, outcome, *covariates]].merge(features[[id_column, *feature_columns]], on=id_column)
    data = merged.dropna(subset=[outcome, *covariates]).copy()
    if data.empty:
        return pd.DataFrame({"feature": feature_columns, "beta": np.nan, "t": np.nan, "p": np.nan, "n": 0, "q": np.nan})

    y = data[outcome].to_numpy(dtype=float)
    x_cov = data[covariates].to_numpy(dtype=float) if covariates else np.empty((data.shape[0], 0))
    x_features = data[feature_columns].fillna(0).to_numpy(dtype=float)
    design = np.column_stack([np.ones(data.shape[0]), x_cov])
    y_res = residualize(y.reshape(-1, 1), design).ravel()
    x_res = residualize(x_features, design)

    denominator = np.sum(x_res * x_res, axis=0)
    numerator = np.sum(x_res * y_res[:, None], axis=0)
    beta = np.full(len(feature_columns), np.nan, dtype=float)
    valid = denominator > np.finfo(float).eps
    beta[valid] = numerator[valid] / denominator[valid]

    resid = y_res[:, None] - x_res * np.nan_to_num(beta, nan=0.0)[None, :]
    df = max(data.shape[0] - len(covariates) - 2, 1)
    sigma2 = np.sum(resid * resid, axis=0) / df
    se = np.sqrt(sigma2 / np.where(valid, denominator, np.nan))
    t_values = beta / se
    p_values = 2.0 * scipy_stats.t.sf(np.abs(t_values), df=df)
    p_values[~np.isfinite(p_values)] = np.nan

    out = pd.DataFrame({"feature": feature_columns, "beta": beta, "t": t_values, "p": p_values, "n": int(data.shape[0])})
    ok = out["p"].notna()
    out["q"] = np.nan
    if ok.any():
        out.loc[ok, "q"] = multipletests(out.loc[ok, "p"], method="fdr_bh")[1]
    return out
