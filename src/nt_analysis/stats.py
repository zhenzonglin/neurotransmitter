from __future__ import annotations

import numpy as np
import pandas as pd
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
    feature_columns = [c for c in features.columns if c != id_column]
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
