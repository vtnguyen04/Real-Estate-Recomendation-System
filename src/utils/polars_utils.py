"""
src/utils/polars_utils.py — Shared Polars utilities for safe DataFrame operations.

Centralizes Categorical-safe fill_null and LightGBM feature preparation.
Used by FeatureEngineer, LGBMRanker, and evaluate.py (DRY).
"""
import polars as pl
import pandas as pd


def safe_fill_null_numeric(df: pl.DataFrame) -> pl.DataFrame:
    """Fill null only on numeric columns. Leaves Categorical/Utf8 untouched."""
    num_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    if num_cols:
        df = df.with_columns([pl.col(c).fill_null(0) for c in num_cols])
    return df


def prepare_features_for_lgbm(df: pl.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Select feature columns, convert Categorical to physical codes, fill nulls,
    and return a pandas DataFrame ready for LightGBM.

    This is the SINGLE source of truth for feature preparation before LightGBM
    inference or training. Any Categorical column gets converted to its physical
    integer code (which LightGBM handles natively).
    """
    available = [c for c in feature_cols if c in df.columns]
    X = df.select(available)
    for c in X.columns:
        if X[c].dtype == pl.Categorical:
            X = X.with_columns(pl.col(c).to_physical().cast(pl.Float32))
    return X.fill_null(0).to_pandas()
