"""
Data Profiling Utilities for EDA.
Provides reusable functions to profile Parquet datasets via PyArrow + Polars.
All EDA scripts should import from this module instead of writing inline logic.
"""
import polars as pl
import pyarrow.dataset as ds
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger("profiler")


def scan_table(path: str) -> pl.LazyFrame:
    """
    Create a Polars LazyFrame from a Parquet directory using PyArrow dataset.
    This is the canonical way to access data — zero-copy, lazy evaluation.
    
    Args:
        path: Path to a directory of Parquet files or a single Parquet file.
    Returns:
        Polars LazyFrame
    """
    dataset = ds.dataset(path, format='parquet')
    return pl.scan_pyarrow_dataset(dataset)


def get_file_count(path: str) -> int:
    """Count the number of Parquet files in a dataset directory."""
    dataset = ds.dataset(path, format='parquet')
    return len(dataset.files)


def get_row_count(lf: pl.LazyFrame) -> int:
    """Get total row count from a LazyFrame (materializes only the count)."""
    return lf.select(pl.len()).collect().item()


def get_schema_info(lf: pl.LazyFrame) -> Dict[str, str]:
    """Extract schema as {column_name: dtype_string}."""
    schema = lf.collect_schema()
    return {col: str(dtype) for col, dtype in schema.items()}


def compute_null_stats(df: pl.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute null value statistics for each column.
    Returns dict of {col: {"count": N, "pct": X.XX}} for columns with nulls > 0.
    """
    total = len(df)
    if total == 0:
        return {}
    
    null_counts = df.null_count().to_dict(as_series=False)
    result = {}
    for col, counts in null_counts.items():
        count = counts[0]
        if count > 0:
            result[col] = {
                "count": count,
                "pct": round((count / total) * 100, 2)
            }
    return result


def compute_cardinality(df: pl.DataFrame, max_unique: int = 200) -> Dict[str, int]:
    """
    Compute cardinality (unique value count) for columns with low cardinality.
    Skips high-cardinality ID columns automatically.
    """
    skip_cols = {'item_id', 'user_id', 'session_id', 'event_id', 'seller_id', 'title', 'query'}
    result = {}
    for col in df.columns:
        if col in skip_cols:
            continue
        try:
            n_unique = df[col].n_unique()
            if n_unique <= max_unique:
                result[col] = n_unique
        except Exception:
            pass
    return result


def compute_value_counts(df: pl.DataFrame, col: str, top_n: int = 20) -> pl.DataFrame:
    """Get value counts for a column, sorted descending, limited to top_n."""
    return df[col].value_counts(sort=True).head(top_n)


def compute_numeric_stats(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Get descriptive statistics for a numeric column."""
    return df[col].describe()


def profile_table(
    path: str,
    table_name: str,
    sample_rows: Optional[int] = None
) -> Dict[str, Any]:
    """
    Full profiling of a Parquet table.
    
    Args:
        path: Path to the Parquet directory.
        table_name: Human-readable name for logging.
        sample_rows: If set, only analyze this many rows (for large tables).
        
    Returns:
        Dict with keys: table_name, files_count, total_rows, sample_size, 
                        schema, null_stats, cardinality
    """
    logger.info(f"Profiling {table_name} from {path}...")
    
    if not os.path.exists(path):
        logger.error(f"Path not found: {path}")
        return {"table_name": table_name, "error": f"Path not found: {path}"}
    
    lf = scan_table(path)
    files_count = get_file_count(path)
    total_rows = get_row_count(lf)
    
    # Materialize sample for detailed analysis
    if sample_rows and total_rows > sample_rows:
        logger.info(f"  Sampling {sample_rows} rows (total: {total_rows})...")
        df = lf.head(sample_rows).collect()
    else:
        logger.info(f"  Loading all {total_rows} rows...")
        df = lf.collect()
    
    schema = get_schema_info(lf)
    null_stats = compute_null_stats(df)
    cardinality = compute_cardinality(df)
    
    result = {
        "table_name": table_name,
        "files_count": files_count,
        "total_rows": total_rows,
        "sample_size": len(df),
        "schema": schema,
        "null_stats": null_stats,
        "cardinality": cardinality,
        "df_sample": df  # Keep the materialized sample for further analysis
    }
    
    logger.info(f"  Done. {files_count} files, {total_rows} total rows, {len(schema)} columns.")
    return result


def profile_single_parquet(
    path: str,
    table_name: str
) -> Dict[str, Any]:
    """
    Profile a single Parquet file (e.g., test_users.parquet).
    """
    logger.info(f"Profiling {table_name} from {path}...")
    
    if not os.path.exists(path):
        logger.error(f"Path not found: {path}")
        return {"table_name": table_name, "error": f"Path not found: {path}"}
    
    df = pl.read_parquet(path)
    schema = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    null_stats = compute_null_stats(df)
    
    result = {
        "table_name": table_name,
        "files_count": 1,
        "total_rows": len(df),
        "sample_size": len(df),
        "schema": schema,
        "null_stats": null_stats,
        "cardinality": compute_cardinality(df),
        "df_sample": df
    }
    
    logger.info(f"  Done. {len(df)} rows, {len(schema)} columns.")
    return result
