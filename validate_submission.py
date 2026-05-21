import argparse
import os
from pathlib import Path
import zipfile

import polars as pl

from config.settings import PipelineConfig


EXPECTED_ROWS = 1_615_680
EXPECTED_USERS = 161_568
EXPECTED_COLUMNS = ["ID", "user_id", "rank", "item_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Kaggle submission format.")
    parser.add_argument("csv_path", nargs="?", default="submission.csv")
    parser.add_argument("--zip_path", default="")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    print(f"Checking {csv_path} ...")

    first_bytes = csv_path.read_bytes()[:3]
    assert first_bytes != b"\xef\xbb\xbf", "CSV has UTF-8 BOM; rules require UTF-8 without BOM."

    df = pl.read_csv(csv_path)
    print(f"Rows: {len(df):,}")
    assert len(df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS:,} rows, got {len(df):,}"

    assert df.columns == EXPECTED_COLUMNS, f"Wrong columns: {df.columns}"
    print("Columns OK: ID, user_id, rank, item_id")

    nulls = {c: df[c].null_count() for c in df.columns}
    assert all(v == 0 for v in nulls.values()), f"Null values found: {nulls}"

    assert df["ID"].n_unique() == len(df), "ID column must be unique."
    assert df["ID"].min() == 1 and df["ID"].max() == len(df), "ID must run from 1 to n rows."

    rank_bad = df.filter((pl.col("rank") < 1) | (pl.col("rank") > 10))
    assert len(rank_bad) == 0, "rank must be in [1, 10]."

    per_user = df.group_by("user_id").agg([
        pl.len().alias("n_rows"),
        pl.col("rank").n_unique().alias("n_ranks"),
        pl.col("item_id").n_unique().alias("n_items"),
        pl.col("rank").min().alias("rank_min"),
        pl.col("rank").max().alias("rank_max"),
    ])
    bad_users = per_user.filter(
        (pl.col("n_rows") != 10)
        | (pl.col("n_ranks") != 10)
        | (pl.col("n_items") != 10)
        | (pl.col("rank_min") != 1)
        | (pl.col("rank_max") != 10)
    )
    assert len(bad_users) == 0, f"Users with invalid rows/ranks/items: {len(bad_users):,}"

    assert df.select(["user_id", "rank"]).n_unique() == len(df), "(user_id, rank) duplicates found."
    assert df.select(["user_id", "item_id"]).n_unique() == len(df), "(user_id, item_id) duplicates found."

    cfg = PipelineConfig()
    test_users = set(pl.read_parquet(os.path.join(cfg.data.test_path, "test_users.parquet"))["user_id"].to_list())
    sub_users = set(df["user_id"].unique().to_list())
    assert len(sub_users) == EXPECTED_USERS, f"Expected {EXPECTED_USERS:,} users, got {len(sub_users):,}"
    assert sub_users == test_users, "Submission users differ from test_users.parquet."
    print("Users OK.")

    valid_items = set(
        pl.scan_parquet(os.path.join(cfg.data.train_path, "dim_listing/*.parquet"))
        .select("item_id")
        .collect()["item_id"]
        .to_list()
    )
    sub_items = set(df["item_id"].unique().to_list())
    invalid_items = sub_items - valid_items
    assert not invalid_items, f"Invalid item_id count: {len(invalid_items):,}"
    print(f"Items OK: {len(sub_items):,} unique items, all in dim_listing.")

    rank1 = (
        df.filter(pl.col("rank") == 1)
        .group_by("item_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(1)
    )
    top_rank1 = rank1["n"][0]
    assert top_rank1 <= EXPECTED_USERS * 0.10, (
        f"Rank-1 top item assigned to {top_rank1:,} users, exceeds 10% threshold."
    )
    print(f"Rank-1 diversity OK: top item assigned to {top_rank1:,} users.")

    if args.zip_path:
        zip_path = Path(args.zip_path)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        assert size_mb <= 100, f"Zip too large: {size_mb:.2f} MB"
        with zipfile.ZipFile(zip_path) as zf:
            assert "submission.csv" in zf.namelist(), "Zip must contain submission.csv"
        print(f"Zip OK: {size_mb:.2f} MB")

    print("ALL SUBMISSION RULES PASS.")


if __name__ == "__main__":
    main()
