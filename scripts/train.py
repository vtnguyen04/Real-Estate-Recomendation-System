"""
scripts/train.py — Train ContactALS + ViewALS + SegPop + LightGBM lambdarank.

All pipeline logic lives in src/pipeline/training_pipeline.py.
"""
import sys
import os
import argparse
import time

import polars as pl
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import PipelineConfig
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.logging import get_logger

logger = get_logger("train")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")


def main():
    config = PipelineConfig()
    parser = argparse.ArgumentParser(description="Train ContactALS + ViewALS + SegPop + LightGBM")
    parser.add_argument("--data_dir", default=config.data.train_path)
    parser.add_argument("--output_dir", default="outputs/models/")
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"TRAIN pipeline  GPU={args.use_gpu}")
    logger.info("=" * 60)

    logger.info("[1/8] Loading preprocessed data...")
    contacts      = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))
    als_contacts  = pl.read_parquet(os.path.join(CACHE_DIR, "als_contact_pairs.parquet"))
    als_pageviews = pl.read_parquet(os.path.join(CACHE_DIR, "als_pageview_pairs.parquet"))
    date_range    = pl.read_parquet(os.path.join(CACHE_DIR, "date_range.parquet"))
    df_listing    = pl.scan_parquet(os.path.join(args.data_dir, "dim_listing/*.parquet")).collect()

    max_date   = date_range["max_date"][0]
    split_date = max_date - timedelta(days=config.validation_days)
    logger.info(f"  Max date: {max_date} | Split: {split_date}")

    pipeline = TrainingPipeline(
        config=config,
        use_gpu=args.use_gpu,
        output_dir=args.output_dir,
        cache_dir=CACHE_DIR,
    )
    pipeline.run(contacts, als_contacts, als_pageviews, df_listing, split_date, ext_logger=logger)

    elapsed = (time.time() - t0) / 60
    logger.info(f"Done. ({elapsed:.1f} min)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
