"""
scripts/preprocess.py — Pre-aggregate large datasets into compact caches.

Steps:
  1. Load fact_user_events → aggregate contacts, pageviews, sessions
  2. (Optional) Integrate PCI data → enrich ALS pairs + cold-start prefs
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import polars as pl

from config.settings import PipelineConfig
from src.data.loader import FactUserEventsLoader
from src.data.preprocessor import DataPreprocessor
from src.data.pci_loader import PCILoader
from src.utils.logging import get_logger

logger = get_logger("preprocess")


def main():
    config = PipelineConfig()
    parser = argparse.ArgumentParser(description="Preprocess data into cache")
    parser.add_argument("--data_dir", default=config.data.train_path)
    parser.add_argument("--test_dir", default=config.data.test_path)
    parser.add_argument("--cache_dir", default=config.cache_dir)
    parser.add_argument("--skip_pci", action="store_true", help="Skip PCI integration step")
    parser.add_argument("--pci_mode", default=config.pci_merge_mode,
                        choices=["existing_only", "all", "test_only"])
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache_dir)

    # Step 1: Aggregate raw events → cache
    logger.info("=" * 60)
    logger.info("[Step 1] Aggregate fact_user_events → cache")
    logger.info("=" * 60)
    lf = FactUserEventsLoader(
        data_path=os.path.join(args.data_dir, "fact_user_events/")
    ).load()
    preprocessor = DataPreprocessor(config, cache_dir)
    snapshot_path = os.path.join(args.data_dir, "fact_listing_snapshot")
    preprocessor.process_and_cache(lf, snapshot_path=snapshot_path)

    # Step 2: PCI integration (optional)
    if config.pci_enabled and not args.skip_pci:
        pci_path = os.path.join(args.data_dir, "fact_post_contact_interactions")
        if os.path.exists(pci_path):
            logger.info("=" * 60)
            logger.info(f"[Step 2] PCI integration (mode={args.pci_mode})")
            logger.info("=" * 60)

            loader = PCILoader(
                pci_path=pci_path,
                dim_listing_path=os.path.join(args.data_dir, "dim_listing"),
                min_lead_count=config.pci_min_lead_count,
                purchased_weight=config.pci_purchased_weight,
            )
            test_users = set(
                pl.read_parquet(os.path.join(args.test_dir, "test_users.parquet"))["user_id"].to_list()
            )
            loader.enrich_als_pairs(cache_dir=cache_dir, mode=args.pci_mode, test_users=test_users)
            if config.model.als_use_weighted:
                loader.enrich_weighted_als_pairs(
                    cache_dir=cache_dir, mode=args.pci_mode, test_users=test_users
                )

            contact_users = set(
                pl.read_parquet(os.path.join(cache_dir, "contact_pairs.parquet"),
                                columns=["user_id"])["user_id"].unique().to_list()
            )
            loader.enrich_cold_prefs(cache_dir=cache_dir, blind_user_ids=test_users - contact_users)
        else:
            logger.info("[Step 2] PCI path not found, skipping")
    else:
        logger.info("[Step 2] PCI integration skipped (--skip_pci or pci_enabled=False)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
