"""
src/data/pci_loader.py — PCI (fact_post_contact_interactions) Data Loader

Loads and processes PCI data to supplement ALS training and cold-start preferences.
Follows INS-058 (density > size) and INS-059/060 strategies.

Usage:
    from src.data.pci_loader import PCILoader
    loader = PCILoader(pci_path, dim_listing_path)
    loader.enrich_als_pairs(cache_dir, mode="existing_only")
    loader.enrich_weighted_als_pairs(cache_dir, mode="existing_only")
    loader.enrich_cold_prefs(cache_dir, blind_user_ids)
"""
import os
from typing import Set, Optional

import polars as pl

from src.utils.logging import get_logger

logger = get_logger("pci_loader")


class PCILoader:
    """
    Loads PCI (fact_post_contact_interactions) data and provides:
    1. ALS supplement pairs (lead interactions not in existing training data)
    2. Blind user preferences (city, category from PCI for users with no fact_user_events)
    3. Full integration pipeline: enrich_als_pairs() + enrich_weighted_als_pairs() + enrich_cold_prefs()
    """

    def __init__(self, pci_path: str, dim_listing_path: str,
                 min_lead_count: int = 1, purchased_weight: float = 3.0):
        """
        Args:
            pci_path: Path to fact_post_contact_interactions parquet directory
            dim_listing_path: Path to dim_listing parquet directory (for city/district)
            min_lead_count: Minimum lead_count to consider as valid interaction
            purchased_weight: Weight multiplier for purchased=True pairs
        """
        self._pci_path = pci_path
        self._dim_listing_path = dim_listing_path
        self._min_lead_count = min_lead_count
        self._purchased_weight = purchased_weight
        self._pci_lazy: Optional[pl.LazyFrame] = None
        self._listing_df: Optional[pl.DataFrame] = None

    def _ensure_loaded(self) -> None:
        """Lazy-load PCI data on first access."""
        if self._pci_lazy is None:
            pci_files = os.path.join(self._pci_path, "**/*.parquet")
            self._pci_lazy = pl.scan_parquet(pci_files)
            logger.info(f"PCI lazy scanner initialized: {self._pci_path}")

    def _ensure_listing(self) -> None:
        """Lazy-load dim_listing on first access."""
        if self._listing_df is None:
            import glob
            listing_files = glob.glob(os.path.join(self._dim_listing_path, "*.parquet"))
            if listing_files:
                self._listing_df = pl.read_parquet(listing_files)
                logger.info(f"dim_listing loaded: {len(self._listing_df):,} items")
            else:
                self._listing_df = pl.DataFrame()
                logger.warning("dim_listing not found, empty DataFrame used")

    # ── Low-level data extraction ──────────────────────────────

    def get_lead_pairs(self) -> pl.DataFrame:
        """
        Extract all (user_id, item_id, weight) pairs from PCI where lead_count >= threshold.
        Weight = lead_count (or lead_count * purchased_weight if purchased=True).
        """
        self._ensure_loaded()
        pairs = (
            self._pci_lazy
            .filter(pl.col("lead_count") >= self._min_lead_count)
            .select(["user_id", "item_id", "lead_count", "purchased", "category"])
            .collect()
        )
        # Compute weight: base = lead_count, boost for purchased
        pairs = pairs.with_columns(
            pl.when(pl.col("purchased") == True)
            .then(pl.col("lead_count") * self._purchased_weight)
            .otherwise(pl.col("lead_count").cast(pl.Float64))
            .alias("weight")
        )
        logger.info(f"PCI lead pairs: {len(pairs):,} ({pairs['user_id'].n_unique():,} users)")
        return pairs

    def get_als_supplement_pairs(
        self,
        existing_pairs: pl.DataFrame,
        mode: str = "existing_only",
        existing_users: Optional[Set[str]] = None,
        test_users: Optional[Set[str]] = None,
    ) -> pl.DataFrame:
        """
        Get NEW (user_id, item_id, weight) pairs from PCI not in existing ALS data.

        Args:
            existing_pairs: Current ALS training pairs (user_id, item_id columns)
            mode: "existing_only" (INS-058 safe), "all", or "test_only"
            existing_users: Set of users already in ALS (for mode="existing_only")
            test_users: Set of test user IDs (for mode="test_only")

        Returns:
            DataFrame with columns: user_id, item_id, weight
        """
        lead_pairs = self.get_lead_pairs()

        # Anti-join to get only NEW pairs
        existing_keys = existing_pairs.select(["user_id", "item_id"]).unique()
        new_pairs = lead_pairs.join(existing_keys, on=["user_id", "item_id"], how="anti")
        logger.info(f"NEW PCI pairs (not in ALS): {len(new_pairs):,}")

        # Apply mode filter (INS-058: density > size)
        if mode == "existing_only" and existing_users:
            new_pairs = new_pairs.filter(pl.col("user_id").is_in(list(existing_users)))
            logger.info(f"  After existing_only filter: {len(new_pairs):,}")
        elif mode == "test_only" and test_users:
            new_pairs = new_pairs.filter(pl.col("user_id").is_in(list(test_users)))
            logger.info(f"  After test_only filter: {len(new_pairs):,}")

        return new_pairs.select(["user_id", "item_id", "weight"])

    def get_blind_user_prefs(self, blind_user_ids: Set[str]) -> pl.DataFrame:
        """
        Build city+category preferences for blind users using PCI data.

        Returns:
            DataFrame with columns: user_id, pref_city, pref_cat
        """
        self._ensure_loaded()
        self._ensure_listing()

        blind_pci = (
            self._pci_lazy
            .filter(pl.col("user_id").is_in(list(blind_user_ids)))
            .select(["user_id", "item_id", "category", "lead_count"])
            .collect()
        )

        if len(blind_pci) == 0:
            logger.warning("No PCI data for blind users")
            return pl.DataFrame(schema={"user_id": pl.Utf8, "pref_city": pl.Utf8, "pref_cat": pl.Int64})

        # Get city from dim_listing
        if len(self._listing_df) > 0 and "city_name" in self._listing_df.columns:
            item_city = self._listing_df.select(["item_id", "city_name"]).unique()
            blind_pci = blind_pci.join(item_city, on="item_id", how="left")
        else:
            blind_pci = blind_pci.with_columns(pl.lit(None).alias("city_name"))

        prefs = (
            blind_pci.group_by("user_id").agg([
                pl.col("city_name").drop_nulls().mode().first().alias("pref_city"),
                pl.col("category").drop_nulls().mode().first().alias("pref_cat"),
            ])
            .filter(pl.col("pref_city").is_not_null() | pl.col("pref_cat").is_not_null())
        )
        logger.info(f"Blind user prefs from PCI: {len(prefs):,}/{len(blind_user_ids):,}")
        return prefs

    # ── High-level integration methods ─────────────────────────

    def enrich_als_pairs(
        self,
        cache_dir: str,
        mode: str = "existing_only",
        test_users: Optional[Set[str]] = None,
    ) -> pl.DataFrame:
        """
        Full pipeline: load existing ALS pairs → get PCI supplement → merge → save with backup.

        Returns:
            Merged ALS pairs DataFrame
        """
        als_path = os.path.join(cache_dir, "als_contact_pairs.parquet")
        existing_pairs = pl.read_parquet(als_path)
        existing_users = set(existing_pairs["user_id"].unique().to_list())
        logger.info(f"Existing ALS pairs: {len(existing_pairs):,} ({len(existing_users):,} users)")

        new_pairs = self.get_als_supplement_pairs(
            existing_pairs=existing_pairs, mode=mode,
            existing_users=existing_users, test_users=test_users,
        )

        if len(new_pairs) == 0:
            logger.info("No new PCI pairs to add")
            return existing_pairs

        # Convert to ALS format — ensure matching dtypes
        pci_als = new_pairs.rename({"weight": "score"}).select(["user_id", "item_id", "score"])
        pci_als = pci_als.with_columns(pl.col("score").cast(pl.Float64))
        if "score" not in existing_pairs.columns:
            existing_pairs = existing_pairs.with_columns(pl.lit(1.0).alias("score"))
        existing_als = existing_pairs.select(["user_id", "item_id", "score"])
        existing_als = existing_als.with_columns(pl.col("score").cast(pl.Float64))

        # Merge + deduplicate (sum scores)
        merged = pl.concat([existing_als, pci_als])
        merged = merged.group_by(["user_id", "item_id"]).agg(pl.col("score").sum())

        # Backup original
        backup_path = als_path + ".backup"
        if not os.path.exists(backup_path):
            os.rename(als_path, backup_path)
            logger.info(f"  Original backed up to {backup_path}")

        merged.write_parquet(als_path)
        density = len(merged) / merged["user_id"].n_unique()
        logger.info(f"  Merged ALS: {len(merged):,} pairs, density={density:.1f}")
        return merged

    def enrich_weighted_als_pairs(
        self,
        cache_dir: str,
        mode: str = "existing_only",
        test_users: Optional[Set[str]] = None,
    ) -> pl.DataFrame:
        """
        Merge PCI lead weights into als_weighted_contact.parquet.

        Training prefers this file when `als_use_weighted=True`, so enriching only
        als_contact_pairs.parquet would silently drop PCI during production train.

        This intentionally mirrors the split-clean eval builder:
        - Base weighted events keep real contacts=3 and other_interaction=1.
        - PCI leads are added as extra weight, including pairs that already exist.
        - `existing_only` keeps ALS density stable by adding only users already in
          the base weighted matrix.

        The method is conservative around already-enriched caches: if no clean
        weighted backup exists and the current weighted file has more rows than
        the raw contact backup, it skips instead of double-counting PCI.
        """
        weighted_path = os.path.join(cache_dir, "als_weighted_contact.parquet")
        if not os.path.exists(weighted_path):
            logger.warning("als_weighted_contact.parquet not found; skip weighted PCI enrichment")
            return pl.DataFrame()

        current_pairs = pl.read_parquet(weighted_path)
        backup_path = weighted_path + ".backup"
        standard_backup_path = os.path.join(cache_dir, "als_contact_pairs.parquet.backup")

        if os.path.exists(backup_path):
            if os.path.exists(standard_backup_path):
                raw_n = pl.scan_parquet(standard_backup_path).select(pl.len()).collect().item()
                if len(current_pairs) <= raw_n:
                    current_pairs.write_parquet(backup_path)
                    base_pairs = current_pairs
                    logger.info(
                        f"Weighted ALS current file looks clean; refreshed backup: {len(base_pairs):,} pairs"
                    )
                else:
                    base_pairs = pl.read_parquet(backup_path)
                    logger.info(f"Weighted ALS base loaded from backup: {len(base_pairs):,} pairs")
            else:
                base_pairs = pl.read_parquet(backup_path)
                logger.info(f"Weighted ALS base loaded from backup: {len(base_pairs):,} pairs")
        else:
            if os.path.exists(standard_backup_path):
                raw_n = pl.scan_parquet(standard_backup_path).select(pl.len()).collect().item()
                if len(current_pairs) > raw_n:
                    logger.warning(
                        "Weighted ALS appears already PCI-enriched but has no clean backup; "
                        "skip weighted PCI enrichment to avoid double-counting. "
                        "Rerun preprocess from raw data to rebuild a clean base."
                    )
                    return current_pairs

            os.rename(weighted_path, backup_path)
            base_pairs = current_pairs
            logger.info(f"  Original weighted ALS backed up to {backup_path}")

        existing_users = set(base_pairs["user_id"].unique().to_list())
        logger.info(f"Base weighted ALS: {len(base_pairs):,} pairs ({len(existing_users):,} users)")

        lead_pairs = self.get_lead_pairs()
        if mode == "existing_only":
            lead_pairs = lead_pairs.filter(pl.col("user_id").is_in(list(existing_users)))
            logger.info(f"  PCI weighted supplement after existing_only filter: {len(lead_pairs):,}")
        elif mode == "test_only" and test_users:
            lead_pairs = lead_pairs.filter(pl.col("user_id").is_in(list(test_users)))
            logger.info(f"  PCI weighted supplement after test_only filter: {len(lead_pairs):,}")
        elif mode == "all":
            logger.info(f"  PCI weighted supplement mode=all: {len(lead_pairs):,}")

        if len(lead_pairs) == 0:
            base_pairs.write_parquet(weighted_path)
            return base_pairs

        pci_weighted = (
            lead_pairs
            .rename({"weight": "score"})
            .select(["user_id", "item_id", "score"])
            .with_columns(pl.col("score").cast(pl.Float64))
        )
        base_weighted = (
            base_pairs
            .select(["user_id", "item_id", "score"])
            .with_columns(pl.col("score").cast(pl.Float64))
        )

        merged = (
            pl.concat([base_weighted, pci_weighted])
            .group_by(["user_id", "item_id"])
            .agg(pl.col("score").sum())
            .with_columns(pl.col("score").cast(pl.Float32))
        )
        merged.write_parquet(weighted_path)
        density = len(merged) / merged["user_id"].n_unique()
        logger.info(
            f"  Merged weighted ALS: {len(merged):,} pairs, "
            f"score_sum={float(merged['score'].sum()):,.1f}, density={density:.1f}"
        )
        return merged

    def enrich_cold_prefs(
        self,
        cache_dir: str,
        blind_user_ids: Set[str],
    ) -> pl.DataFrame:
        """
        Build PCI prefs for blind users and merge into cold_user_prefs.parquet.

        Returns:
            Merged cold prefs DataFrame
        """
        pci_prefs = self.get_blind_user_prefs(blind_user_ids)
        if len(pci_prefs) == 0:
            return pci_prefs

        cold_prefs_path = os.path.join(cache_dir, "cold_user_prefs.parquet")
        if os.path.exists(cold_prefs_path):
            existing_cold = pl.read_parquet(cold_prefs_path)
            existing_uids = set(existing_cold["user_id"].to_list())
            new_pci = pci_prefs.filter(~pl.col("user_id").is_in(list(existing_uids)))
            merged = pl.concat([existing_cold, new_pci], how="diagonal_relaxed")
            logger.info(f"  Cold prefs: existing={len(existing_cold):,}, new PCI={len(new_pci):,}")
        else:
            merged = pci_prefs
            logger.info(f"  Created cold prefs from PCI: {len(pci_prefs):,}")

        merged.write_parquet(cold_prefs_path)
        logger.info(f"  Total cold prefs saved: {len(merged):,}")
        return merged
