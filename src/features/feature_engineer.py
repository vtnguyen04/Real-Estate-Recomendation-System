import polars as pl
from collections import defaultdict
from typing import List, Optional

from src.features.feature_context import FeatureContext
from src.features.base import BaseHeuristicExtractor
from src.features.extractors.collaborative import CollaborativeExtractor
from src.utils.polars_utils import safe_fill_null_numeric
from src.utils.logging import get_logger

logger = get_logger("feature_engineer")





class FeatureEngineer:
    """
    SOLID Orchestrator for Feature Engineering.
    Delegates heuristic scoring to a list of specific BaseHeuristicExtractors.
    Follows OCP — to add new features, pass a new extractor without modifying this class.

    Two modes:
      - Training: extract_for_training(pairs) uses each extractor's build_feature_df()
        (constructor-injected raw data, lazy-cached on first call).
      - Inference: fit_state() fills the shared FeatureContext; extract() uses extract_scores().
    """

    def __init__(self, extractors: List[BaseHeuristicExtractor], context: Optional[FeatureContext] = None):
        self.extractors = extractors
        self.context = context or FeatureContext()
        self._cached_feature_dfs: Optional[list] = None  # [(join_key, df), ...]

    # ── Training ───────────────────────────────────────────────

    def extract_for_training(self, train_pairs: pl.DataFrame) -> pl.DataFrame:
        """
        Join all extractor feature DataFrames onto train_pairs and compute match features.
        Feature DataFrames are built once from each extractor's build_feature_df() and cached.
        """
        if self._cached_feature_dfs is None:
            self._cached_feature_dfs = []
            for ext in self.extractors:
                feat_df = ext.build_feature_df(self.context)
                if feat_df is not None:
                    self._cached_feature_dfs.append((ext.join_key, feat_df))
            logger.info(
                f"Built {len(self._cached_feature_dfs)} feature DataFrames "
                f"from {len(self.extractors)} extractors."
            )

        df = train_pairs
        for join_key, feat_df in self._cached_feature_dfs:
            if join_key and join_key != "pairs":
                df = df.join(feat_df, on=join_key, how="left")
        df = safe_fill_null_numeric(df)

        # Apply pairwise extractors via polymorphism (OCP)
        for ext in self.extractors:
            df = ext.compute_match_features(df)

        return df

    def attach_features_inference(
        self,
        pairs: pl.DataFrame,
        user_stats_df: pl.DataFrame,
        item_stats_df: pl.DataFrame,
        item_meta_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Join pre-built lookup DataFrames onto pairs and compute match features using instance extractors.
        Used by evaluate.py during inference.
        """
        stat_cols = ["user_id"]
        for col in ["event_count", "contact_rate", "pref_city", "pref_cat", "pref_price", "pref_ad_type"]:
            if col in user_stats_df.columns:
                stat_cols.append(col)

        df = pairs.join(user_stats_df.select(stat_cols), on="user_id", how="left")
        df = df.join(item_stats_df, on="item_id", how="left")
        df = df.join(item_meta_df, on="item_id", how="left")
        df = safe_fill_null_numeric(df)

        # Apply pairwise extractors via polymorphism (OCP)
        for ext in self.extractors:
            df = ext.compute_match_features(df)

        return df

    # ── Inference ──────────────────────────────────────────────

    def fit_state(
        self,
        df_listing_collected: pl.DataFrame,
        interactions_collected: pl.DataFrame,
        pageviews_collected: "pl.DataFrame | pl.LazyFrame",
        df_snapshot_collected: Optional[pl.DataFrame] = None,
    ):
        """Fill the shared FeatureContext inference dicts from raw data."""
        self.context.fit(
            df_listing_collected,
            interactions_collected,
            pageviews_collected,
            df_snapshot_collected,
        )

    def extract(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Generate top candidates and compute features for all users in data (Inference Mode).
        For training, use extract_for_training() instead.
        """
        df = data.collect()
        users = df["user_id"].unique().to_list()
        logger.info(f"Extracting candidates for {len(users)} users via {len(self.extractors)} extractors...")

        for ext in self.extractors:
            if isinstance(ext, CollaborativeExtractor):
                ext.prefetch_batch(users, self.context)

        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing

        def process_user(uid):
            features = defaultdict(dict)
            for ext in self.extractors:
                ext.extract_scores(uid, self.context, features)
            final_scores = {}
            for it, f in features.items():
                total = (
                    f.get("score_prev", 0.0) + f.get("score_seller", 0.0) +
                    f.get("score_als", 0.0) + f.get("score_i2i", 0.0) + f.get("score_segpop", 0.0)
                )
                final_scores[it] = total
                f["item_total_contacts"] = float(self.context.item_stats.get(it, {}).get("contacts", 0))
                f["item_total_views"]    = float(self.context.item_stats.get(it, {}).get("views", 0))
            user_rows = []
            for it in sorted(final_scores, key=lambda x: -final_scores[x])[:300]:
                r = {"user_id": uid, "item_id": it, "pre_score": final_scores[it]}
                r.update(features[it])
                user_rows.append(r)
            return user_rows

        n_workers = min(16, multiprocessing.cpu_count())
        logger.info(f"Parallelizing extraction with {n_workers} threads, streaming to disk...")

        written_batches = []
        batch_size = 500

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            batch = []
            for user_rows in executor.map(process_user, users):
                batch.extend(user_rows)
                if len(batch) >= batch_size * 100:
                    df_batch = pl.DataFrame(batch)
                    out = f"/tmp/feat_batch_{len(written_batches)}.parquet"
                    df_batch.write_parquet(out)
                    written_batches.append(out)
                    batch.clear()
            if batch:
                df_batch = pl.DataFrame(batch)
                out = f"/tmp/feat_batch_{len(written_batches)}.parquet"
                df_batch.write_parquet(out)
                written_batches.append(out)

        logger.info(f"Written {len(written_batches)} batches to disk, reading back...")
        return pl.scan_parquet(written_batches)
