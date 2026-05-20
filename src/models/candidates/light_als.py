"""
Lightweight ALS Recommender — Memory-optimized.

Builds matrix from pre-aggregated (user_id, item_id, score) pairs.
Supports GPU via implicit.als.AlternatingLeastSquares.
"""
from __future__ import annotations

import gc
import os
import pickle
from typing import Dict, List, Optional, Set

import numpy as np
import polars as pl
import implicit
from scipy.sparse import csr_matrix, save_npz, load_npz

from src.core.base import BaseRecommender, RecommendationContext
from src.utils.logging import get_logger

logger = get_logger(__name__)

GT_EVENTS = ['view_phone', 'contact_chat', 'other_interaction', 'contact_zalo', 'contact_sms']


class LightALSRecommender(BaseRecommender):
    """
    Memory-efficient ALS trained from pre-aggregated (user_id, item_id, score) pairs.

    fit() accepts either:
      - Raw fact_user_events LazyFrame (filtered internally)
      - Pre-aggregated contact pairs with columns (user_id, item_id, score)
    """

    def __init__(
        self,
        factors: int = 256,
        regularization: float = 0.01,
        iterations: int = 30,
        use_gpu: bool = False,
    ):
        super().__init__(name="light_als")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.use_gpu = use_gpu

        self._model = None
        self._matrix: Optional[csr_matrix] = None
        self._u2i: Dict[str, int] = {}
        self._i2i: Dict[str, int] = {}
        self._i2item: Dict[int, str] = {}

    # ─────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────

    def fit(self, train_data: pl.LazyFrame, **kwargs) -> "LightALSRecommender":
        """
        Build user-item matrix and train ALS.

        Accepts two input formats:
          1. Raw events with is_login / is_contact columns → filters internally.
          2. Pre-aggregated pairs with (user_id, item_id, score) → used directly.
        """
        schema = train_data.collect_schema().names()

        if "is_login" in schema or "is_contact" in schema:
            logger.info("Building contact pairs from raw events (streaming)...")
            all_pairs = (
                train_data
                .filter(pl.col("is_login") == "login")
                .filter(pl.col("is_contact") == 1)
                .group_by(["user_id", "item_id"])
                .agg(pl.len().alias("score"))
                .with_columns(pl.col("score").cast(pl.Float32))
                .collect()
            )
        else:
            logger.info("Using pre-aggregated pairs directly...")
            score_col = "score" if "score" in schema else schema[-1]
            all_pairs = (
                train_data
                .select(["user_id", "item_id", pl.col(score_col).cast(pl.Float32).alias("score")])
                .collect()
            )

        logger.info(f"  Pairs: {len(all_pairs):,}")
        self._build_and_train(all_pairs)
        return self

    def _build_and_train(self, pairs: pl.DataFrame) -> None:
        """Construct CSR matrix from pairs and train the ALS model."""
        users_list = pairs["user_id"].unique().to_list()
        items_list = pairs["item_id"].unique().to_list()
        self._u2i = {u: i for i, u in enumerate(users_list)}
        self._i2i = {it: i for i, it in enumerate(items_list)}
        self._i2item = {i: it for it, i in self._i2i.items()}

        self._matrix = self._pairs_to_csr(pairs)
        del pairs; gc.collect()

        logger.info(f"  Matrix: {self._matrix.shape}, nnz={self._matrix.nnz:,}")
        _use_gpu = self.use_gpu and implicit.gpu.HAS_CUDA
        logger.info(f"  Training ALS: factors={self.factors}, iters={self.iterations}, gpu={_use_gpu}")
        self._model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=_use_gpu,
        )
        self._model.fit(self._matrix)
        logger.info("  ALS trained.")

    def _pairs_to_csr(self, pairs: pl.DataFrame) -> csr_matrix:
        """Convert a (user_id, item_id, score) DataFrame to a CSR matrix."""
        valid = [
            (self._u2i[u], self._i2i[it], float(s))
            for u, it, s in zip(
                pairs["user_id"].to_list(),
                pairs["item_id"].to_list(),
                pairs["score"].to_list(),
            )
            if u in self._u2i and it in self._i2i
        ]
        if not valid:
            return csr_matrix((len(self._u2i), len(self._i2i)), dtype=np.float32)
        r = np.array([x[0] for x in valid], dtype=np.int32)
        c = np.array([x[1] for x in valid], dtype=np.int32)
        v = np.array([x[2] for x in valid], dtype=np.float32)
        return csr_matrix((v, (r, c)), shape=(len(self._u2i), len(self._i2i)))

    def rebuild_matrix(self, pairs: pl.DataFrame) -> None:
        """
        Rebuild the sparse matrix from saved pairs without retraining.
        Call this after load() to restore inference capability.
        """
        logger.info(f"Rebuilding ALS matrix from {len(pairs):,} pairs...")
        self._matrix = self._pairs_to_csr(pairs)
        logger.info(f"  Matrix: {self._matrix.shape}, nnz={self._matrix.nnz:,}")

    # ─────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────

    def recommend_batch(
        self,
        user_ids: List[str],
        n: int = 300,
        filter_already_liked: bool = False,
        valid_items: Optional[Set[str]] = None,
        return_scores: bool = False,
    ) -> "Dict[str, List[str]] | Dict[str, List[tuple]]":
        """
        Batch recommend for multiple users.

        Returns:
            If return_scores=False (default): {user_id: [item_id, ...]}
            If return_scores=True: {user_id: [(item_id, score), ...]}
        """
        if not self._model or self._matrix is None:
            return {}

        warm_indices, warm_uids = [], []
        for uid in user_ids:
            idx = self._u2i.get(uid, -1)
            if idx != -1:
                warm_indices.append(idx)
                warm_uids.append(uid)

        if not warm_indices:
            return {}

        result = {}
        chunk_size = 500
        for start in range(0, len(warm_indices), chunk_size):
            chunk_idx = warm_indices[start:start + chunk_size]
            chunk_uids = warm_uids[start:start + chunk_size]
            idx_arr = np.array(chunk_idx, dtype=np.int32)
            ids_matrix, scores_matrix = self._model.recommend(
                userid=idx_arr,
                user_items=self._matrix[idx_arr],
                N=n,
                filter_already_liked_items=filter_already_liked,
            )
            for uid, item_row, score_row in zip(chunk_uids, ids_matrix, scores_matrix):
                if return_scores:
                    pairs = [
                        (self._i2item[int(i)], float(s))
                        for i, s in zip(item_row, score_row)
                        if int(i) in self._i2item
                    ]
                    if valid_items:
                        pairs = [(it, s) for it, s in pairs if it in valid_items]
                    result[uid] = pairs
                else:
                    items = [self._i2item[int(i)] for i in item_row if int(i) in self._i2item]
                    if valid_items:
                        items = [i for i in items if i in valid_items]
                    result[uid] = items

        return result

    def recommend(self, context: RecommendationContext, candidates=None) -> pl.LazyFrame:
        recs = self.recommend_batch([context.user_id], n=context.num_recommendations)
        items = recs.get(context.user_id, [])
        rows = [
            {"user_id": context.user_id, "item_id": it, "score": float(len(items) - i)}
            for i, it in enumerate(items)
        ]
        return pl.DataFrame(rows).lazy()

    def get_similar_items(self, item_id: str, n: int = 20) -> List[str]:
        idx = self._i2i.get(item_id, -1)
        if idx == -1 or not self._model:
            return []
        ids, _ = self._model.similar_items(idx, N=n + 1)
        return [self._i2item[int(i)] for i in ids[1:] if int(i) in self._i2item]

    def free_matrix(self):
        del self._matrix
        self._matrix = None
        gc.collect()

    # ─────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._model.save(os.path.join(path, "als.npz"))
        with open(os.path.join(path, "als_meta.pkl"), "wb") as f:
            pickle.dump({"u2i": self._u2i, "i2i": self._i2i, "i2item": self._i2item}, f)
        if self._matrix is not None:
            save_npz(os.path.join(path, "als_matrix.npz"), self._matrix)
        logger.info(f"ALS saved to {path}")

    def load(self, path: str) -> "LightALSRecommender":
        npz_path = os.path.join(path, "als.npz")
        # implicit.als.AlternatingLeastSquares is a factory function in 0.7.x;
        # load via the concrete CPU/GPU class directly.
        try:
            import implicit.gpu.als as _gpu_als
            self._model = _gpu_als.AlternatingLeastSquares.load(npz_path)
        except Exception:
            import implicit.cpu.als as _cpu_als
            self._model = _cpu_als.AlternatingLeastSquares.load(npz_path)

        with open(os.path.join(path, "als_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self._u2i = meta["u2i"]
        self._i2i = meta["i2i"]
        self._i2item = meta["i2item"]

        matrix_path = os.path.join(path, "als_matrix.npz")
        if os.path.exists(matrix_path):
            self._matrix = load_npz(matrix_path)
            logger.info(f"ALS loaded from {path}, matrix={self._matrix.shape}")
        else:
            logger.warning(f"No matrix at {matrix_path} — call rebuild_matrix() before recommending.")
        return self
