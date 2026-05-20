import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from datetime import datetime
from typing import Dict, List, Optional, Any

class InteractionMatrixBuilder:
    """
    Constructs a Compressed Sparse Row (CSR) matrix from user interactions
    for use in Collaborative Filtering (ALS) models.
    
    Handles:
    - Business-logic event weighting (Explicit contacts > Implicit views).
    - Temporal decay (Recent events have higher weights).
    - Sparse encoding (Mapping UUIDs to integer indices efficiently).
    """
    def __init__(self, 
                 half_life_days: float = 14.0,
                 config: Optional[Dict[str, Any]] = None):
        self.half_life_days = half_life_days
        self.config = config or {}
        
        self.event_weights = self.config.get('interaction_event_weights', {
            'pageview': 1.0,
            'other_interaction': 1.5,
            'view_phone': 5.0,
            'contact_chat': 5.0,
            'contact_zalo': 5.0,
            'contact_sms': 5.0,
            'lead': 10.0
        })
        
        # Idx mappings for retrieval
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_user = np.array([])
        self.idx_to_item = np.array([])

    def build(self, events: pl.LazyFrame, current_date: datetime) -> csr_matrix:
        """
        Builds the sparse matrix applying weights and time decay.
        Processes everything in LazyFrame before materializing the aggregated pairs.
        """
        # Ensure timestamp is available for decay
        schema = events.collect_schema().names()
        ts_cols = ["timestamp", "event_ts", "date", "last_date"]
        ts_col = next((c for c in ts_cols if c in schema), None)
        
        # Create weights DataFrame for fast join
        weight_df = pl.DataFrame({
            "event_type": list(self.event_weights.keys()),
            "base_weight": list(self.event_weights.values())
        }).lazy()
        
        # 1. Join weights based on event_type
        if "event_type" in schema:
            df = events.join(weight_df, on="event_type", how="left")
            df = df.with_columns(pl.col("base_weight").fill_null(1.0))
        else:
            # Fallback if no event_type exists
            df = events.with_columns(pl.lit(1.0).alias("base_weight"))
        
        # 2. Temporal Decay (Exponential Decay)
        if ts_col is not None:
            # Weight = Base * (0.5 ^ (age_days / half_life))
            df = df.with_columns([
                ((pl.lit(current_date).cast(pl.Datetime) - pl.col(ts_col).cast(pl.Datetime))
                 .dt.total_milliseconds() / (1000.0 * 60 * 60 * 24)).alias("age_days")
            ])
            
            # Cap age_days at 0
            df = df.with_columns([
                pl.when(pl.col("age_days") < 0).then(0.0).otherwise(pl.col("age_days")).alias("age_days")
            ])
            
            # Apply exponential decay
            df = df.with_columns([
                (pl.col("base_weight") * (0.5 ** (pl.col("age_days") / self.half_life_days))).alias("final_weight")
            ])
        else:
            if "score" in schema:
                df = df.with_columns(pl.col("score").alias("final_weight"))
            else:
                df = df.with_columns(pl.col("base_weight").alias("final_weight"))
        
        # 3. Aggregate implicitly to unique (user, item) pairs
        # This drastically reduces the 52GB dataset into a manageable size before collecting to RAM
        agg_df = df.group_by(["user_id", "item_id"]).agg([
            pl.col("final_weight").sum().alias("interaction_score")
        ])
        
        # Collect to memory (Only unique user-item pairs are materialized)
        collected = agg_df.collect()
        
        # Build mappings ONLY if not already built
        if not self.user_to_idx:
            unique_users = collected["user_id"].unique().to_list()
            self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
            self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
            
        if not self.item_to_idx:
            unique_items = collected["item_id"].unique().to_list()
            self.item_to_idx = {it: i for i, it in enumerate(unique_items)}
            self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}
        
        # 4. Map String IDs to Integer Indices for Sparse Matrix
        users = collected["user_id"].to_numpy()
        items = collected["item_id"].to_numpy()
        scores = collected["interaction_score"].to_numpy()
        
        # Vectorized mapping using numpy
        # For huge datasets, we use numpy searchsorted or list comprehensions
        # Dictionary comprehension is extremely fast in Python for 10M records
        valid_mask = np.array([u in self.user_to_idx and it in self.item_to_idx for u, it in zip(users, items)])
        users = users[valid_mask]
        items = items[valid_mask]
        scores = scores[valid_mask]
        
        user_indices = np.array([self.user_to_idx[u] for u in users], dtype=np.int32)
        item_indices = np.array([self.item_to_idx[it] for it in items], dtype=np.int32)
        
        # 5. Build SciPy CSR Matrix
        # Shape: [num_users, num_items]
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        shape = (num_users, num_items)
        matrix = csr_matrix((scores, (user_indices, item_indices)), shape=shape)
        
        return matrix

    def get_user_id(self, idx: int) -> str:
        return self.idx_to_user[idx]

    def get_item_id(self, idx: int) -> str:
        return self.idx_to_item[idx]
        
    def get_user_idx(self, user_id: str) -> int:
        return self.user_to_idx.get(user_id, -1)
        
    def get_item_idx(self, item_id: str) -> int:
        return self.item_to_idx.get(item_id, -1)
