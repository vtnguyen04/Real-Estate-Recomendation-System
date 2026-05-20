import polars as pl
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, List, Optional

from config.settings import PipelineConfig

class ALSMatrixBuilder:
    """
    Centralized loader to build the ALS CSR matrix from pre-cached contact and pageview pairs.
    Adheres to DRY by preventing train/eval/infer scripts from duplicating this logic.
    """
    
    @staticmethod
    def build(als_contacts: pl.DataFrame, als_pageviews: pl.DataFrame, config: Optional[PipelineConfig] = None) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int], List[str]]:
        if config is None:
            config = PipelineConfig()
            
        # Fast ALS matrix build
        contact_scored = als_contacts.with_columns((pl.col("score").cast(pl.Float32) * config.model.als_contact_weight).alias("score"))
        pv_scored = als_pageviews.with_columns(pl.col("view_count").cast(pl.Float32).clip(config.model.als_pageview_min_weight, config.model.als_pageview_max_weight).alias("score")).select(["user_id", "item_id", "score"])
        
        combined = pl.concat([
            contact_scored.select(["user_id", "item_id", "score"]), 
            pv_scored
        ]).group_by(["user_id", "item_id"]).agg(pl.col("score").sum().alias("score"))
        
        users_list = combined["user_id"].unique().to_list()
        items_list = combined["item_id"].unique().to_list()
        
        u2i = {u: i for i, u in enumerate(users_list)}
        i2i_map = {it: i for i, it in enumerate(items_list)}
        
        r_idx = np.array([u2i[u] for u in combined["user_id"].to_list()], dtype=np.int32)
        c_idx = np.array([i2i_map[it] for it in combined["item_id"].to_list()], dtype=np.int32)
        vals = combined["score"].to_numpy().astype(np.float32)
        
        matrix = csr_matrix((vals, (r_idx, c_idx)), shape=(len(users_list), len(items_list)))
        return matrix, u2i, i2i_map, items_list
    

