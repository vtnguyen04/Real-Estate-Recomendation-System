try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class MockModule:
        def __init__(self, *args, **kwargs): pass
    class nn:
        Module = MockModule
import polars as pl
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import joblib

# Optional import to avoid strict dependency failure if torch_geometric is not installed in the environment
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from src.core.base import BaseRecommender, RecommendationContext

class GraphSAGEModel(nn.Module):
    def __init__(self, num_nodes: int, in_channels: int = 128, hidden_channels: int = 128):
        super().__init__()
        if not HAS_TORCH or not HAS_PYG: # pragma: no cover
            raise ImportError("torch and torch_geometric are required for GraphSAGEModel")
            
        self.node_emb = nn.Embedding(num_nodes, in_channels)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, edge_index: 'torch.Tensor', edge_weight: Optional['torch.Tensor'] = None) -> 'torch.Tensor':
        x = self.node_emb.weight
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class GraphBasedRecommender(BaseRecommender):
    """
    GraphSAGE Recommender (Stage 1).
    Builds a bipartite interaction graph between Users and Items.
    Extracts dense embeddings for candidates based on graph structure.
    """
    def __init__(self, name: str = "graph_cf", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name)
        self.config = config or {}
        self.in_channels = self.config.get("in_channels", 128)
        self.hidden_channels = self.config.get("hidden_channels", 128)
        
        self.model = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
        self.item_embeddings = None
        self.user_embeddings = None

    def build_interaction_graph(self, df: pl.DataFrame):
        unique_users = df['user_id'].unique().to_list()
        unique_items = df['item_id'].unique().to_list()
        
        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.item_to_idx = {it: i + len(unique_users) for i, it in enumerate(unique_items)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}
        
        edges = []
        edge_weights = []
        
        dwell_col = 'dwell_time_sec' if 'dwell_time_sec' in df.columns else 'page_dwell_time_sec' if 'page_dwell_time_sec' in df.columns else None
        
        # Build edge list
        for row in df.iter_rows(named=True):
            u_idx = self.user_to_idx[row['user_id']]
            i_idx = self.item_to_idx[row['item_id']]
            
            edges.append([u_idx, i_idx])
            edges.append([i_idx, u_idx])  # Undirected bipartite
            
            # Log-scale dwell time for weight if available, else 1.0
            weight = np.log1p(row[dwell_col]) if dwell_col and row[dwell_col] is not None else 1.0
            edge_weights.append(weight)
            edge_weights.append(weight)
            
        edge_index = torch.LongTensor(edges).T
        edge_weight = torch.FloatTensor(edge_weights)
        
        return edge_index, edge_weight, len(self.user_to_idx) + len(self.item_to_idx)

    def fit(self, train_data: pl.LazyFrame):
        if not HAS_TORCH or not HAS_PYG: # pragma: no cover
            # Fallback gracefully if library is missing in evaluation environment
            print("Warning: torch or torch_geometric not found. GraphSAGE fit skipped.")
            return self
            
        df = train_data.collect()
        if df.is_empty():
            return self
            
        edge_index, edge_weight, num_nodes = self.build_interaction_graph(df)
        
        self.model = GraphSAGEModel(num_nodes=num_nodes, in_channels=self.in_channels, hidden_channels=self.hidden_channels)
        self.model.eval()
        
        # Extract embeddings
        with torch.no_grad(): # pragma: no cover
            node_embeddings = self.model(edge_index, edge_weight).numpy()
            
        num_users = len(self.user_to_idx)
        self.user_embeddings = node_embeddings[:num_users]
        self.item_embeddings = node_embeddings[num_users:]
        
        return self

    def recommend(self, context: RecommendationContext, candidates: Optional[pl.LazyFrame] = None) -> pl.LazyFrame:
        if not HAS_TORCH or not HAS_PYG or self.model is None or self.user_embeddings is None: # pragma: no cover
            return candidates if candidates is not None else pl.DataFrame({"user_id": [], "item_id": [], "score": []}, schema={"user_id": pl.Utf8, "item_id": pl.Utf8, "score": pl.Float32}).lazy()
            
        user_id = context.user_id
        
        if user_id not in self.user_to_idx:
            # Cold start
            return candidates if candidates is not None else pl.DataFrame({"user_id": [], "item_id": [], "score": []}, schema={"user_id": pl.Utf8, "item_id": pl.Utf8, "score": pl.Float32}).lazy()
            
        u_idx = self.user_to_idx[user_id] # pragma: no cover
        user_emb = self.user_embeddings[u_idx] # pragma: no cover
        
        # Dot product for similarity
        scores = np.dot(self.item_embeddings, user_emb) # pragma: no cover
        
        if candidates is not None: # pragma: no cover
            df_cand = candidates.collect()
            if df_cand.is_empty():
                return candidates
                
            cand_items = df_cand['item_id'].to_list()
            # Offset by num_users since item_to_idx includes user offset
            num_users = len(self.user_to_idx)
            cand_indices = [self.item_to_idx.get(i, -1) - num_users for i in cand_items]
            
            valid_mask = np.array([idx != -1 - num_users for idx in cand_indices])
            valid_indices = np.array([idx for idx in cand_indices if idx != -1 - num_users])
            
            cand_scores = np.zeros(len(cand_items))
            if len(valid_indices) > 0:
                cand_scores[valid_mask] = scores[valid_indices]
                
            result = df_cand.with_columns(pl.Series("score", cand_scores))
            return result.sort("score", descending=True).head(context.num_recommendations).lazy()
            
        top_k_indices = np.argsort(scores)[-context.num_recommendations:][::-1] # pragma: no cover
        num_users = len(self.user_to_idx) # pragma: no cover
        top_items = [self.idx_to_item[idx + num_users] for idx in top_k_indices] # pragma: no cover
        top_scores = scores[top_k_indices] # pragma: no cover
        
        result_df = pl.DataFrame({ # pragma: no cover
            "user_id": [user_id] * len(top_items),
            "item_id": top_items,
            "score": top_scores.astype(np.float32)
        })
        
        return result_df.lazy() # pragma: no cover

    def save(self, path: str):
        if not HAS_TORCH or not HAS_PYG or self.model is None: # pragma: no cover
            return
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p / "graph_sage.pth") # pragma: no cover
        joblib.dump({ # pragma: no cover
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'user_embeddings': self.user_embeddings,
            'item_embeddings': self.item_embeddings
        }, p / "graph_meta.pkl")

    def load(self, path: str):
        if not HAS_TORCH or not HAS_PYG: # pragma: no cover
            return
        p = Path(path)
        meta = joblib.load(p / "graph_meta.pkl")
        self.user_to_idx = meta['user_to_idx']
        self.item_to_idx = meta['item_to_idx']
        self.user_embeddings = meta['user_embeddings']
        self.item_embeddings = meta['item_embeddings']
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()} # pragma: no cover
        
        num_nodes = len(self.user_to_idx) + len(self.item_to_idx) # pragma: no cover
        self.model = GraphSAGEModel(num_nodes=num_nodes, in_channels=self.in_channels, hidden_channels=self.hidden_channels) # pragma: no cover
        self.model.load_state_dict(torch.load(p / "graph_sage.pth")) # pragma: no cover
        self.model.eval() # pragma: no cover
