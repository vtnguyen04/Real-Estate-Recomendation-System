import polars as pl
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.core.base import BaseRecommender, RecommendationContext, Recommendation

class ContentRecommender(BaseRecommender):
    """
    Content-Based Recommender using TF-IDF on textual features (e.g., titles).
    Recommends items that are textually similar to what the user historically viewed.
    """
    def __init__(self, top_k: int = 100, max_features: int = 5000):
        super().__init__(name="content_recommender")
        self.top_k = top_k
        self.max_features = max_features
        # TF-IDF maps text to sparse vectors
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words='english')
        
        self.tfidf_matrix = None
        self.item_ids = []
        self.item_to_idx = {}
        
        # Maps user_id -> List of item indices they interacted with
        self.user_history = {}

    def fit(self, train_data: pl.LazyFrame, **kwargs) -> 'BaseRecommender':
        """
        Trains the content recommender.
        Args:
            train_data: LazyFrame containing item snapshots/dimensions (must have 'item_id' and 'title').
            kwargs:
                interactions: LazyFrame of user interactions.
        """
        schema = train_data.collect_schema().names()
        if "item_id" not in schema or "title" not in schema:
            return self
            
        # 1. Fit TF-IDF on item text
        items_df = train_data.select(["item_id", "title"]).drop_nulls().collect()
        self.item_ids = items_df["item_id"].to_list()
        self.item_to_idx = {item: idx for idx, item in enumerate(self.item_ids)}
        
        corpus = items_df["title"].to_list()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        # 2. Build User History from interactions
        interactions = kwargs.get("interactions", None)
        if interactions is not None:
            # Drop duplicates to simply track set of interacted items
            hist_df = interactions.select(["user_id", "item_id"]).unique().collect()
            
            for row in hist_df.iter_rows(named=True):
                u = row["user_id"]
                i = row["item_id"]
                if u not in self.user_history:
                    self.user_history[u] = []
                if i in self.item_to_idx:
                    self.user_history[u].append(self.item_to_idx[i])
                    
        return self

    def recommend(
        self,
        context: RecommendationContext,
        candidates: Optional[pl.LazyFrame] = None
    ) -> pl.LazyFrame:
        
        if not self.item_ids or self.tfidf_matrix is None:
            return pl.DataFrame([]).lazy()
            
        user_idx_history = self.user_history.get(context.user_id, [])
        if not user_idx_history:
            # Pure Cold-Start
            return pl.DataFrame([]).lazy()
            
        # Create user profile vector by averaging TF-IDF vectors of historically viewed items
        user_vector = self.tfidf_matrix[user_idx_history].mean(axis=0)
        
        # Compute cosine similarity against all items
        sim_scores = cosine_similarity(np.asarray(user_vector), self.tfidf_matrix).flatten()
        
        # Exclude items already interacted with
        sim_scores[user_idx_history] = -1.0
        
        # Extract top K indices using argpartition for O(N) performance
        k = min(context.num_recommendations, len(sim_scores))
        top_indices = np.argpartition(sim_scores, -k)[-k:]
        # Sort them in descending order
        top_indices = top_indices[np.argsort(sim_scores[top_indices])[::-1]]
        
        recs = [
            {
                "user_id": context.user_id,
                "item_id": self.item_ids[idx],
                "score": float(sim_scores[idx])
            }
            for idx in top_indices if sim_scores[idx] > 0
        ]
        return pl.DataFrame(recs).lazy()

    def save(self, path: str) -> None:
        import pickle
        with open(path, 'wb') as f:
            pickle.dump((self.vectorizer, self.tfidf_matrix, self.item_ids, self.item_to_idx, self.user_history), f)

    def load(self, path: str) -> 'BaseRecommender':
        import pickle
        with open(path, 'rb') as f:
            (self.vectorizer, self.tfidf_matrix, self.item_ids, self.item_to_idx, self.user_history) = pickle.load(f)
        return self
