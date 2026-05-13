import lightgbm as lgb
import numpy as np
import polars as pl
from typing import Dict, List, Any, Optional
from src.core.base import BaseRecommender, RecommendationContext
from pathlib import Path
import joblib

class MultiTaskLGBMRanker(BaseRecommender):
    """
    Multi-task LightGBM Ranker (Stage 3).
    Combines:
    1. Binary Ranking (LambdaRank) - optimizes for any interaction.
    2. Multi-class Classification - predicts specific contact type (phone, chat, zalo, sms).
    
    Final Score = 0.7 * binary_score + 0.3 * multiclass_weighted_score
    """
    def __init__(self, name: str = "lgbm_ranker", config: Dict[str, Any] = None):
        super().__init__(name=name)
        self.config = config or {}
        self.model_binary = None
        self.model_multiclass = None
        self.feature_cols = []
        
        # Contact weights defined in playbook
        # 0: no_interact, 1: view_phone, 2: chat, 3: zalo, 4: sms
        self.contact_weights = np.array([0.0, 0.8, 0.5, 1.0, 0.9])

    def fit(self, train_data: pl.LazyFrame):
        """
        Expects a collected DataFrame with labels:
        - 'label_binary': 0/1
        - 'label_multiclass': 0-4
        - 'group_id': user_id or session_id for ranking groups
        """
        df = train_data.collect()
        
        if self.feature_cols is None or len(self.feature_cols) == 0:
            # Exclude metadata and labels
            exclude = ['user_id', 'item_id', 'label_binary', 'label_multiclass', 'group_id', 'event_ts']
            self.feature_cols = [c for c in df.columns if c not in exclude]
            
        X = df.select(self.feature_cols).to_pandas()
        y_bin = df['label_binary'].to_numpy()
        y_multi = df['label_multiclass'].to_numpy()
        groups = df.group_by('group_id').agg(pl.len()).select('len').to_numpy().flatten()
        
        # 1. Train Binary Ranker
        train_set_bin = lgb.Dataset(X, label=y_bin, group=groups)
        params_bin = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [10],
            'learning_rate': self.config.get('lr_bin', 0.05),
            'num_leaves': self.config.get('num_leaves_bin', 64),
            'feature_fraction': 0.8,
            'verbose': -1
        }
        self.model_binary = lgb.train(params_bin, train_set_bin, num_boost_round=self.config.get('rounds_bin', 500))
        
        # 2. Train Multi-class Classifier
        train_set_multi = lgb.Dataset(X, label=y_multi)
        params_multi = {
            'objective': 'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'learning_rate': self.config.get('lr_multi', 0.05),
            'num_leaves': self.config.get('num_leaves_multi', 64),
            'verbose': -1
        }
        self.model_multiclass = lgb.train(params_multi, train_set_multi, num_boost_round=self.config.get('rounds_multi', 300))

    def recommend(self, context: RecommendationContext, candidates: pl.LazyFrame = None) -> pl.LazyFrame:
        if candidates is None:
            return pl.LazyFrame([]) # Ranker needs candidates
            
        # Collect candidates to pandas for LightGBM inference
        df_cand = candidates.collect()
        if df_cand.is_empty():
            return candidates
            
        X_test = df_cand.select(self.feature_cols).to_pandas()
        
        # Predict
        bin_scores = self.model_binary.predict(X_test)
        multi_probs = self.model_multiclass.predict(X_test) # [N, 5]
        
        # Weighted multi-class score
        multi_scores = (multi_probs * self.contact_weights).sum(axis=1)
        
        # Combine scores
        final_scores = 0.7 * bin_scores + 0.3 * multi_scores
        
        # Add to dataframe and sort
        result = df_cand.with_columns(pl.Series("score", final_scores))
        return result.sort("score", descending=True).lazy()

    def save(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.model_binary.save_model(str(p / "binary_ranker.txt"))
        self.model_multiclass.save_model(str(p / "multiclass_model.txt"))
        joblib.dump(self.feature_cols, p / "features.pkl")

    def load(self, path: str):
        p = Path(path)
        self.model_binary = lgb.Booster(model_file=str(p / "binary_ranker.txt"))
        self.model_multiclass = lgb.Booster(model_file=str(p / "multiclass_model.txt"))
        self.feature_cols = joblib.load(p / "features.pkl")
