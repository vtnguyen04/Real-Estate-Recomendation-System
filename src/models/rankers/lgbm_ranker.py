import os
import pickle
import lightgbm as lgb
import numpy as np
import polars as pl
from typing import Dict, List, Any, Optional
from src.core.base import BaseRecommender, RecommendationContext
from pathlib import Path
import joblib

from src.utils.polars_utils import prepare_features_for_lgbm

class MultiTaskLGBMRanker(BaseRecommender):
    """
    Multi-task LightGBM Ranker (Stage 3).
    Combines:
    1. Binary Ranking (LambdaRank) - optimizes for any interaction.
    2. Multi-class Classification - predicts specific contact type (phone, chat, zalo, sms).
    
    Final Score = 0.7 * binary_score + 0.3 * multiclass_weighted_score
    """
    def __init__(self, name: str = "lgbm_ranker", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name)
        self.config = config or {}
        self.model_binary: Any = None
        self.model_multiclass: Any = None
        self.feature_cols = []
        
        # Contact weights defined in playbook
        # 0: no_interact, 1: view_phone, 2: chat, 3: zalo, 4: sms
        weights = self.config.get('contact_weights', [0.0, 0.8, 0.5, 1.0, 0.9])
        self.contact_weights = np.array(weights)

    def fit(self, train_data: pl.LazyFrame, val_data: Optional[pl.LazyFrame] = None, **kwargs) -> 'BaseRecommender':
        """
        Expects a collected DataFrame with labels:
        - 'label_binary': 0/1
        - 'label_multiclass': 0-4
        - 'group_id': user_id or session_id for ranking groups
        """
        df = train_data.collect()
        
        # We need query groups for ranking, must sort by group_id (user_id)
        df = df.sort("user_id")
        
        if 'label_multiclass' not in df.columns:
            df = df.with_columns(pl.lit(0).alias('label_multiclass'))
            
        if self.feature_cols is None or len(self.feature_cols) == 0:
            # Exclude metadata and labels
            exclude = ['user_id', 'item_id', 'label', 'label_multiclass', 'group_id', 'event_ts']
            self.feature_cols = [c for c in df.columns if c not in exclude]
            
        X = df.select(self.feature_cols).to_pandas()
        y_bin = df['label'].to_numpy()
        y_multi = df['label_multiclass'].to_numpy()
        
        # Must maintain order for lightgbm groups
        groups = df.group_by('user_id', maintain_order=True).agg(pl.len()).select('len').to_numpy().flatten()
        
        # 1. Train Binary Ranker
        train_set_bin = lgb.Dataset(X, label=y_bin, group=groups)
        valid_sets_bin = [train_set_bin]
        
        if val_data is not None:
            df_val = val_data.collect()
            df_val = df_val.sort("user_id")
            if 'label_multiclass' not in df_val.columns:
                df_val = df_val.with_columns(pl.lit(0).alias('label_multiclass'))
            X_val = df_val.select(self.feature_cols).to_pandas()
            y_bin_val = df_val['label'].to_numpy()
            y_multi_val = df_val['label_multiclass'].to_numpy()
            groups_val = df_val.group_by('user_id', maintain_order=True).agg(pl.len()).select('len').to_numpy().flatten()
            val_set_bin = lgb.Dataset(X_val, label=y_bin_val, group=groups_val, reference=train_set_bin)
            valid_sets_bin.append(val_set_bin)
        
        params_bin = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [10],
            'learning_rate': self.config.get('lr_bin', 0.05),
            'num_leaves': self.config.get('num_leaves_bin', 64),
            'feature_fraction': 0.8,
            'verbose': 1,
            'device_type': 'gpu' if self.config.get('use_gpu', False) else 'cpu'
        }
        
        callbacks = [lgb.log_evaluation(period=20)]
        if val_data is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=30, first_metric_only=True))
            
        self.model_binary = lgb.train(
            params_bin, 
            train_set_bin, 
            num_boost_round=self.config.get('rounds_bin', 500),
            valid_sets=valid_sets_bin,
            valid_names=['train', 'valid'] if val_data is not None else ['train'],
            callbacks=callbacks
        )
        
        # 2. Train Multi-class Classifier
        train_set_multi = lgb.Dataset(X, label=y_multi)
        valid_sets_multi = [train_set_multi]
        if val_data is not None:
            val_set_multi = lgb.Dataset(X_val, label=y_multi_val, reference=train_set_multi)
            valid_sets_multi.append(val_set_multi)
            
        params_multi = {
            'objective': 'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'learning_rate': self.config.get('lr_multi', 0.05),
            'num_leaves': self.config.get('num_leaves_multi', 64),
            'verbose': 1,
            'device_type': 'gpu' if self.config.get('use_gpu', False) else 'cpu'
        }
        
        self.model_multiclass = lgb.train(
            params_multi, 
            train_set_multi, 
            num_boost_round=self.config.get('rounds_multi', 300),
            valid_sets=valid_sets_multi,
            valid_names=['train', 'valid'] if val_data is not None else ['train'],
            callbacks=callbacks
        )
        return self

    def recommend(self, context: RecommendationContext, candidates: Optional[pl.LazyFrame] = None) -> pl.LazyFrame:
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
        
        # Combine scores using configurable weights
        bin_weight = self.config.get('lgbm_bin_weight', 0.7)
        multi_weight = self.config.get('lgbm_multi_weight', 0.3)
        final_scores = bin_weight * bin_scores + multi_weight * multi_scores
        
        # Add to dataframe and sort
        result = df_cand.with_columns(pl.Series("score", final_scores))
        return result.sort("score", descending=True).lazy()

    def save(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.model_binary.save_model(str(p / "binary_ranker.txt"))
        self.model_multiclass.save_model(str(p / "multiclass_model.txt"))
        joblib.dump(self.feature_cols, p / "features.pkl")

    def load(self, path: str) -> 'BaseRecommender':
        p = Path(path)
        self.model_binary = lgb.Booster(model_file=str(p / "binary_ranker.txt"))
        self.model_multiclass = lgb.Booster(model_file=str(p / "multiclass_model.txt"))
        self.feature_cols = joblib.load(p / "features.pkl")
        return self


class LambdarankLGBMRanker(BaseRecommender):
    """
    Single-task lambdarank LightGBM ranker for two-stage recommendation.

    fit() accepts polars DataFrames with columns: user_id, item_id, label, + feature_cols.
    predict() returns raw LightGBM scores as a numpy array.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        use_gpu: bool = False,
        num_leaves: int = 127,
        learning_rate: float = 0.02,
        num_rounds: int = 1000,
        early_stopping: int = 50,
    ):
        super().__init__(name="lgbm_lambdarank")
        self.feature_cols = feature_cols or []
        self.use_gpu = use_gpu
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.num_rounds = num_rounds
        self.early_stopping = early_stopping
        self._model: Optional[lgb.Booster] = None

    def fit(self, train_df: pl.DataFrame, val_df: Optional[pl.DataFrame] = None) -> "LambdarankLGBMRanker":
        train_df = train_df.sort("user_id")
        X = prepare_features_for_lgbm(train_df, self.feature_cols)
        y = train_df["label"].to_numpy()
        groups = (
            train_df.group_by("user_id", maintain_order=True)
            .agg(pl.len()).select("len").to_numpy().flatten()
        )
        lgb_ds = lgb.Dataset(X, label=y, group=groups, free_raw_data=False)

        valid_sets = [lgb_ds]
        valid_names = ["train"]
        callbacks: list = [lgb.log_evaluation(50)]

        if val_df is not None and len(val_df) > 0:
            val_df = val_df.sort("user_id")
            X_val = prepare_features_for_lgbm(val_df, self.feature_cols)
            y_val = val_df["label"].to_numpy()
            groups_val = (
                val_df.group_by("user_id", maintain_order=True)
                .agg(pl.len()).select("len").to_numpy().flatten()
            )
            val_ds = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=lgb_ds)
            valid_sets.append(val_ds)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(self.early_stopping, first_metric_only=True))

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.9,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "verbose": 1,
            "device_type": "gpu" if self.use_gpu else "cpu",
            "seed": 42,
        }
        self._model = lgb.train(
            params, lgb_ds,
            num_boost_round=self.num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        X = prepare_features_for_lgbm(df, self.feature_cols)
        return np.array(self._model.predict(X), dtype=np.float64)

    def recommend(self, context: RecommendationContext, candidates=None) -> pl.LazyFrame:
        raise NotImplementedError("Use predict() for batch inference via InferencePipeline.")

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self._model.save_model(os.path.join(path, "lgbm_ranker.txt"))
        with open(os.path.join(path, "train_meta.pkl"), "wb") as f:
            pickle.dump({"feature_cols": self.feature_cols}, f)

    def load(self, path: str) -> "LambdarankLGBMRanker":
        self._model = lgb.Booster(model_file=os.path.join(path, "lgbm_ranker.txt"))
        with open(os.path.join(path, "train_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        self.feature_cols = meta["feature_cols"]
        return self
