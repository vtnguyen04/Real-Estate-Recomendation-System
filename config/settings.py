"""
Central configuration for the Real Estate Recommendation System.
All hyperparameters, model configs, and operational settings are defined here.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class DataConfig:
    """Configuration for data ingestion and processing."""
    bucket_name: str = "datathon_2026_final"
    train_path: str = "/home/db/rc/datathon/train/"
    test_path: str = "/home/db/rc/datathon/test/"
    cutoff_date: str = "2026-04-09"
    positive_events: List[str] = field(
        default_factory=lambda: [
            "view_phone", "contact_chat", "contact_zalo", "contact_sms", "other_interaction"
        ]
    )
    # EDA R02 H-003: dwell_time_sec is actually milliseconds. 3s = 3000ms raw.
    min_valid_dwell_ms: float = 3000.0
    bot_score_threshold: int = 4
    session_min_items: int = 2
    session_max_items: int = 30


@dataclass
class ModelConfig:
    """Configuration for candidate generation models."""
    # Contact ALS
    als_factors: int = 256
    als_iterations: int = 30
    als_regularization: float = 0.01
    # View ALS
    als_view_factors: int = 64
    als_view_iterations: int = 20
    # Candidate counts PER USER (INS-025: 85.5% GT items are new → SegPop is PRIMARY)
    n_cand_als: int = 50         # secondary: only covers 14.5% of GT
    n_cand_view_als: int = 50    # secondary: browsing signal
    n_cand_segpop: int = 200     # PRIMARY: city+category popularity is the core signal
    # Legacy / unused
    als_contact_weight: float = 5.0
    als_pageview_min_weight: float = 1.0
    als_pageview_max_weight: float = 3.0
    ensemble_weights: Dict[str, float] = field(
        default_factory=lambda: {"als": 0.9, "popularity": 0.1}
    )


@dataclass
class RankerConfig:
    """Configuration for the LightGBM lambdarank ranker."""
    num_leaves: int = 127
    learning_rate: float = 0.01
    n_estimators: int = 2000
    early_stopping_rounds: int = 100
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    min_child_samples: int = 20
    lambdarank_truncation_level: int = 10
    feature_cols: List[str] = field(default_factory=lambda: [
        "score_als", "score_view_als", "score_segpop",
        "is_from_als", "is_from_view_als", "is_from_segpop",
        "event_count", "contact_rate",
        "item_total_contacts", "item_total_views",
        "item_recent_contacts", "item_recency_score", "item_novelty_score",
        "score_prev", "score_seller",
        "item_completeness", "item_photos", "item_has_so_hong",
        "item_is_apartment", "item_is_agent", "item_has_noi_that_cao_cap",
        "item_cat", "item_city",
        "city_match", "cat_match",
        "price_match", "ad_type_match", "listing_age_days",
    ])
    # Legacy multi-task fields (not used by lambdarank pipeline)
    contact_weights: List[float] = field(default_factory=lambda: [0.0, 0.8, 0.5, 1.0, 0.9])
    lgbm_bin_weight: float = 0.7
    lgbm_multi_weight: float = 0.3


@dataclass
class RerankerConfig:
    """Configuration for multi-objective reranking."""
    alpha: float = 0.90   # Accuracy weight (Tuned HIGH for Recall)
    beta: float = 0.05    # Diversity weight
    gamma: float = 0.05   # Fairness weight
    delta: float = 0.00   # Freshness weight
    epsilon: float = 0.00 # Novelty weight


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    validation_days: int = 3
    top_k: int = 10
    top_n_for_rerank: int = 30
    negative_sample_ratio: int = 3
    candidate_half_life_days: float = 7.0
    # SegPop POOL sizes — how many items stored per segment
    # INS-027: GT has 28,706 unique items. Old pool of 50/segment covered only 21.6%.
    # Must store 500+ items per (city,category) to actually cover GT items.
    segpop_global_k: int = 500
    segpop_segment_k: int = 500    # per city
    segpop_cc_k: int = 500         # per (city, category) — the critical segment
    segpop_ccd_k: int = 100        # per (city, category, district)
    # Training pipeline
    n_train_users: int = 50_000
    val_sample: int = 2_000
    cand_batch: int = 5_000
    positive_window_days: int = 14

    def to_dict(self) -> Dict[str, Any]:
        """Flattens the config into a single dictionary."""
        flat_dict = {
            "validation_days": self.validation_days,
            "top_k": self.top_k,
            "top_n_for_rerank": self.top_n_for_rerank,
            "negative_sample_ratio": self.negative_sample_ratio,
            "candidate_half_life_days": self.candidate_half_life_days,
            "segpop_global_k": self.segpop_global_k,
            "segpop_segment_k": self.segpop_segment_k,
            "segpop_cc_k": self.segpop_cc_k,
            "segpop_ccd_k": self.segpop_ccd_k,
        }

        # Add DataConfig
        for k, v in self.data.__dict__.items(): flat_dict[f"data_{k}"] = v
        # Add ModelConfig (aliased to candidate_ for backwards compatibility)
        for k, v in self.model.__dict__.items(): flat_dict[f"candidate_{k.replace('als_', '')}"] = v
        # Add RankerConfig
        for k, v in self.ranker.__dict__.items(): flat_dict[k] = v
        # Add RerankerConfig
        for k, v in self.reranker.__dict__.items(): flat_dict[f"rerank_{k}"] = v

        return flat_dict
