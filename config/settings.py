"""
Central configuration for the Real Estate Recommendation System.
All hyperparameters, model configs, and operational settings are defined here.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataConfig:
    """Configuration for data ingestion and processing."""
    bucket_name: str = "datathon_2026_final"
    train_path: str = "gs://datathon_2026_final/train/"
    test_path: str = "gs://datathon_2026_final/test/"
    cutoff_date: str = "2026-04-09"
    positive_events: List[str] = field(
        default_factory=lambda: ["view_phone", "contact_chat", "contact_zalo", "contact_sms"]
    )
    min_valid_dwell_sec: float = 3.0
    bot_score_threshold: int = 4


@dataclass
class ModelConfig:
    """Configuration for candidate generation models."""
    als_factors: int = 64
    als_iterations: int = 15
    als_regularization: float = 0.1
    ensemble_weights: Dict[str, float] = field(
        default_factory=lambda: {"als": 0.9, "popularity": 0.1}
    )


@dataclass
class RankerConfig:
    """Configuration for the LightGBM ranker."""
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    lambdarank_truncation_level: int = 10


@dataclass
class RerankerConfig:
    """Configuration for multi-objective reranking."""
    alpha: float = 0.65   # Accuracy weight
    beta: float = 0.15    # Diversity weight
    gamma: float = 0.15   # Fairness weight
    delta: float = 0.05   # Freshness weight


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    validation_days: int = 3
    top_k: int = 10
