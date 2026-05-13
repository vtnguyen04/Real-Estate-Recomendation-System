"""
Core interfaces and base classes for the recommendation system.
Defines contracts for models, feature extractors, and rules.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import polars as pl
from pydantic import BaseModel, Field

class RecommendationContext(BaseModel):
    """Context for recommendation request"""
    user_id: str
    timestamp: str  # Could also use datetime if parsing is desired
    num_recommendations: int = Field(default=10, gt=0)
    filters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class Recommendation(BaseModel):
    """Single recommendation item"""
    item_id: str
    score: float
    rank: int = Field(gt=0)
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# CORE INTERFACES
class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation models.
    Enforces interface for training and inference.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.metadata = kwargs

    @abstractmethod
    def fit(self, train_data: pl.LazyFrame, **kwargs) -> 'BaseRecommender':
        """
        Train the model

        Args:
            train_data: Training dataset (LazyFrame for memory efficiency)

        Returns:
            self (for method chaining)
        """
        pass

    @abstractmethod
    def recommend(
        self,
        context: RecommendationContext,
        candidates: Optional[pl.LazyFrame] = None
    ) -> List[Recommendation]:
        """
        Generate recommendations

        Args:
            context: User and request context
            candidates: Optional pre-filtered candidates to score

        Returns:
            List of recommendations sorted by score
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize model to disk"""
        pass

    @abstractmethod
    def load(self, path: str) -> 'BaseRecommender':
        """Load model from disk"""
        pass


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature engineering.
    Uses Template Method pattern to standardize extraction pipeline.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs

    def extract(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """
        Template method - defines extraction workflow
        """
        # Validate input
        self._validate_input(data)

        # Compute features
        features = self._compute_features(data)

        # Post-process
        features = self._post_process(features)

        return features

    @abstractmethod
    def _compute_features(self, data: pl.LazyFrame) -> pl.LazyFrame:
        """Implement feature computation logic"""
        pass

    def _validate_input(self, data: pl.LazyFrame) -> None:
        """
        Optional input validation.
        Should raise ValueError if required columns missing.
        """
        required = self.get_required_columns()
        if required:
            # We must collect schema instead of full dataframe for LazyFrame
            schema = data.collect_schema().names()
            missing = set(required) - set(schema)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

    def _post_process(self, features: pl.LazyFrame) -> pl.LazyFrame:
        """Optional post-processing (e.g., normalization, filling NAs)"""
        return features

    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Define required input columns"""
        pass


class BaseRule(ABC):
    """
    Abstract base class for business rules / post-processing filters.
    """

    def __init__(self, name: str, is_hard_filter: bool = True):
        self.name = name
        self.is_hard_filter = is_hard_filter

    @abstractmethod
    def apply(
        self,
        items: pl.LazyFrame,
        context: RecommendationContext
    ) -> pl.LazyFrame:
        """
        Apply rule to items

        Args:
            items: Current candidate items with scores
            context: Recommendation context

        Returns:
            Filtered or re-scored items
        """
        pass
