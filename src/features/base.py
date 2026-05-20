from abc import ABC, abstractmethod
from typing import Dict, Optional, TYPE_CHECKING
import polars as pl

if TYPE_CHECKING:
    from src.features.feature_context import FeatureContext


class BaseHeuristicExtractor(ABC):
    """
    SOLID interface for all feature extractors.

    Each subclass is responsible for ONE feature group.
    Two modes:
      - Inference: extract_scores() per user (O(1) dict lookups)
      - Training:  build_feature_df() returns a polars DataFrame for batch joins
    Pairwise extractors override compute_match_features() for (user, item) pair features.
    """

    @abstractmethod
    def extract_scores(
        self,
        uid: str,
        context: "FeatureContext",
        features_dict: Dict[str, Dict[str, float]],
    ) -> None:
        """Update features_dict in-place with scores for all candidate items of uid."""

    def build_feature_df(self, context: "FeatureContext") -> Optional[pl.DataFrame]:
        """
        Return a polars DataFrame for efficient batch joins during training.
        Column 'join_key' must match self.join_key.
        Return None if this extractor only works in inference mode.
        """
        return None

    def compute_match_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute pairwise (user, item) features on the joined DataFrame.
        Default: no-op (return unchanged). Override for pairwise extractors.
        """
        return df

    @property
    def join_key(self) -> str:
        """Column to join on when assembling the training feature matrix."""
        return "item_id"
