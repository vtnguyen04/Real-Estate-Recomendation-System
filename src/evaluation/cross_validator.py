import polars as pl
from datetime import timedelta
from typing import Tuple, Optional
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TimeBasedSplitter:
    """
    Splits interaction data into training and validation sets based on time.
    Crucial for recommender systems to avoid future data leakage.
    """
    def __init__(self, validation_days: int = 3, timestamp_col: str = "timestamp"):
        """
        Args:
            validation_days: Number of days at the end of the dataset to use for validation.
            timestamp_col: Name of the timestamp column.
        """
        self.validation_days = validation_days
        self.timestamp_col = timestamp_col

    def split(self, data: pl.LazyFrame) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Splits the LazyFrame into (train, validation).

        Args:
            data: Polars LazyFrame containing interaction data with a timestamp.

        Returns:
            Tuple of (train_lazyframe, val_lazyframe)
        """
        schema = data.collect_schema().names()
        if self.timestamp_col not in schema:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found in data schema: {schema}")

        logger.info(f"Splitting data temporally. Last {self.validation_days} days reserved for validation.")

        # Find the maximum timestamp in the dataset
        # We need to collect just this aggregate to know the split point
        max_ts_df = data.select(pl.col(self.timestamp_col).max()).collect()
        max_ts = max_ts_df[self.timestamp_col][0]

        if max_ts is None:
            raise ValueError("Max timestamp is None. Cannot split an empty dataset.")

        split_point = max_ts - timedelta(days=self.validation_days)
        logger.info(f"Data range ends at {max_ts}. Split point set to {split_point}.")

        # Create train and validation LazyFrames
        train_data = data.filter(pl.col(self.timestamp_col) < split_point)
        val_data = data.filter(pl.col(self.timestamp_col) >= split_point)

        return train_data, val_data
