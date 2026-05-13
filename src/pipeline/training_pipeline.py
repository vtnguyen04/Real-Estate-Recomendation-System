import polars as pl
from typing import Dict, Any, Optional
import os

from src.core.base import RecommendationContext
from src.evaluation.cross_validator import TimeBasedSplitter
from src.features.feature_engineer import FeatureEngineer
from src.pipeline.data_forensics import DataForensicsPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Orchestrates the entire training lifecycle:
    1. Forensics / Data Cleaning
    2. Temporal Train/Val split
    3. Feature Engineering & Matrix building
    4. Model Training (Candidates & Rankers)
    """
    def __init__(
        self,
        candidate_generator: Any,
        feature_engineer: FeatureEngineer,
        ranker: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        self.candidate_generator = candidate_generator
        self.feature_engineer = feature_engineer
        self.ranker = ranker
        self.config = config or {}
        
        self.forensics = DataForensicsPipeline()
        self.splitter = TimeBasedSplitter(
            validation_days=self.config.get("validation_days", 3),
            timestamp_col="timestamp"
        )

    def run(self, raw_events: pl.LazyFrame, item_profile: pl.LazyFrame) -> Dict[str, Any]:
        """
        Executes the training pipeline.
        
        Args:
            raw_events: LazyFrame of raw user interactions.
            item_profile: LazyFrame of item metadata.
            
        Returns:
            Dict containing trained models and validation metrics.
        """
        logger.info("Starting Training Pipeline...")

        # 1. Data Forensics (Clean raw data)
        logger.info("Step 1: Applying Data Forensics...")
        clean_events = self.forensics.clean(raw_events)

        # 2. Temporal Train/Val Split
        logger.info("Step 2: Performing Time-Based Split...")
        train_events, val_events = self.splitter.split(clean_events)

        # 3. Train Candidate Generators (Stage 1)
        logger.info("Step 3: Training Candidate Generator(s)...")
        self.candidate_generator.fit(train_events)

        # 4. Feature Engineering for Ranker (Stage 3)
        logger.info("Step 4: Building Feature Matrix for Ranker...")
        # To train the ranker, we need historical pairs of (user, item) that the user interacted with
        # We join these with the item profiles and user profiles
        
        # Build training feature matrix
        # For a ranker, we need positive examples (interacted) and negative examples.
        # For simplicity in this orchestrator, we assume the feature_engineer 
        # or the ranker itself handles negative sampling if not provided.
        # Assume labels are generated or already present in train_events
        # Ensure label_binary and label_multiclass are carried over
        cols_to_select = ['user_id', 'item_id']
        for col in ['label_binary', 'label_multiclass', 'group_id']:
            if col in train_events.collect_schema().names():
                cols_to_select.append(col)

        train_features = self.feature_engineer.engineer_features(
            candidate_items=train_events.select(cols_to_select).unique(),
            user_profile=None,
            item_profile=item_profile,
            interactions=train_events
        )

        val_features = self.feature_engineer.engineer_features(
            candidate_items=val_events.select(cols_to_select).unique(),
            user_profile=None,
            item_profile=item_profile,
            interactions=val_events
        )

        # 5. Train Ranker (Stage 3)
        logger.info("Step 5: Training ML Ranker...")
        self.ranker.fit(
            train_data=train_features
        )
        
        logger.info("Training Pipeline Completed Successfully.")

        return {
            "candidate_generator": self.candidate_generator,
            "ranker": self.ranker,
            "status": "success"
        }

    def save_models(self, output_dir: str):
        """Saves trained models to disk."""
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(self.candidate_generator, "save"):
            self.candidate_generator.save(os.path.join(output_dir, "candidate_generator.pkl"))
        if hasattr(self.ranker, "save"):
            self.ranker.save(os.path.join(output_dir, "ranker.pkl"))
        logger.info(f"Models saved to {output_dir}")
